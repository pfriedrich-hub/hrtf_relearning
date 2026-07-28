import multiprocessing as mp
import hrtf_relearning as hr
import datetime
import time
import slab
from hrtf_relearning.experiment.analysis.localization.localization_analysis import *
from hrtf_relearning.experiment.localization.localization_helpers.uso_generation import generate_uso
from hrtf_relearning.experiment.localization.localization_helpers.stimulus import make_gapped_pinknoise
from hrtf_relearning.experiment.localization.localization_helpers.make_sequence import make_sequence
from hrtf_relearning.experiment.misc import meta_motion
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from pynput import keyboard
from hrtf_relearning.utils import paths
logging.getLogger().setLevel('INFO')
ROOT = hr.PATH


class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject, hrir, condition, delta, ear=None, stim='noise',
                az_range=(0, 35), sector_size=(7, 14), mirror=False, settings=None):
        # make trial sequence and write to subject-
        date = datetime.datetime.now()
        date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}-{date.strftime("%M")}'

        self.settings = {'kind': 'sectors',
                         'azimuth_range': az_range, 'elevation_range': (-35, 35),
                         'sector_size': sector_size,
                         'targets_per_sector': 3, 'replace': False, 'min_distance': 20,
                         'gain': .2}
        if settings:  # per-run overrides (e.g. targets_per_sector, elevation_range, gain, exclude_midline)
            self.settings.update(settings)
        # alternative setting: play 3 times from each source in the hrir (works well for dome recorded hrirs)
        # self.settings = {'kind': 'standard', 'azimuth_range': (-60, 60), 'elevation_range': (-40, 40),
        #                  'targets_per_speaker': 3, 'min_distance': 10, 'gain': .2}
        self.subject = subject
        self.hrir = hrir
        self.condition = condition
        self.delta = delta
        self.ear = ear
        self.stim_kind = stim
        self.filename = f'{subject.id}_{condition}_d{delta}_{date}_{hrir.name}'
        # metadata
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar
        self.sound_path = paths.BINSIM_DIR / hrir.name / 'sounds'
        self.target = None

        # make sequence
        self.sequence = make_sequence(self.settings, self.hrir_sources)
        self.sequence.name = self.filename
        self.sequence.label = hrir.name
        self.sequence.hrir = hrir.name
        self.sequence.ear = ear
        self.sequence.mirrored = mirror
        self.sequence.stim = stim
        self.sequence.condition = condition
        self.sequence.shift_erb = delta
        # Full modification record embedded in the source SOFA (adds forwarded
        # kw / git hash on top of condition+delta above). Best-effort.
        try:
            from hrtf_relearning.hrtf.modify.edge_shift import read_modification_params
            _sid = hrir.name.split('_')[0]
            self.sequence.hrir_params = read_modification_params(
                paths.SOFA_DIR / _sid / f'{hrir.name}.sofa')
        except Exception:
            self.sequence.hrir_params = None

        self.ue_client = udp_client.SimpleUDPClient(UE_SEND_IP, UE_SEND_PORT)
        self.vr_pose_bridge = None
        self.pose_offset = numpy.array([0.0, 0.0, 0.0])

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def send_ue_state(self, state, target=None):
        self.ue_client.send_message("/loc/state", state)
        if target is not None:
            self.ue_client.send_message("/loc/target", [float(target[0]), float(target[1])])

    def run(self):
        # init pybinsim
        self.osc_client_1 = self._make_osc_client(port=10000)
        self.osc_client_2 = self._make_osc_client(port=10003)
        self.binsim_worker = mp.Process(target=self._binsim_stream, args=(self.hrir.name,))
        self.binsim_worker.start()

        self.vr_pose_bridge = VRPoseBridge(recv_port=UE_RECV_PORT)
        self.vr_pose_bridge.start()
        time.sleep(0.2)

        def calibrate_pose(self):
            if USE_VR_POSE:
                self.pose_offset = self.vr_pose_bridge.get_pose()
                self.ue_client.send_message("/loc/calibrated", 1)
            else:
                self.motion_sensor.calibrate()

        def get_head_pose(self):
            if USE_VR_POSE:
                pose = self.vr_pose_bridge.get_pose() - self.pose_offset
                return pose[:2]  # yaw, pitch
            return self.motion_sensor.get_pose()

        # # init motion sensor
        # self.motion_sensor = self.init_sensor()
        # time.sleep(.2)

        self.play_sound('beep')
        for self.target in self.sequence:
            self.wait_for_button('Look at the Center and press Enter')
            # self.motion_sensor.calibrate()

            self.send_ue_state("center")
            self.wait_for_button('Look at the Center and press Enter')
            self.calibrate_pose()

            self.send_ue_state("stimulus", target=self.target)
            self.play_trial()

            self.send_ue_state("response")

        self.subject.last_sequence = self.sequence
        self.sequence.response_errors = target_p(self.sequence, show=False)
        self.write()
        logging.info('Finished.')
        return

    def play_trial(self):
        # generate stimulus
        self.stim = self.make_stim()  # generate a new stim each trial
        self.stim.write(self.sound_path / 'localization.wav')
        # play stim
        self.play_stimulus()
        time.sleep(self.stim.duration)
        # get response
        self.wait_for_button()
        # response = self.motion_sensor.get_pose()
        response = self.get_head_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress:.1f}% | Target: {self.target} | Response: {response}')
        time.sleep(.25)
        self.sequence.add_response(numpy.array((response, self.target)))
        self.write()  # write to file

    def play_stimulus(self):
        # pose = self.motion_sensor.get_pose()
        pose = self.get_head_pose()
        relative_coords = self.target - pose  # mimic freefield setup
        # find the closest filter idx and send to pybinsim
        relative_coords[0] = (-relative_coords[0] + 360) % 360  # mirror and convert to HRTF convention [0 < az < 360]
        rel_target = numpy.array((relative_coords[0], relative_coords[1], self.hrir_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - self.hrir_sources, axis=1))
        rel_hrtf_coords = self.hrir_sources[filter_idx]
        self.osc_client_1.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                        0, 0, 0])
        logging.debug(f'Set filter for {self.hrir_sources[filter_idx]}')
        # play
        self.osc_client_2.send_message('/pyBinSimLoudness', self.settings['gain'])
        self.osc_client_2.send_message('/pyBinSimFile', str(self.sound_path / 'localization.wav'))
        time.sleep(.5)
        self.osc_client_2.send_message('/pyBinSimLoudness', 0)

    def play_sound(self, kind):
        logging.info(f'Playing {kind} sound')
        name = f'{kind}.wav'
        duration = slab.Sound(self.sound_path / name).duration
        self.osc_client_2.send_message('/pyBinSimLoudness', self.settings['gain'])
        self.osc_client_2.send_message('/pyBinSimFile', str(self.sound_path / name))
        time.sleep(duration)
        self.osc_client_2.send_message('/pyBinSimLoudness', 0)

    @staticmethod
    def _make_osc_client(port, ip='127.0.0.1'):
        return udp_client.SimpleUDPClient(ip, port)

    @staticmethod
    def _binsim_stream(hrir_name):
        import pybinsim
        pybinsim.logger.setLevel(logging.ERROR)
        binsim = pybinsim.BinSim(paths.BINSIM_DIR / hrir_name / f'{hrir_name}_test_settings.txt')
        binsim.stream_start()  # run binsim loop

    @staticmethod
    def init_sensor():
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    def make_stim(self):
        if self.stim_kind == 'noise':
            stim = make_gapped_pinknoise(level=80)
        elif self.stim_kind == 'uso':
            stim = generate_uso(samplerate=self.hrir.samplerate)
        else: raise ValueError('stim must be "noise" or "uso".')
        stim.level = 80
        return stim

    @staticmethod
    def wait_for_button(msg=None):
        if msg: print(msg)
        def on_press(key):
            if key == keyboard.Key.enter:
                listener.stop()  # stop listening once Enter is pressed
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()  # block until listener.stop() is called
import threading
from pythonosc import dispatcher, osc_server, udp_client

USE_VR_POSE = True
UE_SEND_IP = "127.0.0.1"
UE_RECV_PORT = 7001   # Unreal -> Python
UE_SEND_PORT = 7000   # Python -> Unreal


class VRPoseBridge:
    def __init__(self, recv_ip="127.0.0.1", recv_port=7001):
        self.latest_pose = numpy.array([0.0, 0.0, 0.0])  # yaw, pitch, roll
        self._lock = threading.Lock()

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/vr/headpose", self._on_headpose)

        self.server = osc_server.ThreadingOSCUDPServer(
            (recv_ip, recv_port), self.dispatcher
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _on_headpose(self, address, yaw, pitch, roll, *args):
        with self._lock:
            self.latest_pose = numpy.array([float(yaw), float(pitch), float(roll)])

    def get_pose(self):
        with self._lock:
            return self.latest_pose.copy()

def run(subject_id, condition, delta, ear=None, hp='DT990', stim='noise',
       az_range=(0, 35), sector_size=(7, 14), mirror=False, settings=None):
    """Run one localization test block for one subject/condition.

    subject_id : e.g. 'AS'.
    condition  : 'baseline', or an hrtf.modify.edge_shift condition name
        ('rising', 'falling', 'whole'). Resolves the SOFA to load via
        hrtf2binsim's name-based lookup: 'baseline' -> '{subject_id}.sofa',
        else '{subject_id}_{condition}.sofa', both under
        data/hrtf/sofa/{subject_id}/. The SOFA must already exist -- write it
        once via hrtf.modify.edge_shift.save_condition_sofa before
        calling run() (see experiment/protocols/cue_shift.py).
    delta      : the ERB shift magnitude used to build this condition's SOFA
        (edge_shift shift_erb). Recorded in the sequence filename/metadata for
        downstream analysis grouping; does not regenerate the SOFA.
    ear, hp, stim, az_range, sector_size, mirror : per-run overrides of the
        old module-level EAR/HP/STIM/AZ_RANGE/SECTOR_SIZE/MIRROR settings.
    settings   : optional dict merged into the sequence settings, e.g.
        {'targets_per_sector': 1, 'elevation_range': (-35, 35),
         'min_distance': 20, 'exclude_midline': True, 'gain': .2}.

    Returns the written sequence (also available via subject.localization).
    """
    hrir_name = subject_id if condition in (None, 'baseline') else f'{subject_id}_{condition}'
    hrir_settings = dict(name=hrir_name, ear=ear, mirror=mirror,
                         reverb=True, drr=20, hp_filter=True, hp=hp,
                         convolution="cuda", storage="cuda")
    hrir = hrtf2binsim(hrir_settings, overwrite=True)
    subject = hr.Subject(subject_id)

    set_windows_volume(50)
    loc_test = Localization(subject, hrir, condition, delta, ear=ear, stim=stim,
                            az_range=az_range, sector_size=sector_size, mirror=mirror,
                            settings=settings)
    loc_test.run()

    sequence = subject.localization[loc_test.filename]
    plot_dir = paths.subject_plot_dir(subject.id)
    plot_localization(sequence, report_stats=['azimuth', 'elevation'], filepath=plot_dir)
    plot_elevation_response(sequence, filepath=plot_dir)
    plt.show()
    return sequence


if __name__ == "__main__":
    run("AS", "baseline", 0)
