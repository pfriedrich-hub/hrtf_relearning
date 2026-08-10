import multiprocessing as mp
import hrtf_relearning as hr
import datetime
import time
import slab
from hrtf_relearning.experiment.analysis.localization.localization_analysis import *
from hrtf_relearning.experiment.localization.localization_helpers.uso_generation import generate_uso
from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise, RMS_TILT, RMS_CUE, RIPPLE_CUE_MAX)
from hrtf_relearning.experiment.localization.localization_helpers.make_sequence import make_sequence
from pythonosc import udp_client
from hrtf_relearning.experiment.misc import meta_motion
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from pynput import keyboard
from hrtf_relearning.utils import paths

logging.getLogger().setLevel('INFO')
ROOT = hr.PATH

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors0
    """
    def __init__(self, subject, hrir_settings, loc_settings=None):
        self.subject = subject
        date = datetime.datetime.now()
        date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}-{date.strftime("%M")}'

        # Build / refresh binsim files — hp filter loaded automatically from disk
        hrir = hrtf2binsim(hrir_settings, overwrite=True)

        ear    = hrir_settings.get('ear', None)
        mirror = hrir_settings.get('mirror', False)
        hp     = hrir_settings.get('hp', None)

        self.filename = subject.id + '_' + date + '_' + hrir.name
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar
        self.hrir_name = hrir.name
        self.samplerate = hrir.samplerate
        self.sound_path = paths.BINSIM_DIR / hrir.name / 'sounds'
        self.target = None

        if loc_settings is None:
            loc_settings = {
                'kind': 'sectors',
                'azimuth_range': (-180, 180), 'elevation_range': (-35, 35),
                'sector_size': (14, 14),
                'targets_per_sector': 3, 'replace': False, 'min_distance': 20,
                'gain': .2,
                'stim': 'noise',
            }
        self.stim_type = loc_settings.get('stim', 'noise')
        self.settings = loc_settings

        self.sequence = make_sequence(self.settings, self.hrir_sources)
        self.sequence.name = self.filename
        self.sequence.label = hrir.name
        self.sequence.hrir = hrir.name
        self.sequence.ear = ear
        # how the non-listening ear was treated in a monaural run ('flat' delta
        # vs 'envelope'); None when binaural. hrir.name also carries it, but
        # record it explicitly so old and new runs are comparable at a glance.
        self.sequence.other_ear = hrir_settings.get('other_ear', 'flat') if ear else None
        self.sequence.env_n_keep = hrir_settings.get('env_n_keep', None) if ear else None
        self.sequence.mirrored = mirror
        self.sequence.stim = self.stim_type
        # per-trial source-spectrum recipe, appended by make_stim(). Empty for
        # older runs; for 'ripple' it holds the DCT coefficients of that
        # trial's spectral shape, so the stimulus is exactly reconstructible.
        self.sequence.stim_params = []
        self.sequence.stim_settings = self.settings.get('stim_settings', {}) or {}
        self.sequence.hp = hp
        # Record exactly which modification produced this HRTF (read back from
        # the SOFA's embedded params), so a run can always be traced to what was
        # done to it — the name alone is ambiguous (see FD 12:13 vs 12:21).
        self.sequence.hrir_params = self._read_hrir_params(hrir_settings)

    @staticmethod
    def _read_hrir_params(hrir_settings):
        """Modification-params dict embedded in the source SOFA, or None.

        None for an unmodified/free-field HRTF or a SOFA written before param
        embedding. Best-effort: never raise, so it can't block a test."""
        try:
            from hrtf_relearning.hrtf.modify.edge_shift import read_modification_params
            sofa_name = hrir_settings.get('name')
            if not sofa_name:
                return None
            subject_id = hrir_settings.get('subject_id', sofa_name.split('_')[0])
            return read_modification_params(
                paths.SOFA_DIR / subject_id / f'{sofa_name}.sofa')
        except Exception:
            return None

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def run(self):
        # init pybinsim
        self.osc_client_1 = self._make_osc_client(port=10000)
        self.osc_client_2 = self._make_osc_client(port=10003)
        self.binsim_worker = mp.Process(target=self._binsim_stream, args=(self.hrir_name,))
        self.binsim_worker.start()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

        try:
            self.play_sound('beep')
            for self.target in self.sequence:
                self.wait_for_button('Look at the Center and press Enter')
                self.motion_sensor.calibrate()
                self.play_trial()  # generate and play stim, get pose response and write to file
            self.subject.last_sequence = self.sequence
            self.sequence.response_errors = target_p(self.sequence, show=False)
            self.write()
            logging.info('Finished.')
            plot_dir = paths.subject_plot_dir(self.subject.id)
            plot_elevation_response(self.sequence, filepath=plot_dir)
            plot_localization(self.sequence, report_stats=['elevation', 'azimuth'], filepath=plot_dir)
        finally:
            self.motion_sensor.halt()
            try:  # mute before killing so the audio stream stops cleanly
                self.osc_client_2.send_message('/pyBinSimLoudness', 0)
                time.sleep(0.1)
            except Exception:
                pass
            self.binsim_worker.terminate()
            self.binsim_worker.join(timeout=3)
            if self.binsim_worker.is_alive():  # SIGTERM wasn't enough — force-kill
                self.binsim_worker.kill()
                self.binsim_worker.join()

    def play_trial(self):
        # generate stimulus
        self.stim = self.make_stim()  # generate a new stim each trial
        self.stim.write(self.sound_path / 'localization.wav')
        # play stim
        self.play_stimulus()
        time.sleep(self.stim.duration)
        # get response
        self.wait_for_button()
        response = self.motion_sensor.get_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress:.1f}% | Target: {self.target} | Response: {response}')
        time.sleep(.25)
        self.sequence.add_response(numpy.array((response, self.target)))
        self.write()  # write to file

    def play_stimulus(self):
        pose = self.motion_sensor.get_pose()
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
        """Build this trial's stimulus and record what it was.

        'noise'  fixed-spectrum gapped pinknoise (the training stimulus)
        'ripple' the same burst train with a NEW random smooth spectral shape
                 every trial -- use this to ask whether the elevation map
                 survives a source spectrum that moves, i.e. whether learning
                 was a spectral-to-spatial recalibration or a timbre lookup
        'uso'    Mitsuhashi composite; `uso_base` pins the base texture

        The per-trial recipe goes into sequence.stim_params, so the source
        spectrum of every trial can be reconstructed offline.
        """
        stim_settings = self.settings.get('stim_settings', {}) or {}
        if self.stim_type == 'noise':
            stim, params = make_gapped_pinknoise(level=80), {'kind': 'noise'}
        elif self.stim_type == 'ripple':
            stim, params = make_rippled_pinknoise(
                level=80,
                rms_tilt=stim_settings.get('rms_tilt', RMS_TILT),
                rms_cue=stim_settings.get('rms_cue', RMS_CUE),
                flat_rms=stim_settings.get('flat_rms', None),
                ripple_max=stim_settings.get('ripple_max', RIPPLE_CUE_MAX))
        elif self.stim_type == 'uso':
            stim, params = generate_uso(samplerate=self.samplerate,
                                        base=stim_settings.get('uso_base', None),
                                        return_params=True)
        else:
            raise ValueError('stim_type must be "noise", "ripple" or "uso".')
        stim.level = 80
        self.sequence.stim_params.append(params)
        return stim

    @staticmethod
    def wait_for_button(msg=None):
        if msg: print(msg)
        def on_press(key):
            if key == keyboard.Key.enter:
                listener.stop()  # stop listening once Enter is pressed
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()  # block until listener.stop() is called

if __name__ == "__main__":

    # --- SETTINGS ---
    _SUBJECT_ID = "CO"
    _HRIR_NAME = "CO_synth"  # 'KU100', 'kemar', etc.
    _HP = 'DT990'
    _EAR = None  # None (binaural), 'left', or 'right'
    _MIRROR = False  # True to swap left/right spectral cues
    _AZ_RANGE = (-35, 35)
    _SECTOR_SIZE = (14, 14)
    _STIM = 'noise'

    # --- localization / sequence settings ---
    _loc_settings = {
        'kind': 'sectors',  # 'standard' or 'sectors'
        'azimuth_range': _AZ_RANGE,  # (-1, 1) for midline-only, (-180, 180) for full sphere
        'elevation_range': (-35, 35),
        'targets_per_speaker': 3,
        'targets_per_sector': 3,
        'min_distance': 20,  # min angular distance between successive targets (°)
        'gain': .2,
        'stim': _STIM,  # 'noise' or 'uso'
        'sector_size': _SECTOR_SIZE,
        'replace': False
    }

    # --- HRIR / binsim settings ---
    _hrir_settings = {
        'name': _HRIR_NAME,
        'subject_id': _SUBJECT_ID,
        'ear': _EAR,  # None (binaural), 'left', or 'right'
        'mirror': _MIRROR,  # True to swap left/right spectral cues
        'reverb': True,
        'drr': 20,
        'hp_filter': True,
        'hp': _HP,
        'convolution': 'cpu',
        'storage': 'cpu',
    }

    _subject = hr.Subject(_SUBJECT_ID)

    loc_test = Localization(_subject, _hrir_settings, loc_settings=_loc_settings)
    loc_test.run()
    sequence = _subject.localization[loc_test.filename]
    plot_dir = paths.subject_plot_dir(_subject.id)
    plot_localization(sequence, report_stats=['azimuth', 'elevation'], filepath=plot_dir)
    plot_elevation_response(sequence, filepath=plot_dir)
    plt.show()
