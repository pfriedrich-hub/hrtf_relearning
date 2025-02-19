import matplotlib
import threading
matplotlib.use('TkAgg')
from experiment.misc.Pulse_Stream import Pulse_Stream
from hrtf.processing.hrtf2wav import hrtf2wav
import freefield
from pathlib import Path
import argparse
from pythonosc import udp_client
import time
import numpy
import slab
data_dir = Path.cwd() / 'data'
fs = 44100
slab.set_default_samplerate(fs)

filename = 'KU100_HRIR_L2702.sofa'
# todo adjust pulse train to make it intuitive, get a good HRIR for testing,
#  check if one sensor calibration at the beginning is sufficient

class Training:
    def __init__(self, filename='kemar.sofa', target_size=5, target_time=.5, game_time=180, trial_time=30,
                 az_range=(-30, 30), ele_range=(-30, 30)):

        # general parameters
        self.filename = Path(filename)
        self.hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / filename)
        self.target_size = target_size
        self.target_time = target_time
        self.game_time = game_time
        self.trial_time = trial_time
        self.az_range = az_range
        self.ele_range = ele_range
        self.target = None
        self.scores = []

        # init pybinsim streamer and osc messenger
        self.osc_client = self._make_osc_client()
        self.pulse_stream = Pulse_Stream(self.filename.stem)
        self.filter_idx = 0  # filter index in the initial osc message

        # load sounds
        self.sounds = {
                        'coins': slab.Sound.read(data_dir / 'sounds' / 'coins.wav'),
                        'coin': slab.Sound.read(data_dir / 'sounds' / 'coin.wav'),
                        'buzzer': slab.Sound.read(data_dir / 'sounds' / 'buzzer.wav'),
                        }
        for sound, key in zip(self.sounds.values(), self.sounds.keys()):
            self.sounds[key] = sound.resample(fs)
            self.sounds[key].level = 75

    def __repr__(self):
        return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        # init sensor
        freefield.initialize(setup='dome', default=None, device=None, sensor_tracking=True)
        freefield.SENSOR.set_fusion_mode('NDOF')
        while True:
            self.training_session()
            self._wait_for_button('Press Enter to play again.')

    def training_session(self):
        self.game_over, self.score = False, 0        # reset countdown and score
        self.game_start = time.time()
        while not self.game_over:  # loop over trials until end time has passed
            trial_prep = time.time()  # time between trials
            self.set_target()  # get next target
            self._wait_for_button('Press Enter to start.')
            freefield.calibrate_sensor(led_feedback=False, button_control=False)
            self.game_start += time.time() - trial_prep  # count time only while playing
            self.play_trial()
            self.scores.append(self.score)
            print(f'Run {len(self.scores)}: {self.score} points')

    def play_trial(self):
        self.get_distance()
        # filter selection thread: select filter based on relative headpose and osc message to pybinsim
        self.filter_thread = threading.Thread(target=self.set_filter, args=())
        # pulse thread: convert distance to pulse interval and pass to pulse_stream
        self.pulse_thread = threading.Thread(target=self.set_pulse, args=())

        # start trial
        self.filter_thread.start()
        self.pulse_thread.start()
        # self.pulse_stream.interval_duration = 100  # test loops

        self.trial_start = time.time()  # get trial start time
        time_on_target = 0
        count_down = False  # condition for counting time on target
        while True:  # main thread: control game behavior by pose-to-target distance
            self.get_distance()
            print('distance: azimuth %.1f, elevation %.1f, total %.2f'
                  % (self.relative_coords[0], self.relative_coords[1], self.distance), end="\r", flush=True)
            # check for hits
            if self.distance < self.target_size:
                if not count_down:  # start counting time as longs as pose matches target
                    time_on_target, count_down = time.time(), True
            else:
                time_on_target, count_down = time.time(), False  # reset timer if pose no longer matches target
            # end trial if goal conditions are met
            if time.time() > time_on_target + self.target_time:
                if time.time() - self.trial_start <= 3:
                    points = 2
                    self.end_trial('coins')
                else:
                    points = 1
                    self.end_trial('coin')
                self.score += points
                print('Score! %i' % points)
                break
            # end trial after 10 seconds
            if time.time() > self.trial_start + self.trial_time:  #  todo check which variables need to be stored in self
                self.end_trial('buzzer')
                # print('Time out')
                break
            # end training sequence if game time is up
            if time.time() > self.game_start + self.game_time:
                self.game_over = True
                self.end_trial('buzzer')
                break
            else:
                continue
            time.sleep(.001)

    def get_distance(self):
        self.headpose = freefield.get_head_pose()
        self.relative_coords = self.headpose - self.target
        self.distance = numpy.linalg.norm(self.relative_coords)

    def set_filter(self):
        thread = threading.currentThread()
        while getattr(thread, "run", True):
            # convert coordinates to physics / HRTF convention
            if self.relative_coords[0] > 0:
                self.relative_coords[0] = 360 - self.relative_coords[0]
            elif self.relative_coords[0] < 0:
                self.relative_coords[0] *= -1
            # find idx of the nearest filter in the hrtf
            polar = numpy.array((self.relative_coords[0], self.relative_coords[1],
                                 self.hrtf.sources.vertical_polar[0,2]))
            filter_coords = self.hrtf._vertical_polar_to_cartesian(polar[numpy.newaxis, :])
            # filter_coords = self.hrtf._get_coordinates((self.relative_coords[0], self.relative_coords[1],
            #                 self.hrtf.sources.vertical_polar[0,2]), 'spherical').cartesian
            distances = numpy.sqrt(((filter_coords - self.hrtf.sources.cartesian) ** 2).sum(axis=1))
            next_idx = int(numpy.argmin(distances))  # todo check whether this selects the correct filter
            if next_idx != self.filter_idx:
                # change filter
                filter_msg = [0, next_idx, 0, 0, 0, 0, 0]
                self.osc_client.send_message('/pyBinSim', filter_msg)
                self.filter_idx = next_idx
            time.sleep(.001)

    def set_pulse(self):
        thread = threading.currentThread()
        # maximal pulse interval in ms
        max_interval = 500
        # max displacement from center
        max_distance = numpy.linalg.norm(numpy.min((self.az_range, self.ele_range), axis=1) - numpy.array((0, 0)))
        while getattr(thread, "run", True):
            # scale distance with maximal distance
            interval_scale = (self.distance - 2 + 1e-9) / max_distance
            # scale interval logarithmically with distance
            interval = max_interval * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
            self.pulse_stream.interval_duration = interval
            time.sleep(.001)

    def set_target(self, min_dist=45):
        target = (numpy.random.randint(self.az_range[0], self.az_range[1]),
                  numpy.random.randint(self.ele_range[0], self.ele_range[1]))
        if self.target:  # check if target is at least min_dist away from previous target
            diff = numpy.diff((target, self.target), axis=0)[0]
            euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
            while euclidean_dist < min_dist:
                target = (numpy.random.randint(self.az_range[0], self.az_range[1]),
                          numpy.random.randint(self.ele_range[0], self.ele_range[1]))
                diff = numpy.diff((target, self.target), axis=0)[0]
                euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
        print('\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
        self.target = target

    def end_trial(self, sound='buzzer'):
        self.pulse_stream.interval_duration = -1  # pause pybinsim pyaudio stream
        self.filter_thread.run = False
        self.pulse_thread.run = False
        self.sounds[sound].play()  # play end sound #todo play game sounds from target locations
        time.sleep(self.sounds[sound].duration)

    @staticmethod
    def _make_osc_client():
        # Create OSC client
        host = '127.0.0.1'
        mode = 'client'
        ip = '127.0.0.1'
        port = 10000
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default=host)
        parser.add_argument("--mode", default=mode)
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
        return osc_client

    @staticmethod
    def _stop_game():
        freefield.halt()

    @staticmethod
    def _wait_for_button(*msg, button=''):
        response = None
        while response != button:
            if msg: response = input(msg)
            else: response = input('Waiting for button.')
        return response

if __name__ == "__main__":
    if not (data_dir / 'hrtf' / 'wav' / str(Path(filename).stem)).exists():
        hrtf2wav(filename)
    self = Training(filename)
    self.run()