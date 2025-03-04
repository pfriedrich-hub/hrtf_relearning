import argparse
import logging
import time
from pythonosc import udp_client
from experiment.misc import meta_motion
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('INFO')
import pybinsim
import datetime
date = datetime.datetime.now()
date = f'{date.strftime("%d")}_{date.strftime("%m")}'
data_dir = Path.cwd() / 'data'
from experiment.Subject import Subject

subject_id = 'test'
hrtf_name ='KU100_HRIR_L2702'
slab.set_default_samplerate(slab.HRTF(data_dir / 'hrtf' / 'sofa' / f'{hrtf_name}.sofa').samplerate)

class Localization:
    def __init__(self, subject_id, hrtf_name):
        # metadata
        self.filename = subject_id + '_' + date
        self.subject = Subject(subject_id)
        self.hrtf_sources = slab.HRTF(data_dir / 'hrtf' / 'sofa' / f'{hrtf_name}.sofa').sources.vertical_polar
        self.stim_path = data_dir / 'hrtf' / 'wav' / hrtf_name / 'sounds' / 'noise_burst.wav'
        # make trial sequence
        self.sequence = self._make_sequence(10, (-52.5, 52.5), (-37.5, 37.5), 20)
        self.target = None

        # init pybinsim
        self.osc_client = self._init_osc_client()
        self.binsim = self.init_pybinsim()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

    def run(self):
        # enable audio
        self.binsim.config.configurationDict['loudnessFactor'] = 0.5
        for target in self.sequence:
            # center head and calibrate sensor
            input('Look at the center and press enter to calibrate sensor')
            self.motion_sensor.calibrate()
            # generate and play stim
            self.play_trial(target)
            # get head pose response
            self.get_pose()
        # save trial sequence
        self.sequence.save_pickle('test_sequence.pkl')
        # plot
        return

    def play_trial(self, target):
        # generate stimulus
        noise = slab.Sound.pinknoise(duration=0.025, level=90)
        noise = noise.ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise)
        stim.ramp('both', 0.01)
        stim.write(self.stim_path)

        # set filter
        pose = self.motion_sensor.get_pose()
        # set distance for play_session
        relative_coords = target - pose
        # find the closest filter idx and send to pybinsim
        relative_coords[0] = (-relative_coords[0] + 360) % 360  # mirror and convert to HRTF convetion [0 < az < 360]
        rel_target = numpy.array((relative_coords[0], relative_coords[1], self.hrtf_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - self.hrtf_sources, axis=1))
        self.osc_client.send_message('/pyBinSim', [0, int(filter_idx), 0, 0, 0, 0, 0])
        logging.info(f'set filter for {self.hrtf_sources[filter_idx]}')

        # play
        self.osc_client.send_message('/pyBinSimFile', str(self.stim_path))
        # self.binsim.config.configurationDict['loudnessFactor'] = .5
        logging.info(f'playing sound {self.stim_path}')
        time.sleep(stim.duration)
        # self.binsim.config.configurationDict['loudnessFactor'] = 0

    @staticmethod
    def _init_osc_client():
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
        return udp_client.SimpleUDPClient(args.ip, args.port)

    def init_pybinsim(self):
        binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / hrtf_name / f'{hrtf_name}_settings.txt')
        pybinsim.logger.setLevel(logging.INFO)
        binsim.soundHandler.loopSound = False
        binsim.config.configurationDict['loudnessFactor'] = 0
        self.osc_client.send_message('/pyBinSimFile', str(self.stim_path))
        binsim.stream_start()
        return binsim

    @staticmethod
    def _make_sequence(n_trials, az_range, ele_range, min_dist):
        """
        Create a sequence of n_trials target locations
        with more than min_dist angular distance between successive targets
        """
        logging.info('Setting up trial sequence.')
        sequence = []
        target = (0, 0)
        for i in range(n_trials):
            while True:
                prev_tar = target
                next_tar = [numpy.random.randint(az_range[0], az_range[1]),
                            numpy.random.randint(ele_range[0], ele_range[1])]
                if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
                    break
            sequence.append(next_tar)
        return slab.Trialsequence(sequence)

    def init_sensor(self):
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    def get_pose(self):
        input('Aim at sound source and press Enter to confirm.')
        self.osc_client.send_message('/pyBinSimFile', str(data_dir / 'hrtf' / 'wav' /
                                                          hrtf_name / 'sounds' / 'beep.wav'))
        time.sleep(.5)
        self.motion_sensor.get_pose()

if __name__ == "__main__":
    make_wav(hrtf_name)
    self = Localization(subject_id, hrtf_name)
    self.run()


"""

    def play_trial(self):
        # motion_sensor.calibrate()
        # pose = self.motion_sensor.get_pose()
        # distance-to-interval parameters
        max_interval = 250  # interval duration at max distance in ms
        max_distance = numpy.linalg.norm(numpy.subtract([0,0],[az_range[0], ele_range[0]]))  # max distance in degrees
        threshold = target_size  # the distance threshold in degrees below which interval duration should be zero
        steepness = 50  # controls how gradually the interval duration increases

        # game variables
        time_on_target = 0
        count_down = False

        # start
        sensor_state.value = 2  # start sensor tracking
        binsim_state.value = 2  # unmute sound, start pulsing
        running = True
        trial_start = time.time()

        while time.time() - trial_start < trial_time:
            dist = distance.value
            logging.debug(f'play trial: got distance value {dist}')
            if dist <= threshold:  # get interval duration from distance
                interval = 0
            else:
                norm_dist = (dist - threshold) / (max_distance - threshold)
                norm_dist = numpy.clip(norm_dist, 0, 1)  # Keep in range [0, 1]
                interval = int(max_interval * (numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness))) + 100
            interval = max(0, interval)  # ensure non-negative interval
            logging.debug(f'play trial: set interval value {interval}')
            pulse_interval.value = interval
            if dist < target_size:
                if not count_down:
                    time_on_target, count_down = time.time(), True
            else:
                time_on_target, count_down = time.time(), False
            # goal condition
            if count_down and time.time() > time_on_target + target_time:
                if time.time() - trial_start <= 3:
                    play_sound('coins', osc_client, pulse_interval, binsim_state)
                    score.value += 2
                else:
                    play_sound('coin', osc_client, pulse_interval, binsim_state)
                    score.value += 1
                binsim_state.value = 1  # stop pulse stream
                sensor_state.value = 1  # stop sensor tracking
            logging.debug(f'play trial: trial time {time.time() - trial_start}')
            time.sleep(0.005)

    def play_session(game_time, trial_time, target_size, target_time, az_range, ele_range):
        # shared variables
        osc_client = init_osc_client()  # binsim communication
        sensor_state = mp.Value("i", 0)  # sensor_tracking control
        binsim_state = mp.Value("i", 0)  # pulse_stream control
        score = mp.Value("i", 0)  # score counter (play_trial)
        target = mp.Array("i", [0, 0])  # get_target sets target - used by sensor_tracking
        distance = mp.Value("f", 0)  # sensor_tracking updates distance and sends to play_trial
        pulse_interval = mp.Value("i", 0) # play_trial updates interval and sends to pulse_stream
        # p_distance, c_distance = mp.Pipe()  # sensor_tracking updates distance and sends to play_trial
        # p_interval, c_interval = mp.Pipe() # play_trial updates interval and sends to pulse_stream

        # connect sensor and wait for process to run
        tracking_worker = mp.Process(target=head_tracking, args=(distance, target, osc_client, sensor_state))
        tracking_worker.start()
        while sensor_state.value == 0:  # wait for sensor to initialize
            time.sleep(0.1)

        # set up pulse stream (init pybinsim)
        pulse_worker = mp.Process(target=pulse_stream, args=(pulse_interval, binsim_state))
        pulse_worker.start()
        while binsim_state.value == 0:  # wait for pyaudio to initialize
            time.sleep(.01)

        scores = []
        start_time = time.time()
        while time.time() - start_time < game_time:
            set_target(az_range, ele_range, target, min_dist=30)
            trial_worker = mp.Process(target=play_trial, args=(distance, pulse_interval, score, binsim_state, sensor_state,
                                                               osc_client, trial_time, target_size, target_time))
            time.sleep(.5)
            input("Press Enter to play.")
            trial_worker.start()
            trial_worker.join()
            trial_worker.terminate()
            scores.append(score.value)
            print(f'Trial complete: {score.value} points')
            time.sleep(.5)
            sensor_state = 0 # recalibrate sensor
        # end
        play_sound('buzzer', osc_client, pulse_interval)
        print(f'Game Over! Total Score: {sum(scores)}')
        pulse_worker.terminate()
        pulse_worker.join()
        tracking_worker.terminate()
        tracking_worker.join()

    if __name__ == "__main__":
        if not (data_dir / 'wav' / filename).exists():
            hrtf2wav(f'{filename}.sofa')
        play_session(game_time, trial_time, target_size, target_time, tuple(az_range), tuple(ele_range))



"""

        # parser = argparse.ArgumentParser(description="Run the auditory training experiment.")
        # parser.add_argument("--game_time", type=int, default=90, help="Total session duration in seconds")
        # parser.add_argument("--trial_time", type=int, default=10, help="Time limit per trial in seconds")
        # parser.add_argument("--target_size", type=int, default=2, help="Size of the target zone")
        # parser.add_argument("--target_time", type=float, default=1, help="Time required to hold gaze on target")
        # parser.add_argument("--az_range", type=int, nargs=2, default=[-45, 45], help="Azimuth range for target selection")
        # parser.add_argument("--ele_range", type=int, nargs=2, default=[-45, 45], help="Elevation range for target selection")
        # parser.add_argument("--filename", type=str, default='KU100_HRIR_L2702', help="HRTF filename for PyBinSim")
        # args = parser.parse_args()
        #
