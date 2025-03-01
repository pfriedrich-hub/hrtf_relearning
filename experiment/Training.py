import argparse
import logging
import multiprocessing as mp
import time
from pythonosc import udp_client
from experiment.misc import meta_motion
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('INFO')

filename ='KU100_HRIR_L2702'
data_dir = Path.cwd() / 'data' / 'hrtf'

game_time  = 180
trial_time = 80
target_size = 3
target_time = 1
az_range = (-30, 30)
ele_range = (-30, 30)

def init_osc_client():
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
    return udp_client.SimpleUDPClient(args.ip, args.port)

def set_target(az_range, ele_range, target, min_dist=30):
    while True:
        prev_tar = target[:]
        next_tar = [numpy.random.randint(az_range[0], az_range[1]),
                  numpy.random.randint(ele_range[0], ele_range[1])]
        if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
            target[:] = next_tar
            break
    logging.info(f'Set Target to {next_tar}.')

def play_sound(wav_name, osc_client, pulse_interval, binsim_state):
    pulse_interval.value = 0  # stop pulsing to play game sound
    binsim_state.value = 2  # make sure that stream is not muted
    time.sleep(0.2)
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / f'{wav_name}.wav'))
    time.sleep(1.5)
    binsim_state.value = 1  # mute stream after playing sound
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / 'pinknoise.wav'))

# sub processes
def head_tracking(distance, target, osc_client, sensor_state):
    hrtf_sources = slab.HRTF(data_dir / 'sofa' / f'{filename}.sofa').sources.vertical_polar
    # init motion sensor
    device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    while True:
        if sensor_state.value == 0:  # init flag
            logging.info('Calibrating sensor..')
            time.sleep(1)
            motion_sensor.calibrate()
            sensor_state.value = 1  # calibrated flag
        if sensor_state.value == 2:  # head tracking flag
            pose = motion_sensor.get_pose()
            # set distance for play_session
            relative_coords = target[:] - pose
            distance.value = numpy.linalg.norm(relative_coords)
            logging.debug(f'head tracking: set distance value {distance.value}')
            # find the closest filter idx and send to pybinsim
            relative_coords[0] = (-relative_coords[0] + 360) % 360 # mirror and convert to HRTF convetion [0 < az < 360]
            rel_target = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target-hrtf_sources, axis=1))
            osc_client.send_message('/pyBinSim', [0, int(filter_idx), 0, 0, 0, 0, 0])
            logging.debug(f'head tracking: filter coords: {hrtf_sources[filter_idx]}')
        time.sleep(0.005)

def pulse_stream(pulse_interval, binsim_state):
    import pybinsim
    binsim = pybinsim.BinSim(data_dir / 'wav' / filename / f'{filename}_settings.txt')
    pybinsim.logger.setLevel(logging.WARNING)
    binsim.soundHandler.loopSound = True
    binsim.stream_start()

    def smooth_transition(target_value, step=0.2, delay=0.004):
        current_value = binsim.config.configurationDict['loudnessFactor']
        while abs(current_value - target_value) > step:
            current_value += step if target_value > current_value else -step
            binsim.config.configurationDict['loudnessFactor'] = round(current_value, 2)
            time.sleep(delay)
        binsim.config.configurationDict['loudnessFactor'] = target_value

    while not binsim.stream._is_running:
        time.sleep(0.1)
    binsim_state.value = 1  # set flag for completed binsim init

    while True:
        if binsim_state.value == 1:  # Mute sound, stop updating pulse interval
            smooth_transition(0)
        else:  # play
            interval = pulse_interval.value
            logging.debug(f'pulse stream: got interval value {interval}')
            if interval == 0:  # continuous sound
                smooth_transition(0.5)
            else:  # pulse sound
                smooth_transition(0.5)
                time.sleep(interval / 1000)
                smooth_transition(0)
                time.sleep(interval / 1000)
        time.sleep(0.005)


def play_trial(distance, pulse_interval, score, binsim_state, sensor_state, osc_client,
               trial_time, target_size, target_time):
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
        if dist <= threshold:  # interval duration from distance
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
