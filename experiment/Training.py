import argparse
import logging
import multiprocessing as mp
import time
import numpy
from pythonosc import udp_client
# from experiment.misc import meta_motion
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('INFO')

data_dir = Path.cwd() / 'data' / 'hrtf'
fs = 44100
slab.set_default_samplerate(fs)

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

def play_sound(wav_name, osc_client, pulse_interval):
    pulse_interval[0] = 0  # Temporarily set pulse interval to 0 to play game sound
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / 'sounds' / f'{wav_name}.wav'))
    time.sleep(1)
    pulse_interval[0] = -1  # Restore mute after sound  # Play 'coin' sound
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / 'sounds' / 'pinknoise.wav'))

# sub processes
def head_tracking(distance, target, osc_client, filter_name, sensor_state):
    hrtf_sources = slab.HRTF(data_dir / 'sofa' / f'{filter_name}.sofa').sources.vertical_polar

    # init motion sensor
    # device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
    # state = meta_motion.State(device)
    # motion_sensor = meta_motion.Sensor(state)
    # logging.info('Calibrating sensor..')
    # time.sleep(.1)
    # motion_sensor.calibrate()

    sensor_state[0] = 1  # signal other processes that sensor is initialized
    logging.debug('motion sensor running')

    while not sensor_state[0] == 2:
        time.sleep(0.1)

    # start = time.time()
    # while True:
        # headpose = motion_sensor.get_pose()

    # test
    for az, ele in zip(numpy.linspace(-30, target[0], 1000), numpy.linspace(-30, target[1], 1000)):
        headpose = numpy.array((az, ele))
        time.sleep(.01)

        relative_coords = headpose - target[:]
        distance[0] = numpy.linalg.norm(relative_coords)  # set distance for play_session
        logging.debug(f'head tracking: set distance value {distance[0]}')

        if relative_coords[0] < 0:  # todo setup sensor for 0-360° az values (freefield)
            relative_coords[0] = 360 + relative_coords[0]  # convert to (0 < az < 360)
        # relative_coords[0] *= -1  # mirror for counter-clockwise az increase (HRTF (physics) convention)

        # find the closest filter idx and send to pybinsim
        rel_target = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target-hrtf_sources, axis=1))
        osc_client.send_message('/pyBinSim', [0, int(filter_idx), 0, 0, 0, 0, 0])
        logging.debug(f'head tracking: sent filter idx {filter_idx}')
        # logging.info(f'sent filter value at {time.time() - start}')
        time.sleep(0.005)

def pulse_stream(filter_name, pulse_interval, binsim_state):
    import pybinsim
    binsim = pybinsim.BinSim(data_dir / 'wav' / filter_name / f'{filter_name}_settings.txt')
    pybinsim.logger.setLevel(logging.WARNING)
    binsim.soundHandler.loopSound = True
    binsim.stream_start()
    pulse_interval[0] = -1

    while not binsim.stream._is_running:
        time.sleep(0.1)
    binsim_state[0] = 1

    def smooth_transition(target_value, step=0.05, delay=0.002):
        current_value = binsim.config.configurationDict['loudnessFactor']
        while abs(current_value - target_value) > step:
            current_value += step if target_value > current_value else -step
            binsim.config.configurationDict['loudnessFactor'] = round(current_value, 2)
            time.sleep(delay)
        binsim.config.configurationDict['loudnessFactor'] = target_value

    while True:
        interval = pulse_interval[0]
        if interval == -1:  # Mute sound
            smooth_transition(0)
        elif interval < 5:  # continuous sound
            smooth_transition(0.5)
        elif interval > 5:  # pulse sound
            smooth_transition(0.5)
            time.sleep(interval / 1000)
            smooth_transition(0)
            time.sleep(interval / 1000)

        logging.debug(f'got interval value {interval}')
        time.sleep(0.005)

def play_trial(distance, pulse_interval, score, osc_client, trial_time, target_size, target_time):
    # distance-to-interval parameters
    max_interval = 500  # interval duration at max distance in ms
    max_distance = numpy.linalg.norm(numpy.subtract([0,0],[30, 30]))  # max distance in degrees
    threshold = .5  # the distance threshold in degrees below which interval duration should be zero
    steepness = 10  # controls how gradually the interval duration increases

    time_on_target = 0
    count_down = False
    trial_start = time.time()

    while time.time() - trial_start < trial_time:
        dist = distance[0]
        logging.debug(f'play trial: got distance value {dist}')
        if dist <= threshold:  # interval duration from distance
            interval = 0
        else:
            norm_dist = (dist - threshold) / (max_distance - threshold)
            norm_dist = numpy.clip(norm_dist, 0, 1)  # Keep in range [0, 1]
            interval = int(max_interval * (numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)))
        pulse_interval[0] = max(0, interval)  # ensure non-negative interval
        logging.debug(f'play trial: set interval value {max(0, interval)}')

        if dist < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False

        if count_down and time.time() > time_on_target + target_time:
            if time.time() - trial_start <= 3:
                score[0] += 2
                play_sound('coins', osc_client, pulse_interval)
            else:
                score[0] += 1
                play_sound('coin', osc_client, pulse_interval)
            break

        logging.debug(f'trial_time {time.time() - trial_start}')
        time.sleep(0.005)
    pulse_interval[0] = -1

def play_session(game_time, trial_time, target_size, target_time, az_range, ele_range, filter_name):

    # shared variables
    sensor_state = mp.Array("i", [0])  # sensor_tracking control
    binsim_state = mp.Array("i", [0])  # play_trial control
    target = mp.Array("i", [0, 0])  # from get_target to sensor_tracking
    distance = mp.Array("f", [0])  # from sensor_tracking to play_trial
    pulse_interval = mp.Array("i", [-1]) # from play_trial to pulse_stream, initially muted
    score = mp.Array("i", [0])  # from play_trial
    osc_client = init_osc_client()

    # connect sensor and wait for process to run
    tracking_worker = mp.Process(target=head_tracking, args=(distance, target, osc_client,
                                                                    filter_name, sensor_state))
    tracking_worker.start()
    while sensor_state[0] == 0:  # wait for sensor to initialize
        time.sleep(0.1)

    # set up pulse stream (init pybinsim)
    pulse_worker = mp.Process(target=pulse_stream, args=(filter_name, distance, binsim_state))
    pulse_worker.start()
    while binsim_state[0] == 0:  # wait for pyaudio to initialize
        time.sleep(.01)

    input("Press Enter to start the game...")
    scores = []
    sensor_state[0] = 2  # start sensor tracking
    start_time = time.time()

    while time.time() - start_time < game_time:
        set_target(az_range, ele_range, target, min_dist=30)
        trial_worker = mp.Process(target=play_trial, args=(distance, pulse_interval, score,
                                                   osc_client, trial_time, target_size, target_time))
        trial_worker.start()
        trial_worker.join()
        scores.append(score[0])
        print(f'Trial complete: {score} points')
        input("Press Enter to start the next trial...")

    play_sound('buzzer', osc_client, pulse_interval)
    print(f'Game Over! Total Score: {sum(scores)}')
    pulse_worker.terminate()
    pulse_worker.join()
    tracking_worker.terminate()
    tracking_worker.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the auditory training experiment.")
    parser.add_argument("--game_time", type=int, default=200, help="Total session duration in seconds")
    parser.add_argument("--trial_time", type=int, default=90, help="Time limit per trial in seconds")
    parser.add_argument("--target_size", type=int, default=5, help="Size of the target zone")
    parser.add_argument("--target_time", type=float, default=3, help="Time required to hold gaze on target")
    parser.add_argument("--az_range", type=int, nargs=2, default=[-30, 30], help="Azimuth range for target selection")
    parser.add_argument("--ele_range", type=int, nargs=2, default=[-30, 30],
                        help="Elevation range for target selection")
    parser.add_argument("--filter_name", type=str, default='KU100_HRIR_L2702', help="HRTF filename for PyBinSim")
    args = parser.parse_args()
    if not (data_dir / 'wav' / args.filter_name).exists():
        hrtf2wav(f'{args.filter_name}.sofa')
    play_session(args.game_time, args.trial_time, args.target_size, args.target_time, tuple(args.az_range),
         tuple(args.ele_range), args.filter_name)

    # game_time  = 90
    # trial_time = 10
    # target_size = 5
    # target_time = 3
    # az_range = (-30, 30)
    # ele_range = (-30, 30)
    # filter_name = 'KU100_HRIR_L2702'

