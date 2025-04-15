import argparse
import logging
import multiprocessing as mp
from pythonosc import udp_client
# from pythonosc import tcp_client
import pybinsim
from experiment.misc import meta_motion
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('WARNING')
pybinsim.logger.setLevel(logging.WARNING)
# binsim_logger = logging.getLogger('binsim')
# binsim_logger.setLevel('WARNING')

filename ='KU100_HRIR_L2702'
data_dir = Path.cwd() / 'data' / 'hrtf'

amp_scaling = .3
target_size = 5
target_time = .5
az_range = (-45, 45)
ele_range = (-45, 45)
min_dist = 25
game_time  = 90
trial_time = 10

# sub processes
def head_tracking(distance, target, sensor_state):
    osc_client = make_osc_client(port=10000)
    hrtf_sources = slab.HRTF(data_dir / 'sofa' / f'{filename}.sofa').sources.vertical_polar
    # init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    sensor_state.value = 1   # init flag

    while True:
        if sensor_state.value == 2:  # to be calibrated flag
            logging.info('Calibrating sensor..')
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.1)
            sensor_state.value = 1
        elif sensor_state.value == 3:  # head tracking flag
            pose = motion_sensor.get_pose()
            # set distance for play_session
            relative_coords = target[:] - pose
            distance.value = numpy.linalg.norm(relative_coords)
            logging.debug(f'head tracking: set distance value {distance.value}')
            # find the closest filter idx and send to pybinsim
            relative_coords[0] = (-relative_coords[0] + 360) % 360 # mirror and convert to HRTF convetion [0 < az < 360]
            rel_target = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target-hrtf_sources, axis=1))
            rel_hrtf_coords = hrtf_sources[filter_idx]
            osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                            float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                            0, 0, 0])
            logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')
        time.sleep(0.01)    # these intervals mainly determines CPU load

def binsim_stream(binsim_state):
    binsim = pybinsim.BinSim(data_dir / 'wav' / filename / f'{filename}_settings.txt')
    # pybinsim.logger.setLevel(logging.DEBUG)
    binsim.soundHandler.loopSound = True
    binsim.stream_start()
    # logging.info(binsim.stream)
    # while not binsim.stream: #todo
    #     logging.info(binsim.stream)
    #     time.sleep(0.1)
    # binsim_state.value = 1  # set flag for completed binsim init

def pulse_control(pulse_interval, binsim_state):
    osc_client = make_osc_client(port=10003)
    while True:
        if binsim_state.value == 1:  # Mute sound, stop updating interval
            osc_client.send_message('/pyBinSimLoudness', 0)
        elif binsim_state.value == 2:  # play sound with given interval
            interval = pulse_interval.value
            logging.debug(f'pulse stream: got interval value {interval}')
            if interval == 0:  # continuous sound
                osc_client.send_message('/pyBinSimLoudness', 0.5)
            else:  # pulse sound
                osc_client.send_message('/pyBinSimLoudness', 0.5)
                time.sleep(interval / 1000)
                osc_client.send_message('/pyBinSimLoudness', 0)
                time.sleep(interval / 1000)
        time.sleep(0.01)   # these intervals mainly determines CPU load

def play_trial(distance, pulse_interval, score, binsim_state, sensor_state,
               trial_time, target_size, target_time):
    osc_client = make_osc_client(port=10003)  # binsim communication
    logging.debug(f'"play_trial" started')
    # game variables
    score.value = 0
    time_on_target = 0
    count_down = False
    sensor_state.value = 2  # calibrate sensor
    time.sleep(.1)

    # start trial
    sensor_state.value = 3  # start sensor tracking
    binsim_state.value = 2  # unmute sound, start pulse
    trial_start = time.time()

    # osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / 'pinknoise.wav'))

    while time.time() - trial_start < trial_time:  # within trial / game loop until trial time runs out
        # tracking worker -> distance -> pulse worker
        dist = distance.value
        logging.debug(f'play trial: got distance value {dist}')
        interval = distance_to_interval(dist)
        logging.debug(f'play trial: set interval value {interval}')
        pulse_interval.value = interval

        # target condition
        if dist < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False

        # goal condition
        if count_down and time.time() > time_on_target + target_time:
            if time.time() - trial_start <= 3:  # two points if under 3 seconds
                play_sound('coins', osc_client, pulse_interval, binsim_state)
                score.value = 2
            else:
                play_sound('coin', osc_client, pulse_interval, binsim_state)
                score.value = 1
            break

        logging.debug(f'play trial: trial time {time.time() - trial_start}')
        time.sleep(0.01)   # these intervals mainly determines CPU load
    print(f'Trial complete: {score.value} points')
    binsim_state.value = 1  # stop pulse stream
    sensor_state.value = 1  # stop sensor tracking

def play_sound(wav_name, osc_client, pulse_interval, binsim_state):
    logging.debug(f'playing sound file: {wav_name}')
    pulse_interval.value = 0  # stop pulsing to play game sound
    binsim_state.value = 2  # make sure that stream is not muted
    fpath = str(data_dir / 'wav' / filename / 'sounds' / f'{wav_name}.wav')
    osc_client.send_message('/pyBinSimFile', fpath)
    time.sleep(slab.Sound.read(fpath).duration)
    binsim_state.value = 1  # mute stream after playing sound
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / 'pinknoise.wav'))

def play_session(game_time, trial_time, target_size, target_time, az_range, ele_range):
    # shared variables
    osc_client = make_osc_client(port=10000)  # binsim communication
    sensor_state = mp.Value("i", 0)  # sensor_tracking control, set to 0 to start initialization
    binsim_state = mp.Value("i", 0)  # pulse_stream control, set to 0 to start initialization
    score = mp.Value("i", 0)  # within trial score counter
    target = mp.Array("i", [0, 0])  # get_target sets target - used by sensor_tracking
    distance = mp.Value("f", 0)  # sensor_tracking updates distance and sends to play_trial
    pulse_interval = mp.Value("i", 0) # play_trial updates interval and sends to pulse_stream

    # init sensor
    tracking_worker = mp.Process(target=head_tracking, args=(distance, target, sensor_state))
    tracking_worker.start()
    # init pybinsim
    binsim_worker = mp.Process(target=binsim_stream, args=[binsim_state])
    binsim_worker.start()
    # start pulse stream
    pulse_worker = mp.Process(target=pulse_control, args=(pulse_interval, binsim_state))
    pulse_worker.start()

    # wait for init
    while not sensor_state.value == 1:
        time.sleep(0.1) # init binsim and sensor #todo

    # play trials
    scores = []
    start_time = time.time()
    time.sleep(.5)
    while time.time() - start_time < game_time:
        set_target(az_range, ele_range, target, min_dist)
        time.sleep(.1)
        input("Press Enter to play.")
        play_trial(distance, pulse_interval, score, binsim_state, sensor_state, trial_time, target_size, target_time)
        scores.append(score.value)
        # time.sleep(.1)

    # end
    play_sound('buzzer', osc_client, pulse_interval, binsim_state)
    print(f'Game Over! Total Score: {sum(scores)}')
    pulse_worker.terminate()
    pulse_worker.join()
    tracking_worker.terminate()
    tracking_worker.join()


def distance_to_interval(distance):  # todo fix
    max_interval = 250  # interval duration at max distance in ms
    max_distance = numpy.linalg.norm(numpy.subtract([0, 0], [az_range[0], ele_range[0]]))  # max distance in degrees
    steepness = 50  # controls how the interval duration increases
    if distance <= target_size:
        interval = 0
    else:
        norm_dist = (distance - target_size) / (max_distance - target_size)
        norm_dist = numpy.clip(norm_dist, 0, 1)  # Keep in range [0, 1]
        interval = int(max_interval * (numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)))
    interval = max(0, interval)  # ensure non-negative interval
    return interval

def make_osc_client(port=10000):
    host = '127.0.0.1'
    mode = 'client'
    ip = '127.0.0.1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--mode", default=mode)
    parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
    args = parser.parse_args()
    return udp_client.SimpleUDPClient(args.ip, args.port)


def set_target(az_range, ele_range, target, min_dist):
    logging.debug(f'Setting target...')
    while True:
        prev_tar = target[:]
        next_tar = [numpy.random.randint(az_range[0], az_range[1]),
                  numpy.random.randint(ele_range[0], ele_range[1])]
        if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
            target[:] = next_tar
            logging.info(f'Set Target to {next_tar}.')
            break


if __name__ == "__main__":
    make_wav(filename)
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
