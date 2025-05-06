import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
import logging
import multiprocessing as mp
from pythonosc import udp_client
import pybinsim
from experiment.misc import meta_motion
from hrtf.processing.hrtf2wav import *
data_dir = Path.cwd() / 'data' / 'hrtf'

# audio settings
hrtf_file = 'single_notch'
# hrtf_file = 'KU100_HRIR_L2702'
soundfile = 'pinknoise.wav'
# soundfile = 'c_chord_guitar.wav'  # choose file from sounds folder
# soundfile = 'harmonic.wav'

# game settings
target_size = 3
target_time = 30
az_range = (-1, 1)
ele_range = (-1, 1)
min_dist = 0
game_time = 360
trial_time = 30

# logger settings
logging.getLogger().setLevel('INFO')
pybinsim.logger.setLevel(logging.WARNING)

# main functions
def play_session(az_range, ele_range):
    # shared variables
    global osc_client
    osc_client = make_osc_client(port=10003)  # binsim communication
    sensor_state = mp.Value("i", 0)  # sensor_tracking control, 1-initialized / calibrated, 2-calibrate, 3-start tracking
    stream_state = mp.Value("i", 0)  # pulse_stream control, 0-mute, 1-play
    target = mp.Array("i", [0, 0])  # get_target sets target - used by sensor_tracking
    distance = mp.Value("f", 0)  # sensor_tracking updates distance and sends to play_trial
    # init sensor
    tracking_worker = mp.Process(target=head_tracking, args=(distance, target, sensor_state))
    tracking_worker.start()
    # init pybinsim
    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()
    # start stream (muted)
    stream_worker = mp.Process(target=stream_control, args=(stream_state,))
    stream_worker.start()
    # wait for sensor init
    while not sensor_state.value == 1:  # wait for sensor
        time.sleep(0.1)
    while True:  # loop over games
        scores = []
        game_timer = 0  # set game timer
        # play trials until game time is up
        while game_timer < game_time:
            set_target(az_range, ele_range, target, min_dist)
            time.sleep(.01)
            input("Press Enter to play.")
            set_soundfile(soundfile, osc_client)
            game_timer, score = play_trial(distance, stream_state, sensor_state,
                                           trial_time, game_time, game_timer, target_size,
                                           target_time)  # play trial and update game timer
            scores.append(score)
        # end
        play_sound('buzzer', stream_state)
        logging.info(f'Game Over! Total Score: {sum(scores)}')
        if input('Go again? (y/n): ') == 'n':
            break
    logging.info(f'Ending')
    binsim_worker.join()
    stream_worker.join()
    tracking_worker.join()

def play_trial(distance, stream_state, sensor_state, trial_time, game_time, game_timer, target_size, target_time):
    logging.debug('run "play_trial"')
    # trial variables
    score = 0
    trial_timer = 0
    time_on_target = 0
    count_down = False
    sensor_state.value = 2  # calibrate sensor
    time.sleep(.01)  # wait until calbration is complete
    while not sensor_state.value == 3:  # make sure tracking is running
        sensor_state.value = 3  # start tracking
        time.sleep(.1)
    stream_state.value = 1  # play
    logging.debug('Starting trial')
    start_time = time.time()  # get start time
    while trial_timer < trial_time:  # end trial if trial timer is up
        trial_timer = time.time() - start_time  # update trial timer
        if game_timer + trial_timer > game_time:  # end the game if game timer is up
            break
        if distance.value < target_size:  # target condition
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False
        if count_down and time.time() > time_on_target + target_time:  # goal condition
            if trial_timer <= 3:
                play_sound('coins', stream_state)
                score += 2
            else:
                play_sound('coin', stream_state)
                score = 1
            break
        time.sleep(0.01)  # these intervals determine CPU load
    logging.info(f'Trial complete: {score} points')
    game_timer += trial_timer  # update game timer
    stream_state.value = 0  # mute stream
    sensor_state.value = 1  # halt sensor tracking
    time.sleep(.1)
    return game_timer, score

# sub processes
def binsim_stream():
    binsim = pybinsim.BinSim(data_dir / 'wav' / hrtf_file / f'{hrtf_file}_training_settings.txt')
    binsim.soundHandler.loopSound = True
    binsim.stream_start()  # run binsim loop

def stream_control(stream_state):
    osc_client = make_osc_client(port=10003)
    playing = False
    while True:
        # logging.debug(f'playing: {playing}')
        if stream_state.value == 0 and playing:  # Mute sound
            ramp(osc_client, direction='down')
            playing = False
        elif stream_state.value == 1 and not playing:  # play sound
            ramp(osc_client, direction='up')
            playing = True
        time.sleep(0.1)  # these intervals mainly determines CPU load

def head_tracking(distance, target, sensor_state):
    osc_client = make_osc_client(port=10000)
    hrtf_sources = slab.HRTF(data_dir / 'sofa' / f'{hrtf_file}.sofa').sources.vertical_polar
    # init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    sensor_state.value = 1   # init flag
    while True:
        if sensor_state.value == 2:  # to be calibrated flag
            logging.debug('Calibrating sensor..')
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.1)
            sensor_state.value = 1
        elif sensor_state.value == 3:  # head tracking flag
            pose = motion_sensor.get_pose()
            pose_raw = numpy.array((motion_sensor.state.pose.yaw, motion_sensor.state.pose.roll))
            logging.info(f'raw head pose: {pose_raw}')
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

# helpers
def play_sound(wav_name, stream_state):
    logging.debug(f'playing sound file: {wav_name}')
    stream_state.value = 0  # mute stream
    fpath = str(data_dir / 'wav' / hrtf_file / 'sounds' / f'{wav_name}.wav')
    slab.Sound.read(fpath).play()

def ramp(osc_client, direction='up', duration=0.1, n_steps=20):
    global playing
    logging.debug(f'ramping: {direction}')
    envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2  # squared sine window
    multiplier = envelope(numpy.reshape(numpy.linspace(0.0, 1, n_steps), (n_steps, 1)))
    for level in multiplier:
        if direction == 'up':
            level *= 0.5
            playing = True
        elif direction == 'down':
            level = 0.5 * (1 - level)
            playing = False
        osc_client.send_message('/pyBinSimLoudness', level)
        time.sleep(duration / n_steps)

def make_osc_client(port):
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

def set_soundfile(soundfile, osc_client):
    logging.debug(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / hrtf_file / 'sounds' / soundfile))


if __name__ == "__main__":
    make_wav(hrtf_file, overwrite=True)
    play_session(tuple(az_range), tuple(ele_range))
