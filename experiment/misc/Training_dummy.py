import argparse
import logging
import multiprocessing as mp
from pythonosc import udp_client
import pybinsim
from hrtf.processing.hrtf2wav import *
from experiment.Subject import Subject
logging.getLogger().setLevel('INFO')
pybinsim.logger.setLevel(logging.WARNING)

# get subject data
id = ''
subject = Subject(id)

# select HRIR
# filename ='KU100_HRIR_L2702'
filename ='single_notch'

# select wav file for the training stimulus, None will default to pink noise
soundfile = None
# soundfile='c_chord_guitar.wav'

target_size = 3
target_time = 1
az_range = (-45, 45)
ele_range = (-25, 25)
min_dist = 30
game_time  = 180
trial_time = 15
gain = .5

data_dir = Path.cwd() / 'data' / 'hrtf'
hrtf = slab.HRTF(data_dir / 'sofa' / f'{filename}.sofa')
slab.set_default_samplerate(hrtf.samplerate)
sound_dir = data_dir / 'wav' / filename / 'sounds'

# main functions
def play_session(game_time, trial_time, target_size, target_time, az_range, ele_range):
    """
    Play trials until game time is up.
    """
    # shared variables
    global osc_client
    osc_client = make_osc_client(port=10003)  # binsim communication
    sensor_state = mp.Value("i", 0)  # sensor_tracking flag: 1-initialized / calibrated, 2-calibrate, 3-start tracking
    pulse_state = mp.Value("i", 0)  # pulse_stream flag: 0-mute, 1-idle, 2-play pulse with current interval
    target = mp.Array("i", [0, 0])  # get_target sets target - used by sensor_tracking
    distance = mp.Value("f", 0)  # sensor_tracking updates distance and sends to play_trial
    pulse_interval = mp.Value("f", 0) # play_trial updates interval and sends to pulse_stream
    # start head tracker
    tracking_worker = mp.Process(target=head_tracker, args=(distance, target, sensor_state))
    tracking_worker.start()
    # init pybinsim
    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()
    # start pulse maker (muted)
    pulse_worker = mp.Process(target=pulse_maker, args=(pulse_interval, pulse_state))
    pulse_worker.start()
    # wait for sensor init
    while not sensor_state.value == 1:  # wait for sensor
        time.sleep(.1)
    while True:  # loop over games
        scores = []
        game_timer = 0  # set game timer
        while game_timer < game_time:        # play trials until game time is up
            # set_target(az_range, ele_range, target, min_dist
            set_target(target, min_dist)
            game_timer, score = play_trial(distance, pulse_interval, pulse_state, sensor_state,
                       trial_time, game_time, game_timer, target_size, target_time)  # play trial and update game timer
            scores.append(score)
        # end
        pulse_state.value = 1  # stop pulse
        play_sound(osc_client, soundfile='buzzer.wav', duration=None, sleep=True)
        logging.info(f'Game Over! Total Score: {sum(scores)}')
        if input('Go again? (y/n): ') == 'n':
            break
    logging.info(f'Ending')
    binsim_worker.join()
    pulse_worker.join()
    tracking_worker.join()

def play_trial(distance, pulse_interval, pulse_state, sensor_state,
               trial_time, game_time, game_timer, target_size, target_time):
    """
    Play a single trial.
    """
    # trial variables
    score = 0
    trial_timer = 0
    time_on_target = 0
    count_down = False
    time.sleep(.1)
    input("Press Enter to play.")
    sensor_state.value = 2   # calibrate sensor
    time.sleep(.1) # wait until calbration is complete
    while not sensor_state.value == 3: # make sure tracking is running
        sensor_state.value = 3  # start tracking
        time.sleep(.1)
    pulse_interval.value = distance_to_interval(distance.value) # set initial interval value
    time.sleep(.2)  # wait for pulse maker to receive correct interval value before starting pulse
    pulse_state.value = 2 # play pulse
    logging.debug('Starting trial')
    start_time = time.time() # get start time
    while trial_timer < trial_time:  # end trial if trial timer is up
        trial_timer = time.time() - start_time # update trial timer
        if game_timer + trial_timer > game_time:  # end the game if game timer is up
            break
        pulse_interval.value = distance_to_interval(distance.value) # tracking worker->distance_to_interval->pulse worker
        if distance.value < target_size:         # target condition
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False
        if count_down and time.time() > time_on_target + target_time:  # goal condition
            pulse_state.value = 1  # stop pulse
            if trial_timer <= 3:
                play_sound(osc_client, soundfile='coins.wav', duration=None, sleep=True)
                score += 2
            else:
                play_sound(osc_client, soundfile='coin.wav', duration=None, sleep=True)
                score = 1
            break
        time.sleep(.01)   # these intervals determine CPU load
    logging.info(f'Trial complete: {score} points')
    game_timer += trial_timer  # update game timer
    pulse_state.value = 0  # mute sound
    sensor_state.value = 1  # stop sensor tracking
    return game_timer, score

# ----- sub processes ----- #

def binsim_stream():
    binsim = pybinsim.BinSim(data_dir / 'wav' / filename / f'{filename}_training_settings.txt')
    binsim.stream_start()  # run binsim loop

def pulse_maker(pulse_interval, pulse_state):
    import logging
    logging.getLogger().setLevel('INFO')
    osc_client = make_osc_client(port=10003)
    while True:
        logging.debug(f'pulse stream: state {pulse_state.value}')
        if pulse_state.value == 0:  # Mute sound, stop updating interval
            osc_client.send_message('/pyBinSimLoudness', 0)
            target_sound = False  # flag for playing continuous noise
        elif pulse_state.value == 1:  # idle (eg to play game sounds)
            continue
            target_sound = False  # flag for playing continuous noise
        elif pulse_state.value == 2:  # play sound with given interval
            osc_client.send_message('/pyBinSimLoudness', gain)
            interval = pulse_interval.value
            logging.debug(f'pulse stream: got interval value {interval}')
            if interval == 0 and not target_sound:  # play continuous
                play_sound(osc_client, soundfile=soundfile, duration=float(target_time), sleep=False)
                target_sound = True
            elif interval != 0:  # pulse sound
                play_sound(osc_client, soundfile=soundfile, duration=float(interval), sleep=True)
                time.sleep(interval)  # wait for interval duration
                target_sound = False  # flag for playing continuous noise
        time.sleep(0.01)   # these intervals mainly determines CPU load

def head_tracker(distance, target, sensor_state):
    import logging
    logging.getLogger().setLevel('INFO')
    osc_client = make_osc_client(port=10000)
    hrtf_sources = hrtf.sources.vertical_polar
    logging.debug('dummy motion sensor running')
    sensor_state.value = 1  # init flag
    while True:
        if sensor_state.value == 2:  # to be calibrated flag
            logging.debug('Calibrating dummy motion sensor..')
            step_size = .1
            pose = numpy.array([0, 0])
            sensor_state.value = 1
        elif sensor_state.value == 3:  # head tracking dummy flag - start approaching target
            direction = (target - pose)
            direction = direction / numpy.linalg.norm(direction)  # normalize
            pose = pose + step_size * direction
            # set distance for play_session
            relative_coords = target[:] - pose
            distance.value = numpy.linalg.norm(relative_coords)
            logging.debug(f'head tracking: set distance value {distance.value}')
            # find the closest filter idx and send to pybinsim
            relative_coords[0] = (-relative_coords[
                0] + 360) % 360  # mirror and convert to HRTF convetion [0 < az < 360]
            rel_target = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - hrtf_sources, axis=1))
            rel_hrtf_coords = hrtf_sources[filter_idx]
            osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                            float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                            0, 0, 0])
            logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')
            time.sleep(0.01)  # these intervals mainly determines CPU load

# ------- helpers ----- #

def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """ serves as a wrapper and passes the soundfile to pybinsim """
    import logging
    logging.getLogger().setLevel('INFO')
    if duration:
        if soundfile:  # read a soundfile and crop to pulse duration
            sound = slab.Sound.read(sound_dir / soundfile)
            soundfile = 'cropped_' + soundfile
            slab.Sound(sound.data[:int(hrtf.samplerate * duration)]).ramp(duration=.03).write(sound_dir / soundfile)
        else:  # generate noise with pulse duration
            soundfile = 'noise_pulse.wav'
            slab.Sound.pinknoise(duration).ramp(duration=.03).write(sound_dir / soundfile)
    else:
        duration = slab.Sound(sound_dir / soundfile).duration  # get duration of the soundfile
    logging.debug(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(sound_dir / soundfile))  # play
    if sleep:  # wait for sound to finish playing
        time.sleep(duration)

def distance_to_interval(distance):
    max_interval = 350  # max interval duration in ms
    min_interval = 75  # min interval duration before entering target window
    steepness = 5  # controls how the interval duration decreases when approaching the target window
    max_distance = numpy.linalg.norm(numpy.subtract([0, 0], [az_range[0], ele_range[0]]))  # max possible distance
    if distance <= target_size:
        return 0  # fully inside target window → silent
    # Normalize distance: [target_size → max_distance] → [0 → 1]
    norm_dist = (distance - target_size) / (max_distance - target_size)
    norm_dist = numpy.clip(norm_dist, 0, 1)
    # Logarithmic interpolation between min_interval and max_interval
    scale = numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)
    interval = (min_interval + (max_interval - min_interval) * scale).astype(int)
    return int(interval) / 1000  # convert to seconds

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

# def set_target(az_range, ele_range, target, min_dist):
#     logging.debug(f'Setting target...')
#     while True:
#         prev_tar = target[:]
#         next_tar = [numpy.random.randint(az_range[0], az_range[1]),
#                   numpy.random.randint(ele_range[0], ele_range[1])]
#         if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
#             target[:] = next_tar
#             logging.info(f'Set Target to {next_tar}.')
#             break

def set_target(target, min_dist):
    logging.debug(f'Setting target...')
    while True:
        prev_tar = target[:]
        sequence = subject.localization()  #todo append probabilities to localization data after each loc test
        #retrieve here, select sector based on probabilities and assign random target within the sector

        next_tar = [numpy.random.randint(az_range[0], az_range[1]),
                  numpy.random.randint(ele_range[0], ele_range[1])]
        if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
            target[:] = next_tar
            logging.info(f'Set Target to {next_tar}.')
            break

if __name__ == "__main__":
    make_wav(filename, overwrite=False, show=False)
    play_session(game_time, trial_time, target_size, target_time, tuple(az_range), tuple(ele_range))

