import matplotlib
matplotlib.use('Qt5Agg')
import numpy
import slab
import time
import logging
import pybinsim
from pathlib import Path
import multiprocessing as mp
from pythonosc import udp_client
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.misc import meta_motion
logging.getLogger().setLevel('INFO')
pybinsim.logger.setLevel(logging.WARNING)

# --- Subject ID ----
subject = 'PF'

# --- HRTF settings ----

# --- select sofa file
sofa_name ='KU100_HRIR_L2702'
# sofa_name ='single_notch'
# sofa_name ='kemar'

# ---- specify ear for unilateral training, None defaults to binaural training
# ear = None
ear = 'left'

# --- load and process HRIR
hrir = hrtf2binsim(sofa_name, ear, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'wav' / hrir.name

# --- game settings ----
# --- select soundfile for the training stimulus, None defaults to pink noise
soundfile = None
# soundfile='c_chord_guitar.wav'
# soundfile='uso_225ms_9_.wav'

# --- training settings
settings = dict(
    target_size = 5,        # size of target area in degrees
    target_time = 1,        # required time on target to score
    az_range = (-45, 45),   # target azimuth range
    ele_range = (-5, 5),  # target elevation range
    min_dist = 30,          # minimal distance between successive targets in degrees
    game_time  = 180,       # time per session
    trial_time = 15,        # time per trial
    gain = .5               # loudness
    )


# --- main functions
def play_session(): #, game_time, trial_time, target_size, target_time, az_range, ele_range):
    """
    Play trials until game time is up.
    """
    # shared variables
    global osc_client
    osc_client = make_osc_client(port=10003)  # binsim communication
    sensor_state = mp.Value("i", 0)  # sensor_tracking flag: 1-initialized / calibrated, 2-calibrate, 3-start tracking
    pulse_state = mp.Value("i", 0)  # pulse_stream flag: 0-mute, 1-idle, 2-play pulse with current interval
    target = mp.Array("f", [0, 0])  # get_target sets target - used by sensor_tracking
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
        while game_timer < settings['game_time']:        # play trials until game time is up
            set_target(settings['az_range'], settings['ele_range'], target, settings['min_dist'])
            game_timer, score = play_trial(distance, pulse_interval, pulse_state, sensor_state,
                       settings['trial_time'], settings['game_time'], game_timer,
                        settings['target_size'], settings['target_time'])  # play trial and update game timer
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
    # input("Press Enter to play.")
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
                score = 2
            else:
                play_sound(osc_client, soundfile='coin.wav', duration=None, sleep=True)
                score = 1
            break
        time.sleep(.01)   # these intervals determine CPU load
    logging.info(f'Score: {score}!')
    game_timer += trial_timer  # update game timer
    pulse_state.value = 0  # mute sound
    sensor_state.value = 1  # stop sensor tracking
    return game_timer, score

# ----- sub processes ----- #

def binsim_stream():
    logging.info(f'Loading {hrir.name}')
    binsim = pybinsim.BinSim(hrir_dir / f'{hrir.name}_training_settings.txt')
    binsim.stream_start()  # run binsim loop

def pulse_maker(pulse_interval, pulse_state):
    osc_client = make_osc_client(port=10003)
    target_sound = False
    while True:
        if pulse_state.value == 0:  # Mute sound, stop updating interval
            osc_client.send_message('/pyBinSimLoudness', 0)
            target_sound = False  # flag for playing continuous noise
        elif pulse_state.value == 1:  # idle (eg to play game sounds)
            target_sound = False  # flag for playing continuous noise
        elif pulse_state.value == 2:  # play sound with given interval
            osc_client.send_message('/pyBinSimLoudness', settings['gain'])
            interval = pulse_interval.value
            logging.debug(f'pulse stream: got interval value {interval}')
            if interval == 0 and not target_sound:  # play continuous
                play_sound(osc_client, soundfile=soundfile, duration=float(settings['target_time']), sleep=False)
                target_sound = True
            elif interval != 0:  # pulse sound
                play_sound(osc_client, soundfile=soundfile, duration=float(interval), sleep=True)
                time.sleep(interval)  # wait for interval duration
                target_sound = False  # flag for playing continuous noise
        time.sleep(0.01)   # these intervals mainly determines CPU load

def head_tracker(distance, target, sensor_state):
    osc_client = make_osc_client(port=10000)
    sources = hrir.sources.vertical_polar
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
            # pose_raw = numpy.array((motion_sensor.state.pose.yaw, motion_sensor.state.pose.roll))
            # logging.info(f'raw head pose: {pose_raw}')
            # set distance for play_session
            relative_coords = target[:] - pose
            distance.value = numpy.linalg.norm(relative_coords)
            logging.debug(f'head tracking: set distance value {distance.value}')
            # find the closest filter idx and send to pybinsim
            relative_coords[0] = (-relative_coords[0] + 360) % 360 # mirror and convert to HRTF convetion [0 < az < 360]
            rel_target = numpy.array((relative_coords[0], relative_coords[1], sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target-sources, axis=1))
            rel_hrtf_coords = sources[filter_idx]
            osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                            float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                            0, 0, 0])
            logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')
        time.sleep(0.01)    # these intervals mainly determines CPU load

# ------- helpers ----- #
def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """ serves as a wrapper and passes the soundfile to pybinsim """  # todo clunky (might cause the lag?)
    if duration:
        if soundfile:  # read a soundfile and crop to pulse duration
            sound = slab.Sound.read(hrir_dir / 'sounds' / soundfile)
            soundfile = 'cropped_' + soundfile
            (slab.Sound(sound.data[:int(hrir.samplerate * duration)]).ramp(duration=.03)
             .write(hrir_dir / 'sounds' / soundfile))
        else:  # generate noise with pulse duration
            soundfile = 'noise_pulse.wav'
            slab.Sound.pinknoise(duration).ramp(duration=.03).write(hrir_dir / 'sounds' / soundfile)
    else:
        duration = slab.Sound(hrir_dir / 'sounds' / soundfile).duration  # get duration of the soundfile
    logging.debug(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(hrir_dir / 'sounds' / soundfile))  # play
    if sleep:  # wait for sound to finish playing
        time.sleep(duration)

def distance_to_interval(distance):
    max_interval = 350  # max interval duration in ms
    min_interval = 75  # min interval duration before entering target window
    steepness = 5  # controls how the interval duration decreases when approaching the target window
    max_distance = numpy.linalg.norm(numpy.subtract([0, 0], [settings['az_range'][0], settings['ele_range'][0]]))  # max possible distance
    if distance <= settings['target_size']:
        return 0  # fully inside target window → silent
    # Normalize distance: [target_size → max_distance] → [0 → 1]
    norm_dist = (distance - settings['target_size']) / (max_distance - settings['target_size'])
    norm_dist = numpy.clip(norm_dist, 0, 1)
    # Logarithmic interpolation between min_interval and max_interval
    scale = numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)
    interval = (min_interval + (max_interval - min_interval) * scale).astype(int)
    return int(interval) / 1000  # convert to seconds

def make_osc_client(port, ip='127.0.0.1'):
    return udp_client.SimpleUDPClient(ip, port)

def set_target(target, min_dist):
    logging.debug(f'Setting target...')
    sources = hrir.sources.vertical_polar
    az_range = settings['az_range']
    ele_range = settings['ele_range']
    az_range = (az_range[0] % 360, az_range[1] % 360)
    az_mask = ((sources[:, 0] >= az_range[0]) & (sources[:, 0] <= az_range[1])) if az_range[0] <= az_range[1]\
        else ((sources[:, 0] >= az_range[0]) | (sources[:, 0] <= az_range[1]))
    el_mask = (sources[:, 1] >= ele_range[0]) & (sources[:, 1] <= ele_range[1])
    candidates = sources[az_mask & el_mask, :2]
    if candidates.shape[0] == 0:
        raise RuntimeError("No HRIR positions within the given ranges!")
    while True:
        prev_tar = target[:]
        next_tar = [numpy.random.choice(candidates)]
        if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
            target[:] = next_tar
            logging.info("Set Target to [%.1f, %.1f]" % (next_tar[0], next_tar[1]))
            break

# def set_target(target, min_dist):
#     logging.debug(f'Setting target...')
#     while True:
#         prev_tar = target[:]
#         sequence = subject.localization()  #todo append probabilities to localization data after each loc test
#         #retrieve here, select sector based on probabilities and assign random target within the sector
#
#         next_tar = [numpy.random.randint(az_range[0], az_range[1]),
#                   numpy.random.randint(ele_range[0], ele_range[1])]
#         if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= min_dist:
#             target[:] = next_tar
#             logging.info(f'Set Target to {next_tar}.')
#             break
#
# def plot():
#     # todo plot current hrtfs for left and right ear

if __name__ == "__main__":
    play_session()
