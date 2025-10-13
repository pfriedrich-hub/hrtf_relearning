# test_hrtf.py
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy
import slab
import time
import logging
from pathlib import Path
import multiprocessing as mp
from pythonosc import udp_client

# your project imports
from experiment.Subject import Subject
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.misc.training_targets import set_target_probabilistic
from experiment.misc import meta_motion

logging.getLogger().setLevel('INFO')

# ------------------------ CONFIG ------------------------

# Subject / last localization sequence (for training ranges & probabilities)
subject_id = 'PF'
sequence = Subject(subject_id).last_sequence  # must contain .settings and .data

# HRTF selection
sofa_name = 'KU100'
# sofa_name = 'single_notch'
# sofa_name = 'kemar'

# unilateral vs. binaural
# ear = None
ear = 'left'

# Test settings (similar to training but typically longer trial/hold)
settings = dict(
    target_size=3,                # deg
    target_time=1.0,              # time on target to score (seconds)
    az_range=sequence.settings['azimuth_range'],
    ele_range=sequence.settings['elevation_range'],
    min_dist=30,                  # deg
    game_time=300,                # session length
    trial_time=25,                # per trial
    gain=.2,                      # loudness
    hold_on_center=2.0            # additional hold/continuous time after entering target
)

# stimulus (None → internally generated pink pulses)
soundfile = None
# soundfile = 'c_chord_guitar.wav'
# soundfile = 'uso_225ms_9_.wav'

# ------------------------ HRTF LOAD ------------------------

hrir = hrtf2binsim(sofa_name, ear, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'wav' / hrir.name


# ------------------------ LIVE TF PLOTTER ------------------------

def plot_current_tf(filter_idx_shared, redraw_interval_s=0.05):
    """
    Lives in its own process. Opens a Qt figure and plots the TF of the
    current HRTF (hrir[filter_idx]) whenever the filter index changes.
    """
    global hrir
    sources = hrir.sources.vertical_polar

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Current HRTF Transfer Function")
    try:
        fig.canvas.manager.set_window_title("Live HRTF TF")
    except Exception:
        pass

    last_idx = -1
    while True:
        idx = filter_idx_shared.value
        if idx >= 0 and idx != last_idx:
            last_idx = idx
            try:
                ax.cla()
                # slab's helper draws into the provided axis
                hrir[idx].tf(show=True, axis=ax)
                az0, el0 = sources[idx, 0], sources[idx, 1]
                az180 = (az0 + 180) % 360 - 180
                ax.set_title(f"TF idx {idx}  |  az={az180:.1f}°, el={el0:.1f}°")
                ax.grid(True, which='both', linestyle=':', linewidth=0.6)
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception as e:
                ax.cla()
                ax.text(0.5, 0.5, f"Plot error for idx {idx}:\n{e}", ha='center', va='center')
                fig.canvas.draw_idle()
                plt.pause(0.001)

        # throttle loop
        plt.pause(redraw_interval_s)


# ------------------------ MAIN LOOP ------------------------

def play_session():
    """
    Test session: similar to training but allows staying at the center longer.
    """
    # shared state
    global osc_client
    osc_client = make_osc_client(port=10003)  # pyBinSim control

    sensor_state = mp.Value("i", 0)       # 1=ready, 2=calibrate, 3=track
    pulse_state = mp.Value("i", 0)        # 0=mute, 1=idle, 2=play pulses/continuous
    target = mp.Array("f", [0, 0])        # [az, el] in (-180,180], linear el
    distance = mp.Value("f", 0.0)         # updated by head_tracker
    pulse_interval = mp.Value("f", 0.0)   # updated by play_trial
    filter_idx_shared = mp.Value("i", -1) # NEW: share current HRTF index to plotter

    # workers
    plot_worker = mp.Process(target=plot_current_tf, args=(filter_idx_shared,), daemon=True)
    plot_worker.start()

    tracking_worker = mp.Process(target=head_tracker, args=(distance, target, sensor_state, filter_idx_shared))
    tracking_worker.start()

    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()

    pulse_worker = mp.Process(target=pulse_maker, args=(pulse_interval, pulse_state))
    pulse_worker.start()

    # wait for tracker init
    while not sensor_state.value == 1:
        time.sleep(.1)

    while True:
        scores = []
        game_timer = 0.0

        while game_timer < settings['game_time']:
            # use your probabilistic targeting (depends on last localization sequence)
            set_target_probabilistic(target, settings, sequence, hrir)
            game_timer, score = play_trial(
                distance, pulse_interval, pulse_state, sensor_state,
                settings['trial_time'], settings['game_time'], game_timer,
                settings['target_size'], settings['target_time'],
                settings['hold_on_center']
            )
            scores.append(score)

        # end-of-session
        pulse_state.value = 1
        play_sound(osc_client, soundfile='buzzer.wav', duration=None, sleep=True)
        logging.info(f'Game Over! Total Score: {sum(scores)}')
        if input('Go again? (y/n): ').lower().strip() == 'n':
            break

    logging.info('Ending')
    binsim_worker.join()
    pulse_worker.join()
    tracking_worker.join()
    # plot_worker is daemon → will exit with main


def play_trial(distance, pulse_interval, pulse_state, sensor_state,
               trial_time, game_time, game_timer,
               target_size, target_time, hold_on_center):
    """
    One test trial. Similar to training, but keeps the target sound on for
    an additional hold at center (hold_on_center).
    """
    score = 0
    trial_timer = 0.0
    time_on_target = 0.0
    count_down = False

    time.sleep(.1)
    input("Press Enter to play.")
    sensor_state.value = 2   # calibrate
    time.sleep(.1)
    while not sensor_state.value == 3:
        sensor_state.value = 3
        time.sleep(.1)

    pulse_interval.value = distance_to_interval(distance.value)
    time.sleep(.2)
    pulse_state.value = 2
    logging.debug('Starting test trial')

    start_time = time.time()
    held_extra = False

    while trial_timer < trial_time:
        trial_timer = time.time() - start_time
        if game_timer + trial_timer > game_time:
            break

        # Update pulse tempo based on current distance
        pulse_interval.value = distance_to_interval(distance.value)

        # Enter target window
        if distance.value < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False
            held_extra = False

        # goal condition reached
        if count_down and time.time() > time_on_target + target_time:
            # keep the sound on continuously for an extra hold period
            if not held_extra:
                pulse_state.value = 2
                pulse_interval.value = 0.0  # 0 → continuous playback in pulse_maker
                time.sleep(hold_on_center)
                held_extra = True

            # end trial after hold
            pulse_state.value = 1  # idle (stop pulses)
            # reward sound
            if trial_timer <= 3:
                play_sound(osc_client, soundfile='coins.wav', duration=None, sleep=True)
                score = 2
            else:
                play_sound(osc_client, soundfile='coin.wav', duration=None, sleep=True)
                score = 1
            break

        time.sleep(.01)

    logging.info(f'Score: {score}!')
    game_timer += trial_timer
    pulse_state.value = 0  # mute sound
    sensor_state.value = 1 # stop tracking
    return game_timer, score


# ------------------------ SUB-PROCESSES ------------------------

def binsim_stream():
    import pybinsim
    pybinsim.logger.setLevel(logging.ERROR)
    logging.info(f'Loading {hrir.name}')
    binsim = pybinsim.BinSim(hrir_dir / f'{hrir.name}_training_settings.txt')
    binsim.stream_start()


def pulse_maker(pulse_interval, pulse_state):
    osc_client = make_osc_client(port=10003)
    target_sound = False
    while True:
        if pulse_state.value == 0:  # mute
            osc_client.send_message('/pyBinSimLoudness', 0)
            target_sound = False
        elif pulse_state.value == 1:  # idle
            target_sound = False
        elif pulse_state.value == 2:  # play with interval
            osc_client.send_message('/pyBinSimLoudness', settings['gain'])
            interval = pulse_interval.value
            logging.debug(f'pulse stream: got interval value {interval}')
            if interval == 0 and not target_sound:  # continuous
                play_sound(osc_client, soundfile=soundfile, duration=float(settings['target_time']), sleep=False)
                target_sound = True
            elif interval != 0:
                play_sound(osc_client, soundfile=soundfile, duration=float(interval), sleep=True)
                time.sleep(interval)
                target_sound = False
        time.sleep(0.01)


def head_tracker(distance, target, sensor_state, filter_idx_shared):
    osc_client = make_osc_client(port=10000)
    sources = hrir.sources.vertical_polar

    # init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    sensor_state.value = 1

    last_idx = -1
    while True:
        if sensor_state.value == 2:
            logging.debug('Calibrating sensor..')
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.1)
            sensor_state.value = 1

        elif sensor_state.value == 3:
            pose = motion_sensor.get_pose()

            # distance to target in az/el plane
            relative_coords = target[:] - pose
            distance.value = numpy.linalg.norm(relative_coords)
            logging.debug(f'head tracking: set distance value {distance.value}')

            # convert to HRIR convention (0..360 az) & pick nearest filter
            relative_coords[0] = (-relative_coords[0] + 360) % 360
            rel_target = numpy.array((relative_coords[0], relative_coords[1], sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - sources, axis=1))
            rel_hrtf_coords = sources[filter_idx]

            # send to pyBinSim
            osc_client.send_message('/pyBinSim_ds_Filter', [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                0, 0, 0
            ])
            logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')

            # publish to plotter if changed
            if filter_idx != last_idx:
                last_idx = filter_idx
                filter_idx_shared.value = int(filter_idx)

        time.sleep(0.01)


# ------------------------ HELPERS ------------------------

def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """Wrapper that passes the soundfile to pyBinSim. Generates noise if None."""
    if duration:
        if soundfile:  # crop file to duration
            sound = slab.Sound.read(hrir_dir / 'sounds' / soundfile)
            soundfile = 'cropped_' + soundfile
            (slab.Sound(sound.data[:int(hrir.samplerate * duration)]).ramp(duration=.03)
             .write(hrir_dir / 'sounds' / soundfile))
        else:          # generate noise with pulse duration
            soundfile = 'noise_pulse.wav'
            slab.Sound.pinknoise(duration).ramp(duration=.03).write(hrir_dir / 'sounds' / soundfile)
    else:
        # use full length
        if soundfile is None:
            soundfile = 'noise_pulse.wav'
            # ensure it exists (1s default)
            slab.Sound.pinknoise(1.0).ramp(duration=.03).write(hrir_dir / 'sounds' / soundfile)
        duration = slab.Sound(hrir_dir / 'sounds' / soundfile).duration

    logging.debug(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(hrir_dir / 'sounds' / soundfile))
    if sleep:
        time.sleep(duration)


def distance_to_interval(distance):
    max_interval = 350  # ms
    min_interval = 75   # ms
    steepness = 5
    max_distance = numpy.linalg.norm(numpy.subtract(
        [0, 0], [settings['az_range'][0], settings['ele_range'][0]]
    ))
    if distance <= settings['target_size']:
        return 0.0
    # Normalize distance
    norm_dist = (distance - settings['target_size']) / (max_distance - settings['target_size'])
    norm_dist = numpy.clip(norm_dist, 0, 1)
    # log interpolation
    scale = numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)
    interval_ms = (min_interval + (max_interval - min_interval) * scale).astype(int)
    return int(interval_ms) / 1000.0


def make_osc_client(port, ip='127.0.0.1'):
    return udp_client.SimpleUDPClient(ip, port)


# ------------------------ ENTRY ------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer on Windows/Qt
    play_session()
