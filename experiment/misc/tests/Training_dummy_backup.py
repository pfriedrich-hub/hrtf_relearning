# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy
import slab
import time
import logging
import multiprocessing as mp
from pythonosc import udp_client
from experiment.Subject import Subject
from experiment.misc.training_helpers.game_ui import *
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.misc.training_helpers.training_targets import set_target_probabilistic
logging.getLogger().setLevel('INFO')

# --- Subject ID ----
id = 'PF'

# --- sofa file
# hrir ='KU100'
# hrir ='kemar'
hrir ='pf_hr_itld'
# hrir = 'pf'

hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'binsim' / hrir

# ---- for unilateral training specify the side to flatten, None defaults to binaural training
# ear = None
ear = 'left'
reverb = True

# --- Game Settings ----
# --- soundfile for the training stimulus, None defaults to pink noise
soundfile = None
# soundfile='c_chord_guitar.wav'
# soundfile='uso_225ms_9_.wav'

# --- set training area, target size and time
settings = dict(
    target_size = 5,        # size of target area in degrees
    target_time = .5,        # required time on target to score
    # az_range = (-35, 0),   # target azimuth range  # take these from last sequence
    # ele_range = (-35, 35),  # target elevation range
    min_dist = 30,          # minimal distance between successive targets in degrees
    game_time  = 90,       # time per session
    trial_time = 10,        # time per trial
    gain = .2)               # loudness
show = False  # whether to show transfer functions during training


# --- main functions
def play_session(hrir):
    """
    Play trials until game time is up.
    Spawns: head tracker, binsim stream, pulse maker, and the Qt UI process.
    """
    global  osc_client, samplerate, sources, osc_client
    osc_client = make_osc_client(port=10003)

    # meta data
    hrir = hrtf2binsim(hrir, ear, reverb, overwrite=False)  # load and process HRIR
    samplerate = hrir.samplerate
    slab.set_default_samplerate(samplerate)
    sources = hrir.sources.vertical_polar
    sequence = Subject(id).last_sequence  # last localization sequence
    settings['az_range'] = sequence.settings['azimuth_range']   # target azimuth range
    settings['ele_range'] = sequence.settings['elevation_range']  # target elevation range


    # --- SHARED STATE USED BY BOTH GAME & UI (create ONCE here!) ---
    enter_pressed = mp.Value("i", 0)          # UI sets to 1 on Enter/click; game consumes & resets to 0
    ui_state = mp.Value("i", 0)               # 0 idle, 1 await_enter, 2 running, 3 session_over
    current_score = mp.Value("i", 0)          # live score display
    last_goal_points = mp.Value("i", 0)       # latch 1/2 to trigger coin flash
    game_time_left = mp.Value("f", settings['game_time'])
    trial_time_left = mp.Value("f", settings['trial_time'])
    session_total = mp.Value("i", 0)          # final score for highscore check

    # start workers
    sensor_state = mp.Value("i", 0)
    pulse_state = mp.Value("i", 0)
    target = mp.Array("f", [0, 0])
    distance = mp.Value("f", 0)
    pulse_interval = mp.Value("f", 0)
    plot_filter_idx = mp.Value("i", -1)

    tracking_worker = mp.Process(target=head_tracker, args=(distance, target, sources, sensor_state, plot_filter_idx))
    tracking_worker.start()

    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()

    pulse_worker = mp.Process(target=pulse_maker, args=(pulse_interval, pulse_state))
    pulse_worker.start()

    # Launch the UI process (pass EXACT SAME mp.Value objects!)
    from experiment.misc.training_helpers.game_ui import UIShared, run_ui
    ui_shared = UIShared(
        current_score=current_score,
        game_time_left=game_time_left,
        trial_time_left=trial_time_left,
        last_goal_points=last_goal_points,
        session_total=session_total,
        enter_pressed=enter_pressed,
        ui_state=ui_state
    )
    highscore_path = Path.cwd() / "data" / "ui" / "highscore.json"
    ui_worker = mp.Process(target=run_ui, args=(ui_shared, highscore_path, False), daemon=True)
    ui_worker.start()

    # wait for sensor init
    while sensor_state.value != 1:
        time.sleep(.1)

    # ---- GAME LOOP ----
    while True:
        scores = []
        game_timer = 0.0

        # reset UI session state
        current_score.value = 0
        last_goal_points.value = 0
        game_time_left.value = float(settings['game_time'])
        trial_time_left.value = float(settings['trial_time'])
        session_total.value = 0
        ui_state.value = 1                 # show overlay: "Press Enter to start"

        # play trials until session time is up
        while game_timer < settings['game_time']:
            try:
                set_target_probabilistic(target, settings, sequence, hrir)
            except AttributeError:
                logging.debug('Could not load target probabilities')

            game_timer, score = play_trial(
                distance, pulse_interval, pulse_state, sensor_state,
                ui_state, enter_pressed, trial_time_left, game_time_left,
                settings['trial_time'], settings['game_time'],
                game_timer, settings['target_size'], settings['target_time'],
                current_score, last_goal_points
            )
            scores.append(score)
            current_score.value = int(sum(scores))   # keep live score in sync after each trial

            # Prepare next trial prompt (unless time ran out)
            if game_timer < settings['game_time']:
                ui_state.value = 1                   # await Enter for next trial

        # session end
        pulse_state.value = 1
        play_sound(osc_client, soundfile='buzzer.wav', duration=None, sleep=True)
        total = int(sum(scores))
        session_total.value = total
        current_score.value = total
        ui_state.value = 3                           # UI performs highscore check
        logging.info(f'Game Over! Total Score: {total}')

        # ask whether to continue session
        try:
            if input('Go again? (y/n): ').strip().lower() == 'n':
                break
        except EOFError:
            # if running without a TTY, just stop after one session
            break

    logging.info('Ending')
    binsim_worker.join()
    pulse_worker.join()
    tracking_worker.join()

def play_trial(distance, pulse_interval, pulse_state, sensor_state,
               ui_state, enter_pressed, trial_time_left, game_time_left,
               trial_time, game_time, game_timer, target_size, target_time,
               current_score, last_goal_points):
    """
    Play a single trial. Waits for UI 'Enter', runs tracking, pulses sound,
    awards 1 or 2 points, and returns updated game_timer and trial score.
    """
    score = 0
    trial_timer = 0.0
    time_on_target = 0.0
    count_down = False

    # ---- WAIT FOR ENTER FROM UI ----
    ui_state.value = 1  # show overlay
    enter_pressed.value = 0  # clear any stale click
    while enter_pressed.value == 0:  # UI sets this to 1
        time.sleep(0.02)
    enter_pressed.value = 0  # consume
    ui_state.value = 2  # running (overlay hidden)

    # ---- START TRACKING & AUDIO ----
    time.sleep(.1)
    sensor_state.value = 2  # calibrate
    time.sleep(.1)
    while sensor_state.value != 3:  # ensure tracking starts
        sensor_state.value = 3
        time.sleep(.1)

    pulse_interval.value = distance_to_interval(distance.value)
    time.sleep(.2)  # let pulse_maker read interval
    pulse_state.value = 2  # play pulsed sound

    logging.debug('Starting trial')
    start_time = time.time()

    # ---- TRIAL LOOP ----
    while trial_timer < trial_time:
        now = time.time()
        trial_timer = now - start_time

        # update UI timers
        trial_time_left.value = max(0.0, float(trial_time) - trial_timer)
        game_time_left.value = max(0.0, float(game_time) - (game_timer + trial_timer))

        # end the game if session time would be exceeded
        if game_timer + trial_timer > game_time:
            break

        # update pulse interval from current distance
        pulse_interval.value = distance_to_interval(distance.value)

        # target detection
        if distance.value < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False

        # goal condition
        if count_down and time.time() > time_on_target + target_time:
            pulse_state.value = 1  # stop pulse
            # award points (fast = 2, else 1)
            if trial_timer <= 3:
                play_sound(osc_client, soundfile='coins.wav', duration=None, sleep=True)
                score_delta = 2
            else:
                play_sound(osc_client, soundfile='coin.wav', duration=None, sleep=True)
                score_delta = 1

            score += score_delta
            current_score.value += score_delta  # live UI update
            last_goal_points.value = score_delta  # trigger UI flash
            break

        time.sleep(0.01)  # CPU load control

    logging.info(f'Score this trial: {score}')
    game_timer += trial_timer
    pulse_state.value = 0
    sensor_state.value = 1
    return game_timer, score


# ----- sub processes ----- #

def binsim_stream():
    import pybinsim
    pybinsim.logger.setLevel(logging.DEBUG)
    logging.info(f'Loading {hrir_dir.name}')
    binsim = pybinsim.BinSim(hrir_dir / f'{hrir_dir.name}_training_settings.txt')
    binsim.stream_start()  # run binsim loop

def pulse_maker(pulse_interval, pulse_state):
    """
    Drive pyBinSim with a single preloaded noise file and pulse via loudness.
    Avoids reloading files every ~100 ms which can break the Player.
    """
    slab.set_default_samplerate(samplerate)
    osc_client = make_osc_client(port=10003)  # same port you use elsewhere
    base_noise = 'noise_pulse.wav'
    noise_loaded = False
    loud = False  # current loudness state

    # helper: ensure base noise is on the player
    def ensure_noise_loaded():
        nonlocal noise_loaded
        if noise_loaded:
            return
        # create once if missing (1 s pink noise; any content works)
        path = hrir_dir / 'sounds' / base_noise
        if not path.exists():
            slab.Sound.pinknoise(1.0).ramp(duration=.03).write(path)
        # point pyBinSim at the file (settings should have loopSound True, or it will just play once)
        osc_client.send_message('/pyBinSimFile', str(path))
        # start muted
        osc_client.send_message('/pyBinSimLoudness', 0.0)
        noise_loaded = True
    while True:
        state = pulse_state.value
        if state == 0:
            # fully mute, keep file loaded
            if loud:
                osc_client.send_message('/pyBinSimLoudness', 0.0)
                loud = False
        elif state == 1:
            # idle: keep muted so game sounds can play
            if loud:
                osc_client.send_message('/pyBinSimLoudness', 0.0)
                loud = False
        elif state == 2:
            # active pulsing
            ensure_noise_loaded()
            interval = float(pulse_interval.value)
            # inside target → continuous silence
            if interval == 0:
                if loud:
                    osc_client.send_message('/pyBinSimLoudness', 0.0)
                    loud = False
                time.sleep(0.02)
            else:
                # beep: short on, then off for the rest of the interval
                # 25% duty cycle, clamp to sane bounds
                beep = max(0.02, min(0.06, 0.25 * interval))
                if not loud:
                    osc_client.send_message('/pyBinSimLoudness', float(settings['gain']))
                    loud = True
                time.sleep(beep)
                if loud:
                    osc_client.send_message('/pyBinSimLoudness', 0.0)
                    loud = False
                # finish the rest of the interval
                remain = max(0.0, interval - beep)
                time.sleep(remain)
        time.sleep(0.01)  # keep CPU in check

def head_tracker(distance, target, sources, sensor_state, plot_filter_idx):
    import logging
    logging.getLogger().setLevel('INFO')
    osc_client = make_osc_client(port=10000)
    hrtf_sources = sources
    logging.debug('dummy motion sensor running')
    sensor_state.value = 1  # init flag
    while True:
        if sensor_state.value == 2:  # to be calibrated flag
            logging.debug('Calibrating dummy motion sensor..')
            step_size = .1
            pose = numpy.array([0, 0])
            sensor_state.value = 1
            last_idx = -1
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
            if filter_idx != last_idx:  # publish to plotter if changed (debounced)
                last_idx = filter_idx
                plot_filter_idx.value = int(filter_idx)
        time.sleep(0.01)  # these intervals mainly determines CPU load


def plot_current_tf(hrir, filter_idx_shared, redraw_interval_s=0.05, kind='TF'):
    """
    Lives in its own process. Opens a Qt figure and plots the TF of the
    current HRTF (hrir[filter_idx]) whenever the filter index changes.
    """
    # Reuse global `hrir` and its sources
    slab.set_default_samplerate(samplerate)
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Current HRTF Transfer Function")
    fig.canvas.manager.set_window_title("Live HRTF TF")
    last_idx = -1
    while True:
        idx = filter_idx_shared.value
        if idx >= 0 and idx != last_idx:
            last_idx = idx
            try:
                # Clear and replot using slab's built-in viz
                ax.cla()
                # slab.HRTF slice → .tf(show=True, axis=ax) will draw on provided axis
                if kind == 'TF':
                    hrir[idx].channel(0).tf(show=True, axis=ax)
                    hrir[idx].channel(1).tf(show=True, axis=ax)
                    ax.lines[0].set_label('left')
                    ax.lines[1].set_label('right')
                    ax.legend()
                elif kind == 'IR':
                    times = numpy.linspace(0, hrir[idx].n_samples / hrir.samplerate, hrir[idx].n_samples)
                    ax.plot(times, hrir[idx].data[:, 0], label='left')
                    ax.plot(times, hrir[idx].data[:, 1], label='right')
                    ax.legend(loc='upper right')
                # Add some context (az, el)
                az0, el0 = sources[idx, 0], sources[idx, 1]
                az180 = (az0 + 180) % 360 - 180  # to (-180,180]
                ax.set_title(f"TF idx {idx}  |  az={az180:.1f}°, el={el0:.1f}°")
                ax.grid(True, which='both', linestyle=':', linewidth=0.6)
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception as e:
                # Keep the loop alive even if one plot fails
                ax.cla()
                ax.text(0.5, 0.5, f"Plot error for idx {idx}:\n{e}", ha='center', va='center')
                fig.canvas.draw_idle()
                plt.pause(0.001)

        # Throttle the loop to avoid pegging a CPU core
        plt.pause(redraw_interval_s)


# ------- helpers ----- #

def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """ serves as a wrapper and passes the soundfile to pybinsim """
    slab.set_default_samplerate(samplerate)
    if duration:
        if soundfile:  # read a soundfile and crop to pulse duration
            sound = slab.Sound.read(hrir_dir / 'sounds' / soundfile)
            soundfile = 'cropped_' + soundfile
            (slab.Sound(sound.data[:int(samplerate * duration)]).ramp(duration=.03)
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

if __name__ == "__main__":
    play_session(hrir)



# tracker dummy: replace head_tracker() with this function to simulate gaze response on mac

# def head_tracker(distance, target, sensor_state, plot_filter_idx):
#     import logging
#     logging.getLogger().setLevel('INFO')
#     osc_client = make_osc_client(port=10000)
#     hrtf_sources = hrir.sources.vertical_polar
#     logging.debug('dummy motion sensor running')
#     sensor_state.value = 1  # init flag
#     while True:
#         if sensor_state.value == 2:  # to be calibrated flag
#             logging.debug('Calibrating dummy motion sensor..')
#             step_size = .1
#             pose = numpy.array([0, 0])
#             sensor_state.value = 1
#             last_idx = -1
#         elif sensor_state.value == 3:  # head tracking dummy flag - start approaching target
#             direction = (target - pose)
#             direction = direction / numpy.linalg.norm(direction)  # normalize
#             pose = pose + step_size * direction
#             # set distance for play_session
#             relative_coords = target[:] - pose
#             distance.value = numpy.linalg.norm(relative_coords)
#             logging.debug(f'head tracking: set distance value {distance.value}')
#             # find the closest filter idx and send to pybinsim
#             relative_coords[0] = (-relative_coords[
#                 0] + 360) % 360  # mirror and convert to HRTF convetion [0 < az < 360]
#             rel_target = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources[0, 2]))
#             filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - hrtf_sources, axis=1))
#             rel_hrtf_coords = hrtf_sources[filter_idx]
#             osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                                                             float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
#                                                             0, 0, 0])
#             logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')
#             # publish to plotter if changed (debounced)
#             if filter_idx != last_idx:
#                 last_idx = filter_idx
#                 plot_filter_idx.value = int(filter_idx)
#         time.sleep(0.01)  # these intervals mainly determines CPU load

