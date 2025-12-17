import matplotlib
from matplotlib import pyplot as plt
import logging
import time
from pathlib import Path
import multiprocessing as mp
from queue import Empty
import numpy
import slab
from pythonosc import udp_client
import datetime
date = datetime.datetime.now()
import pybinsim
import hrtf_relearning
ROOT = Path(hrtf_relearning.__file__).resolve().parent

from hrtf_relearning.experiment.misc.training_helpers import meta_motion, game_ui
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.experiment.misc.training_helpers.training_targets import set_target_probabilistic
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim

matplotlib.rcParams['figure.raise_window'] = False
logging.getLogger().setLevel('INFO')


# -------------------- Config --------------------
SUBJECT_ID = "test"
HRIR_NAME = "KU100"  # 'KU100', 'kemar', etc.
EAR = None              # or None for binaural

# Sound
SOUND_FILE = None         # None -> pink noise pulses; or 'uso_225ms_9_.wav', etc.
# Graphics
show_ui = True  # todo
SHOW_TF = False  # set to TF or IR to spawn live filter plot

# -------------------- Global HRIR/Sequence --------------------
hrir = hrtf2binsim(HRIR_NAME, EAR, reverb=False, hp_filter=False,
                   convolution='cpu', storage='cpu', overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
HRIR_DIR = ROOT / "data" / "hrtf" / "binsim" / hrir.name
subject = Subject(SUBJECT_ID)
sequence = subject.last_sequence

# Game settings
settings = dict(
    target_size=3,
    target_time=0.5,
    az_range=sequence.settings["azimuth_range"] if sequence else (-30,30),
    ele_range=sequence.settings["elevation_range"] if sequence else (-30,30),
    min_dist=30,
    game_time=90,
    trial_time=20,
    gain=.2)

# -------------------- Helpers --------------------
def make_osc_client(port, ip="127.0.0.1"):
    return udp_client.SimpleUDPClient(ip, port)


def _drain_pose_queue(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except Empty:
            break
    return items

def begin_session(subject):
    """
    Call this once at the beginning of a session to get:
      - session_id (timestamp-based)
      - base_index = current length of subject.trials
    """
    now = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}.{date.strftime("%M")}'
    session_id = datetime.datetime.now().strftime(now)
    base_index = len(subject.trials)
    return {"session_id": session_id, "base_index": base_index}

def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """Wrapper: set /pyBinSimFile to play a file (or synthesize pink noise pulses)."""
    if duration:
        if soundfile:
            sound = slab.Sound.read(HRIR_DIR / "sounds" / soundfile)
            soundfile = "cropped_" + soundfile
            (slab.Sound(sound.data[: int(hrir.samplerate * duration)])
             .ramp(duration=0.03)
             .write(HRIR_DIR / "sounds" / soundfile))
        else:
            soundfile = "noise_pulse.wav"
            slab.Sound.pinknoise(duration, level=77).ramp(duration=0.03).write(HRIR_DIR / "sounds" / soundfile)
    else:
        duration = slab.Sound(HRIR_DIR / "sounds" / soundfile).duration
    logging.debug(f"Setting soundfile: {soundfile}")
    osc_client.send_message("/pyBinSimFile", str(HRIR_DIR / "sounds" / soundfile))
    if sleep:
        time.sleep(duration)

def distance_to_interval(distance):
    max_interval = 350
    min_interval = 75
    steepness = 5
    max_distance = numpy.linalg.norm(numpy.subtract([0, 0], [settings["az_range"][0], settings["ele_range"][0]]))
    if distance <= settings["target_size"]:
        return 0
    norm_dist = (distance - settings["target_size"]) / (max_distance - settings["target_size"])
    norm_dist = numpy.clip(norm_dist, 0, 1)
    scale = numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)
    interval = (min_interval + (max_interval - min_interval) * scale).astype(int)
    return int(interval) / 1000

def plot_current_tf(filter_idx_shared, redraw_interval_s=0.05, kind=SHOW_TF):
    """
    Lives in its own process. Opens a Qt figure and plots the TF of the
    current HRTF (hrir[filter_idx]) whenever the filter index changes.
    """
    # Reuse global `hrir` and its sources
    global hrir
    sources = hrir.sources.vertical_polar  # (N,3), az in [0..360), el linear
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == 'TF':
        ax.set_title("Current Transfer Function")
    if kind == 'IR':
        ax.set_title("Current Impulse Response")
    fig.canvas.manager.set_window_title(f"Live HR{kind}")
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
                ax.set_title(f"{kind} idx {idx}  |  az={az180:.1f}°, el={el0:.1f}°")
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

# -------------------- Subprocess workers --------------------
def binsim_stream():
    # import pybinsim
    pybinsim.logger.setLevel(logging.WARNING)
    logging.info(f"Loading {hrir.name}")
    binsim = pybinsim.BinSim(HRIR_DIR / f"{hrir.name}_training_settings.txt")
    binsim.stream_start()

def pulse_maker(pulse_interval, pulse_state):
    """state: 0 mute, 1 idle, 2 play pulses; interval in seconds (0 => continuous target sound)"""
    osc = make_osc_client(port=10003)
    target_sound = False
    while True:
        if pulse_state.value == 0:
            osc.send_message("/pyBinSimLoudness", 0)
            target_sound = False
        elif pulse_state.value == 1:
            target_sound = False
        elif pulse_state.value == 2:
            osc.send_message("/pyBinSimLoudness", settings["gain"])
            interval = pulse_interval.value
            if interval == 0 and not target_sound:
                play_sound(osc, soundfile=SOUND_FILE, duration=float(settings["target_time"]), sleep=False)
                target_sound = True
            elif interval != 0:
                play_sound(osc, soundfile=SOUND_FILE, duration=float(interval), sleep=True)
                time.sleep(interval)
                target_sound = False
        time.sleep(0.01)

def head_tracker(distance, target, sensor_state, pose_queue, current_trial, plot_filter_idx):
    import logging
    logging.getLogger().setLevel('INFO')
    osc = make_osc_client(port=10000)
    hrtf_sources = hrir.sources.vertical_polar
    #init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    sensor_state.value = 1  # init flag
    while True:
        if sensor_state.value == 2:  # to be calibrated flag
            logging.debug('Calibrating sensor..')
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.1)
            sensor_state.value = 1
            last_idx = -1
        elif sensor_state.value == 3:
            pose = motion_sensor.get_pose()
            pose_raw = numpy.array((motion_sensor.state.pose.yaw, motion_sensor.state.pose.roll))
            logging.debug(f'raw sensor headpose read out: {pose_raw}')
            # set distance for play_session
            try:
                # store: (t_monotonic, current_trial_id, yaw, pitch, roll)
                pose_queue.put((
                    time.time(),
                    int(current_trial.value),
                    float(pose[0]), float(pose[1])
                ))
            except Exception:
                pass
            rel = target[:] - pose
            distance.value = numpy.linalg.norm(rel)
            # find closest HRTF index and send to BinSim
            rel[0] = (-rel[0] + 360) % 360
            rel_target = numpy.array((rel[0], rel[1], hrtf_sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - hrtf_sources, axis=1))
            rel_hrtf_coords = hrtf_sources[filter_idx]
            osc.send_message('/pyBinSim_ds_Filter', [0,0,0,0,0,0,0,0,0,0,
                                                     float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                     0,0,0])
            if filter_idx != last_idx:
                last_idx = filter_idx
                if plot_filter_idx is not None:
                    plot_filter_idx.value = int(filter_idx)
        time.sleep(0.01)

def play_trial(subject, trial_idx, current_trial, target, distance, pulse_interval, pulse_state, sensor_state,
               game_time_left, trial_time, game_time, game_timer,
               target_size, target_time, session_total, last_goal_points, pose_queue):
    """
    Returns: (game_timer, score)
    """
    # Wait-for-Enter phase handled outside this function.
    current_trial.value = int(trial_idx)

    # optional: clear old samples so we only keep this trial's data
    _ = _drain_pose_queue(pose_queue)

    # Start tracking
    score = 0
    trial_timer = 0.0
    time_on_target = 0.0
    count_down = False

    sensor_state.value = 2  # calibrate
    time.sleep(0.1)
    while not sensor_state.value == 3:
        sensor_state.value = 3
        time.sleep(0.05)

    # Start pulser
    pulse_interval.value = distance_to_interval(distance.value)
    time.sleep(0.2)
    pulse_state.value = 2

    logging.debug("Starting trial")
    t0 = time.time()
    while trial_timer < trial_time:
        trial_timer = time.time() - t0
        if game_timer + trial_timer > game_time:
            break

        # update UI timer
        game_time_left.value = max(0.0, game_time - (game_timer + trial_timer))

        # pulse interval based on distance
        pulse_interval.value = distance_to_interval(distance.value)

        # target window / scoring
        if distance.value < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False

        if count_down and time.time() > time_on_target + target_time:  # goal condition
            pulse_state.value = 1  # stop pulse loop (idle but don't mute)

            # Decide score & file
            if trial_timer <= 3.0:
                score = 2
                sfile = 'coins.wav'
                min_audible = 0.35   # let the double-coin poke through
            else:
                score = 1
                sfile = 'coin.wav'
                min_audible = 0.25   # brief head start for single coin

            # 1) UI immediately (coin pop + score bump)
            last_goal_points.value = score
            session_total.value = int(session_total.value + score)

            # 2) Make sure loudness is up for SFX and trigger sound non-blocking
            osc_client.send_message('/pyBinSimLoudness', settings['gain'])

            # Non-blocking SFX trigger
            play_sound(osc_client, soundfile=sfile, duration=None, sleep=False)

            # 3) Give the SFX a short head-start before we exit/mute
            # read actual duration (safe if file is present)
            actual_dur = slab.Sound(HRIR_DIR / 'sounds' / sfile).duration
            time.sleep( actual_dur)

            break

        time.sleep(0.01)

    t1 = time.time()
    logging.info(f"Score: {score}")
    game_timer += trial_timer
    pulse_state.value = 0
    sensor_state.value = 1

    # Collect pose samples; keep those for our trial id & time window
    raw = _drain_pose_queue(pose_queue)
    # raw items are (t_monotonic, trial_id, yaw, pitch, roll)
    trace = [(t, yaw, pitch,) for (t, tid, yaw, pitch) in raw
             if (tid == trial_idx) and (t0 <= t <= t1)]

    # Store on subject
    trial_dict = {
        "trial_idx": int(trial_idx),
        "target": tuple(target.tolist() if hasattr(target, "tolist") else target),
        "pose_trace": trace,  # [(t, yaw, pitch, roll), ...]
        "duration": float(game_timer),
        # add session id if you used begin_session():
        "session_id": globals().get("_current_session_id", None)
    }
    if trial_idx == len(subject.trials):
        subject.trials.append(trial_dict)
    elif 0 <= trial_idx < len(subject.trials):
        subject.trials[trial_idx].update(trial_dict)
    else:
        while len(subject.trials) < (trial_idx + 1):
            subject.trials.append({})
        subject.trials[trial_idx] = trial_dict
    subject.write()
    return game_timer, score


def play_session():
    """
    Main loop: start workers, then run until game_time.
    """
    global osc_client
    osc_client = make_osc_client(port=10003)

    # Shared state for workers
    sensor_state    = mp.Value("i", 0)
    pulse_state     = mp.Value("i", 0)
    target          = mp.Array("f", [0.0, 0.0])
    distance        = mp.Value("f", 0.0)
    pulse_interval  = mp.Value("f", 0.0)
    plot_filter_idx = mp.Value("i", -1)
    current_trial = mp.Value('i', -1)  # -1 = no active trial
    pose_queue = mp.Queue(maxsize=10000)

    # UI shared state
    current_score   = mp.Value("i", 0)   # optional (not used directly)
    session_total   = mp.Value("i", 0)   # what we display as the big number
    game_time_left  = mp.Value("f", float(settings["game_time"]))
    trial_time_left = mp.Value("f", float(settings["trial_time"]))
    last_goal_points = mp.Value("i", 0)  # 0/1/2 → UI coin animation trigger
    enter_pressed    = mp.Value("i", 0)  # UI sets to 1 when user presses Enter
    ui_state         = mp.Value("i", 0)  # 0 idle, 1 awaiting enter, 2 running, 3 over

    # Highscore persistence via Subject
    prev_high = int(getattr(subject, "highscore", 0))
    highscore = mp.Value("i", prev_high)

    # Start UI
    shared = game_ui.UIShared(
        current_score=current_score,
        game_time_left=game_time_left,
        trial_time_left=trial_time_left,
        last_goal_points=last_goal_points,
        session_total=session_total,
        enter_pressed=enter_pressed,
        ui_state=ui_state,
        highscore=highscore)
    ui_proc = mp.Process(target=game_ui.run_ui, args=(shared, ROOT / "data" / "ui" / "highscores.json"))
    ui_proc.start()

    # Start workers
    tracking_worker = mp.Process(target=head_tracker, args=(distance, target, sensor_state, pose_queue,
                                                            current_trial, plot_filter_idx))
    tracking_worker.start()
    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()
    pulse_worker = mp.Process(target=pulse_maker, args=(pulse_interval, pulse_state))
    pulse_worker.start()

    if SHOW_TF:  # start plot_worker
        plot_worker = mp.Process(target=plot_current_tf, args=(plot_filter_idx, 0.05, SHOW_TF), daemon=True)
        plot_worker.start()

    # Wait for tracker init
    while sensor_state.value != 1:
        time.sleep(0.05)

    try:
        while True:  # multiple sessions
            # Reset per-session state
            session_total.value = 0
            last_goal_points.value = 0
            game_time_left.value = float(settings["game_time"])
            enter_pressed.value = 0

            # --- PRE-SESSION PROMPT ---
            ui_state.value = 1  # waiting to start
            while enter_pressed.value == 0:
                time.sleep(0.05)
            enter_pressed.value = 0

            scores = []
            game_timer = 0.0
            game_time_left.value = float(settings["game_time"])

            while game_timer < settings["game_time"]:
                # init new session for recording
                sess = begin_session(subject)
                globals()["_current_session_id"] = sess['session_id']
                trial_idx = sess["base_index"] + 1

                # pick next target
                try:
                    set_target_probabilistic(target, settings, sequence, hrir)
                except AttributeError:
                    logging.debug("Could not load target probabilities")

                # show "Press Enter" overlay and wait for user
                ui_state.value = 1
                enter_pressed.value = 0
                while enter_pressed.value == 0:
                    # keep updating UI timer while we wait
                    game_time_left.value = max(0.0, float(settings["game_time"]) - game_timer)
                    time.sleep(0.05)
                # start trial
                ui_state.value = 2
                enter_pressed.value = 0

                game_timer, score = play_trial(subject, trial_idx, current_trial, target, distance, pulse_interval, pulse_state, sensor_state,
                                            game_time_left,settings["trial_time"], settings["game_time"], game_timer,
                                            settings["target_size"], settings["target_time"], session_total, last_goal_points,
                                               pose_queue)
                scores.append(score)
                # update high score live; persist to Subject
                if session_total.value > highscore.value:
                    highscore.value = int(session_total.value)
                    setattr(subject, "highscore", int(highscore.value))
                    subject.write()  # your Subject.write() persists object

                # if time is up, break
                if game_timer >= settings["game_time"]:
                    break

            # end
            ui_state.value = 3
            pulse_state.value = 1  # idle
            play_sound(osc_client, soundfile='buzzer.wav', duration=None, sleep=True)
            logging.info(f"Game Over! Total Score: {int(session_total.value)}")

            # Show play-again prompt (same big overlay, different text)
            ui_state.value = 3  # session over → "Press Enter to play again"
            enter_pressed.value = 0
            # wait for Enter to start next session
            while enter_pressed.value == 0:
                time.sleep(0.05)
            enter_pressed.value = 0
            # loop continues -> new session

    finally:
        try:
            # Clean up workers
            pulse_state.value = 0
            logging.info("Ending")
            binsim_worker.join()
            pulse_worker.join()
            tracking_worker.join()
            ui_proc.terminate()
            ui_proc.join()
            binsim_worker.terminate()
            pulse_worker.terminate()
            tracking_worker.terminate()
        except Exception:
            pass
        try:
            ui_proc.terminate()
            ui_proc.join()
        except Exception:
            pass




# -------------------- Main --------------------

if __name__ == "__main__":
    try:
        play_session()
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Shutting down.")
