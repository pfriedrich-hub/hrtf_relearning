import logging
import time
from pathlib import Path
import multiprocessing as mp
from queue import Empty
pose_queue = mp.SimpleQueue()
current_trial = mp.Value('i', -1)  # -1 means "no active trial"
import numpy
import slab
from pythonosc import udp_client

from experiment.Subject import Subject
from experiment.misc.training_helpers.training_targets import set_target_probabilistic
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.misc.training_helpers import game_ui

logging.getLogger().setLevel('INFO')

# -------------------- Config --------------------
SUBJECT_ID = "PF"
HRIR_NAME = "pf_hr_itld"  # 'KU100', 'kemar', etc.
EAR = None              # or None for binaural (your unilateral training)
REVERB = True

# Sound/Pulse
SOUND_FILE = None         # None -> pink noise pulses; or 'uso_225ms_9_.wav', etc.

# -------------------- Global HRIR/Sequence --------------------
hrir = hrtf2binsim(HRIR_NAME, EAR, REVERB, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
HRIR_DIR = Path.cwd() / "data" / "hrtf" / "binsim" / hrir.name
sequence = Subject(SUBJECT_ID).last_sequence  # last localization sequence (your code)

# Game settings
settings = dict(
    target_size=3,
    target_time=0.5,
    az_range=sequence.settings["azimuth_range"],
    ele_range=sequence.settings["elevation_range"],
    min_dist=30,
    game_time=90,
    trial_time=10,
    gain=0.2,
)

SHOW_TF = False  # set True to spawn live TF plot process

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
            slab.Sound.pinknoise(duration).ramp(duration=0.03).write(HRIR_DIR / "sounds" / soundfile)
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

# -------------------- Subprocess workers --------------------
def binsim_stream():
    import pybinsim
    pybinsim.logger.setLevel(logging.ERROR)
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

def head_tracker(distance, target, sensor_state, plot_filter_idx):
    """Dummy tracker that 'walks' towards target; updates pyBinSim filters accordingly."""
    import logging
    logging.getLogger().setLevel('INFO')
    osc = make_osc_client(port=10000)
    hrtf_sources = hrir.sources.vertical_polar
    sensor_state.value = 1  # initialized
    while True:
        if sensor_state.value == 2:
            # "calibration"
            step = 0.1
            pose = numpy.array([0.0, 0.0])
            sensor_state.value = 1
            last_idx = -1
        elif sensor_state.value == 3:
            # move towards target
            direction = (target - pose)
            direction = direction / numpy.linalg.norm(direction)
            pose = pose + step * direction

            try:
                # store: (t_monotonic, current_trial_id, yaw, pitch, roll)
                pose_queue.put((
                    time.monotonic(),
                    int(current_trial.value),
                    float(pose[0]), float(pose[1]), float(pose[2])
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

# -------------------- Trials & Session --------------------
def play_trial(subject,
               target,
               pose,
               sensor_state,
               trial_idx,
               ui_shared=None,
               trial_time_s=3.0,
               distance=None,                     # optional: multiprocessing.Value updated by your tracker
               distance_threshold=6.0,            # deg; trial can end early if within this distance
               ui_fps=60.0):
    """
    Full trial with sounds, frame/UI loop, early-finish on distance threshold, and pose-trace logging.
    Keeps head-tracker + pyBinSim logic untouched. Uses your hooks if present; otherwise safe fallbacks.

    Parameters
    ----------
    subject : object
        Your subject container; we'll add/extend subject.trials[trial_idx] with "pose_trace".
    target : array-like
        Shared target [az, el, r] or similar; stored onto the trial entry for provenance.
    pose : any
        Kept only for signature parity with your caller; actual pose samples come from the tracker queue.
    sensor_state : multiprocessing.Value
        Your usual state flag (1/2/3). We'll not modify it here.
    trial_idx : int
        Trial index for storage and tagging.
    ui_shared : optional
        Your UI shared object/dict; we update lightweight fields if present.
    trial_time_s : float
        Hard cap duration.
    distance : optional multiprocessing.Value
        If provided, we allow early completion when distance.value <= distance_threshold.
    distance_threshold : float
        Angular threshold for early success.
    ui_fps : float
        Max UI update rate.

    Returns
    -------
    (game_timer, score)
    """

    import time
    from queue import Empty
    import numpy as numpy
    import logging

    # ---------- helpers ----------
    def _drain_pose_queue(q):
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except Empty:
                break
        return items

    def _ensure_subject_trials(s):
        if not hasattr(s, "trials") or s.trials is None:
            s.trials = []

    def _call_if_exists(name, *args, **kwargs):
        fn = globals().get(name, None)
        if callable(fn):
            try:
                return True, fn(*args, **kwargs)
            except Exception as e:
                logging.debug(f"{name} raised {e}")
        return False, None

    # Fallback audio if you don't have your own hooks:
    def _fallback_play_start_tone():
        try:
            import slab  # keeps your stack consistent
            tone = slab.Sound.tone(frequency=880, duration=0.12, level=0.0)  # short ping
            tone.play()
        except Exception:
            pass  # stay silent if slab/audio is unavailable

    def _fallback_play_success_tone():
        try:
            import slab
            tone = slab.Sound.tone(frequency=1320, duration=0.12, level=0.0)
            tone.play()
        except Exception:
            pass

    # ---------- setup ----------
    _ensure_subject_trials(subject)

    # Mark this trial active so the tracker tags samples with our trial id
    current_trial.value = int(trial_idx)

    # Optional: clear leftover samples before we start
    _ = _drain_pose_queue(pose_queue)

    # Reset basic UI fields if they exist (non-fatal)
    try:
        if ui_shared is not None:
            if hasattr(ui_shared, "trial_idx"):
                ui_shared.trial_idx.value = int(trial_idx)
            if hasattr(ui_shared, "status_text"):
                ui_shared.status_text.value = "Trial running"
            if hasattr(ui_shared, "coins"):
                # no-op here; session code should manage total coins
                pass
    except Exception:
        pass

    # Start audio (prefer your provided function)
    called, _ = _call_if_exists("start_trial_audio", target)
    if not called:
        _fallback_play_start_tone()

    # ---------- main trial loop ----------
    t0 = time.monotonic()
    last_ui = t0
    ui_dt = 1.0 / float(ui_fps)
    success = False

    while True:
        now = time.monotonic()
        elapsed = now - t0

        # UI refresh
        if (now - last_ui) >= ui_dt:
            # Prefer your custom UI updater if present
            _call_if_exists("update_game_ui", ui_shared, elapsed)
            # Also write basic timer into UI if there is a shared value
            try:
                if ui_shared is not None:
                    if hasattr(ui_shared, "game_timer"):
                        ui_shared.game_timer.value = float(elapsed)
            except Exception:
                pass
            last_ui = now

        # Early finish if we're within distance threshold (if provided)
        if distance is not None:
            try:
                if float(distance.value) <= float(distance_threshold):
                    success = True
                    break
            except Exception:
                pass

        # Hard cap
        if elapsed >= float(trial_time_s):
            break

        # Event polling (optional user hook can end the trial)
        called, ev = _call_if_exists("poll_game_events")
        if called and isinstance(ev, dict) and ev.get("done", False):
            success = bool(ev.get("success", False))
            break

        time.sleep(0.003)  # responsive + low CPU

    t1 = time.monotonic()
    game_timer = t1 - t0

    # ---------- teardown ----------
    # Prefer your stop handler; otherwise play a simple success/stop cue
    called, _ = _call_if_exists("stop_trial_audio")
    if not called:
        if success:
            _fallback_play_success_tone()
        # else stay quiet on timeouts to keep semantics simple

    # Mark no active trial to avoid late samples belonging here
    current_trial.value = -1

    # Collect pose samples; keep those for our trial id & time window
    raw = _drain_pose_queue(pose_queue)
    # raw items are (t_monotonic, trial_id, yaw, pitch, roll)
    trace = [(t, yaw, pitch, roll) for (t, tid, yaw, pitch, roll) in raw
             if (tid == trial_idx) and (t0 <= t <= t1)]

    # Store on subject
    tgt_tuple = tuple(target.tolist() if hasattr(target, "tolist") else target)
    trial_dict = {
        "trial_idx": int(trial_idx),
        "target": tgt_tuple,
        "pose_trace": trace,          # [(t, yaw, pitch, roll), ...]
        "success": bool(success),
        "duration": float(game_timer),
    }
    if trial_idx == len(subject.trials):
        subject.trials.append(trial_dict)
    elif 0 <= trial_idx < len(subject.trials):
        subject.trials[trial_idx].update(trial_dict)
    else:
        while len(subject.trials) < (trial_idx + 1):
            subject.trials.append({})
        subject.trials[trial_idx] = trial_dict

    # Update simple UI fields on end
    try:
        if ui_shared is not None:
            if hasattr(ui_shared, "status_text"):
                ui_shared.status_text.value = "Success!" if success else "Time up"
            if hasattr(ui_shared, "last_goal_points") and success:
                ui_shared.last_goal_points.value = 1
    except Exception:
        pass

    # Score: prefer your compute function; else keep subject.score or 0/1 for success
    called, maybe_score = _call_if_exists("compute_trial_score", subject, trial_idx)
    if called and (maybe_score is not None):
        score = maybe_score
    elif hasattr(subject, "score"):
        score = subject.score
    else:
        score = 1 if success else 0

    # Persist if available
    try:
        subject.write()
    except Exception as e:
        logging.debug(f"subject.write() failed: {e}")

    return game_timer, score

def play_trial(distance, pulse_interval, pulse_state, sensor_state,
               ui_state, enter_pressed, game_time_left,
               trial_time, game_time, game_timer,
               target_size, target_time,
               current_score, session_total, last_goal_points):
    """
    Returns: (game_timer, score)
    """
    # Wait-for-Enter phase handled outside this function.
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

    logging.info(f"Score: {score}")
    game_timer += trial_timer
    pulse_state.value = 0
    sensor_state.value = 1
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

    # UI shared state
    current_score   = mp.Value("i", 0)   # optional (not used directly)
    session_total   = mp.Value("i", 0)   # what we display as the big number
    game_time_left  = mp.Value("f", float(settings["game_time"]))
    trial_time_left = mp.Value("f", float(settings["trial_time"]))
    last_goal_points = mp.Value("i", 0)  # 0/1/2 → UI coin animation trigger
    enter_pressed    = mp.Value("i", 0)  # UI sets to 1 when user presses Enter
    ui_state         = mp.Value("i", 0)  # 0 idle, 1 awaiting enter, 2 running, 3 over

    # Highscore persistence via Subject
    subject = Subject(SUBJECT_ID)
    try:
        # try to read previous highscore from subject; fall back to 0
        prev_high = int(getattr(subject, "highscore", 0))
    except Exception:
        prev_high = 0
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
        highscore=highscore,
    )
    ui_proc = mp.Process(target=game_ui.run_ui, args=(shared, Path.cwd() / "data" / "ui" / "highscores.json"))
    ui_proc.start()

    # Start workers
    tracking_worker = mp.Process(target=head_tracker, args=(distance, target, sensor_state, plot_filter_idx))
    tracking_worker.start()
    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()
    pulse_worker = mp.Process(target=pulse_maker, args=(pulse_interval, pulse_state))
    pulse_worker.start()

    # Wait for tracker init
    while sensor_state.value != 1:
        time.sleep(0.05)

    # Game loop
    scores = []
    game_timer = 0.0
    game_time_left.value = float(settings["game_time"])

    while game_timer < settings["game_time"]:
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

        game_timer, score = play_trial(
            distance, pulse_interval, pulse_state, sensor_state,
            ui_state, enter_pressed, game_time_left,
            settings["trial_time"], settings["game_time"], game_timer,
            settings["target_size"], settings["target_time"],
            current_score, session_total, last_goal_points
        )
        scores.append(score)
        # todo overwrites trajectories from play_trial subject (they life in different subprocesses)
        # update high score live; persist to Subject
        if session_total.value > highscore.value:
            highscore.value = int(session_total.value)
            try:
                setattr(subject, "highscore", int(highscore.value))
                subject.write()  # your Subject.write() persists object
            except Exception as e:
                logging.debug(f"Subject.write() failed: {e}")

        # if time is up, break
        if game_timer >= settings["game_time"]:
            break

    # end
    ui_state.value = 3
    pulse_state.value = 1  # idle
    play_sound(osc_client, soundfile='buzzer.wav', duration=None, sleep=True)
    logging.info(f"Game Over! Total Score: {int(session_total.value)}")

    # Clean up workers
    pulse_state.value = 0
    logging.info("Ending")
    binsim_worker.join()
    pulse_worker.join()
    tracking_worker.join()
    ui_proc.terminate()
    ui_proc.join()

    #todo ask to play again in the ui

# -------------------- Main --------------------
if __name__ == "__main__":
    play_session()

