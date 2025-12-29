"""
Training.py (refactored)

Multiprocessing-safe refactor:
- No heavy initialization at module import time (important for spawn on macOS/Windows).
- All side-effectful setup (Subject init, HRIR conversion, pybinsim startup, UI startup) happens inside `main()`.
- Worker processes only receive simple, picklable arguments and do their own lightweight setup.

Notes
-----
- Any process that *must* use pyBinSim or the HRTF object will necessarily initialize those pieces in that process.
  This refactor ensures they are only initialized where needed (usually exactly once per worker).
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, Optional, Tuple

import numpy
import slab
from pythonosc import udp_client

from hrtf_relearning.experiment.misc.training_helpers import meta_motion, game_ui
from hrtf_relearning.experiment.misc.training_helpers.training_targets import set_target_probabilistic


# -------------------- Configuration --------------------

@dataclass(frozen=True)
class TrainingConfig:
    subject_id: str = "TEST"
    hrir_name: str = "KU100"   # see hrtf2binsim(HRIR_NAME, ...)
    ear: str = "both"          # "left" | "right" | "both"
    sound_file: Optional[str] = None  # None -> pink noise pulses
    show_ui: bool = True
    show_tf: Optional[str] = None     # None | "TF" | "IR"
    # BinSim / OSC ports used in this script
    osc_binsim_port: int = 10003
    osc_tracker_port: int = 10000
    # UI OSC port / shared memory is handled by game_ui


def make_osc_client(port: int, ip: str = "127.0.0.1") -> udp_client.SimpleUDPClient:
    return udp_client.SimpleUDPClient(ip, port)


def _drain_pose_queue(q: mp.Queue) -> list:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except Empty:
            break
    return items


def begin_session(subject) -> Dict[str, Any]:
    """
    Call once at the beginning of a session to get:
      - session_id (timestamp-based)
      - base_index = current length of subject.trials
    """
    session_id = datetime.datetime.now().strftime("%d.%m_%H.%M")
    base_index = len(getattr(subject, "trials", []))
    return {"session_id": session_id, "base_index": base_index}


def play_sound(osc_client: udp_client.SimpleUDPClient,
               soundfile: Optional[str] = None,
               duration: Optional[float] = None,
               sleep: bool = False) -> None:
    """
    Trigger playback in BinSim (or your OSC-controlled audio backend).
    """
    if soundfile is None:
        # Use internal noise / pulse mechanism
        osc_client.send_message("/pyBinSimPlay", 1)
        if duration is not None and sleep:
            time.sleep(float(duration))
        return

    osc_client.send_message("/pyBinSimSoundFile", soundfile)
    osc_client.send_message("/pyBinSimPlay", 1)
    if duration is not None and sleep:
        time.sleep(float(duration))


def distance_to_interval(distance: float, settings: Dict[str, Any]) -> float:
    """
    Map distance to pulse interval (seconds).
    If distance is within target window, return 0 (continuous target sound).
    """
    max_interval = 350
    min_interval = 75
    steepness = 5

    max_distance = float(
        numpy.linalg.norm(
            numpy.subtract([0, 0], [settings["az_range"][0], settings["ele_range"][0]])
        )
    )

    if distance <= settings["target_size"]:
        return 0.0

    norm_dist = (distance - settings["target_size"]) / (max_distance - settings["target_size"])
    norm_dist = float(numpy.clip(norm_dist, 0.0, 1.0))

    interval = max_interval / (1 + numpy.exp(-steepness * (norm_dist - 0.5))) + min_interval
    interval = float(numpy.clip(interval, min_interval, max_interval))
    return interval / 1000.0  # ms -> s


def plot_current_tf(filter_idx_shared: mp.Value,
                    hrir_name: str,
                    ear: str,
                    root: Path,
                    redraw_interval_s: float = 0.05,
                    kind: str = "TF") -> None:
    """
    Optional helper process. Loads the HRTF and plots TF/IR for the current filter index.
    This process *will* initialize/load the HRTF (by design).
    """
    import matplotlib
    from matplotlib import pyplot as plt

    from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim

    # Load / build HRTF in this process (spawn-safe)
    hrir = hrtf2binsim(
        hrir_name, ear,
        reverb=True, hp_filter=True,
        convolution="cpu", storage="cpu", overwrite=False
    )
    sources = hrir.sources.vertical_polar

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.canvas.manager.set_window_title(f"Live HR{kind}")

    last_idx = -1
    while True:
        idx = int(filter_idx_shared.value)
        if idx >= 0 and idx != last_idx:
            last_idx = idx
            ax.cla()

            if kind == "TF":
                tf = hrir[idx].spectrum()
                ax.semilogx(tf.frequency, tf.data[:, 0], label="left")
                ax.semilogx(tf.frequency, tf.data[:, 1], label="right")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.legend()
            elif kind == "IR":
                times = numpy.linspace(0, hrir[idx].n_samples / hrir.samplerate, hrir[idx].n_samples)
                ax.plot(times, hrir[idx].data[:, 0], label="left")
                ax.plot(times, hrir[idx].data[:, 1], label="right")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="upper right")

            az0, el0 = sources[idx, 0], sources[idx, 1]
            az180 = (az0 + 180) % 360 - 180
            ax.set_title(f"{kind} idx {idx} | az={az180:.1f}°, el={el0:.1f}°")
            ax.grid(True, which="both", linestyle=":", linewidth=0.6)
            fig.canvas.draw_idle()
            plt.pause(0.001)

        time.sleep(float(redraw_interval_s))


def binsim_stream(binsim_settings_path: Path, log_name: str = "BinSim") -> None:
    """
    Worker that starts BinSim streaming. This process is the *only* one that should touch pybinsim.
    """
    import pybinsim  # delayed import -> prevents init in every spawned worker unnecessarily

    logging.getLogger().setLevel(logging.INFO)
    pybinsim.logger.setLevel(logging.WARNING)
    logging.info("[%s] Loading BinSim settings: %s", log_name, binsim_settings_path)
    binsim = pybinsim.BinSim(str(binsim_settings_path))
    binsim.stream_start()


def pulse_maker(pulse_interval: mp.Value, pulse_state: mp.Value, osc_binsim_port: int) -> None:
    """state: 0 mute, 1 idle, 2 play pulses; interval in seconds (0 => continuous target sound)"""
    logging.getLogger().setLevel(logging.INFO)
    osc = make_osc_client(port=osc_binsim_port)
    target_sound = False

    while True:
        if int(pulse_state.value) == 0:
            osc.send_message("/pyBinSimLoudness", 0)
            target_sound = False
        elif int(pulse_state.value) == 1:
            target_sound = False
        elif int(pulse_state.value) == 2:
            # continuous target tone while on target (interval==0)
            if float(pulse_interval.value) == 0 and not target_sound:
                osc.send_message("/pyBinSimLoudness", 1)
                target_sound = True
            elif float(pulse_interval.value) > 0:
                osc.send_message("/pyBinSimLoudness", 1)
                time.sleep(float(pulse_interval.value))
                osc.send_message("/pyBinSimLoudness", 0)
                time.sleep(float(pulse_interval.value))
                target_sound = False
        time.sleep(0.001)


def head_tracker(distance: mp.Value,
                target: mp.Array,
                sensor_state: mp.Value,
                pose_queue: mp.Queue,
                current_trial: mp.Value,
                plot_filter_idx: Optional[mp.Value],
                hrtf_sources: numpy.ndarray,
                osc_tracker_port: int) -> None:
    """
    Reads head pose, computes distance to target and nearest HRTF index, and sends filter updates via OSC.
    """
    logging.getLogger().setLevel(logging.INFO)
    osc = make_osc_client(port=osc_tracker_port)

    # init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    sensor_state.value = 1  # init flag

    last_idx = -1

    while True:
        if int(sensor_state.value) == 2:  # calibrate
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.01)
            sensor_state.value = 1

        if int(sensor_state.value) == 0:  # tracking off
            time.sleep(0.01)
            continue

        pose = motion_sensor.get_orientation()  # yaw, pitch, roll (deg)
        # store pose trace only during active trials
        if int(current_trial.value) >= 0:
            try:
                pose_queue.put_nowait((time.time(), float(pose[0]), float(pose[1]), float(pose[2])))
            except Exception:
                pass

        # distance to current target (az, el)
        rel = numpy.asarray(target[:], dtype=float) - numpy.asarray([pose[0], pose[1]], dtype=float)
        distance.value = float(numpy.linalg.norm(rel))

        # Find closest HRTF index and send to BinSim via OSC
        rel[0] = (-rel[0] + 360) % 360
        rel_target = numpy.array((rel[0], rel[1], hrtf_sources[0, 2]), dtype=float)
        filter_idx = int(numpy.argmin(numpy.linalg.norm(rel_target - hrtf_sources, axis=1)))
        rel_hrtf_coords = hrtf_sources[filter_idx]

        osc.send_message(
            "/pyBinSim_ds_Filter",
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
             0, 0, 0]
        )

        if filter_idx != last_idx:
            last_idx = filter_idx
            if plot_filter_idx is not None:
                plot_filter_idx.value = int(filter_idx)

        time.sleep(0.01)


def play_trial(subject,
               trial_idx: int,
               session_id: str,
               settings: Dict[str, Any],
               current_trial: mp.Value,
               target: mp.Array,
               distance: mp.Value,
               pulse_interval: mp.Value,
               pulse_state: mp.Value,
               sensor_state: mp.Value,
               game_time_left: mp.Value,
               game_timer: float,
               session_total: mp.Value,
               last_goal_points: mp.Value,
               pose_queue: mp.Queue) -> Tuple[float, int]:
    """
    Run one trial. Returns updated game_timer and score for this trial.
    """
    current_trial.value = int(trial_idx)

    # clear old samples so we only keep this trial's data
    _ = _drain_pose_queue(pose_queue)

    score = 0
    trial_timer = 0.0
    time_on_target = time.time()
    count_down = False

    # start pulses / tracking
    pulse_state.value = 2
    sensor_state.value = 1

    t0 = time.time()
    while trial_timer < float(settings["trial_time"]):
        trial_timer = time.time() - t0

        # stop if game time would exceed max
        if game_timer + trial_timer > float(settings["game_time"]):
            break

        # update UI timer
        game_time_left.value = max(0.0, float(settings["game_time"]) - (game_timer + trial_timer))

        # pulse interval based on distance
        pulse_interval.value = float(distance_to_interval(float(distance.value), settings))

        # target window / scoring
        if float(distance.value) < float(settings["target_size"]):
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            count_down = False

        if count_down and (time.time() - time_on_target) >= float(settings["target_time"]):
            score = int(100)  # or whatever scoring you use
            break

        time.sleep(0.01)

    # stop pulses (keep tracking off)
    pulse_state.value = 1
    sensor_state.value = 0

    # grab pose trace
    trace = _drain_pose_queue(pose_queue)

    # write trial info
    trial_dict = {
        "trial_idx": int(trial_idx),
        "target": (list(target[:]) if hasattr(target[:], "tolist") else target[:]),
        "pose_trace": trace,
        "duration": float(trial_timer),
        "session_id": session_id,
        "score": int(score),
    }

    if trial_idx == len(subject.trials):
        subject.trials.append(trial_dict)
    elif 0 <= trial_idx < len(subject.trials):
        subject.trials[trial_idx] = trial_dict

    # update aggregate
    session_total.value = float(session_total.value) + float(trial_timer)
    if score > 0:
        last_goal_points.value = int(score)

    return float(game_timer + trial_timer), int(score)


def play_session(config: TrainingConfig) -> None:
    """
    Main loop (single process): initialize Subject/HRTF, start worker processes,
    then run trials until `game_time` is elapsed.
    """
    # Delayed imports (prevents side effects on worker import)
    import hrtf_relearning
    from hrtf_relearning.experiment.Subject import Subject
    from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim

    root = Path(hrtf_relearning.__file__).resolve().parent

    # Build/load HRTF ONCE in the main process (to get sources + to create BinSim files if needed)
    hrir = hrtf2binsim(
        config.hrir_name, config.ear,
        reverb=True, hp_filter=True,
        convolution="cpu", storage="cpu", overwrite=False
    )
    slab.set_default_samplerate(hrir.samplerate)
    hrtf_sources = numpy.asarray(hrir.sources.vertical_polar)

    # BinSim settings file path
    hrir_dir = root / "data" / "hrtf" / "binsim" / hrir.name
    binsim_settings_path = hrir_dir / f"{hrir.name}_training_settings.txt"

    # Subject (main process only!)
    subject = Subject(config.subject_id)
    sequence = getattr(subject, "last_sequence", None)

    # Game settings
    settings: Dict[str, Any] = dict(
        target_size=3,
        target_time=0.5,
        az_range=sequence.settings["azimuth_range"] if sequence else (-35, 35),
        ele_range=sequence.settings["elevation_range"] if sequence else (-35, 35),
        min_dist=30,
        game_time=90,
        trial_time=15,
        score_time=6,
        gain=.15,
    )

    # OSC to BinSim (main process uses this for coarse control)
    osc_client = make_osc_client(port=config.osc_binsim_port)

    # Shared state
    sensor_state = mp.Value("i", 0)
    pulse_state = mp.Value("i", 0)
    target = mp.Array("f", [0.0, 0.0])
    distance = mp.Value("f", 0.0)
    pulse_interval = mp.Value("f", 0.0)
    plot_filter_idx = mp.Value("i", -1)
    current_trial = mp.Value("i", -1)
    pose_queue: mp.Queue = mp.Queue(maxsize=10000)

    # UI shared struct
    game_timer = 0.0
    highscore = mp.Value("i", 0)
    current_score   = mp.Value("i", 0)   # optional (not used directly)
    session_total   = mp.Value("i", 0)   # what we display as the big number
    game_time_left  = mp.Value("f", float(settings["game_time"]))
    trial_time_left = mp.Value("f", float(settings["trial_time"]))
    last_goal_points = mp.Value("i", 0)  # 0/1/2 → UI coin animation trigger
    enter_pressed    = mp.Value("i", 0)  # UI sets to 1 when user presses Enter
    ui_state         = mp.Value("i", 0)  # 0 idle, 1 awaiting enter, 2 running, 3 over


    # UI process
    ui_proc = None
    if config.show_ui:
        shared = game_ui.UIShared(
            current_score=current_score,
            game_time_left=game_time_left,
            trial_time_left=trial_time_left,
            last_goal_points=last_goal_points,
            session_total=session_total,
            enter_pressed=enter_pressed,
            ui_state=ui_state,
            highscore=highscore)
        ui_proc = mp.Process(
            target=game_ui.run_ui,
            args=(shared, root / "data" / "ui" / "highscores.json"),
            name="UI",
        )
        ui_proc.start()

    # Optional plot process (loads HRTF in that process)
    plot_proc = None
    if config.show_tf in ("TF", "IR"):
        plot_proc = mp.Process(
            target=plot_current_tf,
            args=(plot_filter_idx, config.hrir_name, config.ear, root, 0.05, config.show_tf),
            name="PlotTF",
        )
        plot_proc.start()

    # Workers
    tracking_worker = mp.Process(
        target=head_tracker,
        args=(distance, target, sensor_state, pose_queue, current_trial, plot_filter_idx,
              hrtf_sources, config.osc_tracker_port),
        name="HeadTracker",
    )
    binsim_worker = mp.Process(
        target=binsim_stream,
        args=(binsim_settings_path, hrir.name),
        name="BinSim",
    )
    pulse_worker = mp.Process(
        target=pulse_maker,
        args=(pulse_interval, pulse_state, config.osc_binsim_port),
        name="PulseMaker",
    )

    tracking_worker.start()
    binsim_worker.start()
    pulse_worker.start()

    try:
        # Wait for workers to report ready
        t_wait = time.time()
        while int(sensor_state.value) == 0:
            if time.time() - t_wait > 10:
                logging.warning("Head tracker did not initialize within 10s.")
                break
            time.sleep(0.05)

        # Main game loop
        while game_timer < float(settings["game_time"]):
            sess = begin_session(subject)
            session_id = sess["session_id"]
            trial_idx = int(sess["base_index"]) + 1

            # pick next target (main process only)
            try:
                set_target_probabilistic(target, settings, sequence, hrir)
            except Exception:
                logging.debug("Could not load target probabilities; using fallback target.")
                # fallback: random target within bounds
                target[0] = float(numpy.random.uniform(settings["az_range"][0], settings["az_range"][1]))
                target[1] = float(numpy.random.uniform(settings["ele_range"][0], settings["ele_range"][1]))

            # show "Press Enter" overlay and wait
            if config.show_ui:
                ui_state.value = 1
                enter_pressed.value = 0
                while int(enter_pressed.value) == 0:
                    game_time_left.value = max(0.0, float(settings["game_time"]) - float(game_timer))
                    time.sleep(0.02)
                ui_state.value = 2  # active trial

            # run trial
            game_timer, score = play_trial(
                subject=subject,
                trial_idx=trial_idx,
                session_id=session_id,
                settings=settings,
                current_trial=current_trial,
                target=target,
                distance=distance,
                pulse_interval=pulse_interval,
                pulse_state=pulse_state,
                sensor_state=sensor_state,
                game_time_left=game_time_left,
                game_timer=game_timer,
                session_total=session_total,
                last_goal_points=last_goal_points,
                pose_queue=pose_queue,
            )

            # brief score display time
            if config.show_ui:
                ui_state.value = 3
                time.sleep(float(settings["score_time"]))
                ui_state.value = 0

            # persist subject after each trial
            try:
                subject.save()
            except Exception:
                logging.exception("Failed to save subject.")

        logging.info("Training finished.")

    finally:
        # shutdown
        try:
            pulse_state.value = 0
            sensor_state.value = 0
            current_trial.value = -1
        except Exception:
            pass

        for proc in (tracking_worker, binsim_worker, pulse_worker, plot_proc, ui_proc):
            if proc is None:
                continue
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=2)
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = TrainingConfig(
        subject_id="TEST",      # TODO: set your subject id
        hrir_name="KU100",      # TODO
        ear="both",
        sound_file=None,
        show_ui=True,
        show_tf=None,           # set to "TF" or "IR" to spawn plotting process
    )
    play_session(config)


if __name__ == "__main__":
    # Explicit spawn makes behaviour consistent across platforms and avoids fork-related audio issues.
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Shutting down.")
