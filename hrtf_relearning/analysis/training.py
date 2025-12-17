import time
from hrtf_relearning.experiment import Subject

SUBJECT_ID = 'PF'

# ---------- Pose trace analysis & visualization ----------

import numpy as numpy
import matplotlib.pyplot as plt

def _unwrap_deg(angle_deg):
    """Unwrap circular angles given in degrees."""
    return numpy.rad2deg(numpy.unwrap(numpy.deg2rad(numpy.asarray(angle_deg, dtype=float))))

def _trace_to_arrays(pose_trace):
    """
    pose_trace: list of (t, yaw, pitch) or (t, yaw, pitch, roll)
    returns dict with arrays: t_rel, yaw, pitch, yaw_unw, pitch_unw, vyaw, vpitch, vspeed
    """
    if len(pose_trace) == 0:
        return None

    t = numpy.asarray([s[0] for s in pose_trace], dtype=float)
    yaw = numpy.asarray([s[1] for s in pose_trace], dtype=float)
    pitch = numpy.asarray([s[2] for s in pose_trace], dtype=float)

    t_rel = t - t[0]
    yaw_unw = _unwrap_deg(yaw)
    pitch_unw = _unwrap_deg(pitch)

    # robust dt (handles any jitter)
    dt = numpy.gradient(t_rel)
    # guard against zeros
    dt = numpy.where(dt == 0.0, numpy.finfo(float).eps, dt)

    vyaw = numpy.gradient(yaw_unw) / dt          # deg/s
    vpitch = numpy.gradient(pitch_unw) / dt      # deg/s
    vspeed = numpy.sqrt(vyaw**2 + vpitch**2)     # resultant speed in angle-space

    return dict(t_rel=t_rel, yaw=yaw, pitch=pitch,
                yaw_unw=yaw_unw, pitch_unw=pitch_unw,
                vyaw=vyaw, vpitch=vpitch, vspeed=vspeed)

def plot_trial_pose(subject, trial_idx, show_velocity=True):
    """
    Plots yaw/pitch vs time (and optionally angular velocities).
    Uses matplotlib only; no external deps beyond your stack.
    """
    trial = subject.trials[trial_idx]
    pose_trace = trial.get("pose_trace", [])
    data = _trace_to_arrays(pose_trace)
    if data is None:
        print(f"No pose samples for trial {trial_idx}")
        return

    t = data["t_rel"]

    # Plot yaw/pitch (degrees)
    plt.figure()
    plt.title(f"Trial {trial_idx} – Yaw/Pitch (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.plot(t, data["yaw"], label="Yaw")
    plt.plot(t, data["pitch"], label="Pitch")
    plt.legend()
    plt.show()

    if show_velocity:
        # Plot angular velocities (deg/s)
        plt.figure()
        plt.title(f"Trial {trial_idx} – Angular Velocity (deg/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("deg/s")
        plt.plot(t, data["vyaw"], label="Yaw velocity")
        plt.plot(t, data["vpitch"], label="Pitch velocity")
        plt.plot(t, data["vspeed"], label="Resultant speed")
        plt.legend()
        plt.show()

# ---------- Optional: replay through pyBinSim at original timing ----------

def replay_trace_with_binsim(subject, trial_idx, hrir, speed=1.0, ip="127.0.0.1", port=10000):
    """
    Replays a recorded pose trace by sending the same *relative* HRTF indices to pyBinSim
    with the original timestamps (scaled by `speed`).
    - `speed` > 1.0 = faster; < 1.0 = slower.
    Uses your vertical_polar HRTF grid and the same mirror/conversion you use in training.
    """
    from pythonosc import udp_client
    osc = udp_client.SimpleUDPClient(ip, port)
    sources = hrir.sources.vertical_polar

    trial = subject.trials[trial_idx]
    pose_trace = trial.get("pose_trace", [])
    if not pose_trace:
        print(f"No pose samples for trial {trial_idx}")
        return

    # Build arrays
    t = numpy.asarray([s[0] for s in pose_trace], dtype=float)
    yaw = numpy.asarray([s[1] for s in pose_trace], dtype=float)
    pitch = numpy.asarray([s[2] for s in pose_trace], dtype=float)

    # relative timing
    t_rel = t - t[0]
    if speed <= 0:
        speed = 1.0
    t_rel = t_rel / float(speed)

    # Helper to convert a (yaw, pitch) "pose" to the relative HRTF coords as in your tracker
    def _pose_to_rel_hrtf_coords(yaw_deg, pitch_deg, tgt_tuple):
        # relative coords: target - pose
        rel_yaw = tgt_tuple[0] - yaw_deg
        rel_pitch = tgt_tuple[1] - pitch_deg
        # mirror az to HRTF convention [0..360)
        rel_yaw_hrtf = (-rel_yaw + 360.0) % 360.0
        return numpy.array((rel_yaw_hrtf, rel_pitch, sources[0, 2]), dtype=float)

    # We need the *target* used in this trial to compute relative HRTF index
    # (stored with the trial by your play_trial code)
    tgt = trial.get("target", None)
    if tgt is None:
        print("This trial is missing its target -> cannot compute relative filter indices.")
        return
    tgt = numpy.asarray(tgt, dtype=float)
    if tgt.shape[0] == 2:
        tgt = numpy.array([tgt[0], tgt[1], sources[0, 2]], dtype=float)

    # Now stream through the trace with the original timing
    t0 = time.time()
    for i in range(len(t_rel)):
        # sleep until the next stamped time
        while (time.time() - t0) < t_rel[i]:
            time.sleep(0.001)

        rel_target = _pose_to_rel_hrtf_coords(yaw[i], pitch[i], tgt)
        # find nearest HRTF index and send
        idx = int(numpy.argmin(numpy.linalg.norm(rel_target - sources, axis=1)))
        rel_hrtf_coords = sources[idx]
        osc.send_message('/pyBinSim_ds_Filter',
                         [0,0,0,0,0,0,0,0,0,0,
                          float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                          0,0,0])

    print(f"Replayed trial {trial_idx} at speed x{speed:.2f}.")

if __name__ == "__main__":
    # After a session, load your subject
    subject = Subject(SUBJECT_ID)

    # 1) Visualize
    plot_trial_pose(subject, trial_idx=subject.trials[-1]["trial_idx"], show_velocity=True)

    # # 2) (Optional) Replay through pyBinSim at original timing (or faster/slower)
    # #    Uses the target stored with the trial to compute relative indices.
    # replay_trace_with_binsim(subject, trial_idx=subject.trials[-1]["trial_idx"], hrir=hrir, speed=1.0)
