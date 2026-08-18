"""Summarise a training trial's head-tracking trace as a handful of numbers.

WHY. Each training trial stores `pose_trace`, a ~48 Hz list of
``(timestamp, azimuth, elevation)`` samples. It is 97.6% of the volume of
`subject.trials` (IR: 7.9 MB with, 190 KB without), which is what kept the
trials out of the JSON archive entirely — so when a pickle was destroyed, the
whole training record went with it. Reducing each trace to ~15 numbers makes the
archive complete at ~70x less volume.

WHAT THE TRACE ACTUALLY IS. Measured on IR (555 usable trials): the head does
reach the target — median closest approach 1.37 deg on trials flagged
``reached_target``, versus 3.31 deg on those that are not — and the trace then
CONTINUES past it to the end of the trial. So the final sample is not the
response: it is wherever the head happened to be when the trial timed out. Any
metric built on the endpoint is meaningless. Closest approach, and the time at
which it happens, are the well-defined quantities.

THE ONE TO CARE ABOUT is `initial_error`: the direction the head first moves,
relative to where the target actually is, measured over INITIAL_WINDOW_S of
sustained movement. `min_distance` and `time_to_target` mostly restate the
game's own score; this is the only metric here that is about the cue.

The window is a deliberate trade-off, so it is a module constant rather than
buried. The task is closed-loop — the target is spatialised continuously
against the head tracker — so nothing is truly open-loop; 200 ms is simply
shorter than an auditory correction can be issued. Lengthening it buys
correlation by importing the correction you were trying to exclude. Measured on
IR (550 trials), initial_error vs trial score:

    window   median err   Spearman rho        p
     0.10 s      51.8         -0.152     3.3e-04
     0.20 s      53.7         -0.185     1.3e-05   <- default
     0.50 s      53.5         -0.207     9.4e-07
     1.50 s      38.8         -0.282     1.8e-11

Do not read the bottom row as the better metric. Pick the window on what you
want it to mean, then keep it fixed across subjects.

CAVEAT. These are summaries, and a summary you did not think of cannot be
recovered later. Raw traces stay in the pickle; only the JSON archive carries
the reduced form. Keep the pickles.
"""
import math

# Movement onset: first sample exceeding this angular speed, sustained.
ONSET_SPEED_DEG_S = 5.0
ONSET_SUSTAIN_SAMPLES = 3
# Window over which the initial heading is measured, from onset.
INITIAL_WINDOW_S = 0.20
# A reversal is a direction change larger than this, to ignore tracker jitter.
REVERSAL_MIN_DEG = 2.0

METRIC_KEYS = (
    "n_samples", "sample_rate", "duration",
    "start_az", "start_el",
    "latency", "initial_error", "initial_el_correct",
    "min_distance", "time_to_target",
    "path_length", "path_efficiency",
    "peak_speed", "mean_speed",
    "el_excursion", "n_reversals",
)


def _clean(trace):
    """(t, az, el) rows, sorted, with non-finite and duplicate-time rows dropped."""
    rows = []
    for row in trace or ():
        if len(row) < 3:
            continue
        t, az, el = float(row[0]), float(row[1]), float(row[2])
        if not all(map(math.isfinite, (t, az, el))):
            continue
        rows.append((t, az, el))
    rows.sort(key=lambda r: r[0])
    out = []
    for r in rows:
        if out and r[0] <= out[-1][0]:
            continue
        out.append(r)
    return out


def pose_metrics(trial):
    """Reduce ``trial['pose_trace']`` to a flat dict of floats.

    Returns ``{}`` when the trace is too short to say anything (fewer than 5
    samples) — an empty dict, not zeros, so a missing trace is never mistaken
    for a measured one.
    """
    rows = _clean((trial or {}).get("pose_trace"))
    if len(rows) < 5:
        return {}

    t = [r[0] for r in rows]
    az = [r[1] for r in rows]
    el = [r[2] for r in rows]
    t0 = t[0]
    duration = t[-1] - t0
    if duration <= 0:
        return {}

    target = (trial or {}).get("target")
    tgt_az, tgt_el = (float(target[0]), float(target[1])) if target else (None, None)

    # per-sample step, speed
    steps, speeds = [], []
    for i in range(1, len(rows)):
        d = math.hypot(az[i] - az[i - 1], el[i] - el[i - 1])
        dt = t[i] - t[i - 1]
        steps.append(d)
        speeds.append(d / dt if dt > 0 else 0.0)

    path_length = sum(steps)

    # --- movement onset: first sustained excursion above threshold ----------
    onset = None
    for i in range(len(speeds) - ONSET_SUSTAIN_SAMPLES):
        if all(speeds[i + k] > ONSET_SPEED_DEG_S
               for k in range(ONSET_SUSTAIN_SAMPLES)):
            onset = i
            break
    latency = (t[onset] - t0) if onset is not None else None

    # --- initial heading vs. true target direction --------------------------
    # Measured from the pose at onset over INITIAL_WINDOW_S. Both vectors start
    # at the same point, so this is purely "did they set off the right way".
    initial_error = initial_el_correct = None
    if onset is not None and tgt_az is not None:
        t_end = t[onset] + INITIAL_WINDOW_S
        j = onset
        while j + 1 < len(t) and t[j + 1] <= t_end:
            j += 1
        if j > onset:
            move = (az[j] - az[onset], el[j] - el[onset])
            want = (tgt_az - az[onset], tgt_el - el[onset])
            nm = math.hypot(*move)
            nw = math.hypot(*want)
            if nm > 1e-9 and nw > 1e-9:
                cos = (move[0] * want[0] + move[1] * want[1]) / (nm * nw)
                initial_error = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                # Elevation alone — the trained dimension. Reported as "did the
                # head set off the right way", 1.0/0.0, not as a displacement:
                # in 200 ms the head moves only a couple of degrees, so the
                # magnitude is dominated by how fast they happened to move and
                # says nothing about accuracy. The sign does.
                if abs(want[1]) > 1e-9 and abs(move[1]) > 1e-9:
                    initial_el_correct = float(move[1] * want[1] > 0)

    # --- closest approach ---------------------------------------------------
    min_distance = time_to_target = None
    straight = None
    if tgt_az is not None:
        dists = [math.hypot(a - tgt_az, e - tgt_el) for a, e in zip(az, el)]
        k = min(range(len(dists)), key=dists.__getitem__)
        min_distance = dists[k]
        time_to_target = t[k] - t0
        straight = math.hypot(az[k] - az[0], el[k] - el[0])

    # path efficiency over the approach segment only (past the target the
    # trace is uncontrolled drift and would dilute the ratio)
    path_efficiency = None
    if straight is not None:
        approach = sum(steps[:k]) if k > 0 else 0.0
        if approach > 1e-9:
            path_efficiency = min(1.0, straight / approach)

    # --- elevation search behaviour ----------------------------------------
    el_excursion = max(el) - min(el)
    n_reversals = 0
    direction = 0
    anchor = el[0]
    for value in el[1:]:
        delta = value - anchor
        if abs(delta) < REVERSAL_MIN_DEG:
            continue
        sign = 1 if delta > 0 else -1
        if direction and sign != direction:
            n_reversals += 1
        direction = sign
        anchor = value

    return {
        "n_samples": len(rows),
        "sample_rate": (len(rows) - 1) / duration,
        "duration": duration,
        "start_az": az[0],
        "start_el": el[0],
        "latency": latency,
        "initial_error": initial_error,
        "initial_el_correct": initial_el_correct,
        "min_distance": min_distance,
        "time_to_target": time_to_target,
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "peak_speed": max(speeds) if speeds else None,
        "mean_speed": path_length / duration,
        "el_excursion": el_excursion,
        "n_reversals": n_reversals,
    }


def add_pose_metrics(trials, overwrite=False):
    """Attach ``trial['pose_metrics']`` in place. Returns how many were added."""
    n = 0
    for trial in trials or ():
        if not trial:
            continue
        if trial.get("pose_metrics") and not overwrite:
            continue
        m = pose_metrics(trial)
        if m:
            trial["pose_metrics"] = m
            n += 1
    return n
