"""Storage format of a training trial's head-tracking trace.

WHY. ``pose_trace`` is ~90% of every subject pickle (GM: 6.1 of 6.75 MB). As a
list of Python float 3-tuples it pickles at 29 bytes per sample; packed as a
float32 array it takes 12. Nothing measurable is lost: the tracker delivers
angles in steps of 1/1024 deg, which float32 holds exactly (anything below 16384 deg),
and times are stored relative to the first sample, so float32 resolves them to
< 1 us over any plausible trial length (measured on GM: max error 0.48 us).

TWO FORMATS, both valid in a trial dict:

    legacy   trial['pose_trace'] = [(t_unix, yaw, pitch), ...]
    packed   trial['pose_trace'] = float32 ndarray (N, 3): (t - t0, yaw, pitch)
             trial['pose_t0']    = t0, the first sample's wall clock (float64)

Never index ``trial['pose_trace']`` directly: in the packed format column 0 is
relative time, so ``trace[0][0]`` is 0.0, not a timestamp, and
``if trial.get('pose_trace')`` raises on an array. Read through
``get_trace`` / ``has_trace`` / ``trace_start`` / ``trace_end``; write through
``pack_trace``.
"""
import numpy

PACKED_DTYPE = numpy.float32
_EMPTY = numpy.empty((0, 3), dtype=float)


def is_packed(trial):
    return isinstance((trial or {}).get("pose_trace"), numpy.ndarray)


def pack_trace(trace):
    """``[(t_unix, yaw, pitch), ...]`` -> ``(array, t0)`` for the packed format.

    Store the result as ``trial['pose_trace'], trial['pose_t0'] = pack_trace(trace)``.
    An empty trace packs to a (0, 3) array and ``t0 = None``.
    """
    rows = numpy.asarray(trace if trace is not None else [], dtype=float)
    if rows.size == 0:
        return numpy.empty((0, 3), dtype=PACKED_DTYPE), None
    rows = rows.reshape(len(rows), -1)[:, :3]
    t0 = float(rows[0, 0])
    packed = rows.astype(PACKED_DTYPE)
    packed[:, 0] = (rows[:, 0] - t0).astype(PACKED_DTYPE)
    return packed, t0


def get_trace(trial):
    """The trace as a float64 (N, 3) array ``(t_unix, yaw, pitch)``, whichever
    format it is stored in. (0, 3) if the trial has none."""
    trace = (trial or {}).get("pose_trace")
    if trace is None:
        return _EMPTY.copy()
    if isinstance(trace, numpy.ndarray):
        out = trace.astype(float)
        if len(out):
            out[:, 0] += float(trial["pose_t0"])
        return out
    if len(trace) == 0:
        return _EMPTY.copy()
    return numpy.asarray(trace, dtype=float).reshape(len(trace), -1)[:, :3]


def has_trace(trial):
    trace = (trial or {}).get("pose_trace")
    return trace is not None and len(trace) > 0


def n_samples(trial):
    trace = (trial or {}).get("pose_trace")
    return 0 if trace is None else len(trace)


def trace_start(trial):
    """Wall clock of the first sample."""
    if is_packed(trial):
        return float(trial["pose_t0"])
    return float(trial["pose_trace"][0][0])


def trace_end(trial):
    """Wall clock of the last sample."""
    trace = trial["pose_trace"]
    if isinstance(trace, numpy.ndarray):
        return float(trial["pose_t0"]) + float(trace[-1, 0])
    return float(trace[-1][0])


def pack_trial(trial):
    """Convert one trial dict to the packed format in place. Returns True if it
    changed. Legacy {} padding and already-packed trials are left alone."""
    if not trial or "pose_trace" not in trial or is_packed(trial):
        return False
    trial["pose_trace"], trial["pose_t0"] = pack_trace(trial["pose_trace"])
    return True


def max_pack_error(original, trial):
    """(max |dt| in s, max |dangle| in deg) between a legacy trace and the
    packed trial made from it — the round-trip check used by the migration."""
    a = numpy.asarray(original, dtype=float).reshape(len(original), -1)[:, :3] \
        if len(original) else _EMPTY
    b = get_trace(trial)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    if not len(a):
        return 0.0, 0.0
    d = numpy.abs(a - b)
    return float(d[:, 0].max()), float(d[:, 1:].max())
