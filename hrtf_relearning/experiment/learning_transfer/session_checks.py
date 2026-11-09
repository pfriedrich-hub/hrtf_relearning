"""Console reminders for the learning_transfer protocol -- PRINT ONLY.

Nothing here stops a cell or blocks a participant; every function prints a
short line and, where a number falls outside what our learners showed, a
"[!]" reminder. The thresholds come from the 2026-11-09 review of 18
donor-detail participants (8 learners, 10 non-learners), see
docs/success_predictors.md for the numbers behind each one.
"""
import datetime

import numpy

from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy, polar_error)

#: free-field (dome) elevation gain below which a participant should not be
#: trained: GM .30, KH .44, IK .48 were below it; the lowest learner was FS .52
MIN_FREEFIELD_GAIN = 0.5
#: own-HRTF (AR) elevation gain: KH .41 was below it; the lowest learner LP .64
MIN_OWN_GAIN = 0.5
#: head pitch at the tracker zero (calibration pose); our blocks sit at about
#: -5..-14 deg, KH's laser offset showed as -20..-27 deg
CALIB_PITCH_RANGE = (-17.0, 0.0)
#: mean response elevation minus target elevation; KH +13..+27, GM +13
MAX_ELEVATION_BIAS = 10.0
#: day-2 post-training gain minus the day-1 A baseline: learners +0.18..+0.39,
#: non-learners <= +0.11 (IK +0.22 the one exception)
DAY2_MIN_GAIN_STEP = 0.15


def _summary(seq):
    """(gain, polar error, elevation bias, calibration pitch) of one block."""
    gain = float(localization_accuracy(seq)[0])
    pe = float(polar_error(seq)[0])
    bias = pitch = numpy.nan
    try:
        d = numpy.asarray(seq.data, float).reshape(-1, 2, 2)
        bias = float(numpy.mean(d[:, 0, 1] - d[:, 1, 1]))
    except (ValueError, TypeError):
        pass
    poses = getattr(seq, 'calibration_poses', None)
    if poses:
        try:
            pitch = float(numpy.mean([p[1] for p in poses]))
        except (TypeError, IndexError):
            pass
    return gain, pe, bias, pitch


def block_check(seq, label=None):
    """One line per localization block: gain, polar error, offset, pitch."""
    gain, pe, bias, pitch = _summary(seq)
    name = label or getattr(seq, 'name', None) or getattr(seq, 'phase', '') or 'block'
    print(f"  check {name}: EG {gain:.2f}  PE {pe:.1f} deg  elevation bias "
          f"{bias:+.1f} deg  calibration pitch {pitch:+.1f} deg")
    lo, hi = CALIB_PITCH_RANGE
    if numpy.isfinite(pitch) and not lo <= pitch <= hi:
        print(f"  [!] calibration pitch {pitch:+.1f} deg is outside {lo:.0f}..{hi:.0f}: "
              f"check the laser / headband seating before the next block (KH -20..-27)")
    if numpy.isfinite(bias) and abs(bias) > MAX_ELEVATION_BIAS:
        print(f"  [!] responses sit {bias:+.1f} deg off the targets on average: "
              f"likely a laser or tracker-zero offset, not perception (KH, GM)")
    return gain, pe, bias, pitch


def baseline_check(seq, kind='own HRTF'):
    """Own-HRTF (AR) block: is this participant worth training?"""
    gain, pe, bias, pitch = block_check(seq, label=f"{kind} baseline")
    floor = MIN_FREEFIELD_GAIN if kind == 'free field' else MIN_OWN_GAIN
    if gain < floor:
        print(f"  [!] {kind} elevation gain {gain:.2f} < {floor:.2f}: this participant "
              f"does not localize well enough to train (your rule; KH .41 own, .44 free field)")
    return gain, pe


def dome_check(subject, day=None):
    """All free-field (dome) blocks of one day, with the training rule."""
    day = day or datetime.date.today().strftime('%d.%m')
    runs = [(k, s) for k, s in subject.localization.items()
            if k.endswith('_dome') and k.split('_')[1] == day]
    if not runs:
        print(f"  no dome blocks on {day} for {subject.id}")
        return None
    gains = []
    for key, seq in runs:
        g, *_ = block_check(seq, label=key)
        gains.append(g)
    best = max(gains)
    if best < MIN_FREEFIELD_GAIN:
        print(f"  [!] best free-field gain {best:.2f} < {MIN_FREEFIELD_GAIN:.2f}: "
              f"do not train this participant (your rule; GM .30, KH .44, IK .48)")
    return best


def head_radius_check(radius, fallback):
    """Reminder when the native is about to be built at the fallback radius."""
    if abs(float(radius) - float(fallback)) < 1e-9:
        print(f"  [!] head radius = FALLBACK {fallback:.4f} m: the fit did not pass. "
              f"Re-seat the mics and redo step 0 before building -- SHE and JO were "
              f"built at the fallback and needed rebuilds mid-experiment")
    else:
        print(f"  head radius {float(radius):.4f} m (fitted)")


def _key_date(key):
    day, month = (int(x) for x in key.split('_')[1].split('.'))
    return month, day


def progress_check(subject, post_seq):
    """Post-training test vs the day-1 A baseline, with the day-2 stopping rule."""
    base = [(k, s) for k, s in subject.localization.items()
            if getattr(s, 'phase', None) == 'baseline_A']
    if not base:
        print("  no day-1 baseline A on file (phase 'baseline_A') -- cannot compare")
        return None
    key0, seq0 = min(base, key=lambda ks: _key_date(ks[0]) + (ks[0],))
    g0, p0, *_ = _summary(seq0)
    g1, p1, *_ = _summary(post_seq)
    d0 = datetime.date(datetime.date.today().year, *_key_date(key0))
    name = getattr(post_seq, 'name', None) or ''
    d1 = datetime.date(datetime.date.today().year, *_key_date(name)) if name.count('_') >= 2 else datetime.date.today()
    day = (d1 - d0).days + 1
    step = g1 - g0
    print(f"  progress, day {day}: post-training EG {g1:.2f} / PE {p1:.1f} deg vs day-1 A "
          f"EG {g0:.2f} / PE {p0:.1f} deg -> gain {step:+.2f}, PE {p0 - p1:+.1f} deg (positive = better)")
    if day >= 2:
        if step < DAY2_MIN_GAIN_STEP:
            print(f"  [!] gain step {step:+.2f} < +{DAY2_MIN_GAIN_STEP:.2f}: every learner so far "
                  f"was at +0.18 or more by day 2, 9 of 10 non-learners at +0.11 or less "
                  f"-- consider stopping this participant")
        else:
            print(f"  on track: learners so far +0.18..+0.39 by day 2")
    return step
