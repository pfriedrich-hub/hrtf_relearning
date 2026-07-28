"""
protocol_helpers.py — bits every protocol needs, in one place.

These were copy-pasted across expectation_transfer.py,
expectation_transfer_verification.py and the learning_transfer protocols,
which meant four slightly different versions of the
same question. Keep the wording identical across protocols so ratings from
different experiments stay comparable.
"""

import logging
import random

from hrtf_relearning.experiment.localization.Localization_AR import Localization

logger = logging.getLogger(__name__)

# Canonical order, least to most of a real ear on the non-listening side. The
# ladder is RUN in a randomised order but always REPORTED in this one.
#   anchor    the participant's own unmodified HRTF, binaural — the ceiling of
#             the whole delivery chain, not just of the ear treatment. Without
#             it the 0-10 scale has no top and ratings cannot be compared
#             between participants.
#   flat      delta impulse: ITD and broadband ILD kept, no spectral shape,
#             ~5.7 dB too bright in 4-16 kHz, and onset-detection ITD jitter of
#             29 us SD / 62 us max across directions.
#   envelope  own n_keep envelope: band level within ~1.7 dB of native, ITD
#             exact to ~0.5 us because the phase is kept.
#   native    own full DTF — the ceiling for the ear treatment alone.
LADDER_RUNGS = ('anchor', 'flat', 'envelope', 'native')

EXTERNALIZATION_PROMPT = (
    "Externalization (0 = entirely inside your head, "
    "10 = felt like a real external loudspeaker): ")


def collect_externalization_rating(loc_test, ask_plausibility=False):
    """Post-block externalization report, stored on the localization sequence.

    Writes ``externalization_rating`` (and optionally
    ``plausibility_response``) onto the sequence and saves the subject, so the
    rating travels with the run it belongs to rather than living in a notebook.

    Parameters
    ----------
    loc_test : Localization
        The block that was just run.
    ask_plausibility : bool, default False
        Also ask whether the participant could tell the sounds were not real
        loudspeakers (as in expectation_transfer.py).
    """
    print("\n--- Post-block question ---")
    while True:
        raw = input(EXTERNALIZATION_PROMPT).strip()
        try:
            rating = float(raw)
            break
        except ValueError:
            print("Please enter a number 0-10.")

    sequence = loc_test.subject.localization[loc_test.filename]
    sequence.externalization_rating = rating
    message = f"externalization={rating}"

    if ask_plausibility:
        answer = input("Could you tell these sounds were NOT real loudspeakers? "
                       "(y/n): ").strip().lower()
        sequence.plausibility_response = answer.startswith("y")
        message += f", told_apart={sequence.plausibility_response}"

    loc_test.subject.write()
    print(f"Recorded: {message}\n")
    return rating


def externalization_check(subject, hrir_settings, loc_settings, label=''):
    """Short block run for the externalization rating only.

    Deliberately NOT on the analysis grid — pass a coarse ``loc_settings``
    (e.g. ``sector_size=(14, 14)``, ``targets_per_sector=1``, ~10 trials). The
    point is the subjective rating, not elevation-gain statistics, and a full
    block per condition is too expensive to run three times on day 1.

    Use it to compare other-ear treatments back to back before committing a
    participant: if externalization is still poor with the least invasive
    setting, the ear treatment is not what is breaking it.
    """
    print("\n" + "=" * 70)
    print(f"EXTERNALIZATION CHECK   {label}")
    print(f"  ear={hrir_settings.get('ear')}  other_ear={hrir_settings.get('other_ear')}"
          f"  az={loc_settings.get('azimuth_range')}")
    print("=" * 70)
    test = Localization(subject, hrir_settings, loc_settings=loc_settings)
    test.run()
    collect_externalization_rating(test)
    return test


def externalization_ladder(subject, settings_for, rungs=LADDER_RUNGS, seed=None,
                           shuffle=True):
    """Run the externalization rungs in a randomised order and summarise them.

    Parameters
    ----------
    subject : hr.Subject
    settings_for : callable
        ``settings_for(rung) -> (hrir_settings, loc_settings)``. The protocol
        owns the SOFA names and grid; this helper only owns the order, the
        tagging and the summary.
    rungs : sequence of str
        Which rungs to run. Defaults to :data:`LADDER_RUNGS`.
    seed : hashable, optional
        Pass the subject id. The order is then random ACROSS participants but
        reproducible for a given one, so a session can be resumed or repeated
        without silently changing the design.
    shuffle : bool, default True
        Randomise the order. Fixed order confounds the comparison with whatever
        happens over the first half hour in the setup — practice, fatigue,
        settling into the head tracker — so only turn this off deliberately.

    Each block is tagged on its sequence with ``ladder_rung``,
    ``ladder_position`` and ``ladder_order``, so the blocks can be found later
    and the order can be modelled.

    Returns ``{rung: rating}``.
    """
    order = list(rungs)
    if shuffle:
        random.Random(seed).shuffle(order)

    print("\n" + "#" * 70)
    print(f"EXTERNALIZATION LADDER   order: {' -> '.join(order)}"
          + (f"   (seed={seed!r})" if seed is not None else ""))
    print(f"{len(order)} short blocks, rating only — NOT on the analysis grid")
    print("#" * 70)

    from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
        localization_accuracy)

    results = {}
    for position, rung in enumerate(order, start=1):
        hrir_settings, loc_settings = settings_for(rung)
        test = externalization_check(
            subject, hrir_settings, loc_settings,
            label=f"rung {position}/{len(order)}: {rung}")
        sequence = test.subject.localization[test.filename]
        sequence.ladder_rung = rung
        sequence.ladder_position = position
        sequence.ladder_order = list(order)
        test.subject.write()

        try:
            gain, ele_rmse, ele_sd, _, _, _ = localization_accuracy(sequence)
        except Exception as exc:      # a 10-trial block can degenerate
            logger.warning('localization_accuracy failed for rung %s: %s', rung, exc)
            gain = ele_rmse = ele_sd = float('nan')
        results[rung] = {
            'rating': getattr(sequence, 'externalization_rating', None),
            'elevation_gain': gain, 'ele_rmse': ele_rmse, 'ele_sd': ele_sd,
            'n_trials': len(sequence.data) if getattr(sequence, 'data', None) else 0,
            'run_position': position,
        }
        print(f"  -> EG {gain:.2f}   ele RMSE {ele_rmse:.1f} deg   "
              f"ele SD {ele_sd:.1f} deg   ({results[rung]['n_trials']} trials)")

    print("\n" + "-" * 74)
    print("EXTERNALIZATION LADDER — canonical order, run position in brackets")
    print(f"  {'rung':>9} {'rating':>7} {'EG':>7} {'RMSE':>7} {'SD':>7} {'n':>4}  run")
    for rung in rungs:
        if rung not in results:
            continue
        row = results[rung]
        rating = row['rating']
        print(f"  {rung:>9} {rating if rating is not None else '--':>7} "
              f"{row['elevation_gain']:>7.2f} {row['ele_rmse']:>7.1f} "
              f"{row['ele_sd']:>7.1f} {row['n_trials']:>4}  [{row['run_position']}]")
    print("-" * 74)
    print("RATING\n"
          "  rises across the rungs   -> the ear treatment is the limiting factor;\n"
          "                              use the highest rung the design allows\n"
          "  flat ~ envelope ~ native -> the ear treatment is NOT the cause; look at\n"
          "                              the delivery chain (HP EQ, midline ILD offset)\n"
          "  anchor also low          -> the whole chain is the problem, and no cue\n"
          "                              manipulation will be measurable until it is\n"
          "                              fixed\n"
          "ELEVATION GAIN — treat as coarse. At ~10 trials the SE of the slope is\n"
          "  roughly 0.18, i.e. +-0.35 at 95%. That separates 'cue destroyed' from\n"
          "  'cue intact' and nothing finer: a difference between two composite\n"
          "  strengths of 0.1-0.2 in EG is NOT resolvable here. Aim for an acute EG\n"
          "  of 0.3-0.5 on the manipulated rungs; if you need to choose between them\n"
          "  on the numbers rather than on the extremes, re-run those two rungs at\n"
          "  ~40 trials.")
    return results
