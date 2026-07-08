"""
learning_transfer.py

Adaptation-transfer experiment protocol runner.

Guides you through the localization tests of the experiment so you never have to
hand-edit parameters in Localization_AR.py between runs:

    Day 1            native      binaural, native SOFA, full field   (familiarization/reference)
                     baseline_A  monaural trained ear, MODIFIED, trained field    (naive ref for A)
                     baseline_D  monaural untrained ear via MIRROR, MODIFIED,
                                 mirrored field                                   (naive ref for D)
    Adaptation days  daily       monaural trained ear, MODIFIED, trained hemifield
                     (training game runs separately -- see Training.py)
    Final day        A           trained ear,   same loc (= trained hemifield)   [baseline retest]
                     B           trained ear,   mirrored loc
                     C           untrained ear, same loc
                     D           untrained ear, mirrored loc                      [MAIN transfer]

The day-1 baselines use the SAME configs as final A and D (same ear/mirror/field/
filter) but pre-training, so pre-vs-post isolates learning. baseline_D delivers the
mirrored to-be-trained-ear filter to the untrained ear (matching D), NOT the
untrained ear's own DTF. All one-sided tests share the matched sampling grid
sector_size=(7,14), elevation_range=(-35,35), targets_per_sector=3, az~=0 excluded.

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as a
plain script. Nothing here loops or blocks on input; rerun any cell as needed
(e.g. redo a single final-day condition).

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT.
------------------------------------------------------------------------------
"""

# %% imports and config ------------------------------------------------------
import csv

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.utils import paths

# The ONLY thing you set per session. Everything else (cue type, trained ear,
# final-day block order) is loaded from the counterbalance sheet below, keyed by
# this id. On day 1, just write each subject's id into the 'subject' column of:
#   data/documentation/exp1_transfer_block_order.csv   (replace an '(assign)' cell)
SUBJECT_ID = "JS"

CSV_PATH = paths.DOCUMENTATION_DIR / "exp1_transfer_block_order.csv"


def _load_subject_params(subject_id, csv_path=CSV_PATH):
    """Look up cue_type, trained_ear and final block order for this subject."""
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject", "").strip() == subject_id:
                cue   = row["cue_type"].strip()
                ear   = row["trained_ear"].strip()
                order = [c.strip() for c in row["block_order"].split("-")]
                return cue, ear, order
    raise ValueError(
        f"Subject '{subject_id}' not found in the 'subject' column of\n  {csv_path}\n"
        f"Add its id to a row there (replacing an '(assign)' cell) before running."
    )


CUE_SET, TRAINED_EAR, FINAL_ORDER = _load_subject_params(SUBJECT_ID)

# SOFA file names (under data/hrtf/sofa/<subject_id>/<name>.sofa)
NATIVE_SOFA   = f"{SUBJECT_ID}"              # individual measured HRTF (day-1 native test)
MODIFIED_SOFA = f"{SUBJECT_ID}_{CUE_SET}"   # modified set (baseline + all training/testing)

HP = "DT990"   # headphone EQ profile

# --- shared localization sampling grid (do not change without re-checking the
#     baseline-vs-final comparability; see project notes) ---
SECTOR_SIZE        = (7, 14)
ELEVATION_RANGE    = (-35, 35)
TARGETS_PER_SECTOR = 3
MIN_DISTANCE       = 20
GAIN               = 0.2
STIM               = "noise"
MIDLINE_TOL        = 1.0   # one-sided tests drop sources with |az| <= this (degrees)

FULL_FIELD = (-35, 35)

# --- derived geometry (do not edit) ---
if TRAINED_EAR == "left":
    UNTRAINED_EAR = "right"
    TRAINED_HEMI  = (-35, 0)
    MIRRORED_HEMI = (0, 35)
elif TRAINED_EAR == "right":
    UNTRAINED_EAR = "left"
    TRAINED_HEMI  = (0, 35)
    MIRRORED_HEMI = (-35, 0)
else:
    raise ValueError("TRAINED_EAR must be 'left' or 'right'.")

# NOTE on the 4 final conditions. The cue filter is ALWAYS the trained ear's
# own filter (ear=TRAINED_EAR). What changes:
#   - mirror=False -> the trained filter stays on the trained ear.
#   - mirror=True  -> mirror_hrtf swaps L/R channels (and negates source az), so
#     the *exact, identical* trained HRIR is delivered to the UNTRAINED ear --
#     NOT the untrained ear's own (flattened/synth) DTF. This is the whole point
#     of the transfer test, and is why we use mirror rather than ear=UNTRAINED.
#   - azimuth selects the physical location: trained vs mirrored hemifield.
# Because mirror negates source azimuth, D (mirror=True, mirrored hemifield) is
# physically identical to A (mirror=False, trained hemifield) with L/R swapped.


def hrir_settings(sofa_name, ear=None, mirror=False):
    return {
        "name": sofa_name,
        "subject_id": SUBJECT_ID,
        "ear": ear,            # None -> binaural; 'left'/'right' -> flatten the other ear
        "mirror": mirror,
        "reverb": True,
        "drr": 20,
        "hp_filter": True,
        "hp": HP,
        "convolution": "cuda",
        "storage": "cuda",
    }


def loc_settings(azimuth_range, exclude_midline=False):
    return {
        "kind": "sectors",
        "azimuth_range": azimuth_range,
        "elevation_range": ELEVATION_RANGE,
        "targets_per_speaker": 3,          # unused for 'sectors'; kept for compatibility
        "targets_per_sector": TARGETS_PER_SECTOR,
        "min_distance": MIN_DISTANCE,
        "gain": GAIN,
        "stim": STIM,
        "sector_size": SECTOR_SIZE,
        "replace": False,
        "exclude_midline": exclude_midline,  # drop az~=0 in one-sided tests
        "midline_tol": MIDLINE_TOL,
    }


# each phase: key -> (label, when, sofa, ear, mirror, azimuth_range, description)
PHASES = {
    "native":     ("Native reference",        "Day 1", NATIVE_SOFA,   None,        False, FULL_FIELD,    "binaural, native HRTF, full field"),
    "baseline_A": ("Baseline A: trained/same", "Day 1", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "naive trained ear, trained filter (matches final A)"),
    "baseline_D": ("Baseline D: untrnd/mirr",  "Day 1", MODIFIED_SOFA, TRAINED_EAR, True,  MIRRORED_HEMI, "naive untrained ear, MIRRORED trained filter (matches final D)"),
    "daily":    ("Daily training test",   "Adaptation days", MODIFIED_SOFA, TRAINED_EAR,   False, TRAINED_HEMI,  "monaural trained ear, trained hemifield"),
    "A":        ("Final A: trained/same", "Final day",       MODIFIED_SOFA, TRAINED_EAR,   False, TRAINED_HEMI,  "trained ear, same locations (baseline retest)"),
    "B":        ("Final B: trained/mirr", "Final day",       MODIFIED_SOFA, TRAINED_EAR,   False, MIRRORED_HEMI, "trained ear, mirrored locations"),
    "C":        ("Final C: untrnd/same",  "Final day",       MODIFIED_SOFA, TRAINED_EAR,   True,  TRAINED_HEMI,  "untrained ear (mirrored trained HRIR), same locations"),
    "D":        ("Final D: untrnd/mirr",  "Final day",       MODIFIED_SOFA, TRAINED_EAR,   True,  MIRRORED_HEMI, "untrained ear (mirrored trained HRIR), mirrored locations  [MAIN]"),
}


def _n_sectors(rng, size):
    import numpy
    return len(numpy.arange(rng[0] + size / 2, rng[1], size))


def _est_trials(az_range):
    return (_n_sectors(az_range, SECTOR_SIZE[0])
            * _n_sectors(ELEVATION_RANGE, SECTOR_SIZE[1])
            * TARGETS_PER_SECTOR)


def _describe(key):
    label, when, sofa, ear, mirror, az, desc = PHASES[key]
    midline = "excluded" if tuple(az) != tuple(FULL_FIELD) else "kept"
    return (f"{label}  [{when}]\n"
            f"      {desc}\n"
            f"      SOFA={sofa}  ear={ear or 'binaural'}  mirror={mirror}\n"
            f"      azimuth={az}  elevation={ELEVATION_RANGE}  sector={SECTOR_SIZE}  "
            f"tps={TARGETS_PER_SECTOR}  midline az=0 {midline}\n"
            f"      ~{_est_trials(az)} trials")


def run_phase(key, subject):
    label, when, sofa, ear, mirror, az, desc = PHASES[key]
    print("\n" + "=" * 70)
    print(f"RUNNING:  {_describe(key)}")
    print("=" * 70)
    one_sided = tuple(az) != tuple(FULL_FIELD)   # drop az~=0 only in one-sided tests
    test = Localization(subject, hrir_settings(sofa, ear=ear, mirror=mirror),
                        loc_settings=loc_settings(az, exclude_midline=one_sided))
    test.run()
    print(f"Done: {test.filename}")


def show_status(subject):
    print("\n" + "-" * 70)
    print(f"SUBJECT: {SUBJECT_ID}    CUE: {CUE_SET}    TRAINED EAR: {TRAINED_EAR}    "
          f"UNTRAINED: {UNTRAINED_EAR}")
    print(f"  (loaded from {CSV_PATH.name})")
    print(f"hemifields -> trained {TRAINED_HEMI}, mirrored {MIRRORED_HEMI}")
    print(f"modified SOFA: {MODIFIED_SOFA}    final-day order: {'-'.join(FINAL_ORDER)}")
    done = list(getattr(subject, "localization", {}).keys())
    if done:
        print(f"\nLocalization runs already on file for {SUBJECT_ID} ({len(done)}):")
        for k in done:
            print(f"   - {k}")
    else:
        print(f"\nNo localization runs on file yet for {SUBJECT_ID}.")
    print("-" * 70)


# %% status check (rerun anytime) --------------------------------------------
subject = hr.Subject(SUBJECT_ID)
show_status(subject)

# %% day 1: native reference ---------------------------------------------------
run_phase("native", subject)

# %% day 1: baseline A -- trained ear, same loc (matches final A) --------------
run_phase("baseline_A", subject)

# %% day 1: baseline D -- untrained ear, mirrored loc (matches final D) --------
run_phase("baseline_D", subject)

# %% adaptation days: daily training test ---------------------------------------
# Rerun this cell once per adaptation day (the training game itself runs
# separately -- see Training.py).
run_phase("daily", subject)

# %% final day: all 4 conditions in this subject's counterbalanced order --------
# Order is loaded per-subject from data/documentation/exp1_transfer_block_order.csv
# (FINAL_ORDER). To redo a single condition instead, use one of the cells below.
print(f"Running final tests in order: {FINAL_ORDER}")
for _key in FINAL_ORDER:
    run_phase(_key, subject)
    subject = hr.Subject(SUBJECT_ID)   # reload after each write

# %% final day: A -- trained ear, same locations (redo individually) -----------
run_phase("A", subject)

# %% final day: B -- trained ear, mirrored locations (redo individually) -------
run_phase("B", subject)

# %% final day: C -- untrained ear, same locations (redo individually) ---------
run_phase("C", subject)

# %% final day: D -- untrained ear, mirrored locations [MAIN] (redo individually)
run_phase("D", subject)
