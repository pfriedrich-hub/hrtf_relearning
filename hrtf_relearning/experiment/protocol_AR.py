"""
Adaptation-transfer experiment protocol runner.

Guides you through the localization tests of the experiment so you never have to
hand-edit parameters in Localization_AR.py between runs:

    Day 1            native     binaural, native SOFA, full field      (native reference)
                     baseline   binaural, MODIFIED SOFA, full field    (transfer baseline)
    Adaptation days  daily      monaural trained ear, MODIFIED, trained hemifield
                     (training game runs separately -- see Training.py)
    Final day        final_A    trained ear,   same loc (= trained hemifield)   [baseline retest]
                     final_B    trained ear,   mirrored loc
                     final_C    untrained ear, same loc
                     final_D    untrained ear, mirrored loc                      [MAIN transfer]

All tests share the matched sampling grid agreed for valid baseline-vs-final
comparison: sector_size=(7,14), elevation_range=(-35,35), targets_per_sector=3.
The one-sided final/daily tests are exact subsets of the full-field baseline grid.

Pick the test from the menu; the script builds the right HRIR/binsim files and
localization settings, shows you what it will do, and runs it after you confirm.

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT, THEN: python protocol_AR.py
------------------------------------------------------------------------------
"""

import csv
import sys
from pathlib import Path

# make this script runnable directly, like Localization_AR.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrtf_relearning as hr
from Localization_AR import Localization

# =============================================================================
# CONFIG  --  edit per participant
# =============================================================================
# The ONLY thing you set per session. Everything else (cue type, trained ear,
# final-day block order) is loaded from the counterbalance sheet below, keyed by
# this id. On day 1, just write each subject's id into the 'subject' column of:
#   data/documentation/exp1_transfer_block_order.csv   (replace an '(assign)' cell)
SUBJECT_ID = "JS"

CSV_PATH = hr.PATH / "data" / "documentation" / "exp1_transfer_block_order.csv"


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

# =============================================================================
# Derived geometry  (do not edit)
# =============================================================================
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


# =============================================================================
# Protocol phases
# =============================================================================
# each phase: key -> (label, when, sofa, ear, mirror, azimuth_range, description)
PHASES = {
    "native":   ("Native reference",      "Day 1",           NATIVE_SOFA,   None,          False, FULL_FIELD,    "binaural, native HRTF, full field"),
    "baseline": ("Transfer baseline",     "Day 1",           MODIFIED_SOFA, None,          False, FULL_FIELD,    "binaural, MODIFIED HRTF, full field"),
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
    print(f"ABOUT TO RUN:  {_describe(key)}")
    print("=" * 70)
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Skipped.")
        return
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


def menu():
    subject = hr.Subject(SUBJECT_ID)
    keys = list(PHASES.keys())
    while True:
        show_status(subject)
        print("\nSelect a test to run:")
        for i, k in enumerate(keys, 1):
            label = PHASES[k][0]
            when  = PHASES[k][1]
            print(f"  {i}. [{k}]  {label}  ({when})")
        print(f"  F. Run ALL final tests in order {FINAL_ORDER}")
        print("  r. Refresh status")
        print("  q. Quit")
        choice = input("\n> ").strip()

        if choice.lower() == "q":
            print("Bye.")
            return
        if choice.lower() == "r":
            subject = hr.Subject(SUBJECT_ID)   # reload from disk
            continue
        if choice.lower() == "f":
            print(f"\nRunning final tests in order: {FINAL_ORDER}")
            for k in FINAL_ORDER:
                run_phase(k, subject)
                subject = hr.Subject(SUBJECT_ID)  # reload after each write
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            run_phase(keys[int(choice) - 1], subject)
            subject = hr.Subject(SUBJECT_ID)      # reload after each write
            continue
        print("Invalid choice.")


if __name__ == "__main__":
    menu()
