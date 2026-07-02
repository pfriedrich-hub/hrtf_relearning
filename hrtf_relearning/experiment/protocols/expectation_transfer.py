"""
expectation_transfer.py

Protocol for the expectation-transfer experiment (see
documentation/expectation_transfer_design.md for hypothesis and full design
rationale). Tests whether real dome-loudspeaker exposure recalibrates
subsequent virtual (AR / pybinsim) localization, against a matched-effort
AR-only control:

    dome group:    AR_pre -> Dome (real speakers)              -> AR_post
    control group: AR_pre -> AR_filler (virtual, same n as dome) -> AR_post

AR_pre and AR_post are IDENTICAL across groups (same HRIR, same
vertical-midline locations shared with the dome speaker layout, same trial
count) -- only the middle block differs. This isolates the real-speaker-
specific effect from mere task repetition/practice (see design doc sec 2-3).

Group ('dome' or 'control') is loaded per-subject from
data/documentation/expectation_transfer_block_order.csv, assigned by
recruitment order (alternating). Assumes the subject already has a recorded
individual HRIR + calibrated headphone filter (HRIR_Recording.py).

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as
a plain script. Only the two block-2 cells matching this subject's assigned
group should be run; the other is guarded to raise instead of silently
running the wrong condition. Rerun any cell as needed (e.g. redo AR_post).
"""

# %% imports and config ------------------------------------------------------
import csv

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.Localization_dome import LocalizationDome

SUBJECT_ID = "JS"   # edit per participant

CSV_PATH = hr.PATH / "data" / "documentation" / "expectation_transfer_block_order.csv"


def _load_group(subject_id, csv_path=CSV_PATH):
    """Look up this subject's group ('dome' or 'control') from the counterbalance sheet."""
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject", "").strip() == subject_id:
                group = row["group"].strip()
                if group not in ("dome", "control"):
                    raise ValueError(f"group must be 'dome' or 'control', got {group!r}")
                return group
    raise ValueError(
        f"Subject '{subject_id}' not found in the 'subject' column of\n  {csv_path}\n"
        f"Add its id to a row there (replacing an '(assign)' cell) before running."
    )


GROUP = _load_group(SUBJECT_ID)

HRIR_NAME = SUBJECT_ID   # individual measured HRIR (native, unmodified)
HP = "DT990"             # headphone EQ profile

TARGETS_PER_SPEAKER = 3  # -> 21 trials/block on the 7 vertical-midline positions,
MIN_DISTANCE = 15        # shared by all four block types (AR_pre/AR_post/dome/AR_filler)
GAIN = 0.2

# Vertical-midline AR settings -- matches the dome speaker layout (see
# Localization_dome.LocalizationDome and HRIR_Recording.py step 5), so AR
# blocks and the dome block sample the same physical directions.
AR_MIDLINE_SETTINGS = {
    "kind": "standard",
    "azimuth_range": (-1, 1),
    "elevation_range": (-35, 35),
    "targets_per_speaker": TARGETS_PER_SPEAKER,
    "min_distance": MIN_DISTANCE,
    "gain": GAIN,
    "stim": "noise",
}

DOME_SETTINGS = {
    "targets_per_speaker": TARGETS_PER_SPEAKER,
    "min_distance": MIN_DISTANCE,
}


def hrir_settings():
    return {
        "name": HRIR_NAME,
        "subject_id": SUBJECT_ID,
        "ear": None,       # binaural
        "mirror": False,
        "reverb": True,
        "drr": 20,
        "hp_filter": True,
        "hp": HP,
        "convolution": "cuda",
        "storage": "cuda",
    }


# %% helper: post-block externalization rating --------------------------------
def collect_externalization_rating(loc_test):
    """
    Short post-block subjective report, console-collected. Run after AR_pre and
    AR_post only (not after dome/AR_filler). New instrumentation -- reword or
    replace freely; see documentation/expectation_transfer_design.md sec 5.
    """
    print("\n--- Post-block questions ---")
    while True:
        raw = input("Externalization (0 = entirely inside your head, "
                     "10 = felt like a real external loudspeaker): ").strip()
        try:
            rating = float(raw)
            break
        except ValueError:
            print("Please enter a number 0-10.")
    plausible = input("Could you tell these sounds were NOT real loudspeakers? (y/n): ").strip().lower()
    sequence = loc_test.subject.localization[loc_test.filename]
    sequence.externalization_rating = rating
    sequence.plausibility_response = plausible.startswith("y")
    loc_test.subject.write()
    print(f"Recorded: externalization={rating}, told_apart={sequence.plausibility_response}\n")


# %% status check (rerun anytime) ----------------------------------------------
subject = hr.Subject(SUBJECT_ID)
print(f"SUBJECT: {SUBJECT_ID}    GROUP: {GROUP}    (loaded from {CSV_PATH.name})")
done = list(getattr(subject, "localization", {}).keys())
if done:
    print(f"Localization runs already on file ({len(done)}):")
    for k in done:
        print(f"   - {k}")
else:
    print("No localization runs on file yet.")

# %% block 1: AR_pre -- naive virtual localization ------------------------------
ar_pre = Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS)
ar_pre.run()
collect_externalization_rating(ar_pre)

# %% block 2: exposure -- DOME, real loudspeakers  [only if GROUP == 'dome'] ----
if GROUP != "dome":
    raise RuntimeError(f"Subject {SUBJECT_ID} is in the '{GROUP}' group -- "
                       f"use the AR_filler cell below instead, not this one.")
subject = hr.Subject(SUBJECT_ID)  # reload after block 1 write
dome = LocalizationDome(subject, DOME_SETTINGS)
dome.run()

# %% block 2: exposure -- AR_filler, virtual  [only if GROUP == 'control'] ------
if GROUP != "control":
    raise RuntimeError(f"Subject {SUBJECT_ID} is in the '{GROUP}' group -- "
                       f"use the DOME cell above instead, not this one.")
subject = hr.Subject(SUBJECT_ID)  # reload after block 1 write
ar_filler = Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS)
ar_filler.run()

# %% block 3: AR_post -- repeat of AR_pre ---------------------------------------
subject = hr.Subject(SUBJECT_ID)  # reload after block 2 write
ar_post = Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS)
ar_post.run()
collect_externalization_rating(ar_post)
