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

Every run this script produces is tagged '_expT-<stage>' on both filename and
sequence.name (stage in AR_pre / dome / AR_filler / AR_post), so later runs in
subject.localization from OTHER experiments on the same participant don't get
mixed up with these -- filter subject.localization keys on '_expT-' to pull
just this protocol's data.

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
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume

SUBJECT_ID = "SZ"   # edit per participant

CSV_PATH = hr.PATH / "experiment" / "protocols" / "documentation" / "expectation_transfer_block_order.csv"



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

# Short protocol code appended to every run's filename/sequence.name, e.g.
# "LS_03.07_14-22_LS_expT-AR_pre". Lets you pick these runs out of a subject's
# localization dict later even after other experiments have added their own.
PROTOCOL_TAG = "expT"

TARGETS_PER_SPEAKER = 3  # -> 21 trials/block on the 7 vertical-midline positions,
MIN_DISTANCE = 15        # shared by all four block types (AR_pre/AR_post/dome/AR_filler)
# pybinsim gain matched by ear to the dome loudspeakers (DT990, MATCH_STIM='native')
# at Windows master volume 50%. Only valid at that OS volume -- set_windows_volume(50)
# below pins it. Dome and AR stimuli now share one synthesis
# (localization_helpers/stimulus.make_gapped_pinknoise); dome loudness was preserved
# so this match still holds until re-calibrated with KEMAR recordings.
# See localization_helpers/match_ar_dome_loudness.py.
GAIN = 0.07

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
        "convolution": "cpu",
        "storage": "cpu",
    }


# %% helper: tag a run with protocol + stage -----------------------------------
def _tag(loc_test, stage):
    """
    Append '_<PROTOCOL_TAG>-<stage>' to a run's filename and sequence.name,
    e.g. stage='AR_pre' -> '..._expT-AR_pre'. Must be called right after
    construction, before .run() -- write() (called during/after the loop)
    uses self.filename as the key into subject.localization, so the tag has
    to be in place before any data is written.
    """
    loc_test.filename = f"{loc_test.filename}_{PROTOCOL_TAG}-{stage}"
    loc_test.sequence.name = loc_test.filename
    return loc_test


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
set_windows_volume(50)   # pin OS volume to the level the GAIN=0.07 match was made at
subject = hr.Subject(SUBJECT_ID)
print(f"SUBJECT: {SUBJECT_ID}    GROUP: {GROUP}    (loaded from {CSV_PATH.name})")
done = list(getattr(subject, "localization", {}).keys())
own = [k for k in done if f"_{PROTOCOL_TAG}-" in k]
other = [k for k in done if k not in own]
if own:
    print(f"expectation_transfer runs already on file ({len(own)}):")
    for k in own:
        print(f"   - {k}")
else:
    print("No expectation_transfer runs on file yet.")
if other:
    print(f"({len(other)} other localization run(s) on file from other protocols, not shown)")

# %% block 1: AR_pre -- naive virtual localization ------------------------------
ar_pre = _tag(Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS), "AR_pre")
ar_pre.run()
collect_externalization_rating(ar_pre)

# %% block 2: exposure -- DOME, real loudspeakers  [only if GROUP == 'dome'] ----
if GROUP != "dome":
    raise RuntimeError(f"Subject {SUBJECT_ID} is in the '{GROUP}' group -- "
                       f"use the AR_filler cell below instead, not this one.")
subject = hr.Subject(SUBJECT_ID)  # reload after block 1 write
dome = _tag(LocalizationDome(subject, DOME_SETTINGS), "dome")
dome.run()

# %% block 2: exposure -- AR_filler, virtual  [only if GROUP == 'control'] ------
if GROUP != "control":
    raise RuntimeError(f"Subject {SUBJECT_ID} is in the '{GROUP}' group -- "
                       f"use the DOME cell above instead, not this one.")
subject = hr.Subject(SUBJECT_ID)  # reload after block 1 write
ar_filler = _tag(Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS), "AR_filler")
ar_filler.run()

# %% block 3: AR_post -- repeat of AR_pre ---------------------------------------
subject = hr.Subject(SUBJECT_ID)  # reload after block 2 write
ar_post = _tag(Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS), "AR_post")
ar_post.run()
collect_externalization_rating(ar_post)
