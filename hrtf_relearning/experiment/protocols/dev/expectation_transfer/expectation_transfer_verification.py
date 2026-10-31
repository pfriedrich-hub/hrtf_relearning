"""
expectation_transfer_verification.py

Go/no-go precursor to expectation_transfer.py. Before committing 16 subjects to
the mechanism study, answer one question with the *current* HRIR pipeline (new
HP equalization): does immediate freefield->VR transfer still fail, and if so,
is the failure contextual (room/presence) or just imperfect rendering?

The historical "VR room = no externalization" observation came from the OLD
signal chain, so it can't tell a real context effect from a rendering artifact.
This script holds the new rendered signal constant and reads out three numbers
per subject:

    Block 1  AR_VR_immediate   VR room, straight after recording, NOTHING in
                               between (no localization at all). The pure
                               replication of the original failure.
    Block 2  dome_ref          real dome loudspeakers (freefield room) =
                               real-ear reference / rendering ceiling.
    Block 3  AR_freefield_primed  AR in the freefield room, right after the
                               dome block = best-case AR (in-room + freshly
                               primed). Shows whether the good state is even
                               reachable with this pipeline.

Two orthogonal comparisons fall out (see expectation_transfer_verification.md):
    FIDELITY  = Block 3 vs Block 2  -> is the rendering faithful at all?
    CONTEXT   = Block 1 vs Block 3  -> does room/presence/priming matter, with
                                       signal held constant?
and the headline: Block 1 vs Block 2 -> is immediate transfer already adequate?

The Block-4 decision cell prints one of three verdicts:
    A  immediate transfer adequate      -> new pipeline fixed it; the mechanism
                                           study (expectation_transfer.py) is
                                           probably unnecessary.
    B  immediate poor, primed good       -> phenomenon persists; run the
                                           mechanism study.
    C  even best-case AR falls short     -> rendering-fidelity problem (HP eq /
                                           spectral-cue capture); debug the
                                           signal chain first, it's not
                                           psychology.

This is a go/no-go on a handful of pilots, NOT an effect-size estimate -- that
is what the matched-control expectation_transfer.py design is for.

ORDER MATTERS AND CANNOT BE UNDONE. Block 1 must be the participant's first
localization of the day, run in the VR room with nothing between it and the
recording -- any AR/dome block run first would prime or warm them up and
destroy the "immediate" reading. You cannot un-prime, so if you run Block 2/3
first this subject's Block 1 is spent. Rerun a *later* cell freely (e.g. redo
Block 3); never re-open Block 1 on the same visit.

Runs are tagged '_verif-<stage>' on filename and sequence.name, so they never
mix with expectation_transfer ('_expT-') or other protocols in
subject.localization. Assumes the subject already has a recorded individual
HRIR + calibrated headphone filter.

Run cell by cell (# %%) in an IDE/console -- do NOT run top-to-bottom.
"""

# %% imports and config ------------------------------------------------------
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.Localization_dome import LocalizationDome
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy,
)
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume

SUBJECT_ID = "XX"        # edit per participant

HRIR_NAME = SUBJECT_ID   # individual measured HRIR (native, unmodified)
HP = "DT990"             # headphone EQ profile

# Appended to every run's filename/sequence.name, e.g. "..._verif-AR_VR_immediate".
# Lets you pull these runs out of a subject's localization dict later.
PROTOCOL_TAG = "verif"

# Shared vertical-midline sampling -- identical to expectation_transfer.py so the
# dome block and the two AR blocks probe the same physical directions and are
# directly comparable. See expectation_transfer_design.md sec 4.
TARGETS_PER_SPEAKER = 3   # -> 21 trials/block on the 7 midline positions
MIN_DISTANCE = 15
GAIN = 0.07               # pybinsim gain matched by ear to the dome at OS volume 50%

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

# --- Decision thresholds (placeholders -- tune after the first pilots) --------
# AR is called "adequate" when its elevation gain reaches at least this fraction
# of the SAME subject's dome (real-ear) gain, AND elevation RMSE is within
# ELE_RMSE_MARGIN degrees of the dome RMSE. Externalization "adequate" at/above
# EXT_ADEQUATE on the 0-10 post-block rating. Gain is the primary criterion;
# it is the cue this whole line of work is about.
ELE_GAIN_ADEQUATE_FRAC = 0.70
ELE_RMSE_MARGIN = 7.5    # degrees above dome RMSE still counts as adequate
EXT_ADEQUATE = 6.0       # 0-10 scale


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
    """Append '_<PROTOCOL_TAG>-<stage>' to filename and sequence.name BEFORE .run()
    (write() keys subject.localization on self.filename, so the tag must be set
    before any data is written)."""
    loc_test.filename = f"{loc_test.filename}_{PROTOCOL_TAG}-{stage}"
    loc_test.sequence.name = loc_test.filename
    return loc_test


# %% helper: post-block externalization rating --------------------------------
def collect_externalization_rating(loc_test):
    """Short console-collected subjective report. Run after the two AR blocks only
    (not after the dome block). Same instrument as expectation_transfer.py."""
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


# %% helper: fetch a tagged run + its stats -----------------------------------
def _latest(subject, stage):
    """Most recent sequence for this stage, or None if not run yet."""
    keys = [k for k in getattr(subject, "localization", {})
            if f"_{PROTOCOL_TAG}-{stage}" in k]
    if not keys:
        return None
    return subject.localization[sorted(keys)[-1]]


def _stats(seq):
    """(elevation_gain, ele_rmse, externalization) for a sequence, NaN/None if absent."""
    if seq is None:
        return None
    ele_gain, ele_rmse, *_ = localization_accuracy(seq)
    ext = getattr(seq, "externalization_rating", None)
    return {"ele_gain": ele_gain, "ele_rmse": ele_rmse, "ext": ext}


# %% status check (rerun anytime) ----------------------------------------------
set_windows_volume(50)   # pin OS volume to the level the GAIN match was made at
subject = hr.Subject(SUBJECT_ID)
print(f"SUBJECT: {SUBJECT_ID}   (verification precursor to expectation_transfer)")
for stage in ("AR_VR_immediate", "dome_ref", "AR_freefield_primed"):
    seq = _latest(subject, stage)
    print(f"   [{'x' if seq is not None else ' '}] {stage}")
print("\nReminder: Block 1 (AR_VR_immediate) must be the FIRST localization of the\n"
      "day, in the VR room, with nothing between it and the HRIR recording.")

# %% block 1: AR_VR_immediate -- record -> straight to VR room, NOTHING between --
# THE headline number. Do not run any other block, and do not let the participant
# localize anything, before this one. If in doubt whether they were primed, abort
# and reschedule -- a contaminated Block 1 is worse than none.
subject = hr.Subject(SUBJECT_ID)
ar_vr = _tag(Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS), "AR_VR_immediate")
ar_vr.run()
collect_externalization_rating(ar_vr)

# %% block 2: dome_ref -- real loudspeakers (freefield room) --------------------
# Real-ear reference / rendering ceiling. Run AFTER Block 1 (return to the
# freefield room). This block primes the participant, so everything after it is
# "primed" by construction -- that is intended for Block 3.
subject = hr.Subject(SUBJECT_ID)  # reload after block 1 write
dome = _tag(LocalizationDome(subject, DOME_SETTINGS), "dome_ref")
dome.run()

# %% block 3: AR_freefield_primed -- best-case AR (in-room + primed) ------------
# AR in the freefield room immediately after the dome block: speakers present,
# freshly primed. Shows whether good AR is reachable with this pipeline at all.
subject = hr.Subject(SUBJECT_ID)  # reload after block 2 write
ar_ff = _tag(Localization(subject, hrir_settings(), AR_MIDLINE_SETTINGS), "AR_freefield_primed")
ar_ff.run()
collect_externalization_rating(ar_ff)

# %% block 4: decision -- print go/no-go verdict -------------------------------
subject = hr.Subject(SUBJECT_ID)
vr   = _stats(_latest(subject, "AR_VR_immediate"))
dome = _stats(_latest(subject, "dome_ref"))
ff   = _stats(_latest(subject, "AR_freefield_primed"))


def _fmt(s, label):
    if s is None:
        return f"   {label:<22} (not run)"
    ext = "  -" if s["ext"] is None else f"{s['ext']:.1f}"
    return (f"   {label:<22} ele_gain={s['ele_gain']:.2f}  "
            f"ele_rmse={s['ele_rmse']:.1f}deg  ext={ext}")


print("\n================ VERIFICATION SUMMARY ================")
print(_fmt(vr,   "AR_VR_immediate"))
print(_fmt(dome, "dome_ref (real-ear)"))
print(_fmt(ff,   "AR_freefield_primed"))
print("-----------------------------------------------------")

if dome is None:
    print("No dome reference yet -- run Block 2 before interpreting AR blocks.")
else:
    g_ref = dome["ele_gain"]
    r_ref = dome["ele_rmse"]
    gain_ok = lambda s: s is not None and s["ele_gain"] >= ELE_GAIN_ADEQUATE_FRAC * g_ref
    rmse_ok = lambda s: s is not None and s["ele_rmse"] <= r_ref + ELE_RMSE_MARGIN
    ext_ok  = lambda s: s is not None and s["ext"] is not None and s["ext"] >= EXT_ADEQUATE

    immediate_ok = gain_ok(vr) and rmse_ok(vr) and ext_ok(vr)
    primed_ok    = gain_ok(ff) and rmse_ok(ff)

    print(f"criteria: ele_gain >= {ELE_GAIN_ADEQUATE_FRAC:.2f} x dome ({ELE_GAIN_ADEQUATE_FRAC * g_ref:.2f}), "
          f"ele_rmse <= dome+{ELE_RMSE_MARGIN:.1f} ({r_ref + ELE_RMSE_MARGIN:.1f}), ext >= {EXT_ADEQUATE:.1f}")
    print("-----------------------------------------------------")

    if immediate_ok:
        print("VERDICT A: immediate freefield->VR transfer is ADEQUATE with the new\n"
              "pipeline. The old failure was likely a rendering artifact now fixed by\n"
              "the HP equalization. The mechanism study (expectation_transfer.py) is\n"
              "probably unnecessary -- your VR transfer baseline can be taken directly.")
    elif primed_ok:
        print("VERDICT B: immediate transfer is INADEQUATE but best-case AR (primed,\n"
              "in-room) reaches near real-ear. The phenomenon PERSISTS with the new\n"
              "pipeline and is contextual, not a signal problem. Proceed to the\n"
              "mechanism study (expectation_transfer.py).")
    else:
        print("VERDICT C: even best-case AR (primed, in-room) falls short of the dome\n"
              "real-ear reference. This points to a RENDERING-FIDELITY issue (HP eq /\n"
              "spectral-cue capture), not a context effect. Debug the signal chain\n"
              "before running the mechanism study -- psychology can't fix a signal the\n"
              "ears never receive.")
print("=====================================================")
