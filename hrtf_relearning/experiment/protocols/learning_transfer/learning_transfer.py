"""
learning_transfer.py

Adaptation-transfer experiment protocol — DONOR-DETAIL cue manipulation.

This is the current protocol. The previous version, which translated the
participant's own spectral detail along the ERB axis, is kept for reference at
protocols/dev/old/learning_transfer_erbshift.py; design, grid, counterbalancing
and the final-day 2x2 are unchanged from it, only the cue manipulation differs.
Instead of moving the participant's own detail, it is REPLACED with a donor's:


    log|H_modified(direction)| = envelope_4( log|H_own| ) + detail( log|H_donor| )

with the participant's own phase and own per-direction broadband level, so ITD
and broadband ILD are exactly as measured and only the spectral shape above the
envelope scale changes. See hrtf.modify.donor_detail.

WHY NOT THE ERB SHIFT. A constant translation keeps the participant's own
pattern intact, so the existing spectral-to-spatial map can still match each
modified DTF to one of its own templates and read out a coherent — merely
displaced — elevation. Van Wanrooij & Van Opstal (2005) saw exactly that: the
listeners whose molds only shifted the main notch showed a bias with preserved
gain and never adapted over the whole study, while those whose spectral cues
were decorrelated did adapt. Measured here, a 1-ERB shift leaves the
own-vs-modified correlation ridge at slope ~1.0, i.e. fully absorbable as a
bias; a donor's detail collapses it.

DONOR SELECTION is the only thing that varies between participants, and it is
made by a fixed rule in hrtf.analysis.donor_selection — see
DONOR_POOL / TARGET_R_MATCH / TOLERANCE / MAX_RIDGE_SLOPE there, and
docs/methods_donor_detail.md for the paragraph this becomes in a paper.
Everything else (n_keep, band, filter bank, target) is identical for everyone.

DONOR SWAPS. The rule produces a ranked shortlist, not a single name, so a
participant who is at floor with the first donor can be moved to the second
without inventing a criterion on the spot. Stage the alternates before the
session with prepare_donor_shortlist(), swap with use_donor(rank=1, reason=...).
WHERE THE DONOR LIVES. The choice is made once, on day 1, and written into the
participant's own file as subject.active_donor (data/results/<id>/<id>.pkl, and
<id>.json alongside it) -- by prepare_donor_shortlist() and build_donor_sofa(),
which leave rank 0 active, or by use_donor() when it is deliberately changed.
Every later session reads it back from there when the config cell runs, so the
participant is always on the donor they were actually trained on rather than
whatever the rule ranks first once the pool has grown. Nothing in this file has
to be edited between sessions and nothing has to be remembered. Only the current
donor is kept in the record; the candidate ranking behind the choice is embedded
in the composite SOFA as GLOBAL_ModificationParams.

The monaural ear treatment is orthogonal and selectable via OTHER_EAR
('flat' | 'envelope' | 'native'); which one to use is still an open question,
tested per participant by the ladder in protocols/dev/ladder.py.

Counterbalancing is read from learning_transfer_block_order.csv IN THIS FOLDER.
Write each subject's id into the 'subject' column before running.

RUN ORDER. Cells top to bottom are the protocol proper, in the order they are
performed:
    day 1            status -> native reference -> build donor -> baseline A/D
    adaptation days  PRE test -> train -> POST test   (three cells, in order)
    final day        the counterbalanced 2x2
Everything under MISC at the bottom is diagnostic and is NOT run as a matter of
course -- the externalization ladder, the cepstral-split QC, the n_keep=8 build
and the other-ear probe live there.

The OS master volume is forced to OS_VOLUME (50%) at the start of every
localization block and every training run, because the pybinsim gain was
matched to the dome at that setting. Off Windows this is a logged no-op.

Run cell by cell (# %%) in an IDE/console -- do NOT run top-to-bottom.

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT.
------------------------------------------------------------------------------
"""

SUBJECT_ID = ("NR")

# %% imports and config #------------------------------------------------------
import csv  # only for the block-order table below; the modification
            # workflow now lives in donor_modification.py next door
import os
import subprocess
import sys
import slab
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.experiment.protocols.protocol_helpers import (
    collect_demographics, collect_externalization_rating, externalization_check,
    externalization_ladder)
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.experiment.protocols.learning_transfer.donor_modification import DonorModification
from hrtf_relearning.hrtf.modify.plot_compare import plot_split_qc
from hrtf_relearning.utils import paths

CSV_PATH = (hr.PATH / "experiment" / "protocols" / "learning_transfer"
            / "learning_transfer_block_order.csv")


def _load_subject_params(subject_id, csv_path=CSV_PATH):
    """Look up trained_ear and final block order for this subject."""
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject", "").strip() == subject_id:
                ear   = row["trained_ear"].strip()
                order = [c.strip() for c in row["block_order"].split("-")]
                return ear, order
    raise ValueError(
        f"Subject '{subject_id}' not found in the 'subject' column of\n  {csv_path}\n"
        f"Add its id to a row there (replacing an '(assign)' cell) before running.")


TRAINED_EAR, FINAL_ORDER = _load_subject_params(SUBJECT_ID)

NATIVE_SOFA = f"{SUBJECT_ID}"      # individual measured HRTF

# The donor is chosen ONCE, on day 1, by prepare_donor_shortlist() or
# build_donor_sofa() -- rank 0 unless use_donor() is called deliberately -- and
# written into the participant's own file (subject.active_donor in
# data/results/<id>/<id>.pkl, mirrored into <id>.json). Every later session
# reads it back from there when `donor` is constructed a few cells down. There
# is nothing to set here and nothing to remember between sessions.
#
# DONOR_OVERRIDE is an escape hatch for rebuilding a subject whose record was
# lost, or for re-deriving an old composite in analysis. Leave it None to run
# the experiment.
DONOR_OVERRIDE = None
DONOR_ID = None                    # resolved from the subject file below
MODIFIED_SOFA = None               # <SUBJECT_ID>_donor_<DONOR_ID>, set below

HP = "DT990"

# --- what the non-listening ear gets in monaural blocks ----------------------
# 'flat' | 'envelope' | 'native' -- hrtf.processing.{flatten,envelope,native}
OTHER_EAR = "envelope"
ENV_NKEEP = selection.N_KEEP
PROBE_OTHER_EAR = "envelope"

# --- which build this subject's filters were made with -----------------------
# 'v1'  Donor detail applied to the finished 475-direction SOFA; the monaural
#       reduction applied again at RENDER time by hrtf2binsim. Every build
#       before 2026-08.
# 'v2'  Donor detail AND the monaural reduction applied to the 19 MEASURED
#       az=0 DTFs, then re-expanded through the spherical head model
#       (hrtf.processing.midline). Three things follow: the ITD comes from the
#       model's interaural phase so native and modified sets share it exactly;
#       the envelope is fitted to the pinna response rather than to the pinna
#       response times the head shadow; and it is averaged over elevation, so
#       the untrained ear carries no elevation-dependent spectral structure at
#       all -- measured, v1 left about half of it there.
#
# Subjects already run stay on v1. The change alters the untrained ear, and
# swapping builds mid-cohort would put a discontinuity in the middle of their
# own pre/post comparison, which is worse than the bias itself.
LEGACY_V1_SUBJECTS = ("FS", "GS", "IR", "TS", "PF")
PIPELINE = "v1" if SUBJECT_ID in LEGACY_V1_SUBJECTS else "v2"

# --- shared localization sampling grid (do not change) -----------------------
SECTOR_SIZE        = (7, 14)
ELEVATION_RANGE    = (-35, 35)
TARGETS_PER_SECTOR = 3
MIN_DISTANCE       = 20
GAIN               = 0.2
# Every block in this file inherits STIM, and that is the point: baselines,
# daily tests and the final 2x2 must all be measured with the SAME stimulus or
# the change scores are meaningless. What must never happen is a mixture within
# a subject.
#   !! TS (10.08) and IR (11.08) had their day-1 blocks run with STIM='uso'
#      because this was left set to 'uso'. See docs/stimulus_spectral_variation.md.
#
# 'ripple' since 2026-08-17. Plain noise has essentially no across-trial source
# variation (0.4 dB SD at 1/6 oct against a ~3.3 dB elevation cue), so the sound
# at the eardrum carries a fixed map from absolute spectrum to elevation and the
# task can be solved by template matching, without ever separating source from
# filter. That cannot distinguish a spectral-to-spatial recalibration from a
# learned timbre->elevation lookup, which is what FS reported doing. The test
# stimulus therefore varies its source spectrum on every trial, in EVERY block,
# so the measure is source-invariant throughout rather than only on the last day.
# Training stays on noise (SOUND_FILE=None -> pink noise).
#
# Subjects run before this date were tested on noise. They are pilots and are
# not pooled with what follows.
STIM               = "ripple"      # -> "ripple" once the depth is settled, below
# Envelope parameters for STIM='ripple'. Empty dict = inherit the defaults in
# localization_helpers.stimulus (the single source of truth); set rms_tilt here
# only to override for a specific cohort, and it is recorded per block in
# sequence.stim_settings either way.
#   !! Depth is NOT yet settled: the current default (8 dB rms, 27 dB median
#      peak-to-trough) sits above the ~20 dB depth at which Macpherson &
#      Middlebrooks report localization degradation. Run the free-field check
#     nm and the AR self-test
#      (cells 7-9) BEFORE flipping STIM to 'ripple' for a participant.
STIM_SETTINGS      = {'rms_tilt': 3}
MIDLINE_TOL        = 1.0
FULL_FIELD = (-35, 35)

if TRAINED_EAR == "left":
    UNTRAINED_EAR, TRAINED_HEMI, MIRRORED_HEMI = "right", (-35, 0), (0, 35)
elif TRAINED_EAR == "right":
    UNTRAINED_EAR, TRAINED_HEMI, MIRRORED_HEMI = "left", (0, 35), (-35, 0)
else:
    raise ValueError("TRAINED_EAR must be 'left' or 'right'.")


# The ladder compares two composite strengths, so more than one modified SOFA
# can exist per subject. n_keep=N_KEEP (4) keeps the plain name; anything else
# gets an _n<k> suffix, so the training/testing SOFA is never ambiguous.
LADDER_N_KEEP = (4, 8)


def loc_settings(azimuth_range, exclude_midline=False):
    return {
        "kind": "sectors",
        "azimuth_range": azimuth_range,
        "elevation_range": ELEVATION_RANGE,
        "targets_per_speaker": 3,
        "targets_per_sector": TARGETS_PER_SECTOR,
        "min_distance": MIN_DISTANCE,
        "gain": GAIN,
        "stim": STIM,
        "stim_settings": STIM_SETTINGS,
        "sector_size": SECTOR_SIZE,
        "replace": False,
        "exclude_midline": exclude_midline,
        "midline_tol": MIDLINE_TOL,
    }

# ---------------------------------------------------------------------------
# Donor manipulation
#
# The machinery moved to donor_modification.DonorModification (next door) on
# 2026-08-19 -- it was ~550 lines of module-level functions reading nine module
# globals and writing three of them back with `global`, which is state that
# belongs to a participant, not to a module.
#
# The thin wrappers below exist so every cell in this file, and every habit
# built around them, keep working unchanged. `donor` is the object; reach for
# it directly (donor.shortlist(), donor.build(), donor.discard_unused()) in
# anything new.
# ---------------------------------------------------------------------------

donor = DonorModification(
    SUBJECT_ID,
    trained_ear=TRAINED_EAR,
    native_sofa=NATIVE_SOFA,
    other_ear=OTHER_EAR,
    env_n_keep=ENV_NKEEP,
    pipeline=PIPELINE,
    donor_id=DONOR_OVERRIDE,
    hp=HP,
)


def _sync():
    """Mirror the object's active donor back onto the module globals.

    DONOR_ID / MODIFIED_SOFA are read by phases() and printed in status lines,
    and re-running the config cell resets them. `donor` is the authority; these
    two follow it.
    """
    global DONOR_ID, MODIFIED_SOFA
    DONOR_ID, MODIFIED_SOFA = donor.donor_id, donor.modified_sofa


def hrir_settings(sofa_name, ear=None, mirror=False, other_ear=None):
    return donor.hrir_settings(sofa_name, ear=ear, mirror=mirror,
                               other_ear=other_ear)


def donor_shortlist(refresh=False, quiet=False):
    return donor.shortlist(refresh=refresh, quiet=quiet)


def build_donor_sofa(overwrite=False, show_qc=True, n_keep=None, rank=0,
                     donor_id=None, set_active=True, quiet=False):
    out = donor.build(overwrite=overwrite, show_qc=show_qc, n_keep=n_keep,
                      rank=rank, donor_id=donor_id, set_active=set_active,
                      quiet=quiet)
    _sync()
    return out


def prepare_donor_shortlist(n=3, mirrored=True, overwrite=False):
    out = donor.prepare_shortlist(n=n, mirrored=mirrored, overwrite=overwrite)
    _sync()
    return out


def use_donor(rank=None, donor_id=None, reason=""):
    out = donor.use_donor(rank=rank, donor_id=donor_id, reason=reason)
    _sync()
    return out


def load_existing_donor():
    out = donor.load_existing()
    _sync()
    return out


def discard_unused_donors(dry_run=True, keep=None):
    return donor.discard_unused(dry_run=dry_run, keep=keep)


def show_donor_log():
    return donor.show_log()


# Take whatever the participant's file says into the module globals, so every
# cell below is on this subject's donor the moment the config cell has run --
# no build, no load_existing_donor(), no DONOR_ID to remember. Before day 1
# there is nothing recorded yet and both stay None, which is what the build
# cell is for.
_sync()
if DONOR_ID and donor.donor_from_record:
    _sofa = paths.SOFA_DIR / SUBJECT_ID / f"{MODIFIED_SOFA}.sofa"
    print(f"donor {DONOR_ID} (from {SUBJECT_ID}'s subject file) -> "
          f"{MODIFIED_SOFA}" + ("" if _sofa.exists() else "   [!] SOFA MISSING"))
elif DONOR_ID:
    print(f"donor {DONOR_ID} (DONOR_OVERRIDE) -> {MODIFIED_SOFA}")
else:
    print(f"no donor recorded for {SUBJECT_ID} yet -- run the day-1 build cell")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phases():
    """Built on demand so MODIFIED_SOFA is picked up after the build cell."""
    return {
        "native":     ("Native reference",         "Day 1", NATIVE_SOFA,   None,        False, FULL_FIELD,    "binaural, native HRTF, full field"),
        "baseline_A": ("Baseline A: trained/same", "Day 1", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "naive trained ear, modified filter (matches final A)"),
        "baseline_D": ("Baseline D: untrnd/mirr",  "Day 1", MODIFIED_SOFA, TRAINED_EAR, True,  MIRRORED_HEMI, "naive untrained ear, MIRRORED modified filter (matches final D)"),
        "daily":      ("Daily training test",      "Adaptation days", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "monaural trained ear, trained hemifield"),
        "A":          ("Final A: trained/same",    "Final day", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "trained ear, same locations (baseline retest)"),
        "B":          ("Final B: trained/mirr",    "Final day", MODIFIED_SOFA, TRAINED_EAR, False, MIRRORED_HEMI, "trained ear, mirrored locations"),
        "C":          ("Final C: untrnd/same",     "Final day", MODIFIED_SOFA, TRAINED_EAR, True,  TRAINED_HEMI,  "untrained ear (mirrored modified HRIR), same locations"),
        "D":          ("Final D: untrnd/mirr",     "Final day", MODIFIED_SOFA, TRAINED_EAR, True,  MIRRORED_HEMI, "untrained ear (mirrored modified HRIR), mirrored locations  [MAIN]"),
    }


def _n_sectors(rng, size):
    import numpy
    return len(numpy.arange(rng[0] + size / 2, rng[1], size))


def _est_trials(az_range):
    return (_n_sectors(az_range, SECTOR_SIZE[0])
            * _n_sectors(ELEVATION_RANGE, SECTOR_SIZE[1]) * TARGETS_PER_SECTOR)


def _describe(key):
    label, when, sofa, ear, mirror, az, desc = phases()[key]
    midline = "excluded" if tuple(az) != tuple(FULL_FIELD) else "kept"
    other = f"{OTHER_EAR}" if ear else "-"
    return (f"{label}  [{when}]\n      {desc}\n"
            f"      SOFA={sofa}  ear={ear or 'binaural'}  other ear={other}  mirror={mirror}\n"
            f"      azimuth={az}  elevation={ELEVATION_RANGE}  sector={SECTOR_SIZE}  "
            f"tps={TARGETS_PER_SECTOR}  midline az=0 {midline}\n"
            f"      ~{_est_trials(az)} trials")


OS_VOLUME = 50   # Windows master slider, %. The pybinsim gain was matched to the
                 # dome at this setting (match_ar_dome_loudness.py), so every
                 # localization test and training run forces it before starting;
                 # a moved slider silently invalidates the presentation level.


def _fix_output_level():
    """Force the OS master volume to OS_VOLUME. No-op off Windows."""
    if not set_windows_volume(OS_VOLUME):
        print(f"  [!] OS volume NOT set programmatically — check the slider is "
              f"at {OS_VOLUME}% before continuing")


def run_phase(key, subject, other_ear=None):
    label, when, sofa, ear, mirror, az, desc = phases()[key]
    if sofa is None:
        raise RuntimeError("MODIFIED_SOFA is not set — run build_donor_sofa() "
                           "or load_existing_donor() first")
    print("\n" + "=" * 70)
    print(f"RUNNING:  {_describe(key)}"
          + (f"\n      OTHER EAR OVERRIDE: {other_ear}" if other_ear else ""))
    print("=" * 70)
    _fix_output_level()
    one_sided = tuple(az) != tuple(FULL_FIELD)
    test = Localization(subject,
                        hrir_settings(sofa, ear=ear, mirror=mirror, other_ear=other_ear),
                        loc_settings=loc_settings(az, exclude_midline=one_sided))
    test.run()
    print(f"Done: {test.filename}")
    return test


def run_anchor(subject):
    """Short native-HRTF block + rating -- the top of the 0-10 scale.

    'A real external loudspeaker' is not a sound the participant has heard over
    these headphones, so an unanchored rating is a number about their
    imagination. This is the participant's OWN unmodified HRTF played
    binaurally: the ceiling of the whole delivery chain, and the closest thing
    to a 10 the setup can produce. ~10 trials, about a minute.

    DAY 1 ONLY as of 2026-08-19. It used to run every day so ratings stayed
    comparable across days; the protocol now takes only the day-1 ratings
    (native reference + the two baselines), so there is nothing left on the
    later days for a same-day anchor to make comparable. Paul's call: if
    externalization holds for the modified ears on day 2 it holds to the end,
    and a daily ~1-minute block plus rating is not worth what it buys.

    On day 1 the `native` phase serves this purpose -- same own-HRTF binaural
    block, more trials -- so this function is no longer called by the protocol
    proper. Kept, and reachable from MISC, for the case it was also good at:
    if you suspect the delivery chain (headphone seat, HP filter, OS volume)
    has moved mid-experiment, run it and compare against day 1. A drop there is
    a chain problem, not adaptation.
    """
    settings = loc_settings(FULL_FIELD)
    settings.update(sector_size=(14, 14), targets_per_sector=1)   # ~10 trials
    return externalization_check(subject, hrir_settings(NATIVE_SOFA, ear=None),
                                 settings, label=f"{SUBJECT_ID} anchor (own HRTF, binaural)")


def ladder_settings(rung):
    """(hrir_settings, loc_settings) for one rung of the externalization ladder.

    Coarse grid on purpose (~10 trials): these blocks are for the rating, not
    for elevation-gain statistics. 'anchor' is the participant's own unmodified
    HRTF played binaurally — the ceiling of the whole delivery chain, which is
    what gives the 0-10 scale a top.
    """
    settings = loc_settings(TRAINED_HEMI, exclude_midline=True)
    settings.update(sector_size=(14, 14), targets_per_sector=1)   # ~10 trials
    if rung == "anchor":
        return hrir_settings(NATIVE_SOFA, ear=None), settings
    if rung.startswith("donor_n"):
        # composite STRENGTH: same donor, same other-ear treatment, only n_keep
        # differs. Lower n_keep hands over more of the cue.
        n_keep = int(rung.split("_n")[1])
        return (hrir_settings(donor.modified_name(DONOR_ID, n_keep), ear=TRAINED_EAR),
                settings)
    # ear TREATMENT: n_keep=4 composite on the trained ear, other ear varies
    return hrir_settings(MODIFIED_SOFA, ear=TRAINED_EAR, other_ear=rung), settings


TRAINING_SCRIPT = hr.PATH / "experiment" / "training" / "Training_AR.py"


def run_training(hrir_name=None, ear=None, az_range=None):
    """Launch Training_AR.py with this subject's modified HRIR and ear settings."""
    hrir_name = MODIFIED_SOFA if hrir_name is None else hrir_name
    if hrir_name is None:
        raise RuntimeError("MODIFIED_SOFA is not set — build or load the donor SOFA first")
    ear = TRAINED_EAR if ear is None else ear
    az_range = TRAINED_HEMI if az_range is None else az_range

    sofa_path = paths.SOFA_DIR / SUBJECT_ID / f"{hrir_name}.sofa"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Modified HRIR not found:\n  {sofa_path}")

    print("-" * 64)
    print(f"TRAINING   subject={SUBJECT_ID}   ear={ear}   az_range={az_range}")
    print(f"           HRIR={hrir_name}.sofa   HP={HP}")
    print(f"           other ear={OTHER_EAR} (n_keep={ENV_NKEEP})")
    print("-" * 64)
    _fix_output_level()

    env = dict(os.environ,
               TRAINING_SUBJECT_ID=SUBJECT_ID,
               TRAINING_HRIR_NAME=hrir_name,
               TRAINING_EAR=ear,
               TRAINING_OTHER_EAR=OTHER_EAR,
               TRAINING_ENV_NKEEP=str(ENV_NKEEP),
               TRAINING_NATIVE_SOFA=NATIVE_SOFA,
               TRAINING_AZ_RANGE=f"{az_range[0]},{az_range[1]}",
               TRAINING_HP=HP)
    subprocess.run(
        [sys.executable, "-m", "hrtf_relearning.experiment.training.Training_AR"],
        env=env, cwd=str(hr.PATH.parent), check=False)


def show_status(subject):
    print("\n" + "-" * 70)
    print(f"SUBJECT: {SUBJECT_ID}    TRAINED EAR: {TRAINED_EAR}    UNTRAINED: {UNTRAINED_EAR}")
    print(f"manipulation: donor detail (n_keep={selection.N_KEEP}), donor="
          f"{DONOR_ID or '(not selected yet)'}   other ear={OTHER_EAR}")
    print(f"hemifields -> trained {TRAINED_HEMI}, mirrored {MIRRORED_HEMI}")
    print(f"modified SOFA: {MODIFIED_SOFA}    final-day order: {'-'.join(FINAL_ORDER)}")
    done = list(getattr(subject, "localization", {}).keys())
    if done:
        print(f"\nLocalization runs on file for {SUBJECT_ID} ({len(done)}):")
        for k in done:
            print(f"   - {k}")
    else:
        print(f"\nNo localization runs on file yet for {SUBJECT_ID}.")
    print("-" * 70)







# %% status check (rerun anytime) --------------------------------------------
subject = hr.Subject(SUBJECT_ID)
collect_demographics(subject)      # once per participant; skipped if on file
show_status(subject)

# %% day 1: native reference (original HRIR, full field) ----------------------
# Doubles as the first anchor: this is the best the chain can sound, so its
# rating defines the top of the 0-10 scale for everything that follows.
native = run_phase("native", subject)
collect_externalization_rating(native)

# %% BEFORE THE SESSION: stage the top 3 donors -------------------------------
# Run this with nobody in the rig. Builds the rank 0/1/2 composites AND their
# pyBinSim databases (mirrored and un-mirrored), so that if the participant
# turns out to be at floor with the rank-0 donor you can swap in seconds
# instead of rebuilding filters while they wait. Leaves rank 0 active.
prepare_donor_shortlist(n=3)

# %% day 1: select the donor and build the modified HRTF -- run ONCE ----------
# Prints the full candidate ranking, the chosen donor and why, writes
# <SUBJECT_ID>_donor_<DONOR>.sofa with the selection embedded, plus a ranking
# CSV and before/after figures, and records the donor in the subject file --
# from here on every session picks it up automatically.
# Redundant if prepare_donor_shortlist() was run; harmless to run anyway.

build_donor_sofa(overwrite=False)
subject = hr.Subject(SUBJECT_ID)

# %% later sessions: confirm which composite is loaded ------------------------bjm
# NOT required -- the config cell already resolved the donor from the subject
# file. This re-reads the SOFA's embedded params and prints them, so you can see
# that what is on disk is what was built (donor, r_match, ridge, fallback flag)
# before running a block on it.
load_existing_donor()

# %% IN SESSION: participant is at floor -- swap to the next donor ------------
# Only when the composite has abolished the cue rather than degraded it (a
# block at chance tells you nothing). Moves down the rule's ranked list, which
# was fixed before the participant heard anything -- see selection.shortlist.
# WRITE WHAT YOU SAW in reason=: the swap is a data-dependent decision and has
# to be reportable as one. The new donor replaces the old one in the subject
# file, so later sessions follow the swap on their own.
#   use_donor(rank=1, reason
#   ="EG 0.03 on baseline A, responses at chance")
# use_donor(rank=1, reason="")

# %% day 1: baseline A -- trained ear, same loc (matches final A) -------------
baseline_A = run_phase("baseline_A", subject)
collect_externalization_rating(baseline_A)

# %% day 1: baseline D -- untrained ear, mirrored loc (matches final D) -------
baseline_D = run_phase("baseline_D", subject)
collect_externalization_rating(baseline_D)
# Condition identity (ear / mirror / hemifield) is carried into the sequence
# name by run_phase(), so every figure titled from it says which cell it is --
# see `_condition_tag`.

# ---------------------------------------------------------------------------
# ADAPTATION DAYS
# anchor -> PRE test -> train -> POST test, so within-session change is
# separable from overnight consolidation and every rating has a same-day top.
# Run the four cells in order.
# ---------------------------------------------------------------------------

# %% adaptation day: 1. PRE-training test -------------------------------------
subject = hr.Subject(SUBJECT_ID)
daily_pre = run_phase("daily", subject)

# %% adaptation day: 2. TRAIN --------------------------------------------------
run_training()

# %% adaptation day: 3. POST-training test ------------------------------------
subject = hr.Subject(SUBJECT_ID)
daily_post = run_phase("daily", subject)

# %% final day: all 4 conditions in this subject's counterbalanced order -------
subject = hr.Subject(SUBJECT_ID)
print(f"Running final tests in order: {FINAL_ORDER}")
for key in FINAL_ORDER:
    run_phase(key, subject)

# %% final day: A -- trained ear, same locations (redo individually) -----------
run_phase("A", subject)

# %% final day: B -- trained ear, mirrored locations (redo individually) -------
run_phase("B", subject)

# %% final day: C -- untrained ear, same locations (redo individually) ---------
run_phase("C", subject)

# %% final day: D -- untrained ear, mirrored locations [MAIN] -----------------
run_phase("D", subject)


# ===========================================================================
# MISC — diagnostics, not part of the per-participant protocol.
# Nothing below runs as a matter of course. Reach for it when something looks
# wrong, or on the odd participant where the extra measurement is worth the
# time. Each cell stands alone; the config cell at the top must have been run.
# ===========================================================================

# %% misc: QC the cepstral split the manipulation depends on ------------------
# Envelope (red) should be smooth and roughly elevation-invariant; if it tracks
# elevation, the split is freezing part of the cue instead of separating it.
# Worth a look on the first few participants and whenever a donor composite
# looks odd in the before/after figure.
plot_split_qc(slab.HRTF(str(paths.SOFA_DIR / SUBJECT_ID / f"{NATIVE_SOFA}.sofa")),
              envelope_n_keep=selection.N_KEEP, ear=TRAINED_EAR,
              band=selection.DEFAULT_BAND)

# %% misc: build the second composite strength (n_keep=8) ---------------------
# Half as much of the cue handed over. Only needed as a rung of the
# externalization ladder below; the training/testing SOFA stays the n_keep=4 one.
build_donor_sofa(overwrite=False, show_qc=False, n_keep=8)

# %% misc: externalization + acute-degradation ladder -------------------------
# ~50 trials plus ratings, so it is NOT run on every participant. Use it when
# externalization is in doubt, when picking OTHER_EAR for a new cohort, or to
# check that a donor composite lands in the intended acute-degradation range.
# Requires the n_keep=8 SOFA from the cell above.
#
# Blocks of ~10 trials in a per-subject RANDOM order, each followed by the 0-10
# rating; elevation gain is reported alongside.
#   anchor     own unmodified HRTF, binaural      <- ceiling of the whole chain
#   flat       other ear = delta impulse          }
#   native     other ear = own full DTF           } ear treatment
#   donor_n4   composite n_keep=4, other ear = OTHER_EAR   }
#   donor_n8   composite n_keep=8, other ear = OTHER_EAR   } composite strength
# donor_n4 is the condition the experiment actually runs. Read the EG column for
# the acute degradation (target 0.3-0.5) and the rating column for
# externalization -- but see the caveat the summary prints about 10-trial EG.
subject = hr.Subject(SUBJECT_ID)
externalization_ladder(
    subject, ladder_settings, seed=SUBJECT_ID,
    rungs=("anchor", "flat", "native", "donor_n4", "donor_n8"))

# %% misc: daily PROBE (only meaningful when OTHER_EAR = 'native') ------------
# Repeats the daily test with the other ear reduced, so relearning of the
# modified cue can be told apart from reweighting toward the intact ear.
subject = hr.Subject(SUBJECT_ID)
run_phase("daily", subject, other_ear=PROBE_OTHER_EAR)

# %% misc: externalization ratings so far, in order ---------------------------
# Every rating on file for this participant, with its block, so drift in the
# anchor can be told apart from drift in the conditions. Anchors are the
# ~10-trial native binaural blocks; they should stay roughly flat across days.
subject = hr.Subject(SUBJECT_ID)
print(f"{'run':46s} {'n':>4s} {'ear':6s} {'mir':5s} {'rating':>6s}")
for _name, _seq in subject.localization.items():
    _rating = getattr(_seq, "externalization_rating", None)
    if _rating is None:
        continue
    _n = len(getattr(_seq, "data", []) or [])
    _tag = " <- anchor" if (_n <= 12 and getattr(_seq, "hrir", "") == NATIVE_SOFA) else ""
    print(f"{_name:46s} {_n:4d} {str(getattr(_seq, 'ear', None)):6s} "
          f"{str(getattr(_seq, 'mirrored', None)):5s} {_rating:6.1f}{_tag}")

# %% misc: which donors this participant has been on --------------------------
# The full swap history from the subject pickle: every donor this participant
# was on, when, and the reason recorded at the time. Read it before comparing
# blocks across days -- a swap mid-experiment means the two are not the same
# manipulation.
show_donor_log()

# %% misc: re-anchor the externalization scale mid-experiment -----------------
# ~10 trials of the participant's own HRTF, binaural, plus a rating. NOT part
# of the protocol any more (day-1 ratings only). Use it if you suspect the
# delivery chain has drifted -- headphone seating, HP filter, OS volume -- and
# compare the number against day 1. A drop here is a chain problem, not
# adaptation.
subject = hr.Subject(SUBJECT_ID)
run_anchor(subject)

# %% misc: force the OS output level on its own -------------------------------
# run_phase() and run_training() already do this. Use it when checking levels
# by ear outside a block, or after someone has touched the volume slider.
_fix_output_level()
