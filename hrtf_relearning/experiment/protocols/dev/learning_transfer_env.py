"""
learning_transfer_env.py

PILOT VARIANT of experiment/protocols/learning_transfer/learning_transfer.py.
Everything (design, grid, counterbalancing, ERB-shift modification, final-day 2x2)
is identical EXCEPT what the non-listening ear receives:

    learning_transfer.py   other ear = FLAT      (flatten_dtf: one delta at the
                           onset; ITD + broadband ILD kept, no spectral shape)
    THIS FILE              other ear = ENVELOPE  (envelope_dtf: its own coarse
                           cepstral envelope, ENV_NKEEP coefficients; ITD +
                           broadband ILD kept, fine detail removed)

WHY. With a flat other ear the two ears stop looking like they belong to the same
head: no pinna, no canal resonance, no head-shadow colouration on one side.
Externalization collapses toward the middle of the head, and an internalized
percept is a poor teacher — there is little pressure to recalibrate a spatial map
for a sound that is not out in space. Poor externalization is the suspected cause
of the weak monaural learning seen so far.

The fix keeps the untrained ear plausible without giving it back an elevation
cue. The cepstral split is the SAME one the shift uses on the trained ear
(Kulkarni & Colburn 1998): log|H| = envelope(n_keep) + detail. On the trained ear
the detail is transported along the ERB axis; on the untrained ear it is removed.
Both ears keep the identical untouched envelope, so ENV_NKEEP = SHIFT_ENV_NKEEP
by construction — one number defines "coarse" for the whole paradigm.

CAVEAT to keep in mind when reading the data: the envelope is direction-dependent
(head shadow, broad concha resonance), so the untrained ear is not literally
cue-free the way a flat ear is. At n_keep=4 it has ~2 extrema across the spectrum
and cannot resolve pinna notches (0.5-2 ripples/octave), so it should not support
elevation, but it does co-vary with azimuth. If the pilot shows better learning,
condition C/D still isolate transfer, but any azimuth-based improvement should be
checked against the flat cohort. hrtf/processing/envelope.py run directly prints
the residual detail RMS on the processed ear (native vs envelope vs flat) — do
that once before running a subject.

NOT INTERCHANGEABLE WITH THE FLAT COHORT. The binsim database name carries the
mode (``<sofa>_<ear>_env4`` vs ``<sofa>_<ear>``), so runs never collide on disk
and each localization sequence records ``other_ear`` / ``env_n_keep``. Subjects
run under this file are a separate group; do not pool them with
learning_transfer.py subjects without modelling the ear treatment.

Day 1 also collects an externalization rating after each monaural block (0-10
console prompt) — the whole point of the change, and the flat cohort has no such
number, so collect it consistently or the comparison rests on hearsay.

Counterbalancing is read from learning_transfer_env_block_order.csv IN THIS
FOLDER (separate sheet from the flat cohort). Write each subject's id into the
'subject' column before running.

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as a
plain script. See the parent script's header for the cell-running notes.

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT.
------------------------------------------------------------------------------
"""

SUBJECT_ID = ("XX")

# %% imports and config #------------------------------------------------------
import csv
import os
import subprocess
import sys

import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.hrtf.modify.shift_spectral_detail import shift_spectral_detail, describe
from hrtf_relearning.hrtf.modify.plot_compare import plot
from hrtf_relearning.utils import paths

# The ONLY thing you set per session. Everything else (trained ear, final-day
# block order) is loaded from the counterbalance sheet next to this script, keyed
# by this id. On day 1, write each subject's id into the 'subject' column of:
#   learning_transfer_env_block_order.csv   (this folder; replace an '(assign)' cell)

CSV_PATH = (hr.PATH / "experiment" / "protocols" / "dev"
            / "learning_transfer_env_block_order.csv")


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
        f"Add its id to a row there (replacing an '(assign)' cell) before running."
    )


TRAINED_EAR, FINAL_ORDER = _load_subject_params(SUBJECT_ID)

# SOFA file names (under data/hrtf/sofa/<subject_id>/<name>.sofa)
NATIVE_SOFA   = f"{SUBJECT_ID}"          # individual measured HRTF (day-1 native baseline)
MODIFIED_SOFA = f"{SUBJECT_ID}_shift"    # ERB-shift modified set (built below; baseline + all training/testing)

HP = "DT990"   # headphone EQ profile

# --- modification (ERB shift) on the TRAINED ear -- unchanged from the parent
#     protocol; see hrtf.modify.shift_spectral_detail ---
SHIFT_BAND      = (5700, 11300)  # Trapeau et al. 2016 peak-VSI octave (selection window)
SHIFT_ERB       = 1      # ERB displacement of the fine detail (tune per pilot)
SHIFT_ENV_NKEEP = 4      # Fourier coeffs kept for the coarse envelope (Kulkarni & Colburn 1998)
SHIFT_SKIRT     = 0.1    # taper on the selection window [octaves]; 0 = hard edges (no ghosting)
SHIFT_EQ_RMS    = True   # match per-ERB detail RMS between source and target

# --- THE CHANGE: what the non-listening ear gets in every monaural block ------
# 'envelope' -> hrtf.processing.envelope.envelope_dtf, ENV_NKEEP coefficients.
# Set OTHER_EAR = 'flat' to reproduce the parent protocol exactly (useful as a
# within-subject A/B on day 1 before committing a subject to the pilot).
OTHER_EAR = "envelope"
ENV_NKEEP = SHIFT_ENV_NKEEP   # same 'coarse' as the trained ear's held-fixed envelope

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
#     NOT the untrained ear's own DTF. This is the whole point of the transfer
#     test, and is why we use mirror rather than ear=UNTRAINED.
#   - azimuth selects the physical location: trained vs mirrored hemifield.
# The ear reduction happens BEFORE the mirror (see hrtf2binsim), so with
# OTHER_EAR='envelope' the envelope-only ear is mirrored along with everything
# else and the listening ear always receives the trained filter, exactly as in
# the flat version.


def hrir_settings(sofa_name, ear=None, mirror=False):
    return {
        "name": sofa_name,
        "subject_id": SUBJECT_ID,
        "ear": ear,              # None -> binaural; 'left'/'right' -> reduce the other ear
        "other_ear": OTHER_EAR,  # 'envelope' (this pilot) | 'flat' (parent protocol)
        "env_n_keep": ENV_NKEEP,
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


def build_modified_sofa(overwrite=True, show_qc=True):
    """Build the ERB-shift modified HRTF from the subject's native SOFA and write
    it to data/hrtf/sofa/<subject>/<subject>_shift.sofa (= MODIFIED_SOFA).

    Identical to the parent protocol -- the ear reduction is NOT baked into the
    SOFA, it is applied at binsim build time, so this file is shared with the
    flat cohort and can be reused if a subject was already recorded.
    """
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    out_path = sofa_dir / f"{MODIFIED_SOFA}.sofa"
    if out_path.exists() and not overwrite:
        print(f"{out_path.name} already exists (overwrite=False) -- skipping build")
        return out_path
    native = slab.HRTF(str(sofa_dir / f"{NATIVE_SOFA}.sofa"))
    describe(SHIFT_BAND, SHIFT_ERB)
    modified = shift_spectral_detail(native, shift_erb=SHIFT_ERB, band=SHIFT_BAND,
                            envelope_n_keep=SHIFT_ENV_NKEEP, skirt_octaves=SHIFT_SKIRT,
                            equalize_rms=SHIFT_EQ_RMS)
    sofa_dir.mkdir(parents=True, exist_ok=True)
    modified.write_sofa(str(out_path))
    print(f"wrote {out_path}")
    if show_qc:
        # before/after HRTF image (native vs modified, median plane)
        fig = plot(native, modified, kind="image", ear="right")
        plot_dir = paths.subject_plot_dir(SUBJECT_ID)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"{MODIFIED_SOFA}_before_after.png", bbox_inches="tight")
    return out_path


def check_other_ear(save=True):
    """QC the ear reduction BEFORE running a subject.

    Prints what the non-listening ear still carries (native vs envelope vs flat)
    and saves a before/after image of that ear. What you want to see: the
    elevation SD well down from native (the cue that must not survive) while the
    azimuth SD stays close to native (head shadow, what keeps the ear
    plausible). If the elevation SD is still near native, lower ENV_NKEEP.
    """
    from hrtf_relearning.hrtf.processing.envelope import (
        envelope_dtf, residual_detail_db, direction_variation_db)
    from hrtf_relearning.hrtf.processing.flatten import flatten_dtf

    sofa_path = paths.SOFA_DIR / SUBJECT_ID / f"{MODIFIED_SOFA}.sofa"
    hrtf = slab.HRTF(str(sofa_path))
    hrtf.name = sofa_path.stem
    other = UNTRAINED_EAR

    env = envelope_dtf(hrtf, ear=TRAINED_EAR, n_keep=ENV_NKEEP)
    flat = flatten_dtf(hrtf, ear=TRAINED_EAR)
    native_db, env_db = residual_detail_db(hrtf, env, ear=TRAINED_EAR)
    _, flat_db = residual_detail_db(hrtf, flat, ear=TRAINED_EAR)
    print(f"{SUBJECT_ID}: {other} (untrained) ear, 4-16 kHz")
    print(f"   detail RMS:  native {native_db:.2f} dB | "
          f"envelope(n_keep={ENV_NKEEP}) {env_db:.2f} dB | flat {flat_db:.2f} dB")
    for label, h in (("native", hrtf), (f"envelope({ENV_NKEEP})", env), ("flat", flat)):
        el_sd, az_sd = direction_variation_db(h, ear=TRAINED_EAR)
        print(f"   spectral SD {label:>14}:  elevation {el_sd:5.2f} dB | "
              f"azimuth {az_sd:5.2f} dB")

    fig = plot(hrtf, env, "image", ear=other)
    if save:
        plot_dir = paths.subject_plot_dir(SUBJECT_ID)
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_png = plot_dir / f"{MODIFIED_SOFA}_env{ENV_NKEEP}_{other}_ear.png"
        fig.savefig(out_png, bbox_inches="tight")
        print(f"wrote {out_png}")
    return fig


TRAINING_SCRIPT = hr.PATH / "experiment" / "training" / "Training_AR.py"


def run_training(hrir_name=None, ear=None, az_range=None):
    """Launch the AR training game (Training_AR.py) in its own process with this
    subject's modified HRIR, trained ear, trained hemifield AND the envelope-only
    other ear (passed as TRAINING_OTHER_EAR / TRAINING_ENV_NKEEP, so the spawned
    multiprocessing workers inherit it -- if these were left unset the training
    game would silently run with a FLAT other ear while the tests ran with an
    envelope one).

    Blocks until you press ESC at a game-over prompt (or close the game window);
    the training process then disconnects the motion sensor and stops pybinsim
    before returning, so the daily test cell can be run next in the same console.
    Do NOT Ctrl+C to stop training.
    """
    hrir_name = MODIFIED_SOFA if hrir_name is None else hrir_name
    ear = TRAINED_EAR if ear is None else ear
    az_range = TRAINED_HEMI if az_range is None else az_range

    # guard: the modified HRIR must exist (build_modified_sofa cell) before training
    sofa_path = paths.SOFA_DIR / SUBJECT_ID / f"{hrir_name}.sofa"
    if not sofa_path.exists():
        raise FileNotFoundError(
            f"Modified HRIR not found:\n  {sofa_path}\n"
            f"Run the 'build the modified HRTF (ERB shift)' cell first "
            f"(build_modified_sofa()).")

    print("-" * 64)
    print(f"TRAINING   subject={SUBJECT_ID}   ear={ear}   az_range={az_range}")
    print(f"           HRIR={hrir_name}.sofa   HP={HP}")
    print(f"           other ear={OTHER_EAR} (n_keep={ENV_NKEEP})")
    print("-" * 64)

    env = dict(os.environ,
               TRAINING_SUBJECT_ID=SUBJECT_ID,
               TRAINING_HRIR_NAME=hrir_name,
               TRAINING_EAR=ear,
               TRAINING_OTHER_EAR=OTHER_EAR,
               TRAINING_ENV_NKEEP=str(ENV_NKEEP),
               TRAINING_AZ_RANGE=f"{az_range[0]},{az_range[1]}",
               TRAINING_HP=HP)
    # Launch as a package MODULE (-m), not by file path, and with cwd set to the
    # repo root (the parent of the package dir) -- see the parent protocol for
    # why (spawn workers re-import the main module and must resolve the real
    # installed package, not the repo-root namespace package of the same name).
    subprocess.run(
        [sys.executable, "-m", "hrtf_relearning.experiment.training.Training_AR"],
        env=env, cwd=str(hr.PATH.parent), check=False,
    )


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
    other = f"{OTHER_EAR} (n_keep={ENV_NKEEP})" if ear else "-"
    return (f"{label}  [{when}]\n"
            f"      {desc}\n"
            f"      SOFA={sofa}  ear={ear or 'binaural'}  other ear={other}  mirror={mirror}\n"
            f"      azimuth={az}  elevation={ELEVATION_RANGE}  sector={SECTOR_SIZE}  "
            f"tps={TARGETS_PER_SECTOR}  midline az=0 {midline}\n"
            f"      ~{_est_trials(az)} trials")


def run_phase(key, subject):
    """Run one phase. Returns the Localization object so an externalization
    rating can be attached to the run it belongs to."""
    label, when, sofa, ear, mirror, az, desc = PHASES[key]
    print("\n" + "=" * 70)
    print(f"RUNNING:  {_describe(key)}")
    print("=" * 70)
    one_sided = tuple(az) != tuple(FULL_FIELD)   # drop az~=0 only in one-sided tests
    test = Localization(subject, hrir_settings(sofa, ear=ear, mirror=mirror),
                        loc_settings=loc_settings(az, exclude_midline=one_sided))
    test.run()
    print(f"Done: {test.filename}")
    return test


def collect_externalization_rating(loc_test):
    """Post-block externalization report, console-collected.

    This is the outcome the whole manipulation targets, so collect it after
    EVERY monaural block (baselines, daily, final). Same 0-10 wording as
    expectation_transfer.py, so the numbers are comparable across protocols.
    """
    print("\n--- Post-block question ---")
    while True:
        raw = input("Externalization (0 = entirely inside your head, "
                    "10 = felt like a real external loudspeaker): ").strip()
        try:
            rating = float(raw)
            break
        except ValueError:
            print("Please enter a number 0-10.")
    sequence = loc_test.subject.localization[loc_test.filename]
    sequence.externalization_rating = rating
    loc_test.subject.write()
    print(f"Recorded: externalization={rating}\n")


def show_status(subject):
    print("\n" + "-" * 70)
    print(f"SUBJECT: {SUBJECT_ID}    TRAINED EAR: {TRAINED_EAR}    "
          f"UNTRAINED: {UNTRAINED_EAR}")
    print(f"  (loaded from {CSV_PATH.name})")
    print(f"PILOT VARIANT: other ear = {OTHER_EAR} (n_keep={ENV_NKEEP}) "
          f"-- not poolable with the flat cohort")
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

# %% day 1: native reference (original HRIR, full field) ----------------------
run_phase("native", subject)

# %% day 1: build the modified HRTF (ERB shift) -- run ONCE per subject ---------
# Reads <subject>.sofa, writes <subject>_shift.sofa (= MODIFIED_SOFA). The ear
# reduction is NOT in this file -- it is applied when the binsim database is
# built, so the same SOFA serves the envelope and flat versions.
build_modified_sofa(overwrite=False)
subject = hr.Subject(SUBJECT_ID)   # reload after SOFA write

# %% day 1: QC the untrained ear (run before the first monaural block) ---------
# Detail RMS on the untrained ear: native vs envelope vs flat, plus a before/after
# image of that ear. Lower ENV_NKEEP if too much fine structure survives.
check_other_ear()

# %% day 1: baseline A -- trained ear, same loc (matches final A) --------------
baseline_A = run_phase("baseline_A", subject)
collect_externalization_rating(baseline_A)

# %% day 1: baseline D -- untrained ear, mirrored loc (matches final D) --------
baseline_D = run_phase("baseline_D", subject)
collect_externalization_rating(baseline_D)

# %% adaptation days: TRAIN -----------------------------------------------------
# Rerun once per adaptation day. Training launches in its own process with the
# trained ear + trained hemifield on the modified HRIR AND the envelope-only
# other ear; play the games (e.g. 15), then press ESC at the GAME OVER prompt to
# quit cleanly (sensor disconnects, pybinsim stops).
run_training()

# %% adaptation days: daily TEST -------------------------------------------------
# Run after the training block above, once per adaptation day.
subject = hr.Subject(SUBJECT_ID)   # reload after training appended trials
daily = run_phase("daily", subject)
collect_externalization_rating(daily)

# %% final day: all 4 conditions in this subject's counterbalanced order --------
# Order is loaded per-subject from learning_transfer_env_block_order.csv.
# To redo a single condition instead, use one of the cells below.
subject = hr.Subject(SUBJECT_ID)   # reload after training appended trials
print(f"Running final tests in order: {FINAL_ORDER}")
for key in FINAL_ORDER:
    collect_externalization_rating(run_phase(key, subject))

# %% final day: A -- trained ear, same locations (redo individually) -----------
collect_externalization_rating(run_phase("A", subject))

# %% final day: B -- trained ear, mirrored locations (redo individually) -------
collect_externalization_rating(run_phase("B", subject))

# %% final day: C -- untrained ear, same locations (redo individually) ---------
collect_externalization_rating(run_phase("C", subject))

# %% final day: D -- untrained ear, mirrored locations [MAIN] (redo individually)
collect_externalization_rating(run_phase("D", subject))
