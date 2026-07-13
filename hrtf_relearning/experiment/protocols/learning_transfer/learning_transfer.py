"""
learning_transfer.py

Adaptation-transfer experiment protocol runner.

Session 1 (record the individual HRIR, calibrate headphones, dome-externalization
check and midline AR localization) runs FIRST from HRIR_Recording.py. This script
then picks up from the full-field baseline, builds the modified HRTF, and runs all
localization tests so you never have to hand-edit Localization_AR.py between runs:

    Day 1            native      binaural, native SOFA, full field   (original-HRIR baseline)
                     >> build the modified HRTF (ERB shift) <<
                     baseline_A  monaural trained ear, MODIFIED, trained field    (naive ref for A)
                     baseline_D  monaural untrained ear via MIRROR, MODIFIED,
                                 mirrored field                                   (naive ref for D)
    Adaptation days  daily       monaural trained ear, MODIFIED, trained hemifield
                     (training game runs separately -- see Training.py)
    Final day        A           trained ear,   same loc (= trained hemifield)   [baseline retest]
                     B           trained ear,   mirrored loc
                     C           untrained ear, same loc
                     D           untrained ear, mirrored loc                      [MAIN transfer]

MODIFICATION (the cue manipulation). The modified HRTF is built here by translating
each direction's fine spectral structure a constant distance along the ERB-number
axis inside one octave centred on 8 kHz (5657-11314 Hz), with the coarse envelope
held fixed -- a bijective, relearnable remap of the elevation cue rather than a
destroyed or conflicting cue (Kulkarni & Colburn 1998 cepstral split; magnitude-only,
original phase kept). See hrtf.processing.modify.shift_band and
learning_transfer_methods.md (this folder) for the full method.

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
from pathlib import Path

import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.hrtf.processing.modify import shift_band, octave_band, plot_split_qc
from hrtf_relearning.utils import paths

# The ONLY thing you set per session. Everything else (trained ear, final-day
# block order) is loaded from the counterbalance sheet next to this script, keyed
# by this id. On day 1, write each subject's id into the 'subject' column of:
#   learning_transfer_block_order.csv   (this folder; replace an '(assign)' cell)
SUBJECT_ID = "JS"

CSV_PATH = Path(__file__).resolve().parent / "learning_transfer_block_order.csv"


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

# --- modification (ERB shift) -- see build_modified_sofa() and modify.shift_band ---
SHIFT_CENTER    = 8000   # band centre [Hz]; 1 octave -> 5657-11314 Hz (= VSI band)
SHIFT_OCTAVES   = 1.0
SHIFT_ERB       = 2.5    # ERB displacement of the fine detail (tune per pilot; factor 1.4 ~= 3.0 ERB)
SHIFT_ENV_NKEEP = 4      # Fourier coeffs kept for the coarse envelope (Kulkarni & Colburn 1998)
SHIFT_SKIRT     = 0.25   # cosine taper outside the band [octaves]
SHIFT_EQ_RMS    = True   # match in-band detail RMS per direction/ear

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


def build_modified_sofa(overwrite=True, show_qc=True):
    """Build the ERB-shift modified HRTF from the subject's native SOFA and write
    it to data/hrtf/sofa/<subject>/<subject>_shift.sofa (= MODIFIED_SOFA).

    Translates each direction's fine spectral detail by SHIFT_ERB along the
    ERB-number axis inside the SHIFT_CENTER +/- SHIFT_OCTAVES band, envelope held
    fixed, magnitude-only (original phase). Run ONCE per subject once the native
    recording exists; baseline_A/D, daily and final all load the result.
    Saves the split-QC panel (envelope vs full log-mag across elevation) so you
    can confirm a clean shift, not a frozen-cue conflict, before testing.
    """
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    out_path = sofa_dir / f"{MODIFIED_SOFA}.sofa"
    if out_path.exists() and not overwrite:
        print(f"{out_path.name} already exists (overwrite=False) -- skipping build")
        return out_path
    native = slab.HRTF(str(sofa_dir / f"{NATIVE_SOFA}.sofa"))
    low_hz, high_hz = octave_band(SHIFT_CENTER, fraction=SHIFT_OCTAVES)
    print(f"shift_band: {low_hz:.0f}-{high_hz:.0f} Hz, shift={SHIFT_ERB} ERB")
    modified = shift_band(native, low_hz, high_hz, shift_erb=SHIFT_ERB,
                          envelope_n_keep=SHIFT_ENV_NKEEP, skirt_octaves=SHIFT_SKIRT,
                          equalize_band_rms=SHIFT_EQ_RMS)
    sofa_dir.mkdir(parents=True, exist_ok=True)
    modified.write_sofa(str(out_path))
    print(f"wrote {out_path}")
    if show_qc:
        fig = plot_split_qc(native, SHIFT_ENV_NKEEP, ear="right", band=(low_hz, high_hz))
        plot_dir = paths.subject_plot_dir(SUBJECT_ID)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"{MODIFIED_SOFA}_split_qc.png", bbox_inches="tight")
    return out_path


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
    print(f"SUBJECT: {SUBJECT_ID}    TRAINED EAR: {TRAINED_EAR}    "
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

# %% day 1: native reference (original HRIR, full field) ----------------------
run_phase("native", subject)

# %% day 1: build the modified HRTF (ERB shift) -- run ONCE per subject ---------
# Reads <subject>.sofa, writes <subject>_shift.sofa (= MODIFIED_SOFA), and shows
# the split-QC panel. Tune SHIFT_ERB in the config block if the pilot says so,
# then rerun this cell (overwrite=True). baseline_A/D, daily and final all load
# the file written here.
build_modified_sofa(overwrite=True)
subject = hr.Subject(SUBJECT_ID)   # reload after SOFA write

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
