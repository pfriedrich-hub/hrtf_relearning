"""
experiment_procedure.py
=======================
Master script for the HRTF re-learning experiment.
Run sections top-to-bottom on recording day; on daily sessions jump straight
to the DAILY ROUTINE block.  Each step is a standalone call — comment out
or re-run individual steps as needed, and adjust the settings dicts below
before each run.

Sections
--------
1. IMPORTS & SUBJECT
2. SETTINGS  ← adjust hrir_settings / loc_settings / training_settings here
3. RECORDING DAY
   3a. HRIR recording
   3b. HP equalization
   3c. Acoustic sanity check
   3d. Dome localization   (behavioral, standalone)
   3e. VR localization     (behavioral, standalone)
4. DAILY ROUTINE
   4a. VR localization     (standalone, with daily parameters)
   4b. Training session    (standalone)
"""

# =============================================================================
# 1.  IMPORTS & SUBJECT
# =============================================================================
import logging
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import slab
import freefield

import hrtf_relearning as hr
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.hrtf.record.record_hrir import record_hrir
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import (
    calibrate_headphones, load_hp_filter,
)
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.experiment.Localization.Localization_dome import LocalizationDome
from hrtf_relearning.experiment.Localization.Localization_AR import Localization
from hrtf_relearning.experiment.HRIR_Recording import acoustic_test, compare_localization
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_localization, plot_elevation_response,
)
import hrtf_relearning.experiment.Training as Training

logging.getLogger().setLevel("INFO")
freefield.set_logger("info")
ROOT = hr.PATH

# =============================================================================
# 2.  SETTINGS  ← edit before each run
# =============================================================================

SUBJECT_ID   = "NKa"
HP_ID        = "MYSPHERE"       # headphone model
REFERENCE_ID = "ref_03.04"     # dome calibration reference

# --- HRIR settings (passed to hrtf2binsim and Localization) -----------------
hrir_settings = dict(
    name        = SUBJECT_ID + "_s1",   # binsim folder / filter set name
    subject_id  = SUBJECT_ID,
    ear         = None,                  # None | 'left' | 'right'
    mirror      = False,
    reverb      = True,
    drr         = 20,
    hp_filter   = True,
    hp          = HP_ID,
    convolution = "cuda",
    storage     = "cuda",
)

# --- Localization settings ---------------------------------------------------
loc_settings = dict(
    kind              = "sectors",
    azimuth_range     = (-60, 60),
    elevation_range   = (-35, 35),
    sector_size       = (14, 14),
    targets_per_sector = 3,
    replace           = False,
    min_distance      = 20,
    gain              = 0.2,
    stim              = "noise",
)

# --- Training settings (daily, can override Training module globals) ---------
training_settings = dict(
    target_size     = 4,
    target_time     = 0.5,
    min_dist        = 30,
    game_time       = 90,      # seconds per game
    trial_time      = 10,
    score_time      = 3,
    gain            = 0.10,
    azimuth_range   = (0, 35),
    elevation_range = (-35, 35),
)

# --- Recording-day parameters ------------------------------------------------
FS           = 48828
N_DIRECTIONS = 3
N_RECORDINGS = 10
N_REC_HP     = 3

slab.set_default_samplerate(FS)

# Subject object (shared across all steps)
subject  = Subject(SUBJECT_ID)
plot_dir = ROOT / "data" / "results" / "plot" / SUBJECT_ID

# =============================================================================
# 3.  RECORDING DAY
# =============================================================================

# -----------------------------------------------------------------------------
# 3a.  HRIR RECORDING
# -----------------------------------------------------------------------------
hrir = record_hrir(
    subject_id   = SUBJECT_ID,
    reference_id = REFERENCE_ID,
    n_directions = N_DIRECTIONS,
    n_recordings = N_RECORDINGS,
    fs           = FS,
    show         = True,
    overwrite    = False,
)

# -----------------------------------------------------------------------------
# 3b.  HP EQUALIZATION
# -----------------------------------------------------------------------------
logging.warning("--------- Check HP jack & model ---------")
_hp_filter_path = ROOT / "data" / "hrtf" / "rec" / SUBJECT_ID / f"{HP_ID}_equalization.npz"
try:
    hp_filter = load_hp_filter(_hp_filter_path, "slab")
    logging.info(f"HP filter loaded from disk: {_hp_filter_path.name}")
except FileNotFoundError:
    hp_filter = calibrate_headphones(
        subject_id     = SUBJECT_ID,
        hp_id          = HP_ID,
        n_rec          = N_REC_HP,
        show           = True,
        save_freefield = False,
    )

# -----------------------------------------------------------------------------
# 3c.  ACOUSTIC SANITY CHECK
# -----------------------------------------------------------------------------
acoustic_test(hrir, hp_filter, subject_id=SUBJECT_ID, hp_id=HP_ID, show=True)

# -----------------------------------------------------------------------------
# 3d.  DOME LOCALIZATION  (real speakers — standalone call)
# -----------------------------------------------------------------------------
dome_loc_settings = dict(
    targets_per_speaker = 3,
    min_distance        = 15,
    gain                = 1.0,
)
dome_loc = LocalizationDome(subject, hrir_settings, loc_settings=dome_loc_settings)
dome_loc.run()
plot_elevation_response(subject.localization[dome_loc.filename], filepath=plot_dir)

# -----------------------------------------------------------------------------
# 3e.  VR LOCALIZATION  (pybinsim — standalone call)
# -----------------------------------------------------------------------------
logging.warning("--------- Switch HP jack to PC ---------")
vr_loc = Localization(subject, hrir_settings, loc_settings=loc_settings)
vr_loc.run()
plot_elevation_response(subject.localization[vr_loc.filename], filepath=plot_dir)

# Comparison: dome vs VR
compare_localization(
    dome_seq   = dome_loc.sequence,
    vr_seq     = vr_loc.sequence,
    subject_id = SUBJECT_ID,
    filepath   = plot_dir,
)

# =============================================================================
# 4.  DAILY ROUTINE  (run from here on subsequent days)
# =============================================================================
# Adjust hrir_settings / loc_settings / training_settings above before running.

# -----------------------------------------------------------------------------
# 4a.  VR LOCALIZATION  (standalone — run once or more per session)
# -----------------------------------------------------------------------------
# Quick-edit: update hrir_settings / loc_settings above, then re-run this block.
vr_loc = Localization(subject, hrir_settings, loc_settings=loc_settings)
vr_loc.run()
plot_localization(
    subject.localization[vr_loc.filename],
    report_stats = ["azimuth", "elevation"],
    filepath     = plot_dir,
)
plot_elevation_response(subject.localization[vr_loc.filename], filepath=plot_dir)
plt.show()

# -----------------------------------------------------------------------------
# 4b.  TRAINING SESSION  (standalone — run daily after localization)
# -----------------------------------------------------------------------------
# Push any overrides into the Training module before calling play_session().
Training.SUBJECT_ID = SUBJECT_ID
Training.HRIR_NAME  = hrir_settings["name"]
Training.EAR        = hrir_settings.get("ear")
Training.HP         = HP_ID
Training.AZ_RANGE   = training_settings["azimuth_range"]
Training.settings.update(training_settings)
# Reload HRIR inside Training with current settings
Training.hrir = hrtf2binsim(hrir_settings)
Training.subject = Subject(SUBJECT_ID)

Training.play_session()
