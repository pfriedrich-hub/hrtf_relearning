"""Central path definitions for the hrtf_relearning package.

All directories are derived from the installed package root (hrtf_relearning.PATH).
Import from here instead of rebuilding paths in individual modules:

    from hrtf_relearning.utils.paths import SOFA_DIR, PLOT_DIR
"""
from pathlib import Path

# Absolute path to the installed package root (same as hrtf_relearning.PATH)
PATH = Path(__file__).resolve().parent.parent  # package root (utils/ -> hrtf_relearning/)

# --- data ---
DATA_DIR = PATH / "data"

# HRTF data
HRTF_DIR = DATA_DIR / "hrtf"
SOFA_DIR = HRTF_DIR / "sofa"          # measured + modified HRTFs, per subject
BINSIM_DIR = HRTF_DIR / "binsim"      # wav/filter lists for pybinsim
REC_DIR = HRTF_DIR / "rec"            # raw in-ear recordings (dome)
REC_MESM_DIR = HRTF_DIR / "rec_mesm"  # raw in-ear recordings (MESM)

# Results
RESULTS_DIR = DATA_DIR / "results"
PLOT_DIR = RESULTS_DIR / "plot"       # per-subject figures: PLOT_DIR / subject_id
BACKUP_DIR = RESULTS_DIR / "backup"   # subject pickle backups
ARCHIVE_DIR = RESULTS_DIR / "archive" # archived tuning sequences

# Sounds / assets
SOUNDS_DIR = DATA_DIR / "sounds"
IMG_DIR = DATA_DIR / "img"
UI_DIR = DATA_DIR / "ui"              # game UI assets (fonts, icons)

# Documentation / protocol assets
DOCUMENTATION_DIR = DATA_DIR / "documentation"

# Analysis output (not subject data)
ANALYSIS_RESULTS_DIR = PATH / "analysis_results"
