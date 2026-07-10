"""Central path definitions for the hrtf_relearning package.

All directories are derived from the installed package root (hrtf_relearning.PATH).
Import from here instead of rebuilding paths in individual modules:

    from hrtf_relearning.utils.paths import SOFA_DIR, subject_plot_dir

Per-subject results live together under RESULTS_DIR/<id>/:
    RESULTS_DIR/<id>/<id>.pkl    subject pickle          (subject_pkl)
    RESULTS_DIR/<id>/<id>.json   scoreboard/data backup  (subject_backup)
    RESULTS_DIR/<id>/plots/      figures                 (subject_plot_dir)
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

# Results — everything for one subject lives under RESULTS_DIR/<id>/
RESULTS_DIR = DATA_DIR / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"  # archived tuning sequences


def subject_dir(subject_id):
    """RESULTS_DIR/<id>/ — holds the pickle, JSON backup and plots/ subfolder."""
    return RESULTS_DIR / subject_id


def subject_pkl(subject_id):
    """RESULTS_DIR/<id>/<id>.pkl — authoritative subject pickle."""
    return RESULTS_DIR / subject_id / f"{subject_id}.pkl"


def subject_backup(subject_id):
    """RESULTS_DIR/<id>/<id>.json — plain-text scoreboard/data backup."""
    return RESULTS_DIR / subject_id / f"{subject_id}.json"


def subject_plot_dir(subject_id):
    """RESULTS_DIR/<id>/plots/ — per-subject figures."""
    return RESULTS_DIR / subject_id / "plots"


def subject_pkls():
    """Every existing RESULTS_DIR/<id>/<id>.pkl.

    Active subjects only: the pilot archive (RESULTS_DIR/pilot/...) has no
    pilot/pilot.pkl, so it is naturally excluded."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(d / f"{d.name}.pkl" for d in RESULTS_DIR.iterdir()
                  if d.is_dir() and (d / f"{d.name}.pkl").exists())

# Sounds / assets
SOUNDS_DIR = DATA_DIR / "sounds"
IMG_DIR = DATA_DIR / "img"
UI_DIR = DATA_DIR / "ui"              # game UI assets (fonts, icons)

# Documentation / protocol assets
DOCUMENTATION_DIR = DATA_DIR / "documentation"

# Analysis output (not subject data)
ANALYSIS_RESULTS_DIR = PATH / "analysis_results"
