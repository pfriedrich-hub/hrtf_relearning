"""
protocol_edge_shift.py

Cell-by-cell protocol for the spectral-edge elevation experiment (see
documentation/elevation_spectral_cue_models.md for the model background).
Manipulation itself lives in hrtf.modify.edge_shift; this script just
sequences it: build condition SOFAs -> verify against the model -> run the
VR localization blocks (Localization_VR.run).

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom
as a plain script. Nothing here loops or blocks on input; rerun any cell as
needed (e.g. re-verify a condition, redo a block).

Conditions (hrtf.modify.edge_shift), sharing one shift magnitude DELTA_ERB:
    baseline  unmodified individual HRTF                     (reference)
    whole     shift the WHOLE notch up                       (A: dose/sanity check -- all models agree)
    rising    shift the RISING edge up, notch minimum pinned  (B: rising-edge vs notch-CF -- the key contrast)
    falling   shift the FALLING edge down, notch minimum pinned (C: falling-edge control)
"""

# %% imports and config ------------------------------------------------------
import slab

import hrtf_relearning as hr
from hrtf_relearning.hrtf.modify.edge_shift import (
    save_condition_sofa, verify_condition, compare_waterfall, print_notch_summary,
)
from hrtf_relearning.experiment.localization.Localization_VR import run as run_localization
from hrtf_relearning.utils import paths

SUBJECT_ID = "SZ"                            # edit per participant
DELTA_ERB = 1.5                              # shift magnitude (ERB), shared by whole/rising/falling
CONDITIONS = ("whole", "rising", "falling")  # baseline needs no manipulation

ROOT = hr.PATH
SOFA_DIR = paths.SOFA_DIR / SUBJECT_ID
BASE_SOFA_PATH = SOFA_DIR / f"{SUBJECT_ID}.sofa"  # individual measured HRTF (already recorded)

# %% load baseline HRTF -------------------------------------------------------
base_hrtf = slab.HRTF(str(BASE_SOFA_PATH))
base_hrtf.name = SUBJECT_ID

# %% build + write condition SOFAs (run once per subject/delta) --------------
condition_hrtfs = {}
condition_sofa_paths = {}
for condition in CONDITIONS:
    path = SOFA_DIR / f"{SUBJECT_ID}_{condition}.sofa"
    hrtf_new, reports = save_condition_sofa(base_hrtf, condition, DELTA_ERB, path)
    condition_hrtfs[condition] = hrtf_new
    condition_sofa_paths[condition] = path
    print(f"{condition:8s} -> {path.name}")
    print_notch_summary(reports, label="  notches found")
    # a baseline-vs-condition waterfall QC is auto-saved by save_condition_sofa
    # to <id>/plots/hrtf/<name>_waterfall.png

# %% verify one condition against the model before testing it ----------------
# change `condition` and rerun this cell to check a different one
condition = "rising"
report = verify_condition(base_hrtf, condition_hrtfs[condition],
                          BASE_SOFA_PATH, condition_sofa_paths[condition])
print(f"{condition}: model-predicted polar error "
     f"{report['baseline']['mean_pe']:.1f} -> {report['condition']['mean_pe']:.1f} deg "
     f"(delta {report['delta_pe']:+.1f})  |  "
     f"quadrant error delta {report['delta_qe']:+.1f}%")

# %% visual sanity check: median-plane waterfall, baseline vs condition ------
compare_waterfall(base_hrtf, condition_hrtfs[condition],
                  labels=('baseline', condition))

# %% run baseline localization block ------------------------------------------
run_localization(SUBJECT_ID, "baseline", 0)

# %% run whole-notch block (A: dose / sanity check) ----------------------------
run_localization(SUBJECT_ID, "whole", DELTA_ERB)

# %% run rising-edge block (B: the key contrast) -------------------------------
run_localization(SUBJECT_ID, "rising", DELTA_ERB)

# %% run falling-edge block (C: control) ---------------------------------------
run_localization(SUBJECT_ID, "falling", DELTA_ERB)
