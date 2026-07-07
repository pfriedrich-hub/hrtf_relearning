"""
shift_edge_pilot.py

Cell-by-cell protocol for the shift-edge PILOT. All localization tests run in AR
(HRTF over headphones, Localization_AR) on the VERTICAL MIDLINE only, matched to
the dome speaker layout. Detector/manipulation config: Iida Gaussian, all valid
notches in 4-15 kHz, half-height edges (hrtf.processing.edge_shift).

Phases:
  1. Externalization / localization transfer baseline (AR - dome - AR).
     --> USE THE EXISTING PROTOCOL: experiment/protocols/expectation_transfer.py
     (AR_pre -> Dome -> AR_post). Not rebuilt here.
  2. Whole-notch DOSE search: shift every notch up by delta for
     delta in {0.25, 0.5, 1.0, 1.5}, run in quick succession, 3 trials per
     midline speaker. Find the smallest delta that shifts perceived elevation.
  3. Edge shift at the chosen delta (CHOSEN_DELTA) -- compare against the
     whole-notch condition at the same delta.
  4. Edge only: flatten the DTF, keep only the rising edges (DCN edge-isolation).
     *** WIP: edge_only_ir currently has a low-frequency artifact -- QC with
     dev/waterfall_edge_qc.py 0.5 edge_only before running participants. ***

Run cell by cell (# %%). Nothing blocks on input; rerun any cell as needed.
Build a condition's SOFA (build cell) before running its AR block.
"""

# %% imports and config ------------------------------------------------------
import slab

import hrtf_relearning as hr
from hrtf_relearning.hrtf.processing.edge_shift import (
    save_condition_sofa, compare_tf, print_notch_summary, parametric_summary,
    hrtf_to_array,
)
from hrtf_relearning.experiment.localization.Localization_AR import Localization

SUBJECT_ID = "GS"                        # edit per participant
HP = "DT990"
NOTCH_BAND = (4000., 15000.)             # manipulate all valid notches here
FEATURE_KW = dict(band=NOTCH_BAND)
WHOLE_DELTAS = (0.25, 0.5, 1.0, 1.5)     # dose staircase (ERB), quick succession
CHOSEN_DELTA = 1.5                       # set to the threshold from phase 2, then run phases 3-4

TARGETS_PER_SPEAKER = 3
MIN_DISTANCE = 15
GAIN = 0.2

ROOT = hr.PATH
SOFA_DIR = ROOT / "data" / "hrtf" / "sofa" / SUBJECT_ID
BASE_SOFA_PATH = SOFA_DIR / f"{SUBJECT_ID}.sofa"

# Vertical-midline AR settings (match the dome layout / expectation_transfer.py)
AR_MIDLINE_SETTINGS = {
    "kind": "standard",
    "azimuth_range": (-1, 1),
    "elevation_range": (-35, 35),
    "targets_per_speaker": TARGETS_PER_SPEAKER,
    "min_distance": MIN_DISTANCE,
    "gain": GAIN,
    "stim": "noise",
}


def _label(mode, delta):
    return f"{mode}_{delta:g}".replace(".", "p") if mode != "edge_only" else "edge_only"


def hrir_settings(label):
    """binaural, vertical-midline; `name` resolves data/hrtf/sofa/<subj>/<subj>_<label>.sofa."""
    return {
        "name": f"{SUBJECT_ID}_{label}",
        "subject_id": SUBJECT_ID,
        "ear": None, "mirror": False,
        "reverb": True, "drr": 20,
        "hp_filter": True, "hp": HP,
        "convolution": "cpu", "storage": "cpu",
    }


def run_ar(subject, label):
    """Build the binsim files for <subj>_<label>.sofa and run one AR midline block."""
    loc = Localization(subject, hrir_settings(label), AR_MIDLINE_SETTINGS)
    loc.run()
    return subject.localization[loc.filename]


# %% load baseline HRTF -------------------------------------------------------
subject = hr.Subject(SUBJECT_ID)
base_hrtf = slab.HRTF(str(BASE_SOFA_PATH))
base_hrtf.name = SUBJECT_ID

# %% sanity: valid notches + rising edges across the median plane ------------
arr, fs = hrtf_to_array(base_hrtf)
vp = base_hrtf.sources.vertical_polar
az = (vp[:, 0] + 180) % 360 - 180
med = sorted([i for i in range(len(vp)) if abs(az[i]) <= 2 and -35 <= vp[i, 1] <= 35],
             key=lambda i: vp[i, 1])
for i in med[::max(1, len(med) // 12)]:
    feats = parametric_summary(arr[i, :, 0], fs, feature_kw=FEATURE_KW)["features"]
    pairs = ["{:.1f}->{:.1f}kHz".format(f["f_hz"] / 1000, (f.get("f_edge_rise_hz") or 0) / 1000)
             for f in feats]
    print("el={:+6.1f}  notch->edge: {}".format(vp[i, 1], pairs))
# todo plot!!

# %% PHASE 1 -- transfer baseline: run experiment/protocols/expectation_transfer.py
# (AR_pre -> Dome -> AR_post). Nothing to do here; kept as a pointer.

# %% PHASE 2a -- build whole-notch dose SOFAs --------------------------------
whole_labels = {}
for delta in WHOLE_DELTAS:
    label = _label("whole", delta)
    _, reports = save_condition_sofa(base_hrtf, "whole", delta,
                                     SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa", feature_kw=FEATURE_KW)
    whole_labels[delta] = label
    print(f"whole delta={delta:>4} -> {SUBJECT_ID}_{label}.sofa")
    print_notch_summary(reports, label="  notches shifted")
# todo plot sofa for confirmation!!

# %% PHASE 2b -- run one whole-notch dose block (quick succession) ------------
# set DELTA_TO_RUN to each WHOLE_DELTAS value in turn (ascending), rerun this
# cell, until localized elevation clearly follows the shift -> put it in
# CHOSEN_DELTA above.
DELTA_TO_RUN = 1.5
run_ar(subject, whole_labels[DELTA_TO_RUN])


# %% PHASE 3a -- build edge-shift SOFA at the chosen delta --------------------
edge_label = _label("edge", CHOSEN_DELTA)
_, edge_reports = save_condition_sofa(base_hrtf, "edge", CHOSEN_DELTA,
                                     SOFA_DIR / f"{SUBJECT_ID}_{edge_label}.sofa", feature_kw=FEATURE_KW)
print(f"edge delta={CHOSEN_DELTA} -> {SUBJECT_ID}_{edge_label}.sofa")
print_notch_summary(edge_reports, label="  edges shifted")

# %% PHASE 3b -- run edge-shift block (compare vs whole at same delta) --------
run_ar(subject, edge_label)

# %% PHASE 4a -- build edge-only (flatten) SOFA  [WIP: QC first] --------------
# edge_only ignores delta. *** verify dev/waterfall_edge_qc.py 0.5 edge_only
# looks clean before running participants -- current build has a LF artifact. ***
save_condition_sofa(base_hrtf, "edge_only", 0.0,
                    SOFA_DIR / f"{SUBJECT_ID}_edge_only.sofa", feature_kw=FEATURE_KW)
print(f"edge_only -> {SUBJECT_ID}_edge_only.sofa")

# %% PHASE 4b -- run edge-only block -----------------------------------------
run_ar(subject, "edge_only")

# %% visual sanity: baseline vs a condition (edit `label`) -------------------
label = edge_label
compare_tf(base_hrtf, slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa")))
