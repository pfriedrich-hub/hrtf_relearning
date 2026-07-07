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


# --- QC plotting -------------------------------------------------------------
def plot_median_features(hrtf, med, ear=0, band=(3000., 16000.), feature_kw=None,
                         title=None, show=True):
    """Median-plane QC: DTF magnitude (elevation x frequency) with each notch
    center (o) and its half-height rising edge (^) overlaid, so the cue the
    edge-shift manipulation acts on is visible before any SOFA is built.

    `med` is the elevation-sorted median-plane index list. Returns the figure."""
    import numpy as np
    import matplotlib.pyplot as plt

    arr, fs = hrtf_to_array(hrtf)
    vp = hrtf.sources.vertical_polar
    els = np.array([vp[i, 1] for i in med])

    # magnitude image on the shared rfft grid (parametric_summary's 4x-nextpow2)
    n = arr.shape[-1]
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    fsel = (freqs >= band[0]) & (freqs <= band[1])
    fb = freqs[fsel]
    img = np.empty((len(med), fb.size))
    notch_pts, edge_pts = [], []  # (freq_khz, elevation)
    for row, i in enumerate(med):
        mag = np.abs(np.fft.rfft(arr[i, :, ear], nfft))
        img[row] = 20.0 * np.log10(np.maximum(mag[fsel], 1e-9))
        feats = parametric_summary(arr[i, :, ear], fs, feature_kw=feature_kw)["features"]
        for f in feats:
            if f.get("f_hz"):
                notch_pts.append((f["f_hz"] / 1000, vp[i, 1]))
            if f.get("f_edge_rise_hz"):
                edge_pts.append((f["f_edge_rise_hz"] / 1000, vp[i, 1]))

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(fb / 1000, els, img, cmap="viridis", shading="nearest")
    fig.colorbar(mesh, ax=ax, label="magnitude (dB)")
    if notch_pts:
        nx, ny = zip(*notch_pts)
        ax.scatter(nx, ny, marker="o", s=45, facecolors="none",
                   edgecolors="w", linewidths=1.4, label="notch center")
    if edge_pts:
        ex, ey = zip(*edge_pts)
        ax.scatter(ex, ey, marker="^", s=45, c="#D85A30",
                   edgecolors="k", linewidths=0.5, label="rising edge (half-height)")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(title or f"{getattr(hrtf, 'name', '')} median-plane notch/edge (ear {ear})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


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
plot_median_features(base_hrtf, med, feature_kw=FEATURE_KW,
                     title=f"{SUBJECT_ID} baseline - median-plane notch/edge")

# %% PHASE 1 -- transfer baseline: run experiment/protocols/expectation_transfer.py
# (AR_pre -> Dome -> AR_post). Nothing to do here; kept as a pointer.

# %% PHASE 2a -- build whole-notch dose SOFAs --------------------------------
whole_labels = {}
for delta in WHOLE_DELTAS:
    label = _label("whole", delta)
    manip_hrtf, reports = save_condition_sofa(base_hrtf, "whole", delta,
                                     SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa", feature_kw=FEATURE_KW)
    whole_labels[delta] = label
    print(f"whole delta={delta:>4} -> {SUBJECT_ID}_{label}.sofa")
    print_notch_summary(reports, label="  notches shifted")
    # confirm the written SOFA: baseline vs manipulated across the median arc,
    # plus the shifted notch/edge overlay for this dose.
    fig = compare_tf(base_hrtf, manip_hrtf, show=False)
    fig.suptitle(f"{SUBJECT_ID} whole delta={delta:g} ERB", y=1.02)
    plot_median_features(manip_hrtf, med, feature_kw=FEATURE_KW,
                         title=f"{SUBJECT_ID} whole delta={delta:g} ERB - notch/edge")

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
