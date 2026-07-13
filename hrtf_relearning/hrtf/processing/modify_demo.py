"""
modify_demo.py — run block by block (# %%) to see the ERB shift in action.

Loads a subject's measured HRTF, applies modify.shift_detail, and shows:
  - original vs modified median-plane transfer function (waterfall or image)
  - the coarse/fine split QC (envelope vs full, stacked by elevation)
  - VSI and VSI-dissimilarity in the peak-VSI band

Run cell by cell (# %%) in an IDE/console -- do NOT run top-to-bottom. Nothing
blocks on input; rerun any cell after changing SHIFT_ERB / SHIFT_BAND / N_KEEP to
see the effect. The last cell (optional) writes <id>_shift.sofa.

This is a thin harness around hrtf.processing.modify; the manipulation itself
lives there (shift_detail). See the learning_transfer methods note for the design.
"""

# %% imports and config ------------------------------------------------------
import slab

from hrtf_relearning.utils import paths
from hrtf_relearning.hrtf.processing.modify import shift_detail, plot, plot_split_qc
from hrtf_relearning.hrtf.analysis.vsi import (
    vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity,
)

SUB_ID     = 'AS'            # subject with a measured <id>.sofa under data/hrtf/sofa/<id>/

# --- shift parameters (edit + rerun the shift cell to explore) ---
SHIFT_BAND = (5700, 11300)   # Trapeau peak-VSI octave; None -> whole spectrum
SHIFT_ERB  = 2.5             # ERB displacement (factor 1.4 ~= 3.0 ERB)
N_KEEP     = 4               # cosine coeffs kept for the coarse envelope (M)
SKIRT      = 0.25            # raised-cosine taper on the selection window [octaves]
EQ_RMS     = True            # match in-band detail energy per direction/ear

PLOT_KIND  = 'waterfall'     # 'waterfall' (per-elevation spectra stacked) | 'image'
EAR        = 'right'
VSI_BW     = (5700, 11300)

sofa_dir = paths.SOFA_DIR / SUB_ID


# %% load the measured HRTF --------------------------------------------------
hrtf = slab.HRTF(str(sofa_dir / f'{SUB_ID}.sofa'))
hrtf.name = SUB_ID
print(hrtf)


# %% apply the ERB shift -----------------------------------------------------
hrtf_shift = shift_detail(
    hrtf,
    shift_erb=SHIFT_ERB,
    band=SHIFT_BAND,
    envelope_n_keep=N_KEEP,
    skirt_octaves=SKIRT,
    equalize_rms=EQ_RMS,
)
print(f'shifted band={SHIFT_BAND} Hz by {SHIFT_ERB} ERB (M={N_KEEP})')


# %% VSI metrics (peak-VSI band) ---------------------------------------------
vsi_o = _vsi(hrtf,       bandwidth=VSI_BW)
vsi_m = _vsi(hrtf_shift, bandwidth=VSI_BW)
vsi_d = _vsi_dissimilarity(hrtf, hrtf_shift, bandwidth=VSI_BW)
print(f'VSI  orig={vsi_o:.3f}  mod={vsi_m:.3f}  dissimilarity={vsi_d:.3f}')


# %% original vs modified transfer function (median plane) -------------------
# Look here for the shift: the notch/peak pattern should slide up (or down) by a
# constant ERB step inside the shaded band, envelope unchanged, out-of-band smooth.
fig = plot(hrtf, hrtf_shift, PLOT_KIND, ear=EAR,
           vsi_orig=vsi_o, vsi_mod=vsi_m, vsi_dis=vsi_d, vsi_bw=VSI_BW)


# %% split QC: envelope (M) vs full, stacked by elevation --------------------
# The red envelope must be smooth AND roughly elevation-invariant; if it still
# tracks elevation, M is freezing a cue that carries elevation -> lower N_KEEP.
qc = plot_split_qc(hrtf, N_KEEP, ear=EAR, band=SHIFT_BAND)


# %% (optional) write the modified SOFA --------------------------------------
out_path = sofa_dir / f'{SUB_ID}_shift.sofa'
hrtf_shift.write_sofa(str(out_path))
print('wrote', out_path)
