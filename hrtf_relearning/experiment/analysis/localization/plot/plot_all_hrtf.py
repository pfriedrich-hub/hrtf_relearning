# %% backfill baseline-vs-modified WATERFALL QC plots for existing HRTFs
# For each top-level subject folder in data/hrtf/sofa/ this overlays the
# recorded (base) <ID>.sofa against every modified <ID>_*.sofa as a
# median-plane waterfall (baseline grey vs modified red, stacked by elevation)
# -- the same QC figure save_condition_sofa now auto-produces when a shifted
# copy is generated. Figures are saved to data/results/plot/<ID>/hrtf/.
#
# This is the waterfall view (for edge/notch shifts), NOT the image overlay.
# Run cell by cell (# %%); re-run any cell as needed. Existing PNGs are skipped
# unless OVERWRITE=True.
import logging
import slab
from hrtf_relearning.utils import paths
from hrtf_relearning.hrtf.processing.edge_shift import save_waterfall_qc

logging.basicConfig(level=logging.INFO)

EARS = ('left', 'right')
SMOOTHING = 'raw'          # 'raw' | 'gaussian' | int n_keep (cepstral)
OVERWRITE = False          # skip conditions whose waterfall PNG already exists


# %% collect top-level subjects (folders in SOFA_DIR holding <ID>.sofa)
def subject_dirs():
    out = {}
    for folder in sorted(paths.SOFA_DIR.iterdir()):
        if folder.is_dir() and (folder / f'{folder.name}.sofa').exists():
            out[folder.name] = folder
    return out


# %% plotting
def plot_subject_waterfalls(subject_id, sofa_dir, ears=EARS,
                            smoothing=SMOOTHING, overwrite=OVERWRITE):
    """Waterfall QC of recorded vs each modified HRTF for one subject."""
    out_dir = paths.PLOT_DIR / subject_id / 'hrtf'
    base = slab.HRTF(str(sofa_dir / f'{subject_id}.sofa'))
    base.name = subject_id
    modified = sorted(p for p in sofa_dir.glob(f'{subject_id}_*.sofa'))
    if not modified:
        logging.info(f'{subject_id}: no modified HRTFs.')
    for mod_path in modified:
        out_png = out_dir / f'{mod_path.stem}_waterfall.png'
        if out_png.exists() and not overwrite:
            logging.info(f'{subject_id}: {out_png.name} exists — skipping.')
            continue
        manip = slab.HRTF(str(mod_path))
        save_waterfall_qc(base, manip, mod_path, ears=ears, smoothing=smoothing)
        logging.info(f'{subject_id}: wrote {out_png.relative_to(paths.PLOT_DIR)}')


# %% run for all top-level subjects
if __name__ == '__main__':
    for subject_id, sofa_dir in subject_dirs().items():
        plot_subject_waterfalls(subject_id, sofa_dir)

# %% single subject (adjust as needed)
# subs = subject_dirs(); plot_subject_waterfalls('GS', subs['GS'], overwrite=True)
