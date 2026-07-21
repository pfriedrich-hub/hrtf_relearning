# %% backfill WATERFALL plots for existing HRTFs
# For each top-level subject folder in data/hrtf/sofa/ this saves:
#   <ID>_recorded_waterfall.png  -- the recorded (base) <ID>.sofa alone
#   <ID>_<mod>_waterfall.png     -- baseline vs each modified <ID>_*.sofa
# as median-plane waterfalls (stacked by elevation, both ears) -- the same QC
# figures save_condition_sofa auto-produces when a shifted copy is generated.
# Figures are saved to data/results/<ID>/plots/hrtf/.
#
# Run cell by cell (# %%); re-run any cell as needed. Existing PNGs are skipped
# unless OVERWRITE=True.
import logging
import slab
from hrtf_relearning.utils import paths
from hrtf_relearning.hrtf.modify.edge_shift import (
    save_waterfall_qc, save_recorded_hrtf_waterfall,
)

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
    """Waterfall QC for one subject: the recorded HRTF, plus recorded-vs-each
    modified HRTF."""
    out_dir = paths.subject_plot_dir(subject_id) / 'hrtf'
    base = slab.HRTF(str(sofa_dir / f'{subject_id}.sofa'))
    base.name = subject_id
    # recorded HRTF (single-HRTF waterfall) — always produced, even with no mods
    rec_png = out_dir / f'{subject_id}_recorded_waterfall.png'
    if overwrite or not rec_png.exists():
        save_recorded_hrtf_waterfall(base, subject_id=subject_id,
                                     ears=ears, smoothing=smoothing)
        logging.info(f'{subject_id}: wrote {rec_png.relative_to(paths.RESULTS_DIR)}')
    else:
        logging.info(f'{subject_id}: {rec_png.name} exists — skipping.')
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
        logging.info(f'{subject_id}: wrote {out_png.relative_to(paths.RESULTS_DIR)}')


# %% run for all top-level subjects
if __name__ == '__main__':
    for subject_id, sofa_dir in subject_dirs().items():
        plot_subject_waterfalls(subject_id, sofa_dir)

# %% single subject (adjust as needed)
# subs = subject_dirs(); plot_subject_waterfalls('GS', subs['GS'], overwrite=True)
