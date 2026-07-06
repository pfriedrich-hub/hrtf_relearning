"""
waterfall_edge_qc.py  (dev/analysis)

Quick-check the edge-shift manipulation: line-based median-plane waterfalls that
overlay the ORIGINAL (grey) and edge-SHIFTED (red) DTFs stacked by elevation, so
original vs modified are directly comparable in one plot (no image/heatmap).

Applies edge_shift.edge_shift_ir(mode='edge', shift_erb=DELTA) with the pilot
config (Gaussian detector, all valid notches in 4-15 kHz, half-height edge,
fixed-shape rigid shift, next-cue ceiling). Left ear, median plane.

Outputs (analysis_results/edge_qc/):
  _montage.png      -- all subjects at a glance (grid of mini-waterfalls)
  <subject>.png     -- full-size per-subject waterfall

Run: python3 hrtf_relearning/hrtf/processing/dev/waterfall_edge_qc.py [DELTA]
"""
import sys
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUTDIR = ROOT / "analysis_results/edge_qc"
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

DELTA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MODE = sys.argv[2] if len(sys.argv) > 2 else 'edge'   # 'edge' | 'whole' | 'edge_only'
FMT = sys.argv[3] if len(sys.argv) > 3 else 'png'     # 'png' | 'pdf' (vector) | 'svg'
FEATURE_KW = dict(band=(4000., 15000.))
F_LO, F_HI = 3000., 16000.
N_EL = 11            # elevations to draw per subject
OFFSET = 35.0        # dB offset between stacked elevations
VARIANTS = ("_notch", "_shift", "_synth", "_rising", "_falling", "_whole", "_edge",
            "_test", "_full", "_molds", "_0", "_4", "_s_", "_01", "__")


def load(sofa):
    with h5py.File(sofa, "r") as f:
        return f["Data.IR"][:], f["SourcePosition"][:], float(np.array(f["Data.SamplingRate"]).ravel()[0])


def median_dirs(sp, n=N_EL):
    azn = (sp[:, 0] + 180) % 360 - 180
    el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]
    med = med[(el[med] >= -35) & (el[med] <= 35)]
    med = med[np.argsort(el[med])]
    if len(med) > n:
        med = med[np.linspace(0, len(med) - 1, n).astype(int)]
    return med, el


def raw_db(x, fs):
    """RAW (unsmoothed) log-magnitude -- the actual presented DTF. Detection
    smooths internally to find edges, but the manipulation is applied to the
    raw spectrum, so QC shows the raw before/after the subject really hears."""
    nfft = int(2 ** np.ceil(np.log2(len(x))) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    return fr, 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x, nfft)), 1e-9))


def waterfall(ax, sofa, delta, mini=False):
    ir, sp, fs = load(sofa)
    med, el = median_dirs(sp)
    if len(med) < 3:
        ax.set_visible(False); return 0
    off = 0.0
    for d in med:
        x0 = ir[d, 0, :]
        if MODE == 'edge_only':
            x1 = es.edge_only_ir(x0, fs, feature_kw=FEATURE_KW)
        else:
            x1 = es.edge_shift_ir(x0, fs, delta, mode=MODE, strict=False, feature_kw=FEATURE_KW)
        fr, L0 = raw_db(x0, fs); _, L1 = raw_db(x1, fs)
        b = (fr >= F_LO) & (fr <= F_HI)
        ax.plot(fr[b] / 1000, L0[b] + off, color="0.6", lw=0.8 if mini else 1.2)
        ax.plot(fr[b] / 1000, L1[b] + off, color="#D62728", lw=0.8 if mini else 1.2)
        if not mini:
            ax.text(F_HI / 1000 * 1.02, L0[b][-1] + off, f"{el[d]:+.0f}", fontsize=7, va="center")
        off += OFFSET
    ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 16])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(F_LO / 1000, F_HI / 1000)
    ax.set_yticks([])
    return 1


def subject_list():
    subs = [("JS", SOFA / "JS/JS.sofa"), ("CA", SOFA / "CA/CA.sofa")]
    for p in sorted((SOFA / "pilot").glob("*.sofa")):
        if any(v in p.stem for v in VARIANTS):
            continue
        subs.append((p.stem, p))
    return subs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    subs = subject_list()
    # montage
    ncol = 5
    nrow = int(np.ceil(len(subs) / ncol))
    figm, axm = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow))
    axm = np.atleast_2d(axm)
    for k, (name, path) in enumerate(subs):
        ax = axm[k // ncol, k % ncol]
        try:
            ok = waterfall(ax, path, DELTA, mini=True)
        except Exception as e:
            ax.set_visible(False); print(f"{name}: ERROR {type(e).__name__}"); continue
        if ok:
            ax.set_title(name, fontsize=9)
    for k in range(len(subs), nrow * ncol):
        axm[k // ncol, k % ncol].set_visible(False)
    modlabel = "edge-only (flatten)" if MODE == 'edge_only' else f"{MODE} +{DELTA} ERB"
    figm.legend(handles=[Line2D([], [], color="0.6", lw=1.2, label="original"),
                         Line2D([], [], color="#D62728", lw=1.2, label=modlabel)],
                loc="upper right", fontsize=9)
    figm.suptitle(f"QC {modlabel} (4-15 kHz): median-plane DTFs, original vs modified", y=1.0)
    figm.tight_layout()
    mpath = OUTDIR / f"_montage_{MODE}.{FMT}"
    figm.savefig(mpath, dpi=120, bbox_inches="tight")
    print("saved", mpath)
    # per-subject full size
    for name, path in subs:
        fig, ax = plt.subplots(figsize=(7, 8))
        try:
            if not waterfall(ax, path, DELTA):
                plt.close(fig); continue
        except Exception as e:
            plt.close(fig); print(f"{name}: ERROR {type(e).__name__}"); continue
        ax.set_xlabel("frequency (kHz)"); ax.set_title(f"{name}: median-plane DTF, original vs {modlabel}")
        ax.legend(handles=[Line2D([], [], color="0.6", lw=1.2, label="original"),
                           Line2D([], [], color="#D62728", lw=1.2, label=modlabel)], fontsize=8)
        fig.tight_layout(); fig.savefig(OUTDIR / f"{name}_{MODE}.{FMT}", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved per-subject figures to", OUTDIR)


if __name__ == "__main__":
    main()
