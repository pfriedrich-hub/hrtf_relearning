"""
edge_only_cascade_qc.py  (dev/analysis)

Visualise the edge-only "cascade" condition (edge_shift.edge_only_ir): a coarse
low-n_keep cepstral baseline (externalisation landscape) with ONLY the rising
edges of the higher-coefficient fine detail added back on top -- falling edges
(the descent into each notch) flattened -- so each DTF becomes an upward
staircase / cascade.

Two figures (analysis_results/edge_cascade/):
  <subject>_mechanism.png : single elevations, showing raw / de-rippled original,
      the low-n_keep baseline, and the cascade result overlaid -- makes the
      "keep rising, flatten falling" construction explicit.
  <subject>_waterfall.png : median-plane DTFs stacked by elevation, original
      (grey) vs cascade (red), for n_keep_baseline = 4 and = 8 side by side.

Run: python3 hrtf_relearning/hrtf/processing/dev/edge_only_cascade_qc.py [SUBJECT]
"""
import sys
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUTDIR = ROOT / "analysis_results/edge_cascade"
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "JS"
BAND = (4000., 15000.)        # relevant-cue selection band (select_features)
TILT = 4.0                    # log-frequency attenuation of the overlay (dB/oct)
F_LO, F_HI = 3000., 16000.    # display band
EAR = 0                       # left
N_EL = 11
OFFSET = 30.0
NK = (4, 8)


def sofa_path(name):
    for p in (SOFA / name / f"{name}.sofa",
              SOFA / "pilot" / name / f"{name}.sofa",
              SOFA / "pilot" / f"{name}.sofa"):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load(p):
    with h5py.File(p, "r") as f:
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
    nfft = int(2 ** np.ceil(np.log2(len(x))) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    return fr, 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x, nfft)), 1e-9))


def curves(x, fs, nk):
    """raw dB, low-n_keep baseline dB, de-rippled dB, cascade result dB (all on
    the raw_db frequency grid)."""
    nfft = int(2 ** np.ceil(np.log2(len(x))) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    mag = np.abs(np.fft.rfft(x, nfft))
    raw = 20.0 * np.log10(np.maximum(mag, 1e-9))
    base = es._cepstral_smooth_db(mag, nk)
    deripple = es._gaussian_smooth_db(mag, fr, es.SIGMA_HZ_DEFAULT)
    casc_ir = es.edge_only_ir(x, fs, n_keep_baseline=nk, tilt_db_per_oct=TILT,
                              feature_kw=dict(band=BAND))
    _, casc = raw_db(casc_ir, fs)
    return fr, raw, base, deripple, casc


# ---------------- mechanism figure ----------------
def mechanism_fig(path, name):
    ir, sp, fs = load(path)
    med, el = median_dirs(sp)
    els = [med[np.argmin(np.abs(el[med] - t))] for t in (-30, 0, 30)]
    fig, axes = plt.subplots(len(els), len(NK), figsize=(11, 8), sharex=True)
    for r, d in enumerate(els):
        for c, nk in enumerate(NK):
            ax = axes[r, c]
            fr, raw, base, drp, casc = curves(ir[d, EAR, :], fs, nk)
            b = (fr >= F_LO) & (fr <= F_HI)
            ax.plot(fr[b] / 1000, raw[b], color="0.8", lw=0.9, label="raw original")
            ax.plot(fr[b] / 1000, drp[b], color="0.45", lw=1.2, label="de-rippled original")
            ax.plot(fr[b] / 1000, base[b], color="#1f77b4", lw=1.8, ls="--", label=f"baseline (n_keep={nk})")
            ax.plot(fr[b] / 1000, casc[b], color="#D62728", lw=2.0, label="edge-only (relevant cue)")
            ax.axvspan(BAND[0] / 1000, BAND[1] / 1000, color="0.95", zorder=0)
            ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 16])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.set_xlim(F_LO / 1000, F_HI / 1000)
            if c == 0:
                ax.set_ylabel(f"el {el[d]:+.0f}°\nmagnitude (dB)")
            if r == 0:
                ax.set_title(f"n_keep = {nk}")
            if r == len(els) - 1:
                ax.set_xlabel("frequency (kHz)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{name}: edge-only construction (median plane, left ear)\n"
                 "low-n_keep baseline + rising edge of the relevant cue only (falling flank flattened)")
    fig.tight_layout()
    out = OUTDIR / f"{name}_mechanism.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------- waterfall figure ----------------
def waterfall_fig(path, name):
    ir, sp, fs = load(path)
    med, el = median_dirs(sp)
    fig, axes = plt.subplots(1, len(NK), figsize=(11, 8.5), sharey=True)
    for c, nk in enumerate(NK):
        ax = axes[c]
        off = 0.0
        for d in med:
            x0 = ir[d, EAR, :]
            x1 = es.edge_only_ir(x0, fs, n_keep_baseline=nk, tilt_db_per_oct=TILT,
                                 feature_kw=dict(band=BAND))
            fr, L0 = raw_db(x0, fs); _, L1 = raw_db(x1, fs)
            b = (fr >= F_LO) & (fr <= F_HI)
            ax.plot(fr[b] / 1000, L0[b] + off, color="0.7", lw=1.1)
            ax.plot(fr[b] / 1000, L1[b] + off, color="#D62728", lw=1.3)
            ax.text(F_HI / 1000 * 1.02, L1[b][-1] + off, f"{el[d]:+.0f}", fontsize=7, va="center")
            off += OFFSET
        ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 16])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlim(F_LO / 1000, F_HI / 1000)
        ax.set_yticks([])
        ax.set_xlabel("frequency (kHz)")
        ax.set_title(f"n_keep = {nk}")
    axes[0].legend(handles=[Line2D([], [], color="0.7", lw=1.2, label="original"),
                            Line2D([], [], color="#D62728", lw=1.3,
                                   label=f"edge-only, relevant cue (tilt {TILT:.0f} dB/oct)")],
                   loc="upper left", fontsize=9)
    fig.suptitle(f"{name}: median-plane DTFs, original vs edge-only "
                 f"(relevant cues {BAND[0]/1000:.0f}-{BAND[1]/1000:.0f} kHz, tilt {TILT:.0f} dB/oct)", y=0.995)
    fig.tight_layout()
    out = OUTDIR / f"{name}_waterfall.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = sofa_path(SUBJECT)
    m = mechanism_fig(p, SUBJECT)
    w = waterfall_fig(p, SUBJECT)
    print("saved", m)
    print("saved", w)


if __name__ == "__main__":
    main()
