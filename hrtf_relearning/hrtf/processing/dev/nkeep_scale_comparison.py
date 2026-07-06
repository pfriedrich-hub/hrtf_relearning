"""
nkeep_scale_comparison.py  (dev/analysis, not part of the pipeline)

Multi-scale comparison of the cepstral notch-identity step on a real DTF (JS),
to pin `n_keep` in edge_shift.py against the perceptual spectral-resolution
literature (see experiment/protocols/documentation/spectral_resolution_literature.md).

For a median-plane direction it overlays the raw log-magnitude, several cepstral
smoothings (n_keep), and a ~1-ERB gammatone bank (~Baumgartner front end), and
marks the notch minima detect_notches would find (find_peaks, prominence 3 dB) at
each scale. It shows where JS's fine multi-notch cluster (~8-11 kHz) collapses
into a single perceptual notch, and where over-smoothing starts to erase the
genuinely separate ~16 kHz notch.

Run: python3 hrtf_relearning/hrtf/processing/dev/nkeep_scale_comparison.py
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa/JS/JS.sofa"
OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs/JS_nkeep_scale_comparison.png")

spec = iu.spec_from_file_location("es", ES)
es = iu.module_from_spec(spec); spec.loader.exec_module(es)

F_LO, F_HI = 3000.0, 17500.0
PROM = 3.0
CLUSTER = (8000.0, 11500.0)   # JS fine multi-notch region

# ---- load JS ----
with h5py.File(SOFA, "r") as f:
    ir_all = f["Data.IR"][:]
    sp = f["SourcePosition"][:]
    fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
azn = (sp[:, 0] + 180) % 360 - 180
el = sp[:, 1]
med = np.where(np.abs(azn) <= 2)[0]
med = med[np.argsort(el[med])]
EAR = 0  # left


def spectrum(ir):
    n = len(ir)
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)          # same grid as edge_shift_ir
    mag = np.abs(np.fft.rfft(ir, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, mag


def smoothed(ir, n_keep):
    freqs, mag = spectrum(ir)
    Lc = es._cepstral_smooth_db(mag, n_keep)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    return freqs[band], Lc[band]


def raw_logmag(ir):
    freqs, mag = spectrum(ir)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    return freqs[band], 20 * np.log10(np.maximum(mag[band], 1e-9))


def gammatone(ir):
    freqs, mag = spectrum(ir)
    fc, ec, Lb = es.erb_filterbank_rms(freqs, mag, flow=700, fhigh=18000, spacing=1.0)
    win = (fc >= F_LO) & (fc <= F_HI)
    return fc[win], Lb[win]


def notches(fb, L):
    idx, _ = find_peaks(-L, prominence=PROM)
    return fb[idx], L[idx]


# ---- target direction ----
d8 = med[np.argmin(np.abs(el[med] - 8.3))]
ir8 = ir_all[d8, EAR, :]

NK_LIST = [60, 30, 16, 12, 8]
colors = {60: "#d62728", 30: "#ff7f0e", 16: "#1f77b4", 12: "#2ca02c", 8: "#9467bd"}

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1, 1], hspace=0.32, wspace=0.22)

# ===== Panel A: overlay at el=8.3 =====
axA = fig.add_subplot(gs[:, 0])
axA.axvspan(CLUSTER[0] / 1000, CLUSTER[1] / 1000, color="0.9", zorder=0,
            label="fine multi-notch cluster")
fr, Lr = raw_logmag(ir8)
axA.plot(fr / 1000, Lr, color="0.6", lw=1.0, alpha=0.8, label="raw log-mag")
for nk in NK_LIST:
    fb, L = smoothed(ir8, nk)
    axA.plot(fb / 1000, L, color=colors[nk], lw=2.0, label=f"n_keep={nk}")
    nf, nl = notches(fb, L)
    axA.plot(nf / 1000, nl, "v", color=colors[nk], ms=9, mec="k", mew=0.5, zorder=5)
fg, Lg = gammatone(ir8)
axA.plot(fg / 1000, Lg, color="k", lw=1.6, ls="--", label="gammatone ~1 ERB")
ng, nlg = notches(fg, Lg)
axA.plot(ng / 1000, nlg, "s", color="k", ms=8, zorder=5)
axA.set_xscale("log")
axA.set_xticks([3, 4, 5, 6, 8, 10, 12, 16])
axA.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axA.set_xlim(F_LO / 1000, F_HI / 1000)
axA.set_xlabel("frequency (kHz, log)")
axA.set_ylabel("magnitude (dB)")
axA.set_title(f"JS left DTF, el = {el[d8]:.1f}°  —  notch identity vs smoothing scale\n"
              "(▼ = detected notch minimum, prominence ≥ 3 dB)")
axA.legend(loc="lower left", fontsize=8, ncol=2)
axA.grid(True, which="both", alpha=0.25)

# ===== Panel B (top): detected notch CF vs n_keep (this direction) =====
axB = fig.add_subplot(gs[0, 1])
xs = [150, 60, 30, 16, 12, 8]
for i, nk in enumerate(xs):
    fb, L = smoothed(ir8, nk)
    nf, _ = notches(fb, L)
    axB.plot([nk] * len(nf), nf / 1000, "o", color="#1f77b4", ms=7)
fg, Lg = gammatone(ir8)
ng, _ = notches(fg, Lg)
axB.plot([4] * len(ng), ng / 1000, "s", color="k", ms=7, label="gammatone")
axB.axhspan(CLUSTER[0] / 1000, CLUSTER[1] / 1000, color="0.9", zorder=0)
axB.axvline(16, color="#1f77b4", ls=":", alpha=0.7)
axB.set_xscale("log")
axB.set_xticks([4, 8, 12, 16, 30, 60, 150])
axB.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axB.set_xlabel("n_keep  (4 = gammatone)")
axB.set_ylabel("detected notch CF (kHz)")
axB.set_title(f"el = {el[d8]:.1f}°: cluster collapses at n_keep ≈ 16")
axB.set_ylim(F_LO / 1000, F_HI / 1000)
axB.grid(True, alpha=0.25)

# ===== Panel B (bottom): notch count vs n_keep across all median-plane dirs =====
axC = fig.add_subplot(gs[1, 1])
nk_sweep = [150, 100, 60, 40, 30, 20, 16, 12, 10, 8, 6]
counts = np.zeros((len(med), len(nk_sweep)))
for j, d in enumerate(med):
    ir = ir_all[d, EAR, :]
    for k, nk in enumerate(nk_sweep):
        fb, L = smoothed(ir, nk)
        nf, _ = notches(fb, L)
        counts[j, k] = len(nf)
mean_c = counts.mean(0); lo_c = counts.min(0); hi_c = counts.max(0)
axC.fill_between(nk_sweep, lo_c, hi_c, color="#1f77b4", alpha=0.15, label="min–max across 19 dirs")
axC.plot(nk_sweep, mean_c, "-o", color="#1f77b4", label="mean")
axC.axvspan(12, 16, color="#2ca02c", alpha=0.12, label="recommended 12–16")
axC.axvline(60, color="#d62728", ls=":", label="current (60)")
axC.set_xscale("log")
axC.set_xticks([6, 8, 12, 16, 30, 60, 150])
axC.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axC.set_xlabel("n_keep")
axC.set_ylabel("# notches in 3–17.5 kHz")
axC.set_title("notch count vs smoothing (JS median plane, left)")
axC.legend(fontsize=8, loc="upper left")
axC.grid(True, alpha=0.25)

fig.suptitle("edge_shift.py notch identity: n_keep=60 over-segments a real DTF; "
             "~16 matches the perceptual scale", fontsize=13, y=0.98)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)

# ---- numbers to console ----
print(f"\nel={el[d8]:.1f} left ear, detected notch CFs (Hz):")
for nk in [150, 60, 30, 16, 12, 8]:
    fb, L = smoothed(ir8, nk); nf, _ = notches(fb, L)
    print(f"  n_keep={nk:3d}: {np.round(nf).astype(int)}")
fg, Lg = gammatone(ir8); ng, _ = notches(fg, Lg)
print(f"  gammatone : {np.round(ng).astype(int)}")
print(f"\nmean notch count across 19 median-plane dirs:")
for k, nk in enumerate(nk_sweep):
    print(f"  n_keep={nk:3d}: {mean_c[k]:.2f}  (range {int(lo_c[k])}-{int(hi_c[k])})")
