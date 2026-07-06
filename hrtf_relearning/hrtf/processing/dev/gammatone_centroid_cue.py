"""
gammatone_centroid_cue.py  (dev/analysis)

Payoff test for gammatone-domain notch detection: summarise the broad
elevation notch by its DEPTH-WEIGHTED CENTROID (not the minimum, which locks
onto the stable deep part; not a hard edge, which jumps on the gentle low
shoulder). Centroid is sub-band precise, matches verify_binned's read-out, and
tracks elevation cleanly in a typical ear.

Finding: CA centroid climbs ~7.5->10 kHz with elevation (Spearman ~0.95);
JS's own notch is atypically stationary (~0.6 kHz, non-monotonic), so JS is a
poor case for judging the detector.

Run: python3 hrtf_relearning/hrtf/processing/dev/gammatone_centroid_cue.py
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA_DIR = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUT_DIR = Path("/sessions/friendly-exciting-planck/mnt/outputs")

spec = iu.spec_from_file_location("es", ES)
es = iu.module_from_spec(spec); spec.loader.exec_module(es)

BAND = (6000.0, 12500.0)


def load(sofa):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0] + 180) % 360 - 180
    el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]
    med = med[np.argsort(el[med])]
    return ir, fs, el, med


def gam(ir, fs, flow=2500, fhigh=18000):
    n = len(ir); nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(ir, nfft))
    return es.erb_filterbank_rms(fr, mag, flow=flow, fhigh=fhigh, spacing=1.0)


def centroid(ir, fs):
    fc, ec, L = gam(ir, fs)
    w = (fc >= BAND[0]) & (fc <= BAND[1]); idx = np.where(w)[0]
    a, b = idx[0], idx[-1]
    ref = max(L[a], L[b]); d = np.maximum(ref - L[a:b + 1], 0.0)
    if d.sum() <= 0: return np.nan
    return float(es.erb_to_hz(np.sum(ec[a:b + 1] * d) / d.sum()))


def raw_logmag(ir, fs):
    n = len(ir); nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(ir, nfft))
    b = (fr >= 3000) & (fr <= 17500)
    return fr[b], 20 * np.log10(np.maximum(mag[b], 1e-9))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.4))

# ---- Panel A: centroid vs elevation, CA vs JS, both ears ----
styles = {("CA", 0): ("#185FA5", "-o", "CA left"),
          ("CA", 1): ("#0F6E56", "-s", "CA right"),
          ("JS", 0): ("#D85A30", "--o", "JS left"),
          ("JS", 1): ("#993C1D", "--s", "JS right")}
for subj in ("CA", "JS"):
    ir, fs, el, med = load(SOFA_DIR / subj / f"{subj}.sofa")
    for ear in (0, 1):
        c = np.array([centroid(ir[d, ear, :], fs) for d in med])
        e = np.array([el[d] for d in med]); m = ~np.isnan(c)
        rho = spearmanr(e[m], c[m]).correlation
        col, mk, lab = styles[(subj, ear)]
        axA.plot(e[m], c[m] / 1000, mk, color=col, lw=1.8, ms=5, label=f"{lab} (ρ={rho:+.2f})")
axA.set_xlabel("elevation (deg)"); axA.set_ylabel("notch centroid (kHz)")
axA.set_title("gammatone notch centroid tracks elevation\n(CA typical; JS atypically stationary)")
axA.grid(True, alpha=0.3); axA.legend(fontsize=9)

# ---- Panel B: CA left spectra marching up with elevation ----
ir, fs, el, med = load(SOFA_DIR / "CA" / "CA.sofa")
show = [-33, -16, 0, 16, 33]
dirs = [med[np.argmin(np.abs(el[med] - e))] for e in show]
cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(dirs)))
for d, col in zip(dirs, cmap):
    fc, ec, L = gam(ir[d, 0, :], fs)
    w = (fc >= 3000) & (fc <= 17500)
    axB.plot(fc[w] / 1000, L[w], color=col, lw=2, label=f"el {el[d]:.0f}°")
    cz = centroid(ir[d, 0, :], fs)
    axB.axvline(cz / 1000, color=col, lw=1.2, ls=":")
axB.set_xscale("log")
axB.set_xticks([3, 5, 8, 12, 17]); axB.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axB.set_xlim(3, 17.5)
axB.set_xlabel("frequency (kHz)"); axB.set_ylabel("magnitude (dB)")
axB.set_title("CA left: broad notch + centroid (dotted) climb with elevation")
axB.grid(True, which="both", alpha=0.25); axB.legend(fontsize=8)

fig.suptitle("Detector answer: coarse gammatone identity + depth-weighted centroid recovers the elevation cue",
             fontsize=13)
fig.tight_layout()
p = OUT_DIR / "gammatone_centroid_cue.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("saved", p)
PY_DONE = True
