"""
two_notch_resolvability.py  (dev/analysis)

Test whether JS's mobile lower notch (N1, ~7 kHz at low el) is resolvable as a
feature distinct from the ~stationary upper notch (N2, ~10-11 kHz) -- the
precondition for the rising-edge manipulation to mean anything (if N1 and N2
merge below perceptual resolution, only the notch SHAPE matters and moving the
edge == moving the center; the edge is special only under an edge-detector
model).

Tracks the two lowest prominent notches per elevation on a moderately fine
cepstral curve (resolves ~0.3-0.5 octave, coarse enough to reject finer
phantoms), computes N1-N2 separation in octaves, and compares against the
model-independent resolution limits:
  - Macpherson & Middlebrooks 2003: usable spectral detail 0.5-2 ripples/oct
    -> a 2 ripple/oct feature = 0.5 octave period = the FINE edge of usable.
  - Iida 2007: N1 and N2 both used when parametrically separable.
  - Goupell 2010 / Baumgartner: ~9-12 log channels over 0.3-16 kHz (~0.4-0.5 oct).

Run: python3 hrtf_relearning/hrtf/processing/dev/two_notch_resolvability.py
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import spearmanr
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA_DIR = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUT_DIR = Path("/sessions/friendly-exciting-planck/mnt/outputs")

spec = iu.spec_from_file_location("es", ES)
es = iu.module_from_spec(spec); spec.loader.exec_module(es)

NK = 40                      # resolves ~0.3 oct, rejects finer phantoms
WIN = (6000.0, 13000.0)
PROM = 3.0


def load(subj):
    with h5py.File(SOFA_DIR / subj / f"{subj}.sofa", "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]; med = med[np.argsort(el[med])]
    return ir, fs, el, med


def notch_cfs(ir, fs):
    n = len(ir); nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(ir, nfft))
    Lc = es._cepstral_smooth_db(mag, NK)
    b = (fr >= WIN[0]) & (fr <= WIN[1]); fb, L = fr[b], Lc[b]
    idx, props = find_peaks(-L, prominence=PROM)
    order = np.argsort(fb[idx])
    return fb[idx][order], props["prominences"][order]


def two_tracks(subj, ear):
    ir, fs, el, med = load(subj)
    els, lo, hi, sep = [], [], [], []
    for d in med:
        cf, pr = notch_cfs(ir[d, ear, :], fs)
        els.append(el[d])
        if len(cf) >= 2:
            lo.append(cf[0]); hi.append(cf[-1]); sep.append(np.log2(cf[-1] / cf[0]))
        elif len(cf) == 1:
            lo.append(cf[0]); hi.append(cf[0]); sep.append(0.0)
        else:
            lo.append(np.nan); hi.append(np.nan); sep.append(np.nan)
    return np.array(els), np.array(lo), np.array(hi), np.array(sep)


fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- Panel A: JS two notch tracks (left ear) ---
els, lo, hi, sep = two_tracks("JS", 0)
axes[0].plot(els, lo / 1000, "-o", color="#D85A30", label="lower notch N1")
axes[0].plot(els, hi / 1000, "-s", color="#185FA5", label="upper notch N2")
axes[0].set_xlabel("elevation (deg)"); axes[0].set_ylabel("notch CF (kHz)")
axes[0].set_title(f"JS left: two-notch structure (n_keep={NK})")
axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
m = ~np.isnan(lo)
r_lo = spearmanr(els[m], lo[m]).correlation
r_hi = spearmanr(els[m], hi[m]).correlation
axes[0].text(0.03, 0.97, f"N1 ρ={r_lo:+.2f}\nN2 ρ={r_hi:+.2f}", transform=axes[0].transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# --- Panel B: separation vs elevation vs resolution limits ---
axes[1].axhspan(0.0, 0.5, color="#D85A30", alpha=0.10)
axes[1].axhspan(0.5, 1.5, color="#1D9E75", alpha=0.10)
axes[1].axhline(0.5, color="#1D9E75", ls="--", lw=1.5, label="0.5 oct = 2 ripples/oct (Macpherson upper limit)")
for subj, ear, col, mk in [("JS", 0, "#D85A30", "o"), ("CA", 0, "#185FA5", "s")]:
    els, lo, hi, sep = two_tracks(subj, ear)
    axes[1].plot(els, sep, mk + "-", color=col, label=f"{subj} left N1–N2 separation")
axes[1].set_xlabel("elevation (deg)"); axes[1].set_ylabel("N1–N2 separation (octaves)")
axes[1].set_title("resolvable (green) vs merged (orange)")
axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8, loc="upper right")
axes[1].set_ylim(-0.02, 1.0)

# --- Panel C: JS raw spectra, low vs high elevation, two notches marked ---
ir, fs, el, med = load("JS")
for e, col in [(-33, "#04342C"), (0, "#1D9E75"), (33, "#9FE1CB")]:
    d = med[np.argmin(np.abs(el[med] - e))]
    n = len(ir[d, 0, :]); nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(ir[d, 0, :], nfft))
    b = (fr >= 4000) & (fr <= 16000)
    axes[2].plot(fr[b] / 1000, 20 * np.log10(mag[b]), color=col, lw=1.6, label=f"el {el[d]:.0f}°")
axes[2].set_xscale("log")
axes[2].set_xticks([4, 6, 8, 10, 13, 16]); axes[2].get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axes[2].set_xlim(4, 16)
axes[2].set_xlabel("frequency (kHz)"); axes[2].set_ylabel("magnitude (dB)")
axes[2].set_title("JS left raw: N1 climbs 7→10 kHz, N2 ~stationary")
axes[2].grid(True, which="both", alpha=0.25); axes[2].legend(fontsize=9)

fig.suptitle("Can JS's mobile lower notch be resolved from the stationary upper notch? "
             "Separation vs the perceptual resolution limit", fontsize=13)
fig.tight_layout()
p = OUT_DIR / "JS_two_notch_resolvability.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("saved", p)

# ---- console ----
els, lo, hi, sep = two_tracks("JS", 0)
print("\nJS left:  el    N1(kHz)  N2(kHz)  sep(oct)")
for e, a, b, s in zip(els, lo, hi, sep):
    print(f"        {e:6.1f}   {a/1000:5.2f}   {b/1000:5.2f}   {s:5.2f}"
          if not np.isnan(a) else f"        {e:6.1f}    --      --      --")
print(f"\nfraction of elevations with sep >= 0.5 oct (resolvable): {(sep[~np.isnan(sep)] >= 0.5).mean():.2f}")
print(f"N1 range {np.nanmin(lo)/1000:.1f}-{np.nanmax(lo)/1000:.1f} kHz ; N2 range {np.nanmin(hi)/1000:.1f}-{np.nanmax(hi)/1000:.1f} kHz")
