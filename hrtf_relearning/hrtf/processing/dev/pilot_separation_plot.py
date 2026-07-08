"""Scatter of per-ear N1-N2 resolvability across the pilot set (see
pilot_separation_screen.py). Top-right quadrant = both ears resolvable."""
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import dct, idct

ROOT = Path(__file__).resolve().parents[4]
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs/pilot_separation_scatter.png")
NK, WIN, PROM, BAR, EL = 40, (5500., 13500.), 3.0, 0.5, (-35., 35.)
DUM = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test", "MSc"}


def smooth(mag, nk):
    lm = np.log(np.maximum(mag, np.finfo(float).tiny)); c = dct(lm, type=2, norm="ortho")
    c[nk:] = 0.0; return 20*np.log10(np.maximum(np.exp(idct(c, type=2, norm="ortho")), 1e-9))


def fracs(sofa):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0]+180) % 360-180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]; med = med[(el[med] >= EL[0]) & (el[med] <= EL[1])]
    if len(med) < 5: return None
    out = []
    for ear in (0, 1):
        s = []
        for d in med:
            x = ir[d, ear, :]; n = len(x); nfft = int(2**np.ceil(np.log2(n))*4)
            fr = np.fft.rfftfreq(nfft, 1/fs); mag = np.abs(np.fft.rfft(x, nfft))
            L = smooth(mag, NK); b = (fr >= WIN[0]) & (fr <= WIN[1]); fb, Lb = fr[b], L[b]
            idx, _ = find_peaks(-Lb, prominence=PROM)
            s.append(np.log2(np.sort(fb[idx])[-1]/np.sort(fb[idx])[0]) if len(idx) >= 2 else 0.0)
        out.append((np.array(s) >= BAR).mean())
    return out


subs = [("JS", SOFA/"JS/JS.sofa"), ("CA", SOFA/"CA/CA.sofa")]
for p in sorted((SOFA/"pilot").rglob("*.sofa")):
    if any(x in p.stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                 "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")): continue
    subs.append((p.stem, p))

fig, ax = plt.subplots(figsize=(8, 8))
ax.axhspan(0.5, 1.02, xmin=0.5/1.02, color="#1D9E75", alpha=0.08)
ax.axvline(0.5, color="0.6", ls="--", lw=1); ax.axhline(0.5, color="0.6", ls="--", lw=1)
nreal = 0; both = []
for name, p in subs:
    try: fr = fracs(p)
    except Exception: continue
    if fr is None: continue
    dummy = name in DUM
    col = "0.7" if dummy else ("#185FA5" if min(fr) >= 0.5 else ("#EF9F27" if max(fr) >= 0.5 else "#D85A30"))
    ax.scatter(fr[0], fr[1], c=col, s=70 if not dummy else 45, edgecolor="k", linewidth=0.4, zorder=3,
               marker="o" if not dummy else "x")
    ax.annotate(name, (fr[0], fr[1]), fontsize=7.5, xytext=(4, 3), textcoords="offset points",
                color="0.5" if dummy else "black")
    if not dummy:
        nreal += 1
        if min(fr) >= 0.5: both.append(name)
ax.plot([0, 1], [0, 1], color="0.85", lw=1, zorder=0)
ax.set_xlabel("left ear: fraction of elevations with N1–N2 sep ≥ 0.5 oct")
ax.set_ylabel("right ear: fraction ≥ 0.5 oct")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title(f"Pilot-set N1–N2 resolvability (tested el {EL[0]:.0f}..{EL[1]:.0f}°)\n"
             f"both ears resolvable (top-right): {len(both)}/{nreal} — {', '.join(both)}", fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT); print("both-ears resolvable:", both)
