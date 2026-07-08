"""
pilot_gates_screen.py  (dev/analysis)

Combined resolvability screen for the rising-edge experiment, applying BOTH
literature gates to the mobile lower notch N1 per subject/ear/elevation:

  gate 1 (separation): N1-N2 separation > 0.5 octave   (Macpherson 2 ripples/oct)
  gate 2 (width):      N1 half-depth width > 0.30 octave (Moore et al. 1989 -
                       narrow notches, bw < ~0.25 fc ~ 0.36 oct at 8 kHz, are
                       undetectable even if infinitely deep)

Measurement resolution n_keep=40 (finer than the 0.5-oct bar, coarser than raw
ripple; DCT-II lifter, matches edge_shift._cepstral_smooth_db). Reports the
fraction of tested elevations (-35..35) clearing separation-only vs BOTH gates,
and who survives both.

Run: python3 hrtf_relearning/hrtf/processing/dev/pilot_gates_screen.py
"""
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import dct, idct

ROOT = Path(__file__).resolve().parents[4]
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs")
NK, WIN, PROM = 40, (5500., 13500.), 3.0
SEP_BAR, WID_BAR, EL = 0.5, 0.30, (-35., 35.)
DUM = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test", "MSc"}


def smooth(mag, nk):
    lm = np.log(np.maximum(mag, np.finfo(float).tiny)); c = dct(lm, type=2, norm="ortho")
    c[nk:] = 0.0; return 20*np.log10(np.maximum(np.exp(idct(c, type=2, norm="ortho")), 1e-9))


def halfdepth_width_oct(fb, L, i, prom):
    """half-depth width (octaves) of the notch at index i with depth `prom`."""
    thr = L[i] + 0.5 * prom
    lo = i
    while lo > 0 and L[lo] < thr:
        lo -= 1
    hi = i
    while hi < len(L) - 1 and L[hi] < thr:
        hi += 1
    def xcross(a, b):
        if L[b] == L[a]:
            return fb[a]
        t = np.clip((thr - L[a]) / (L[b] - L[a]), 0, 1)
        return fb[a] + t * (fb[b] - fb[a])
    f_lo = xcross(lo, min(lo + 1, i)) if lo < i else fb[i]
    f_hi = xcross(hi, max(hi - 1, i)) if hi > i else fb[i]
    if f_hi <= f_lo:
        return 0.0
    return float(np.log2(f_hi / f_lo))


def screen(sofa):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]; med = med[(el[med] >= EL[0]) & (el[med] <= EL[1])]
    if len(med) < 5:
        return None
    out = []
    for ear in (0, 1):
        sep_ok, both_ok = [], []
        for d in med:
            x = ir[d, ear, :]; n = len(x); nfft = int(2**np.ceil(np.log2(n))*4)
            fr = np.fft.rfftfreq(nfft, 1/fs); mag = np.abs(np.fft.rfft(x, nfft))
            L = smooth(mag, NK); b = (fr >= WIN[0]) & (fr <= WIN[1]); fb, Lb = fr[b], L[b]
            idx, props = find_peaks(-Lb, prominence=PROM)
            if len(idx) >= 2:
                order = np.argsort(fb[idx]); ii = idx[order]; pr = props["prominences"][order]
                sep = np.log2(fb[ii[-1]] / fb[ii[0]])
                w1 = halfdepth_width_oct(fb, Lb, ii[0], pr[0])   # width of N1 (lowest)
                sep_ok.append(sep >= SEP_BAR)
                both_ok.append(sep >= SEP_BAR and w1 >= WID_BAR)
            else:
                sep_ok.append(False); both_ok.append(False)
        out.append((np.mean(sep_ok), np.mean(both_ok)))
    return out


subs = [("JS", SOFA/"JS/JS.sofa"), ("CA", SOFA/"CA/CA.sofa")]
for p in sorted((SOFA/"pilot").rglob("*.sofa")):
    if any(x in p.stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                 "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")): continue
    subs.append((p.stem, p))

rows = []
print(f"{'subject':10s} {'sep-only(best)':>14s} {'BOTH(best)':>11s}  verdict")
print("-" * 52)
for name, p in subs:
    try:
        r = screen(p)
    except Exception as e:
        print(f"{name:10s}  ERROR {type(e).__name__}"); continue
    if r is None:
        continue
    sep_best = max(r[0][0], r[1][0]); both_best = max(r[0][1], r[1][1])
    dummy = name in DUM
    v = "PASS" if both_best >= 0.5 else ("marginal" if both_best >= 0.2 else "fail")
    print(f"{name:10s} {100*sep_best:13.0f}% {100*both_best:10.0f}%  {v}{' [dummy]' if dummy else ''}")
    if not dummy:
        rows.append((name, sep_best, both_best))

rows.sort(key=lambda r: r[2], reverse=True)
passers = [r[0] for r in rows if r[2] >= 0.5]
print(f"\n=== real subjects: {len(rows)} ===")
print(f"clear BOTH gates (>=50% of tested el): {len(passers)} -> {passers}")
print(f"(vs separation-only >=50%: {sum(1 for r in rows if r[1] >= 0.5)})")

# figure: separation-only vs both, sorted
fig, ax = plt.subplots(figsize=(9, 8))
names = [r[0] for r in rows]; y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color="#B5D4F4", height=0.7, label="separation gate only")
ax.barh(y, [r[2] for r in rows], color="#185FA5", height=0.42, label="separation + width (Moore)")
ax.axvline(0.5, color="#D85A30", ls="--", lw=1.5, label="viability bar (50% of elevations)")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("fraction of tested elevations (−35…35°) clearing the gate")
ax.set_xlim(0, 1); ax.legend(fontsize=9, loc="lower right")
ax.set_title(f"Pilot screen with both gates — {len(passers)}/{len(rows)} subjects clear separation + width\n"
             f"(sep>0.5 oct, N1 half-depth width>0.30 oct)", fontsize=11)
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT/"pilot_gates_screen.png", dpi=140, bbox_inches="tight")
print("saved", OUT/"pilot_gates_screen.png")
