"""
pilot_width_sensitivity.py  (dev/analysis)

How many pilot subjects remain viable for the rising-edge arm as the N1
width (detectability) gate is tightened, on top of the separation gate
(sep > 0.5 oct). The answer is highly threshold-sensitive and the crossover
coincides with Moore et al. (1989)'s notch-detectability limit (~0.25 fc ~
1.5-2 ERB), so real HRTF lower notches sit right at the edge of detectability.

Run: python3 hrtf_relearning/hrtf/processing/dev/pilot_width_sensitivity.py
"""
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import dct, idct

SOFA = Path(__file__).resolve().parents[4] / "hrtf_relearning/data/hrtf/sofa"
OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs/pilot_width_sensitivity.png")
NK, WIN, PROM, SEP = 40, (5500., 13500.), 3.0, 0.5
DUM = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test", "MSc"}


def smooth(mag, nk):
    lm = np.log(np.maximum(mag, np.finfo(float).tiny)); c = dct(lm, type=2, norm="ortho")
    c[nk:] = 0.0; return 20*np.log10(np.maximum(np.exp(idct(c, type=2, norm="ortho")), 1e-9))


def per_el(x, fs):
    n = len(x); nfft = int(2**np.ceil(np.log2(n))*4)
    fr = np.fft.rfftfreq(nfft, 1/fs); mag = np.abs(np.fft.rfft(x, nfft))
    L = smooth(mag, NK); b = (fr >= WIN[0]) & (fr <= WIN[1]); fb, Lb = fr[b], L[b]
    idx, pp = find_peaks(-Lb, prominence=PROM)
    if len(idx) < 2: return None
    o = np.argsort(fb[idx]); ii = idx[o]; pr = pp["prominences"][o]; i = ii[0]
    sep = np.log2(fb[ii[-1]] / fb[ii[0]])
    thr = Lb[i] + 0.5*pr[0]; lo = i
    while lo > 0 and Lb[lo] < thr: lo -= 1
    hi = i
    while hi < len(Lb)-1 and Lb[hi] < thr: hi += 1
    hw = np.log2(fb[hi] / fb[lo]) if hi > lo else 0.0
    return sep, hw


def load(s):
    with h5py.File(s, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0]+180) % 360-180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]; med = med[(el[med] >= -35) & (el[med] <= 35)]
    return ir, fs, med


subs = [("JS", SOFA/"JS/JS.sofa"), ("CA", SOFA/"CA/CA.sofa")]
for p in sorted((SOFA/"pilot").glob("*.sofa")):
    if any(x in p.stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                 "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")): continue
    subs.append((p.stem, p))

bars = np.arange(0.0, 0.401, 0.02)
counts = np.zeros(len(bars))
for name, p in subs:
    if name in DUM: continue
    try: ir, fs, med = load(p)
    except Exception: continue
    if len(med) < 5: continue
    best = np.zeros(len(bars))
    for ear in (0, 1):
        rr = [per_el(ir[d, ear, :], fs) for d in med]
        for k, w in enumerate(bars):
            frac = np.mean([(r is not None and r[0] >= SEP and r[1] >= w) for r in rr])
            best[k] = max(best[k], frac)
    counts += (best >= 0.5).astype(float)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axvspan(0.25, 0.36, color="#D85A30", alpha=0.12, label="Moore detectability limit (~0.25fc, ~1.5–2 ERB)")
ax.axvspan(0.13, 0.20, color="#1D9E75", alpha=0.10, label="~1 ERB @7 kHz")
ax.plot(bars, counts, "-o", color="#185FA5", lw=2)
ax.set_xlabel("N1 half-depth width gate (octaves)")
ax.set_ylabel("viable subjects (of 25)\nsep>0.5 oct AND width>gate, ≥50% of el, best ear")
ax.set_title("Rising-edge viability collapses as the notch-width gate approaches\n"
             "Moore's detectability limit — the lower notches are marginally detectable", fontsize=11)
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
ax.set_ylim(0, 12)
fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
for w, c in zip(bars, counts):
    if abs(w*100 % 5) < 1e-6:
        print(f"  width>={w:.2f} oct: {int(c)} subjects")
