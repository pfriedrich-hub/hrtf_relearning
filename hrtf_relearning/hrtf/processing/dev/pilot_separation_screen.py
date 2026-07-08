"""
pilot_separation_screen.py  (dev/analysis)

Screen the pilot set for N1-N2 resolvability: for each subject/ear, measure the
separation between the two prominent notches (n_keep=40 measurement resolution)
across median-plane elevations, and report how much of the range clears the
~0.5-octave (2 ripples/oct, Macpherson) bar needed for the rising-edge
manipulation to target a resolved feature.

Run: python3 hrtf_relearning/hrtf/processing/dev/pilot_separation_screen.py
"""
from pathlib import Path
import sys
import numpy as np
import h5py
from scipy.signal import find_peaks
from scipy.fft import dct, idct


def smooth_db(mag, n_keep):
    """Fast truncated-cosine (low-quefrency lifter) smoothing of log-magnitude,
    DCT-II equivalent of edge_shift._cepstral_smooth_db (avoids the per-call
    2049x2049 lstsq; adequate for a resolvability screen)."""
    log_mag = np.log(np.maximum(mag, np.finfo(float).tiny))
    c = dct(log_mag, type=2, norm="ortho")
    c[n_keep:] = 0.0
    sm = idct(c, type=2, norm="ortho")
    return 20.0 * np.log10(np.maximum(np.exp(sm), 1e-9))


ROOT = Path(__file__).resolve().parents[4]
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"

NK, WIN, PROM = 40, (5500.0, 13500.0), 3.0
BAR = 0.5   # octaves
EL_RANGE = (-35.0, 35.0)   # tested elevation range

DUMMIES = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test", "MSc"}


def paths():
    out = [("JS", SOFA / "JS/JS.sofa"), ("CA", SOFA / "CA/CA.sofa")]
    for p in sorted((SOFA / "pilot").rglob("*.sofa")):
        stem = p.stem
        if any(x in stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                   "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")):
            continue
        out.append((stem, p))
    return out


def sep_track(sofa):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]
    med = med[(el[med] >= EL_RANGE[0]) & (el[med] <= EL_RANGE[1])]
    med = med[np.argsort(el[med])]
    if len(med) < 5:
        return None
    seps = []
    for ear in (0, 1):
        s = []
        for d in med:
            x = ir[d, ear, :]
            n = len(x); nfft = int(2 ** np.ceil(np.log2(n)) * 4)
            fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(x, nfft))
            Lc = smooth_db(mag, NK)
            b = (fr >= WIN[0]) & (fr <= WIN[1]); fb, L = fr[b], Lc[b]
            idx, _ = find_peaks(-L, prominence=PROM)
            if len(idx) >= 2:
                cf = np.sort(fb[idx]); s.append(np.log2(cf[-1] / cf[0]))
            else:
                s.append(0.0)
        seps.append(np.array(s))
    return seps  # [left, right]


print(f"{'subject':10s} {'ear':>3s}  {'med.sep':>7s} {'max.sep':>7s} {'%>=0.5oct':>9s}  verdict")
print("-" * 60)
rows = []
for name, p in paths():
    try:
        res = sep_track(p)
    except Exception as e:
        print(f"{name:10s}  ERROR {type(e).__name__}")
        continue
    if res is None:
        print(f"{name:10s}  (no median-plane sampling)")
        continue
    tag = " [dummy]" if name in DUMMIES else ""
    best_frac = 0.0
    for ear, s in zip("LR", res):
        med_s = np.median(s); max_s = s.max(); frac = (s >= BAR).mean()
        best_frac = max(best_frac, frac)
        v = "resolvable" if frac >= 0.5 else ("marginal" if frac >= 0.2 else "merged")
        print(f"{name:10s} {ear:>3s}  {med_s:7.2f} {max_s:7.2f} {100*frac:8.0f}%  {v}{tag}")
    rows.append((name, best_frac, name in DUMMIES))

print("\n=== summary (best ear) ===")
real = [r for r in rows if not r[2]]
res_ = [r for r in real if r[1] >= 0.5]
marg = [r for r in real if 0.2 <= r[1] < 0.5]
mer = [r for r in real if r[1] < 0.2]
print(f"real subjects: {len(real)}")
print(f"  resolvable (>=50% of tested elevations clear 0.5 oct): {len(res_)}  -> {[r[0] for r in res_]}")
print(f"  marginal   (20-50%): {len(marg)}  -> {[r[0] for r in marg]}")
print(f"  merged     (<20%):   {len(mer)}  -> {[r[0] for r in mer]}")
