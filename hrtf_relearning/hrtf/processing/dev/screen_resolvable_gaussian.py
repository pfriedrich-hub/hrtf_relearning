"""
screen_resolvable_gaussian.py  (dev/analysis)

Screen subjects with the single-notch-first detector: a DTF is USABLE for the
edge-shift manipulation as soon as it has one perceptually valid notch
(select_features gate = in-band 5-11 kHz AND depth >=4 dB (Moore) AND half-depth
width >=0.17 oct (Moore)). Two notches are counted as separate features only
when >=0.5 oct apart (Macpherson). Reports, per subject: % of median-plane
elevations that are usable, and the mean number of features.

Usage: python3 screen_resolvable_gaussian.py [pilot|aachen]
"""
import os
import sys
from pathlib import Path
import numpy as np, h5py
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
AACHEN = Path(os.environ.get("AACHEN_DIR", "/sessions/friendly-exciting-planck/mnt/aachen_database"))
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

EL = (-35.0, 35.0)
DUM = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test", "MSc"}


def median_dirs(sp, aachen):
    if aachen:
        x, y, z = sp[:, 0], sp[:, 1], sp[:, 2]; r = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x)); el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1, 1)))
        med = np.where((np.abs(az) <= 3) & (x > 0))[0]
    else:
        azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
        med = np.where(np.abs(azn) <= 2)[0]
    med = med[(el[med] >= EL[0]) & (el[med] <= EL[1])]
    med = med[np.argsort(el[med])]
    # subsample to <=17 elevations for speed
    if len(med) > 17:
        med = med[np.linspace(0, len(med) - 1, 17).astype(int)]
    return med


def screen(sofa, aachen):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    med = median_dirs(sp, aachen)
    if len(med) < 5:
        return None
    best = 0.0
    best_nfeat = 0.0
    for ear in (0, 1):
        res, nfeat = [], []
        for d in med:
            try:
                g = es.parametric_summary(ir[d, ear, :], fs)["gates"]
                res.append(bool(g["usable"]))
                nfeat.append(int(g["n_features"]))
            except Exception:
                res.append(False)
                nfeat.append(0)
        frac = float(np.mean(res))
        if frac >= best:
            best, best_nfeat = frac, float(np.mean(nfeat))
    return best, best_nfeat


group = sys.argv[1] if len(sys.argv) > 1 else "pilot"
if group == "pilot":
    subs = [("JS", SOFA/"JS/JS.sofa", False), ("CA", SOFA/"CA/CA.sofa", False)]
    for p in sorted((SOFA/"pilot").rglob("*.sofa")):
        if any(x in p.stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                     "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")): continue
        subs.append((p.stem, p, False))
else:
    subs = [(p.stem, p, True) for p in sorted(AACHEN.glob("*.sofa"))]

rows = []
for name, path, aach in subs:
    try:
        r = screen(path, aach)
    except Exception as e:
        print(f"{name:14s} ERROR {type(e).__name__}"); continue
    if r is None:
        continue
    f, nfeat = r
    dummy = name in DUM
    v = "PASS" if f >= 0.5 else ("marginal" if f >= 0.2 else "fail")
    print(f"{name:14s} usable {100*f:5.0f}%  mean_feat {nfeat:.2f}  {v}{' [dummy]' if dummy else ''}")
    if not dummy:
        rows.append((name, f))

rows.sort(key=lambda r: r[1], reverse=True)
passers = [r[0] for r in rows if r[1] >= 0.5]
marg = [r[0] for r in rows if 0.2 <= r[1] < 0.5]
print(f"\n=== {group}: {len(rows)} real subjects ===")
print(f"usable (>=50% of el): {len(passers)} -> {passers}")
print(f"marginal (20-50%): {len(marg)} -> {marg}")
