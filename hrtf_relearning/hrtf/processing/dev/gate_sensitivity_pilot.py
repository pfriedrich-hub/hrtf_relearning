"""
gate_sensitivity_pilot.py  (dev/analysis)

How many pilot subjects become "resolvable" as the literature-derived gates are
loosened toward the timbre-resolution floor. Collects the raw gate values
(separation_oct, saddle_db, n1_width_oct) per subject/ear/elevation from
edge_shift.parametric_summary (Gaussian detector), then sweeps thresholds
offline. Saddle fixed at 2.5 dB (Moore wide-saddle bound).

Run: python3 gate_sensitivity_pilot.py
"""
from pathlib import Path
import numpy as np, h5py
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

EL = (-35., 35.)
DUM = {"MSc"}
SADDLE_BAR = 2.5


def med_dirs(sp):
    azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]; med = med[(el[med] >= EL[0]) & (el[med] <= EL[1])]
    return med[np.argsort(el[med])]


def collect(sofa):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    med = med_dirs(sp)
    if len(med) < 5: return None
    per_ear = []
    for ear in (0, 1):
        rows = []
        for d in med:
            try:
                g = es.parametric_summary(ir[d, ear, :], fs)["gates"]
                rows.append((g["separation_oct"], g["saddle_db"], g["n1_width_oct"]))
            except Exception:
                rows.append((None, None, None))
        per_ear.append(rows)
    return per_ear


subs = [("JS", SOFA/"JS/JS.sofa"), ("CA", SOFA/"CA/CA.sofa")]
for p in sorted((SOFA/"pilot").glob("*.sofa")):
    if any(x in p.stem for x in ("_notch", "_shift", "_synth", "_rising", "_falling",
                                 "_whole", "_test", "_full", "_molds", "_0", "_4", "_s_")): continue
    if p.stem in DUM: continue
    subs.append((p.stem, p))

data = {}
for name, path in subs:
    r = collect(path)
    if r is not None:
        data[name] = r
n = len(data)


def resolvable_frac(rows, sep_bar, wid_bar):
    ok = []
    for sep, sad, wid in rows:
        ok.append(sep is not None and sad is not None and wid is not None
                  and sep >= sep_bar and sad >= SADDLE_BAR and wid >= wid_bar)
    return np.mean(ok) if ok else 0.0


def count_viable(sep_bar, wid_bar, thr=0.5):
    c = 0
    for name, per_ear in data.items():
        best = max(resolvable_frac(per_ear[0], sep_bar, wid_bar),
                   resolvable_frac(per_ear[1], sep_bar, wid_bar))
        if best >= thr: c += 1
    return c


sep_grid = [0.50, 0.40, 0.30, 0.25, 0.20]
wid_grid = [0.17, 0.13, 0.10, 0.00]
print(f"pilot real subjects: {n}   (saddle bar fixed at {SADDLE_BAR} dB)")
print("viable count (>=50% of elevations, best ear), rows=separation bar (oct), cols=N1 width bar (oct)\n")
print("  sep\\wid  " + "  ".join(f"{w:>5.2f}" for w in wid_grid))
for s in sep_grid:
    print(f"  {s:>5.2f}    " + "  ".join(f"{count_viable(s, w):>5d}" for w in wid_grid))
print("\n(current gate = sep 0.50 / width 0.17 -> top-left; timbre floor ~ sep 0.20 / width 0.17)")

# also: how many clear at >=30% of elevations (a laxer viability criterion)
print("\nsame, but viability = >=30% of elevations:")
print("  sep\\wid  " + "  ".join(f"{w:>5.2f}" for w in wid_grid))
for s in sep_grid:
    print(f"  {s:>5.2f}    " + "  ".join(f"{count_viable(s, w, 0.30):>5d}" for w in wid_grid))
