"""
diagnose_fused_edge_interference.py  (dev/analysis)

Quantify how often the CURRENT edge-shift manipulation moves an edge that lies
in the INTERIOR of a perceptually fused notch pair, rather than on the outer
flank of a resolved feature.

Why this matters
----------------
edge_shift_ir() builds its manipulated feature set with
    select_features(notches_all, sep_min_oct=0.0)
i.e. it shifts EVERY detected notch's edge, regardless of whether two notches
are separately resolvable. The perceptual (default) set uses
    select_features(notches_all, sep_min_oct=SEP_MIN_OCT=0.5)   # Macpherson 2003
which fuses a pair closer than ~0.5 octave (also gated by Moore 1989 saddle
depth / notch width inside select_features).

For a fused adjacent pair (N_lo, N_hi) with sep < 0.5 oct:
  - under mode='rising'  the LOWER notch's rising edge points UP into the
    interior saddle  -> INTERIOR edge (perception cannot resolve it)
  - under mode='falling' the UPPER notch's falling edge points DOWN into the
    interior saddle  -> INTERIOR edge
The outer flanks (N_lo falling, N_hi rising) are legitimate merged-feature edges.

This script reports, per subject and pooled:
  - fraction of median-plane directions containing >=1 fused adjacent pair
  - interior-edge rate for rising and falling modes
    = (# interior edges shifted) / (# edges shifted) under the sep=0 set
  - mean feature count sep=0 vs sep=0.5 (how many phantom features fusion removes)

Usage: python3 diagnose_fused_edge_interference.py
Outputs a table to stdout, a CSV, and a summary figure.
"""
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import importlib.util as iu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUT = Path(os.environ.get("OUT_DIR", "/sessions/intelligent-quirky-brahmagupta/mnt/outputs"))
OUT.mkdir(parents=True, exist_ok=True)

spec = iu.spec_from_file_location("es", ES)
es = iu.module_from_spec(spec)
spec.loader.exec_module(es)

EL = (-35.0, 35.0)
SEP_LIMIT = es.SEP_MIN_OCT          # 0.5 oct (Macpherson)
DUM = {"KU100", "FABIAN", "MRT01", "kemar", "kemar_pir", "universal", "test",
       "mit_kemar_normal_pinna"}
# skip modified / derived sofas -- we want measured DTFs only
SKIP = ("_notch", "_shift", "_synth", "_rising", "_falling", "_whole", "_edge",
        "_test", "_full", "_molds", "_s_", "_0", "_1", "_4", "__")


def median_dirs(f, sp):
    """Median-plane, front, el in [-35,35], sorted by elevation, <=17 samples."""
    typ = ""
    try:
        typ = (f["SourcePosition"].attrs.get("Type", b"") or b"")
        typ = typ.decode() if isinstance(typ, bytes) else str(typ)
    except Exception:
        pass
    if "cart" in typ.lower():
        x, y, z = sp[:, 0], sp[:, 1], sp[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1, 1)))
        med = np.where((np.abs(az) <= 3) & (x > 0))[0]
    else:
        az = (sp[:, 0] + 180) % 360 - 180
        el = sp[:, 1]
        med = np.where(np.abs(az) <= 2)[0]
    med = med[(el[med] >= EL[0]) & (el[med] <= EL[1])]
    med = med[np.argsort(el[med])]
    if len(med) > 17:
        med = med[np.linspace(0, len(med) - 1, 17).astype(int)]
    return med


def features_for_ir(ir, fs, sep):
    """detect_notches + select_features(sep_min_oct=sep), frequency-sorted."""
    n = len(ir)
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mag = np.abs(np.fft.rfft(ir, nfft))
    L = 20.0 * np.log10(np.maximum(mag, es.EPS))
    e, _Le, _band, _fb = es._to_erb_grid(L, freqs, 3000.0, 17500.0)
    notches = es.detect_notches(e, freqs, mag)
    feats = es.select_features(notches, sep_min_oct=sep)
    return feats


def analyse_direction(ir, fs):
    """Return dict of counts for one DTF (one direction, one ear)."""
    f0 = features_for_ir(ir, fs, 0.0)       # what edge_shift_ir actually shifts
    f5 = features_for_ir(ir, fs, SEP_LIMIT)  # perceptually resolved set
    f0.sort(key=lambda d: d["f_hz"])
    # adjacent fused pairs within the sep=0 set
    fused_lo, fused_hi = set(), set()
    for i in range(len(f0) - 1):
        if np.log2(f0[i + 1]["f_hz"] / f0[i]["f_hz"]) < SEP_LIMIT:
            fused_lo.add(i)          # lower member -> interior RISING edge
            fused_hi.add(i + 1)      # upper member -> interior FALLING edge
    n_rise = sum(1 for d in f0 if d.get("edge_rise") is not None)
    n_fall = sum(1 for d in f0 if d.get("edge_fall") is not None)
    int_rise = sum(1 for i, d in enumerate(f0)
                   if i in fused_lo and d.get("edge_rise") is not None)
    int_fall = sum(1 for i, d in enumerate(f0)
                   if i in fused_hi and d.get("edge_fall") is not None)
    return dict(nfeat0=len(f0), nfeat5=len(f5),
                has_fused=int(len(fused_lo) > 0),
                n_rise=n_rise, n_fall=n_fall,
                int_rise=int_rise, int_fall=int_fall)


def analyse_subject(path):
    with h5py.File(path, "r") as f:
        ir = f["Data.IR"][:]
        sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
        med = median_dirs(f, sp)
    if len(med) < 5:
        return None
    agg = dict(ndir=0, nfeat0=0, nfeat5=0, has_fused=0,
               n_rise=0, n_fall=0, int_rise=0, int_fall=0)
    for d in med:
        for ear in (0, 1):
            try:
                r = analyse_direction(ir[d, ear, :], fs)
            except Exception:
                continue
            agg["ndir"] += 1
            for k in ("nfeat0", "nfeat5", "has_fused", "n_rise", "n_fall",
                      "int_rise", "int_fall"):
                agg[k] += r[k]
    return agg if agg["ndir"] else None


def collect():
    subs = []
    for stem in ("JS", "CA", "GS", "SS", "AS"):
        p = SOFA / stem / f"{stem}.sofa"
        if p.exists():
            subs.append((stem, p))
    for p in sorted((SOFA / "pilot").rglob("*.sofa")):
        if any(x in p.stem for x in SKIP):
            continue
        subs.append((p.stem, p))
    for p in sorted((SOFA / "database").glob("*.sofa")):
        subs.append((p.stem, p))
    # de-dup by stem
    seen, uniq = set(), []
    for name, p in subs:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, p))
    return uniq


def main():
    rows = []
    for name, path in collect():
        try:
            a = analyse_subject(path)
        except Exception as ex:
            print(f"{name:16s} ERROR {type(ex).__name__}: {ex}")
            continue
        if a is None:
            continue
        dummy = name in DUM
        fused_dir = 100 * a["has_fused"] / a["ndir"]
        ri = 100 * a["int_rise"] / max(a["n_rise"], 1)
        fi = 100 * a["int_fall"] / max(a["n_fall"], 1)
        rows.append(dict(name=name, dummy=dummy, ndir=a["ndir"],
                         mf0=a["nfeat0"] / a["ndir"], mf5=a["nfeat5"] / a["ndir"],
                         fused_dir=fused_dir, rise_int=ri, fall_int=fi,
                         n_rise=a["n_rise"], int_rise=a["int_rise"],
                         n_fall=a["n_fall"], int_fall=a["int_fall"]))

    rows.sort(key=lambda r: r["rise_int"], reverse=True)
    print(f"\n{'subject':16s} {'dir':>4s} {'feat/dir(0)':>11s} {'feat/dir(.5)':>12s}"
          f" {'%dir fused':>10s} {'%rise int':>9s} {'%fall int':>9s}")
    print("-" * 78)
    real = [r for r in rows if not r["dummy"]]
    for r in rows:
        tag = " [dummy]" if r["dummy"] else ""
        print(f"{r['name']:16s} {r['ndir']:4d} {r['mf0']:11.2f} {r['mf5']:12.2f}"
              f" {r['fused_dir']:9.0f}% {r['rise_int']:8.0f}% {r['fall_int']:8.0f}%{tag}")

    # pooled over real subjects (edge-weighted, not subject-weighted)
    tot = {k: sum(r[k] for r in real) for k in ("n_rise", "int_rise", "n_fall", "int_fall")}
    ndir = sum(r["ndir"] for r in real)
    nfused = sum(r["fused_dir"] * r["ndir"] / 100 for r in real)
    print("\n=== POOLED over real subjects ===")
    print(f"subjects: {len(real)}   directions (ear x el): {ndir}")
    print(f"directions with >=1 fused pair: {100*nfused/ndir:.0f}%")
    print(f"rising-edge shifts that are INTERIOR:  {tot['int_rise']}/{tot['n_rise']}"
          f" = {100*tot['int_rise']/max(tot['n_rise'],1):.0f}%")
    print(f"falling-edge shifts that are INTERIOR: {tot['int_fall']}/{tot['n_fall']}"
          f" = {100*tot['int_fall']/max(tot['n_fall'],1):.0f}%")
    mf0 = np.mean([r["mf0"] for r in real])
    mf5 = np.mean([r["mf5"] for r in real])
    print(f"mean features/direction: sep=0 {mf0:.2f}  ->  sep=0.5 {mf5:.2f}"
          f"  ({100*(mf0-mf5)/mf0:.0f}% are phantom / fused-away)")

    # CSV
    import csv
    csvp = OUT / "fused_edge_interference.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", csvp)

    # figure: per-subject interior-rising-edge rate (real subjects, sorted)
    real_sorted = sorted(real, key=lambda r: r["rise_int"], reverse=True)
    names = [r["name"] for r in real_sorted]
    ri = [r["rise_int"] for r in real_sorted]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(names))))
    y = np.arange(len(names))
    cols = ["#D85A30" if v >= 33 else ("#EF9F27" if v >= 10 else "#1D9E75") for v in ri]
    ax.barh(y, ri, color=cols, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of rising-edge shifts that land on an INTERIOR (fused) flank")
    ax.set_xlim(0, 100)
    pooled = 100 * tot["int_rise"] / max(tot["n_rise"], 1)
    ax.axvline(pooled, color="k", ls="--", lw=1)
    ax.text(pooled + 1, len(names) - 0.5, f"pooled {pooled:.0f}%", fontsize=9)
    ax.set_title("Interior-edge interference of the current sep=0 edge-shift\n"
                 "(rising mode; interior = lower member of a <0.5-oct fused pair)",
                 fontsize=11)
    fig.tight_layout()
    figp = OUT / "fused_edge_interference.png"
    fig.savefig(figp, dpi=140, bbox_inches="tight")
    print("wrote", figp)


if __name__ == "__main__":
    main()
