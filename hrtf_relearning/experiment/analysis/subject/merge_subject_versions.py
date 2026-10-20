"""Merge two copies of the SAME subject recorded on different machines.

Typical case: a participant ran on the freefield-lab PC and the VR-lab PC on
the same day, one copy got committed, the other has runs the commit lacks.

Usage:
    python merge_subject_versions.py A.pkl B.pkl --out OUT_DIR [--apply]

A and B are two Subject pickles for the same id; each may have its <id>.json
archive next to it. Dry-run by default: prints the diff and writes nothing.
With --apply, writes OUT_DIR/<id>.pkl and OUT_DIR/<id>.json (backing up any
existing files there to .bak first).

Unlike merge_subjects.py (which is for DIFFERENT people's records) this keeps
every Subject field and writes the complete archive (trials included).

Merge rules
  localization  union by run key, sorted chronologically. A key in both with
                different content -> the copy with more completed trials wins,
                and the conflict is reported. Runs that exist only in an
                archive (removed from the pickle on purpose) stay archive-only.
  trials        union by (session_id, trial_idx, t_start), sorted by t_start;
                when both have a trial, the copy that still has its raw
                pose_trace wins.
  highscore     max.
  demographics / active_donor / donor_screen / head_radius
                whichever copy has a value; if both differ -> refuse unless
                --prefer A|B is given.
  last_sequence the chronologically newest of the two pointers, repointed at
                the merged run object.
"""

import argparse
import json
import pickle
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

FIELDS = ("demographics", "active_donor", "donor_screen", "head_radius")
_KEY_TIME = re.compile(r"_(\d{2})\.(\d{2})_(\d{2})-(\d{2})")


def run_time(key: str):
    """(month, day, hour, minute) from a run key like DA_20.10_09-36_..."""
    m = _KEY_TIME.search(key)
    if not m:
        return (99, 99, 99, 99)
    d, mo, h, mi = map(int, m.groups())
    return (mo, d, h, mi)


def n_done(seq) -> int:
    """Completed trials of a Trialsequence (or its archived dict)."""
    data = seq.get("data") if isinstance(seq, dict) else getattr(seq, "data", None)
    return sum(1 for x in (data or []) if x is not None)


def same_run(a, b) -> bool:
    return _to_jsonable(a) == _to_jsonable(b)


def trial_key(t):
    return (t.get("session_id"), t.get("trial_idx"), t.get("t_start"))


def has_trace(t) -> bool:
    tr = t.get("pose_trace")   # list, or a packed numpy array
    return tr is not None and len(tr) > 0


def is_empty(v) -> bool:
    return v is None or v == {} or v == []


def _to_jsonable(obj):
    """Copy of Subject._to_jsonable — kept local so this runs without the package."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return {
            "__class__": f"{type(obj).__module__}.{type(obj).__name__}",
            **{k: _to_jsonable(v) for k, v in vars(obj).items()
               if not k.startswith("_")},
        }
    return repr(obj)


def load(pkl: Path):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    js = pkl.with_suffix(".json")
    archive = {}
    if js.exists():
        with open(js, encoding="utf-8") as f:
            archive = json.load(f) or {}
    return data, archive


def describe(tag, path, d, arch):
    loc = d.get("localization") or {}
    tr = [t for t in d.get("trials") or [] if t]
    sessions = sorted({t.get("session_id") for t in tr})
    print(f"{tag}: {path}")
    print(f"   {len(loc)} runs, {len(tr)} trials (sessions {sessions}), "
          f"highscore={d.get('highscore', 0)}, "
          f"last_sequence={getattr(d.get('last_sequence'), 'name', None)}")
    only_arch = sorted(set(arch.get("localization") or {}) - set(loc))
    if only_arch:
        print(f"   archive-only runs: {only_arch}")


def merge(a, b, prefer=None):
    la, lb = a.get("localization") or {}, b.get("localization") or {}
    loc, report = {}, []
    for key in sorted(set(la) | set(lb), key=lambda k: (run_time(k), k)):
        if key in la and key in lb:
            if same_run(la[key], lb[key]):
                loc[key] = la[key]
                report.append(f"   =  {key}  (identical in both)")
            else:
                pick = "A" if n_done(la[key]) >= n_done(lb[key]) else "B"
                loc[key] = la[key] if pick == "A" else lb[key]
                report.append(f"   !  {key}  DIFFERS: A {n_done(la[key])} done, "
                              f"B {n_done(lb[key])} done -> kept {pick}")
        elif key in la:
            loc[key] = la[key]
            report.append(f"   A  {key}  ({n_done(la[key])}/{getattr(la[key], 'n_trials', '?')})")
        else:
            loc[key] = lb[key]
            report.append(f"   B  {key}  ({n_done(lb[key])}/{getattr(lb[key], 'n_trials', '?')})")

    trials = {}
    n_dup = 0
    for t in [t for t in a.get("trials") or [] if t] + [t for t in b.get("trials") or [] if t]:
        k = trial_key(t)
        if k in trials:
            n_dup += 1
            if not has_trace(trials[k]) and has_trace(t):
                trials[k] = t
        else:
            trials[k] = t
    trials = sorted(trials.values(), key=lambda t: (t.get("t_start") or 0))

    out = {"id": a["id"], "localization": loc, "trials": trials,
           "highscore": max(int(a.get("highscore") or 0), int(b.get("highscore") or 0))}
    conflicts = []
    for f in FIELDS:
        va, vb = a.get(f), b.get(f)
        if is_empty(va):
            out[f] = vb
        elif is_empty(vb) or _to_jsonable(va) == _to_jsonable(vb):
            out[f] = va
        elif prefer:
            out[f] = va if prefer == "A" else vb
            conflicts.append(f"   {f}: differs, kept {prefer} (--prefer)")
        else:
            conflicts.append(f"   {f}: A={va!r}  B={vb!r}")
            out[f] = None

    names = [getattr(x.get("last_sequence"), "name", None) for x in (a, b)]
    names = [n for n in names if n in loc]
    last = max(names, key=lambda k: (run_time(k), k)) if names else None
    out["last_sequence"] = loc.get(last)
    return out, report, conflicts, n_dup


def archived_trials(trials):
    """Subject._archivable_trials without recomputing pose metrics."""
    return [_to_jsonable({k: v for k, v in t.items() if k not in ("pose_trace", "pose_t0")})
            for t in trials]


def write(out_dir: Path, merged, arch_a, arch_b):
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / f"{merged['id']}.pkl"
    js = pkl.with_suffix(".json")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in (pkl, js):
        if p.exists():
            shutil.copy2(p, p.with_name(f"{p.name}.premerge-{stamp}"))
    data = {k: merged[k] for k in ("id", "localization", "trials", "last_sequence",
                                   "highscore", *FIELDS)}
    tmp = pkl.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(pkl)
    # append-only archive: union of both archives, current runs take precedence
    loc = {**(arch_a.get("localization") or {}), **(arch_b.get("localization") or {}),
           **_to_jsonable(merged["localization"])}
    loc = dict(sorted(loc.items(), key=lambda kv: (run_time(kv[0]), kv[0])))
    payload = {
        "id": merged["id"],
        "highscore": int(merged["highscore"] or 0),
        "demographics": dict(merged["demographics"] or {}),
        "active_donor": _to_jsonable(merged["active_donor"] or {}),
        "donor_screen": _to_jsonable(merged["donor_screen"] or {}),
        "head_radius": _to_jsonable(merged["head_radius"]),
        "last_sequence": getattr(merged["last_sequence"], "name", None),
        "localization": loc,
        "trials": archived_trials(merged["trials"]),
    }
    tmp = js.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(js)
    return pkl, js


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--prefer", choices=("A", "B"), default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    a, arch_a = load(args.a)
    b, arch_b = load(args.b)
    if a.get("id") != b.get("id"):
        sys.exit(f"id mismatch: {a.get('id')!r} vs {b.get('id')!r}")
    describe("A", args.a, a, arch_a)
    describe("B", args.b, b, arch_b)

    merged, report, conflicts, n_dup = merge(a, b, args.prefer)
    print("\nlocalization:")
    print("\n".join(report))
    print(f"\ntrials: {len(merged['trials'])} after merge ({n_dup} present in both)")
    print(f"highscore: {merged['highscore']}")
    print(f"last_sequence: {getattr(merged['last_sequence'], 'name', None)}")
    if conflicts:
        print("\nfield conflicts:")
        print("\n".join(conflicts))
        if not args.prefer:
            sys.exit("\nrefusing to merge: resolve with --prefer A|B")
    if not args.apply:
        print("\n[dry-run] nothing written; add --apply")
        return
    pkl, js = write(args.out, merged, arch_a, arch_b)
    print(f"\nwrote {pkl}\nwrote {js}")


if __name__ == "__main__":
    main()
