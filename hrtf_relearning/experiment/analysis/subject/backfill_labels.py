"""Backfill seq.label for all existing Subject pkl files.

Extracts the condition name from the localization dict key by stripping the
leading '{subject_id}_{dd.mm}_{HH-MM}_' prefix, and sets it as seq.label
if the attribute is currently missing or empty.

Usage:
    python backfill_labels.py [--results-dir PATH] [--dry-run]
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = _pkg_root / "data" / "results"

KEY_RE = re.compile(r"^.+?_\d{2}\.\d{2}_\d{2}[:\-]\d{2}_(.+)$")


def condition_from_key(key: str) -> str:
    """Extract condition label from a localization dict key."""
    m = KEY_RE.match(key)
    return m.group(1) if m else key


def _to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        import numpy as np
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
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
            **{k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")},
        }
    return repr(obj)


def backfill(pkl_path: Path, dry_run: bool = False) -> int:
    """Return number of sequences updated."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    loc = data.get("localization") or {}
    updated = 0
    for key, seq in loc.items():
        current_label = getattr(seq, "label", None)
        if current_label:
            continue  # already set
        label = condition_from_key(key)
        if dry_run:
            print(f"  {key!r} → label={label!r}")
        else:
            seq.label = label
        updated += 1

    if updated and not dry_run:
        tmp = pkl_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        tmp.replace(pkl_path)
        # update json backup in backup subfolder
        backup_dir = pkl_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        json_path = backup_dir / pkl_path.with_suffix(".json").name
        try:
            payload = {"id": data["id"], "localization": _to_jsonable(loc)}
            tmp_j = json_path.with_suffix(".json.tmp")
            with open(tmp_j, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_j.replace(json_path)
        except Exception as e:
            print(f"  [warn] JSON backup failed: {e}")

    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pkls = sorted(args.results_dir.glob("*.pkl"))
    if not pkls:
        sys.exit(f"No pkl files found in {args.results_dir}")

    total = 0
    for p in pkls:
        n = backfill(p, dry_run=args.dry_run)
        if n:
            print(f"{'[dry] ' if args.dry_run else ''}{p.name}: {n} label(s) set")
        else:
            print(f"  {p.name}: up to date")
    print(f"\n{'Would set' if args.dry_run else 'Set'} {total} label(s) total.")


if __name__ == "__main__":
    main()
