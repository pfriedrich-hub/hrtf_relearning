"""Merge multiple Subject pkl files into one.

Usage:
    python merge_subjects.py PRIMARY [SECONDARY ...] [--output OUTPUT] [--dry-run]

The PRIMARY file is updated in-place (or written to --output).
SECONDARY files are merged into it. The originals are not deleted.

Merge rules:
  localization  : dict update — keys are timestamped run-names so collisions
                  are unlikely; any collision is reported and skipped.
  trials        : lists are concatenated.
  highscore     : max of all files.
  last_sequence : kept from PRIMARY (most recent authoritative state).
  id            : kept from PRIMARY.
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path


def _to_jsonable(obj):
    """Recursively convert obj to JSON-serializable primitives (mirrors Subject._to_jsonable)."""
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
            **{k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")},
        }
    return repr(obj)


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def merge(primary: dict, others: list[dict], verbose: bool = True) -> dict:
    merged = {
        "id": primary["id"],
        "localization": dict(primary.get("localization") or {}),
        "trials": list(primary.get("trials") or []),
        "last_sequence": primary.get("last_sequence"),
        "highscore": primary.get("highscore", 0),
    }

    for src in others:
        src_id = src.get("id", "?")

        # --- localization ---
        for key, seq in (src.get("localization") or {}).items():
            if key in merged["localization"]:
                print(f"  [SKIP] localization key already exists, skipping: {key!r}")
            else:
                merged["localization"][key] = seq

        # --- trials ---
        src_trials = src.get("trials") or []
        merged["trials"].extend(src_trials)
        if verbose:
            print(f"  [{src_id}] +{len(src.get('localization') or {})} runs, "
                  f"+{len(src_trials)} trials")

        # --- highscore ---
        merged["highscore"] = max(merged["highscore"], src.get("highscore", 0))

    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("primary", type=Path, help="Primary pkl file (merged result goes here)")
    parser.add_argument("secondary", type=Path, nargs="+", help="Files to merge in")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write result here instead of overwriting primary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing anything")
    args = parser.parse_args()

    primary_path = args.primary
    output_path = args.output or primary_path

    if not primary_path.exists():
        sys.exit(f"Primary file not found: {primary_path}")
    for p in args.secondary:
        if not p.exists():
            sys.exit(f"Secondary file not found: {p}")

    primary = load(primary_path)
    secondaries = [load(p) for p in args.secondary]

    print(f"Primary : {primary_path.name}  "
          f"({len(primary.get('localization') or {})} runs, "
          f"{len(primary.get('trials') or [])} trials, "
          f"highscore={primary.get('highscore', 0)})")
    for p, s in zip(args.secondary, secondaries):
        print(f"Merge in: {p.name}  "
              f"({len(s.get('localization') or {})} runs, "
              f"{len(s.get('trials') or [])} trials, "
              f"highscore={s.get('highscore', 0)})")

    merged = merge(primary, secondaries)

    print(f"\nResult: {len(merged['localization'])} runs, "
          f"{len(merged['trials'])} trials, "
          f"highscore={merged['highscore']}")

    if args.dry_run:
        print("\n[dry-run] Nothing written.")
        return

    # Back up primary before overwriting
    if output_path == primary_path:
        backup = primary_path.with_suffix(".pkl.bak")
        shutil.copy2(primary_path, backup)
        print(f"\nBacked up primary → {backup.name}")

    save(output_path, merged)
    print(f"Saved → {output_path}")

    # Regenerate JSON backup in the backup subfolder
    backup_dir = output_path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    json_path = backup_dir / output_path.with_suffix(".json").name
    try:
        payload = {"id": merged["id"], "localization": _to_jsonable(merged["localization"])}
        tmp_json = json_path.with_suffix(".json.tmp")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_json.replace(json_path)
        print(f"JSON backup → {json_path}")
    except Exception as e:
        print(f"[warn] Could not write JSON backup: {e}")


if __name__ == "__main__":
    main()
