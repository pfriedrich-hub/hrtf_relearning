"""Restore localization data from a JSON backup and merge into a Subject pkl.

Used when the pkl is missing but a JSON backup exists, and/or when the
subject id in the JSON is wrong and needs to be remapped.

Usage:
    python restore_from_json.py JSON_FILE TARGET_PKL [--id-remap OLD NEW] [--dry-run]

Example (remap UG → JS and merge into JS.pkl):
    python restore_from_json.py JS_fflab.json JS.pkl --id-remap UG JS

The TARGET_PKL is updated in-place (backed up first).
If TARGET_PKL doesn't exist yet, a fresh one is created.
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


def _reconstruct_trialsequence(d: dict):
    """Reconstruct a slab.Trialsequence from its _to_jsonable dict representation."""
    import slab
    cls = slab.psychoacoustics.Trialsequence
    obj = object.__new__(cls)
    for k, v in d.items():
        if k == "__class__":
            continue
        setattr(obj, k, v)
    return obj


def load_json_localization(json_path: Path, id_remap: tuple[str, str] | None = None
                           ) -> tuple[str, dict]:
    """Load localization from a JSON backup.

    Returns (subject_id, localization_dict) where values are Trialsequence objects.
    id_remap=(old, new): rename run keys that start with old_ to new_.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    subject_id = data["id"]
    old_prefix, new_prefix = id_remap if id_remap else (None, None)

    localization = {}
    for key, val in data["localization"].items():
        # Remap key prefix if requested
        if old_prefix and key.startswith(old_prefix + "_"):
            key = new_prefix + key[len(old_prefix):]

        # Reconstruct Trialsequence from dict
        if isinstance(val, dict) and val.get("__class__", "").endswith("Trialsequence"):
            try:
                val = _reconstruct_trialsequence(val)
            except Exception as e:
                print(f"  [warn] Could not reconstruct Trialsequence for {key!r}: {e}")

        localization[key] = val

    return subject_id, localization


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_file", type=Path, help="JSON backup file to restore from")
    parser.add_argument("target_pkl", type=Path, help="Target pkl to merge into (or create)")
    parser.add_argument("--id-remap", nargs=2, metavar=("OLD", "NEW"),
                        help="Rename run-key prefix OLD → NEW (e.g. --id-remap UG JS)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.json_file.exists():
        sys.exit(f"JSON file not found: {args.json_file}")

    print(f"Loading {args.json_file.name} ...")
    json_id, json_loc = load_json_localization(args.json_file, args.id_remap)
    remap_msg = f" (keys remapped {args.id_remap[0]}→{args.id_remap[1]})" if args.id_remap else ""
    print(f"  id={json_id!r}, {len(json_loc)} runs{remap_msg}")

    # Load or create target
    if args.target_pkl.exists():
        target = load_pkl(args.target_pkl)
        print(f"\nTarget {args.target_pkl.name}: "
              f"{len(target.get('localization') or {})} runs, "
              f"{len(target.get('trials') or [])} trials, "
              f"highscore={target.get('highscore', 0)}")
    else:
        target_id = args.id_remap[1] if args.id_remap else json_id
        target = {"id": target_id, "localization": {}, "trials": [],
                  "last_sequence": None, "highscore": 0}
        print(f"\nTarget {args.target_pkl.name}: does not exist, will create fresh")

    # Merge localization
    existing_keys = set(target.get("localization") or {})
    added, skipped = 0, 0
    merged_loc = dict(target.get("localization") or {})
    for key, val in json_loc.items():
        if key in existing_keys:
            print(f"  [SKIP] key already exists: {key!r}")
            skipped += 1
        else:
            merged_loc[key] = val
            added += 1

    target["localization"] = merged_loc
    print(f"\nResult: +{added} runs added, {skipped} skipped → "
          f"{len(merged_loc)} total runs")

    # --- everything else the archive now carries ---------------------------
    # Older archives hold only id/highscore/demographics/localization, so each
    # of these is restored only when present AND when the target has nothing
    # better. Trials are the important one: they used to be archived nowhere,
    # so a rebuilt pickle silently came back with the training record empty.
    with open(args.json_file, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}

    json_trials = raw.get("trials")
    if json_trials is None:
        print("  [!] this archive predates trial backup — training data cannot "
              "be restored from it (raw pose traces are gone either way)")
    elif len(json_trials) > len(target.get("trials") or []):
        n_before = len(target.get("trials") or [])
        target["trials"] = json_trials
        print(f"  trials: {n_before} → {len(json_trials)} restored from archive "
              f"(pose_trace summarised as pose_metrics, raw samples not recoverable)")
    else:
        print(f"  trials: keeping target's {len(target.get('trials') or [])} "
              f"(archive has {len(json_trials)})")

    for field in ("demographics", "active_donor"):
        if raw.get(field) and not target.get(field):
            target[field] = raw[field]
            print(f"  {field}: restored from archive")

    if raw.get("highscore") and int(raw["highscore"]) > int(target.get("highscore") or 0):
        target["highscore"] = int(raw["highscore"])
        print(f"  highscore: restored {target['highscore']} from archive")

    # last_sequence is archived as a key; repoint it at the reconstructed run
    name = raw.get("last_sequence")
    if name and not target.get("last_sequence"):
        target["last_sequence"] = merged_loc.get(name)
        print(f"  last_sequence: repointed at {name!r}"
              if name in merged_loc else
              f"  [!] last_sequence {name!r} not among the restored runs")

    if args.dry_run:
        print("\n[dry-run] Nothing written.")
        return

    if args.target_pkl.exists():
        backup = args.target_pkl.with_suffix(".pkl.bak")
        shutil.copy2(args.target_pkl, backup)
        print(f"\nBacked up → {backup.name}")

    save_pkl(args.target_pkl, target)
    print(f"Saved → {args.target_pkl}")

    # Regenerate JSON backup alongside the pickle: RESULTS_DIR/<id>/<id>.json
    json_out = args.target_pkl.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = {"id": target["id"], "highscore": int(target.get("highscore") or 0),
                   "demographics": target.get("demographics") or {},
                   "active_donor": target.get("active_donor") or {},
                   "last_sequence": getattr(target.get("last_sequence"), "name", None),
                   "localization": _to_jsonable(target["localization"]),
                   "trials": [_to_jsonable({k: v for k, v in (t or {}).items()
                                            if k != "pose_trace"})
                              for t in (target.get("trials") or [])]}
        tmp = json_out.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(json_out)
        print(f"JSON backup → {json_out}")
    except Exception as e:
        print(f"[warn] Could not write JSON backup: {e}")


if __name__ == "__main__":
    main()
