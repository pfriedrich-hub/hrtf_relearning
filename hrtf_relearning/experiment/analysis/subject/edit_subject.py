"""Interactive Subject editor.

Lists localization runs for a subject and lets you remove selected entries.

Usage:
    python edit_subject.py SUBJECT_ID
    python edit_subject.py AH
"""

import json
import pickle
import shutil
import sys
from pathlib import Path

# script lives at hrtf_relearning/experiment/analysis/subject/edit_subject.py
# so package root is 4 levels up
_pkg_root = Path(__file__).resolve().parents[3]
results_dir = _pkg_root / "data" / "results"
backup_dir = results_dir / "backup"


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


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save(path: Path, data: dict) -> None:
    shutil.copy2(path, path.with_suffix(".pkl.bak"))
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def write_json_backup(data: dict, pkl_path: Path) -> None:
    backup_dir.mkdir(exist_ok=True)
    json_path = backup_dir / pkl_path.with_suffix(".json").name
    payload = {"id": data["id"], "localization": _to_jsonable(data["localization"])}
    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(json_path)


def seq_summary(key: str, seq) -> str:
    label = getattr(seq, "label", "") or ""
    n = getattr(seq, "n_trials", "?")
    finished = getattr(seq, "finished", None)
    status = "✓" if finished else "…"
    parts = [status, f"{n:>3} trials" if isinstance(n, int) else f"  ? trials"]
    if label and label not in key:
        parts.append(f"[{label}]")
    return "  ".join(parts)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: edit_subject.py SUBJECT_ID")

    subject_id = sys.argv[1]
    pkl_path = results_dir / f"{subject_id}.pkl"

    if not pkl_path.exists():
        sys.exit(f"No file found: {pkl_path}")

    data = load(pkl_path)
    loc = data.get("localization") or {}

    if not loc:
        print(f"{subject_id}: no localization entries.")
        return

    keys = list(loc.keys())
    print(f"\nSubject {subject_id!r} — {len(keys)} localization run(s):\n")
    for i, k in enumerate(keys):
        summary = seq_summary(k, loc[k])
        print(f"  [{i+1:>2}]  {k:<45}  {summary}")

    print("\nEnter numbers to remove (e.g. 1 3 5), or press Enter to cancel:")
    raw = input("> ").strip()
    if not raw:
        print("Cancelled.")
        return

    try:
        indices = [int(x) - 1 for x in raw.split()]
    except ValueError:
        sys.exit("Invalid input.")

    to_remove = []
    for i in indices:
        if 0 <= i < len(keys):
            to_remove.append(keys[i])
        else:
            print(f"  [warn] index {i+1} out of range, skipped")

    if not to_remove:
        print("Nothing to remove.")
        return

    print(f"\nWill remove {len(to_remove)} run(s):")
    for k in to_remove:
        print(f"  - {k}")
    print("\nConfirm? [y/N]")
    if input("> ").strip().lower() != "y":
        print("Cancelled.")
        return

    for k in to_remove:
        del data["localization"][k]

    save(pkl_path, data)
    write_json_backup(data, pkl_path)
    print(f"\nDone. {len(data['localization'])} run(s) remaining. Backup → {pkl_path.with_suffix('.pkl.bak').name}")


if __name__ == "__main__":
    main()
