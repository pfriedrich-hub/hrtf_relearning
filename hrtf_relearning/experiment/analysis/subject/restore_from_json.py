"""Restore localization data from a JSON backup and merge into a Subject pkl.

Used when the pkl is missing but a JSON backup exists, and/or when the
subject id in the JSON is wrong and needs to be remapped.

QUICK FIX for a broken pkl (the one to remember):

    hrtf restore LP

rebuilds RESULTS_DIR/LP/LP.pkl from LP.json, keeping the broken file next to it
as LP.pkl.broken_<timestamp>. Does nothing if the pkl still loads (add --force
to rebuild anyway). See restore_subject() below.

Usage (merge / id-remap, the older general tool):
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


def restore_subject(subject_id: str, force: bool = False) -> int:
    """Rebuild RESULTS_DIR/<id>/<id>.pkl from <id>.json in one go.

    For the case where the pkl is destroyed (text-editor UTF-8 round-trip, see
    hrtf_relearning/utils/integrity.py) and has to be usable again NOW, e.g.
    with the participant waiting. Unlike main(), this restores every field the
    archive carries -- including donor_screen and head_radius -- and never
    rewrites the JSON, which is the only intact copy at that point.

    The broken pkl is moved aside, not deleted. Lost for good: only the raw
    pose_trace of training trials (the archive holds pose_metrics instead).
    """
    import datetime
    from hrtf_relearning.utils import paths
    from hrtf_relearning.utils.integrity import is_mangled_pickle

    pkl = paths.subject_pkl(subject_id)
    json_path = paths.subject_backup(subject_id)
    if not json_path.exists():
        print(f"No JSON archive at {json_path} -- nothing to restore from.")
        return 1

    if pkl.exists():
        if is_mangled_pickle(pkl):
            state = "destroyed by a text editor (UTF-8 round-trip)"
        else:
            try:
                load_pkl(pkl)
                state = None
            except Exception as exc:
                state = f"unreadable ({type(exc).__name__}: {exc})"
        if state is None and not force:
            print(f"{pkl.name} loads fine -- nothing to do. "
                  f"(hrtf restore {subject_id} --force rebuilds it anyway.)")
            return 0
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = pkl.with_name(f"{pkl.name}.broken_{stamp}")
        pkl.rename(aside)
        print(f"{pkl.name}: {state or 'healthy, rebuilt on --force'}")
        print(f"  moved aside -> {aside.name}")
    else:
        print(f"{pkl.name} missing -- creating it from {json_path.name}")

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    _, localization = load_json_localization(json_path)
    trials = []
    for t in raw.get("trials") or []:
        t = dict(t)
        if isinstance(t.get("target"), list):
            t["target"] = tuple(t["target"])    # pickle schema stores a tuple
        trials.append(t)
    data = {
        "id": raw.get("id", subject_id),
        "localization": localization,
        "trials": trials,
        "last_sequence": localization.get(raw.get("last_sequence")),
        "highscore": raw.get("highscore", 0),
        "demographics": raw.get("demographics") or {},
        "active_donor": raw.get("active_donor") or {},
        "donor_screen": raw.get("donor_screen") or {},
        "head_radius": raw.get("head_radius"),
    }
    save_pkl(pkl, data)

    # prove it: load through the same class the experiment uses
    from hrtf_relearning.experiment.misc.Subject import Subject
    s = Subject(subject_id)
    donor = s.active_donor or {}
    print(f"Rebuilt {pkl.name}: {len(s.localization)} runs, {len(s.trials)} "
          f"training trials, highscore {s.highscore}")
    print(f"  last run:    {next(reversed(s.localization), None)}")
    print(f"  donor:       {donor.get('donor', '-')} ({donor.get('donor_ear', '-')} ear)")
    print(f"  donor screen {'on record' if s.donor_screen else 'NOT on record'}")
    if raw.get("trials") is None:
        print("  [!] archive predates trial backup -- training record is EMPTY")
    print("  lost: raw head-tracker traces of training trials (pose_metrics kept)")
    return 0


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
    else:
        # MERGE, never replace: the target's own trials carry the raw
        # pose_trace, the archive's only pose_metrics. Replacing them wholesale
        # (the old behaviour) threw away every surviving trace — found on FP
        # 2026-09-29, where it would have stripped 366 intact trials.
        def _trial_key(t):
            return (t.get("session_id"), t.get("trial_idx"), t.get("t_start"))
        have = {_trial_key(t) for t in (target.get("trials") or []) if t}
        new = []
        for t in json_trials:
            if not t or _trial_key(t) in have:
                continue
            t = dict(t)
            if isinstance(t.get("target"), list):
                t["target"] = tuple(t["target"])   # pickle schema stores a tuple
            new.append(t)
        n_before = len(target.get("trials") or [])
        target["trials"] = list(target.get("trials") or []) + new
        print(f"  trials: kept {n_before}, +{len(new)} from archive "
              f"(archive has {len(json_trials)}); archived trials carry "
              f"pose_metrics only, their raw samples are not recoverable")

    for field in ("demographics", "active_donor"):
        if raw.get(field) and not target.get(field):
            target[field] = raw[field]
            print(f"  {field}: restored from archive")

    if raw.get("highscore") and int(raw["highscore"]) > int(target.get("highscore") or 0):
        target["highscore"] = int(raw["highscore"])
        print(f"  highscore: restored {target['highscore']} from archive")

    # last_sequence is archived as a key; repoint it at the reconstructed run
    name = raw.get("last_sequence")
    # the archive wins when it names a run that only the archive had: then it
    # is newer than whatever the target pointed at
    if name and (not target.get("last_sequence")
                 or (name in merged_loc and name not in existing_keys)):
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
                                            if k not in ("pose_trace", "pose_t0")})
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
