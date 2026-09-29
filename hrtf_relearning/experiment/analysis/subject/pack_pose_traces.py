"""Convert stored training traces to the packed float32 format, in place.

Rewrites ``trial['pose_trace']`` in existing subject pickles from a list of
``(t_unix, yaw, pitch)`` tuples to a float32 (N, 3) array plus
``trial['pose_t0']`` — see ``hrtf_relearning/utils/pose_trace.py`` for the
format and why nothing measurable is lost. About 2.4x smaller per pickle.

Only ``pose_trace``/``pose_t0`` change. Every other key of every trial, the
localization runs, last_sequence etc. are carried over untouched, and the JSON
archive is not touched (it never held the trace). Already-packed trials are
skipped, so running it twice is a no-op.

SAFETY. Dry run by default: prints what it would do and the size it would
save. With ``--apply``, each pickle is

  1. copied to ``<id>.pkl.prepack.bak`` (never overwritten if it exists),
  2. rewritten through a .tmp file and an atomic rename,
  3. re-read from disk and compared trial by trial against the originals —
     every non-trace field must be equal and every trace must round-trip
     within 1e-5 s / 1e-4 deg. On any mismatch the backup is copied back.

Mangled (UTF-8 round-tripped) pickles are skipped with a warning.

Do NOT run it on a subject whose training script is open: that process holds
the old list-format trials in memory and its next ``subject.write()`` would
overwrite the packed file (harmlessly — it just un-does the migration — but
the gain is lost). Run it between sessions.

Usage::

    python pack_pose_traces.py                    # dry run, all active subjects
    python pack_pose_traces.py GM LS              # dry run, just these
    python pack_pose_traces.py --pilot            # also RESULTS_DIR/pilot/**
    python pack_pose_traces.py --apply            # do it
    python pack_pose_traces.py --apply --exclude SH
"""
import argparse
import pickle
import shutil
import sys
from pathlib import Path

from hrtf_relearning.utils import paths
from hrtf_relearning.utils.integrity import is_mangled_pickle
from hrtf_relearning.utils.pose_trace import (
    is_packed, pack_trial, max_pack_error)

MAX_DT_S = 1e-5
MAX_DANGLE_DEG = 1e-4


def _targets(ids, pilot):
    if ids:
        out = []
        for sid in ids:
            p = Path(sid)
            out.append(p if p.suffix == ".pkl" else paths.subject_pkl(sid))
        return out
    out = list(paths.subject_pkls())
    if pilot:
        out += sorted(p for p in (paths.RESULTS_DIR / "pilot").rglob("*.pkl")
                      if "backup" not in p.parts)
    return out


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _verify(path, originals, data):
    """Re-read `path` and check it against the pre-pack trial list.
    `originals` maps trial position -> the legacy trace list; `data` is the
    in-memory packed dict that was written. Returns a list of problems."""
    problems = []
    back = _load(path)
    if set(back) != set(data):
        problems.append(f"top-level keys differ: {set(back) ^ set(data)}")
    trials, written = back.get("trials") or [], data.get("trials") or []
    if len(trials) != len(written):
        return problems + [f"trial count {len(trials)} != {len(written)}"]
    for i, (a, b) in enumerate(zip(trials, written)):
        other_a = {k: v for k, v in (a or {}).items() if k not in ("pose_trace", "pose_t0")}
        other_b = {k: v for k, v in (b or {}).items() if k not in ("pose_trace", "pose_t0")}
        if other_a != other_b:
            problems.append(f"trial {i}: non-trace fields differ")
        if i in originals:
            dt, dang = max_pack_error(originals[i], a)
            if dt > MAX_DT_S or dang > MAX_DANGLE_DEG:
                problems.append(f"trial {i}: round-trip error {dt:.2e} s / {dang:.2e} deg")
    return problems


def process(path, apply):
    name = path.relative_to(paths.RESULTS_DIR) if paths.RESULTS_DIR in path.parents else path
    if not path.exists():
        print(f"  {name}: missing, skipped")
        return 0
    if is_mangled_pickle(path):
        print(f"  {name}: MANGLED pickle, skipped (see utils/integrity.py)")
        return 0
    size_before = path.stat().st_size
    data = _load(path)
    trials = data.get("trials") or []

    originals = {i: list(t["pose_trace"]) for i, t in enumerate(trials)
                 if t and "pose_trace" in t and not is_packed(t)}
    if not originals:
        print(f"  {name}: nothing to pack ({size_before / 1e6:.2f} MB)")
        return 0
    for i in originals:
        pack_trial(trials[i])
    # worst round-trip error before anything touches disk
    worst_dt = max(max_pack_error(originals[i], trials[i])[0] for i in originals)
    worst_da = max(max_pack_error(originals[i], trials[i])[1] for i in originals)
    size_after = len(pickle.dumps(data))
    print(f"  {name}: {len(originals)} trials, {size_before / 1e6:.2f} -> "
          f"{size_after / 1e6:.2f} MB, worst error {worst_dt:.1e} s / {worst_da:.1e} deg")
    if worst_dt > MAX_DT_S or worst_da > MAX_DANGLE_DEG:
        print(f"    ERROR beyond tolerance — not written")
        return 0
    if not apply:
        return size_before - size_after

    backup = path.with_name(path.name + ".prepack.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)
    shutil.copymode(backup, path)

    problems = _verify(path, originals, data)
    if problems:
        shutil.copy2(backup, path)
        print(f"    VERIFY FAILED, original restored from {backup.name}:")
        for p in problems[:10]:
            print(f"      {p}")
        return 0
    print(f"    written and verified (backup: {backup.name})")
    return size_before - path.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="subject ids or .pkl paths (default: all active)")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--pilot", action="store_true", help="include RESULTS_DIR/pilot/**")
    ap.add_argument("--exclude", nargs="*", default=[], help="subject ids to skip")
    args = ap.parse_args(argv)

    targets = [p for p in _targets(args.ids, args.pilot) if p.stem not in args.exclude]
    print(("APPLYING" if args.apply else "DRY RUN") + f" — {len(targets)} pickle(s)")
    saved = sum(process(p, args.apply) for p in targets)
    print(f"{'Saved' if args.apply else 'Would save'} {saved / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
