"""Rebuild GS.pkl after the pickle was destroyed by a UTF-8 re-encode.

What happened
-------------
RESULTS_DIR/GS/GS.pkl was at some point read as text and written back as
UTF-8, so every non-ASCII byte became U+FFFD (EF BF BD). The pickle header
``80 04 95 ..`` became ``EF BF BD 04 EF BF BD ..`` -- hence

    _pickle.UnpicklingError: invalid load key, '\\xef'.

That transformation is lossy and cannot be undone: the 1.78 MB pickle
committed on 2026-07-28/29 is gone, and with it the training ``trials``
recorded between 2026-07-10 and 2026-07-28 (the JSON archive only keeps
localization runs, never trials). GS is the only subject affected.

What can be recovered, and from where
-------------------------------------
    e4b0661  GS/GS.pkl    intact pickle, 2026-07-10   12 runs (07.07/08.07),
                                                      10 trials, highscore 2
    ce9e607  GS/GS.json   valid JSON archive           3 runs (28.07),
                                                      highscore 12
    working  GS/GS.pkl    live pickle, restarted       8 runs (10.08),
                          from empty after the loss    0 trials

The 2026-07-28 JSON is *not* a superset: it holds only the three 28.07 runs,
so restoring from it alone (restore_from_json.py) would silently drop the
twelve 07.07/08.07 runs. Nor is the live pickle: after the corruption the
record was restarted empty, so it holds only the 10.08 session. This script
merges all three sources:

    localization  12 runs (07-10 pickle) + 3 (07-28 JSON) + 8 (live pickle),
                  in that order, i.e. chronological
    trials        10, from the 07-10 pickle (07-10..07-28 trials are lost)
    highscore     max over all three sources
    demographics  from the live pickle, if present
    last_sequence repointed at the newest surviving run

The live pickle's own runs always win on a key collision -- it is the only
source that can have been edited since.

Both git sources are read by blob sha with ``git cat-file blob`` (never
``git show``, which would apply text filters), so nothing extra needs to be
committed.

Usage
-----
    python recover_GS.py --dry-run     # report what would be written
    python recover_GS.py               # write GS.pkl + refresh GS.json

The existing GS.pkl / GS.json are copied to *.pre-recovery first.
"""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hrtf_relearning.utils import paths
from hrtf_relearning.experiment.analysis.subject.restore_from_json import (
    load_json_localization)

SUBJECT_ID = "GS"

# Blob shas, not commit paths: the results folders were reorganised and the
# newer commits contain the corrupted copy.
PKL_BLOB = "212f0bbf58e772fc86dc68473908aad09ed43ddc"   # e4b0661 GS/GS.pkl  (intact)
JSON_BLOB = "0e96eeadaf68945546293669ca86742c03fa079c"  # ce9e607 GS/GS.json (valid)

REPO_ROOT = Path(__file__).resolve().parents[4]


def git_blob(sha: str) -> bytes:
    """Raw bytes of a git blob. cat-file, not show: no smudge/eol filters."""
    return subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "blob", sha],
                          check=True, capture_output=True).stdout


def load_pkl_blob(sha: str) -> dict:
    data = git_blob(sha)
    if not data.startswith(b"\x80"):
        sys.exit(f"blob {sha[:8]} is not a pickle (starts {data[:4]!r}) -- "
                 "it is probably the UTF-8-mangled copy.")
    return pickle.loads(data)


def load_json_blob(sha: str) -> tuple:
    """load_json_localization() takes a path, so stage the blob in a temp dir."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "GS_archive.json"
        tmp.write_bytes(git_blob(sha))
        return load_json_localization(tmp)


def restore_array_dtypes(restored: dict, templates) -> list:
    """JSON has no arrays, so a run reconstructed from the archive comes back
    with lists where a natively pickled Trialsequence holds numpy arrays
    (``trials``, ``conditions``, ``response_errors``). Analysis code indexes
    and broadcasts those, so cast them back.

    Which attributes are arrays is taken from the intact sequences, unioned
    over all of them -- a single template is not enough, e.g. aborted runs
    carry an empty list where finished ones have an ndarray of
    response_errors. Returns the attribute names that were cast."""
    import numpy as np
    array_attrs = {k for t in templates for k, v in vars(t).items()
                   if isinstance(v, np.ndarray)}
    cast = []
    for seq in restored.values():
        for attr in array_attrs:
            val = getattr(seq, attr, None)
            if isinstance(val, list):
                setattr(seq, attr, np.asarray(val))
                cast.append(attr)
    return sorted(set(cast))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the merge without writing anything")
    args = ap.parse_args()

    pkl_path = paths.subject_pkl(SUBJECT_ID)
    json_path = paths.subject_backup(SUBJECT_ID)

    old = load_pkl_blob(PKL_BLOB)
    print(f"07-10 pickle : {len(old['localization'])} runs, "
          f"{len(old['trials'])} trials, highscore {old['highscore']}")

    _, archived = load_json_blob(JSON_BLOB)
    cast = restore_array_dtypes(archived, old["localization"].values())
    print(f"07-28 archive: {len(archived)} runs "
          f"(list -> ndarray: {', '.join(cast) or 'nothing'})")

    # The live pickle: everything recorded after the record was restarted
    # empty, plus any demographics collected since.
    live = {}
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                live = pickle.load(f)
            print(f"live pickle  : {len(live.get('localization') or {})} runs, "
                  f"{len(live.get('trials') or [])} trials, "
                  f"highscore {live.get('highscore') or 0}, "
                  f"demographics {live.get('demographics') or '(none)'}")
        except Exception as e:
            print(f"live pickle  : unreadable ({e}) -- nothing carried over")
    live_loc = live.get("localization") or {}
    demographics = live.get("demographics") or {}

    # 07.07/08.07 first, then 28.07, then the live runs: insertion order stays
    # chronological. The live pickle wins on a collision -- it is the only
    # source that can have been edited since.
    localization = dict(old["localization"])
    from_archive = [k for k in archived if k not in localization]
    for key in from_archive:
        localization[key] = archived[key]
    from_live = list(live_loc)
    for key in from_live:
        localization[key] = live_loc[key]

    # Trials only ever existed in the pickle; the live record starts empty.
    # Re-running the script must not append the 07-10 trials a second time, so
    # skip them if the live list already starts with them.
    old_trials = list(old["trials"])
    live_trials = list(live.get("trials") or [])
    if live_trials[:len(old_trials)] == old_trials:
        print("             (live pickle already carries the 07-10 trials -- "
              "re-run, not appending them again)")
        trials = live_trials
    else:
        trials = old_trials + live_trials

    json_highscore = json.loads(git_blob(JSON_BLOB)).get("highscore") or 0
    highscore = max(int(old.get("highscore") or 0), int(json_highscore),
                    int(live.get("highscore") or 0))
    last_sequence = next(reversed(localization.values())) if localization else None

    data = {
        "id": SUBJECT_ID,
        "localization": localization,
        "trials": trials,
        "last_sequence": last_sequence,
        "highscore": highscore,
        "demographics": demographics,
    }

    print(f"\nmerged       : {len(localization)} runs "
          f"(+{len(from_archive)} from the archive, +{len(from_live)} live), "
          f"{len(trials)} trials, highscore {highscore}")
    print(f"last_sequence: {getattr(last_sequence, 'name', None)}")
    for i, (key, seq) in enumerate(localization.items(), 1):
        src = ("live   " if key in from_live else
               "archive" if key in from_archive else "pickle ")
        print(f"  [{i:>2}] {src}  {key:<48} {getattr(seq, 'n_trials', '?')} trials")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    for path in (pkl_path, json_path):
        pre = path.with_suffix(path.suffix + ".pre-recovery")
        if pre.exists():
            # A second run must not overwrite the pre-recovery copy with the
            # output of the first one -- that is the only untouched original.
            print(f"\nkept existing {pre.name} (not overwritten)")
        elif path.exists():
            shutil.copy2(path, pre)
            print(f"\nsaved {path.name} -> {pre.name}")

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(pkl_path)
    print(f"wrote {pkl_path}")

    # Refresh the JSON archive through Subject so it gets the standard
    # (append-merged) layout including demographics.
    from hrtf_relearning.experiment.misc.Subject import Subject
    s = Subject(SUBJECT_ID)
    s._write_backup()
    print(f"wrote {json_path}")
    s.print_localization()


if __name__ == "__main__":
    main()
