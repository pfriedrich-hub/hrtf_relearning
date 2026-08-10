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
    e4b0661  GS/GS.pkl    intact pickle, 2026-07-10   12 runs, 10 trials,
                                                      highscore 2
    ce9e607  GS/GS.json   valid JSON archive           3 runs (28.07),
                                                      highscore 12
    working  GS/GS.pkl    empty record written today  demographics only

The 2026-07-28 JSON is *not* a superset: it holds only the three 28.07 runs,
so restoring from it alone (restore_from_json.py) would silently drop the
twelve 07.07/08.07 runs. This script merges all three sources instead:

    localization  12 runs from the 07-10 pickle + 3 runs from the 07-28 JSON
    trials        10, from the 07-10 pickle (07-10..07-28 trials are lost)
    highscore     max of the two sources
    demographics  from the current pickle, if present
    last_sequence repointed at the newest surviving run

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
    print(f"07-28 archive: {len(archived)} runs")

    # Demographics were collected today into the empty replacement pickle;
    # keep them if they are there.
    demographics = {}
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                demographics = pickle.load(f).get("demographics") or {}
            print(f"current pkl  : demographics {demographics or '(none)'}")
        except Exception as e:
            print(f"current pkl  : unreadable ({e}) -- no demographics carried over")

    # 07.07/08.07 first, then 28.07: insertion order stays chronological.
    localization = dict(old["localization"])
    added = [k for k in archived if k not in localization]
    for key in added:
        localization[key] = archived[key]

    json_highscore = json.loads(git_blob(JSON_BLOB)).get("highscore") or 0
    highscore = max(int(old.get("highscore") or 0), int(json_highscore))
    last_sequence = next(reversed(localization.values())) if localization else None

    data = {
        "id": SUBJECT_ID,
        "localization": localization,
        "trials": old["trials"],
        "last_sequence": last_sequence,
        "highscore": highscore,
        "demographics": demographics,
    }

    print(f"\nmerged       : {len(localization)} runs "
          f"(+{len(added)} from the archive), {len(data['trials'])} trials, "
          f"highscore {highscore}")
    print(f"last_sequence: {getattr(last_sequence, 'name', None)}")
    for i, (key, seq) in enumerate(localization.items(), 1):
        src = "archive" if key in added else "pickle "
        print(f"  [{i:>2}] {src}  {key:<48} {getattr(seq, 'n_trials', '?')} trials")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    for path in (pkl_path, json_path):
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".pre-recovery"))
            print(f"\nsaved {path.name} -> {path.name}.pre-recovery")

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
