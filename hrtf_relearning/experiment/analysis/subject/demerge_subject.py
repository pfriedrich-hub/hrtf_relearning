"""
De-merge a subject whose data was accidentally combined with an earlier
participant of the same initials.

Case: this week's JS (cue=synth, left, started 23.06) was written into the
existing JS.pkl that already held last week's JS (16-19.06). Git still has the
clean pre-merge file (commit a4b9360, 19.06) = the OLD participant complete,
including training trials. This week's runs were appended on top, so:

    OLD participant  = the git pre-merge pickle            -> moved to pilot/JS.pkl
    THIS week's JS   = current pickle minus the old entries -> stays results/JS.pkl

Localization split is by key (old keys come straight from the pre-merge file);
trials split assumes chronological append (current.trials == old.trials + new),
which is how the merge happened.

USAGE (run in your normal env, where slab + hrtf_relearning import):
    python demerge_subject.py            # DRY RUN: prints what it would do
    python demerge_subject.py --apply    # actually back up + write the files
"""

import pickle
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# allow running directly even if the package isn't pip-installed in this env:
# walk up to the repo root (the dir that contains the hrtf_relearning package)
for _p in Path(__file__).resolve().parents:
    if (_p / "hrtf_relearning" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
import hrtf_relearning as hr

# ---------------------------------------------------------------- config
SUBJECT_ID        = "JS"
GIT_PREMERGE_REV  = "a4b9360"           # 19.06 commit: old participant only
REPO_ROOT         = hr.PATH.parent      # parent of the hrtf_relearning package
RESULTS_DIR       = hr.PATH / "data" / "results"
CURRENT_PKL       = RESULTS_DIR / f"{SUBJECT_ID}.pkl"
OLD_DEST          = RESULTS_DIR / "pilot" / f"{SUBJECT_ID}.pkl"   # archive old participant here
REL_PKL           = CURRENT_PKL.relative_to(REPO_ROOT).as_posix()  # path git knows

APPLY = "--apply" in sys.argv


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _git_premerge_bytes():
    """Raw bytes of the pre-merge pickle from git (exact old-participant file)."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{GIT_PREMERGE_REV}:{REL_PKL}"]
    )


def main():
    print(f"{'APPLY' if APPLY else 'DRY RUN'} — de-merging {SUBJECT_ID}\n")

    current = _load_pkl(CURRENT_PKL)
    old_bytes = _git_premerge_bytes()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        tf.write(old_bytes)
        old_tmp = Path(tf.name)
    old = _load_pkl(old_tmp)

    cur_loc, old_loc = current.get("localization", {}), old.get("localization", {})
    cur_tr,  old_tr  = current.get("trials", []),       old.get("trials", [])

    old_keys = set(old_loc)
    new_keys = [k for k in cur_loc if k not in old_keys]          # preserves order
    missing  = old_keys - set(cur_loc)

    # --- sanity checks ---
    print(f"current loc runs : {len(cur_loc)}")
    print(f"old (pre-merge)  : {len(old_loc)}")
    print(f"-> stays with JS : {len(new_keys)} runs")
    for k in new_keys:
        print(f"      {k}")
    if missing:
        print(f"\n!! {len(missing)} pre-merge keys NOT in current file (unexpected): {missing}")
    print(f"\ntrials: current={len(cur_tr)}  old={len(old_tr)}  -> new JS gets {max(0,len(cur_tr)-len(old_tr))}")
    prefix_ok = len(cur_tr) >= len(old_tr)
    if not prefix_ok:
        print("!! current has FEWER trials than pre-merge — NOT a clean append; aborting.")
        return

    # --- build this-week's JS (old entries removed) ---
    new_subject = {
        "id": SUBJECT_ID,
        "localization": {k: cur_loc[k] for k in new_keys},
        "trials": cur_tr[len(old_tr):],                  # appended tail only
        "last_sequence": cur_loc[new_keys[-1]] if new_keys else None,
        "highscore": 0 if len(cur_tr) == len(old_tr) else current.get("highscore", 0),
    }

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply to:")
        print(f"  1. back up {CURRENT_PKL.name} -> {SUBJECT_ID}_MERGED_backup_<ts>.pkl")
        print(f"  2. write OLD participant  -> {OLD_DEST}")
        print(f"  3. overwrite             -> {CURRENT_PKL}  (this week's JS only)")
        old_tmp.unlink(missing_ok=True)
        return

    # --- apply ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = RESULTS_DIR / f"{SUBJECT_ID}_MERGED_backup_{ts}.pkl"
    shutil.copy2(CURRENT_PKL, backup)
    print(f"backed up merged file -> {backup}")

    OLD_DEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OLD_DEST, "wb") as f:          # exact pre-merge bytes = old participant
        f.write(old_bytes)
    print(f"wrote OLD participant -> {OLD_DEST}")

    with open(CURRENT_PKL, "wb") as f:
        pickle.dump(new_subject, f)
    print(f"wrote this-week JS    -> {CURRENT_PKL}")

    old_tmp.unlink(missing_ok=True)
    print("\nDone. (JSON backups refresh next time each subject is saved via Subject.write().)")


if __name__ == "__main__":
    main()
