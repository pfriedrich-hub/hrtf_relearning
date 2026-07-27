"""Interactive Subject editor.

Lists localization runs for a subject and lets you remove selected entries.
Thin wrapper over Subject.remove_localization_by_index (see
experiment/misc/Subject.py) — that class holds the reusable editing API for
use from scripts or `# %%` cells; this is just the interactive front end.

Usage:
    python edit_subject.py SUBJECT_ID
    python edit_subject.py AH
    python edit_subject.py AH --prune   # drop unfinished + duplicate runs
"""

import sys
from hrtf_relearning.experiment.misc.Subject import Subject
from hrtf_relearning.utils import paths


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: edit_subject.py SUBJECT_ID [--prune]")

    subject_id = sys.argv[1]
    if not paths.subject_pkl(subject_id).exists():
        sys.exit(f"No file found: {paths.subject_pkl(subject_id)}")

    subject = Subject(subject_id)
    if not subject.localization:
        print(f"{subject_id}: no localization entries.")
        return

    subject.print_localization()

    if "--prune" in sys.argv:
        removed = subject.prune_localization(write=False)
        if not removed:
            print("\nNothing redundant to prune.")
            return
        print(f"\nWould prune {len(removed)} redundant run(s):")
        for k in removed:
            print(f"  - {k}")
    else:
        print("\nEnter numbers to remove (e.g. 1 3 5), or press Enter to cancel:")
        raw = input("> ").strip()
        if not raw:
            print("Cancelled.")
            return
        try:
            indices = [int(x) for x in raw.split()]
        except ValueError:
            sys.exit("Invalid input.")
        keys = list(subject.localization.keys())
        removed = [keys[i - 1] for i in indices if 1 <= i <= len(keys)]
        if not removed:
            print("Nothing to remove.")
            return
        print(f"\nWill remove {len(removed)} run(s):")
        for k in removed:
            print(f"  - {k}")

    print("\nConfirm? [y/N]")
    if input("> ").strip().lower() != "y":
        print("Cancelled.")
        return

    subject.remove_localization(removed)  # backs up, fixes last_sequence, writes
    print(f"\nDone. {len(subject.localization)} run(s) remaining. "
          f"Backup → {paths.subject_pkl(subject_id).with_suffix('.pkl.bak').name}")


if __name__ == "__main__":
    main()
