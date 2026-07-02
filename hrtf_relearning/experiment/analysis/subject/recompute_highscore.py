"""
Recompute each subject's highscore directly from recorded trials, rather
than trusting the persisted Subject.highscore field.

highscore is a "best ever" value that only ratchets upward during live
play (Training_AR.py: `if session_total.value > highscore.value: ...`) and
is never re-derived from the trial log afterward. It can outlive the data
that justifies it in a couple of ways:
  - it was inflated by an old bug where session totals didn't reset
    cleanly across game boundaries, before explicit game_idx/trial_in_game
    bookkeeping was added (commit "Consolidate and harden training
    trajectory analysis"); or
  - it survived a merge: merge_subjects.py takes max(highscore) across the
    merged files, so a stale inflated value from one side carries forward
    even if the merged trial data no longer supports it.

This groups each subject's trials by (session_id, game_idx), sums 'score'
per group, and takes the max as the ground-truth best single game. Only
trials with game_idx bookkeeping (added mid-June 2026) are considered;
subjects with no such trials are reported but left untouched, since game
boundaries can't be reliably recovered for older data.

Usage:
    python -m hrtf_relearning.experiment.analysis.subject.recompute_highscore           # report only
    python -m hrtf_relearning.experiment.analysis.subject.recompute_highscore --apply   # also fix + write
"""
import argparse
from collections import defaultdict

from hrtf_relearning.experiment.misc.Subject import Subject, results_dir


def true_best_game(trials):
    """Ground-truth best single-game total from trial records, or None if
    the trials have no game_idx bookkeeping to group by."""
    by_game = defaultdict(int)
    has_bookkeeping = False
    for t in trials:
        if not isinstance(t, dict) or "game_idx" not in t:
            continue
        has_bookkeeping = True
        key = (t.get("session_id"), t.get("game_idx"))
        by_game[key] += int(t.get("score", 0) or 0)
    if not has_bookkeeping:
        return None
    return max(by_game.values()) if by_game else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write corrected highscores (default: report only)")
    args = parser.parse_args()

    pkls = sorted(results_dir.glob("*.pkl"))
    if not pkls:
        print(f"No subject .pkl files found in {results_dir}")
        return

    for p in pkls:
        subject_id = p.stem
        subject = Subject(subject_id)
        true_high = true_best_game(subject.trials)
        stored = int(subject.highscore or 0)

        if true_high is None:
            print(f"{subject_id}: stored highscore={stored}, no game_idx bookkeeping in trials — skipped")
            continue

        if true_high == stored:
            print(f"{subject_id}: OK (highscore={stored})")
            continue

        flag = "TOO HIGH" if stored > true_high else "TOO LOW"
        print(f"{subject_id}: stored highscore={stored}, true best single game={true_high}  <-- {flag}")
        if args.apply:
            subject.highscore = true_high
            subject.write()  # rewrites .pkl + refreshes the JSON backup used by the scoreboard
            print(f"  -> corrected to {true_high}")


if __name__ == "__main__":
    main()
