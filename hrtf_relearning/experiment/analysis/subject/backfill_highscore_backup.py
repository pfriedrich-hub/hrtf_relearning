"""
One-off migration: refresh data/results/<ID>/<ID>.json for every existing
subject so it includes the 'highscore' field.

Subject._write_backup() only started writing 'highscore' recently (added so
game_ui.read_scoreboard() can render the cross-participant scoreboard
without unpickling other subjects' full .pkl files, which can contain
slab objects). Subjects who haven't played a new trial since that change
still have an old backup missing the field, so they won't show up on the
scoreboard until this runs (or they play again).

This just calls Subject(id).write() for every subject in data/results/*.pkl,
which rewrites the .pkl (unchanged) and regenerates the JSON backup with
the current highscore. Safe to re-run any time.

Run once, locally — needs slab installed (same environment as training,
since Subject.last_sequence can be a slab.Trialsequence object):
    python -m hrtf_relearning.experiment.analysis.subject.backfill_highscore_backup
"""
from hrtf_relearning.experiment.misc.Subject import Subject, results_dir
from hrtf_relearning.utils import paths


def main():
    pkls = paths.subject_pkls()
    if not pkls:
        print(f"No subject .pkl files found in {results_dir}")
        return

    print(f"Refreshing {len(pkls)} subject backup(s) in {results_dir} ...")
    for p in pkls:
        subject_id = p.stem
        try:
            subject = Subject(subject_id)
            subject.write()
            print(f"  {subject_id}: highscore={subject.highscore}")
        except Exception as e:
            print(f"  {subject_id}: FAILED ({e})")


if __name__ == "__main__":
    main()
