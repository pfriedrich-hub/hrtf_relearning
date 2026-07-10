"""
Standalone visual preview of the training GameWindow (score display,
coin/sparkle effects, and the cross-participant scoreboard) without any of
the training hardware: no TDT DSPs, no head tracker, no pybinsim audio
backend, no slab. Useful for quickly iterating on GUI/layout changes.

Runs the real GameWindow + UIShared in a single process and drives
ui_state through a fake session on a QTimer, so a full cycle
(start prompt -> scoring trials -> game over -> scoreboard reveal ->
play-again prompt) loops automatically every ~15 seconds. Press Enter/
Space or click the button same as in a real session.

By default the scoreboard is populated with fake demo rows in an isolated
temp directory, so this never touches real participant data in
data/results/<id>/<id>.json. Pass --live to read the actual current
scoreboard.

The scoreboard only appears once the current subject_id has an entry in the
backup data (day-1 beginners just see their own score instead, so they're
not immediately compared to established participants). Pass --new-player
to preview that path.

Usage:
    python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview
    python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview CA
    python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview CA --live
    python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview --new-player

To A/B test the retro pixel font (see game_ui._PIXEL_FONT_CHOICES), set
HRTF_PIXEL_FONT before launching, e.g.:
    HRTF_PIXEL_FONT=dotgothic16 python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview
    HRTF_PIXEL_FONT=handjet     python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview
    HRTF_PIXEL_FONT=vt323       python -m hrtf_relearning.experiment.training.training_helpers.game_ui_preview
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import tempfile
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from hrtf_relearning.experiment.training.training_helpers import game_ui

FAKE_GAME_TIME = 12.0    # seconds per fake "game" (real sessions use settings['game_time'], usually 90s)
FAKE_TRIAL_TIME = 1.6    # seconds between fake scoring events


def _make_fake_backup_dir(subject_id: str, include_current_player: bool = True) -> Path:
    """Populate an isolated temp dir with fake per-participant JSON backups,
    mirroring the real RESULTS_DIR/<id>/<id>.json layout, so the scoreboard has
    something to show without touching real data/results/.

    Deliberately uses IDs that can't collide with real subject codes (2-3
    real-looking letters like "JS"/"CA") so fake demo numbers are never
    mistaken for real participant data. Pass --live to see the real
    scoreboard instead.

    include_current_player=False leaves subject_id out of the fake data
    entirely, previewing the "new participant, not listed yet" path (see
    --new-player) — GameWindow should then just show the plain score with
    no scoreboard.

    When included, the current player's fake highscore is deliberately set
    above every demo row (not just given a mid-range random value), so they
    always land in the visible top of the table regardless of whatever
    score they end up with in a given dummy run — no relying on the "..."
    fallback for an out-of-range row.
    """
    tmp = Path(tempfile.mkdtemp(prefix="hrtf_scoreboard_preview_"))
    fake_rows = {"DEMO-A": 142, "DEMO-B": 165, "DEMO-C": 88, "DEMO-D": 97, "DEMO-E": 58, "DEMO-F": 133}
    if include_current_player and subject_id not in fake_rows:
        fake_rows[subject_id] = max(fake_rows.values()) + random.randint(10, 50)
    for sid, score in fake_rows.items():
        payload = {"id": sid, "highscore": score, "localization": {}}
        sub = tmp / sid
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp


class FakeSessionDriver(QtCore.QObject):
    """Mimics play_session()'s ui_state machine (Training.py) on a QTimer,
    without any hardware/audio workers, so GameWindow can be watched
    end-to-end on a loop.
    """
    def __init__(self, shared: game_ui.UIShared, parent=None):
        super().__init__(parent)
        self.shared = shared
        self._trial_timer = QtCore.QTimer(self)
        self._trial_timer.setInterval(int(FAKE_TRIAL_TIME * 1000))
        self._trial_timer.timeout.connect(self._score_trial)

        # Single persistent poller for "wait until GameWindow sets
        # enter_pressed"; which phase runs next is just data (_next_phase),
        # so we never need to dynamically connect/disconnect signals.
        self._enter_poll = QtCore.QTimer(self)
        self._enter_poll.setInterval(50)
        self._enter_poll.timeout.connect(self._check_enter)
        self._next_phase = None

        self._game_elapsed = 0.0
        self._start_prompt()

    def _wait_for_enter(self, next_phase):
        self._next_phase = next_phase
        self.shared.enter_pressed.value = 0
        self._enter_poll.start()

    def _check_enter(self):
        if self.shared.enter_pressed.value:
            self.shared.enter_pressed.value = 0
            self._enter_poll.stop()
            phase, self._next_phase = self._next_phase, None
            if phase is not None:
                phase()

    def _start_prompt(self):
        self.shared.session_total.value = 0
        self.shared.last_goal_points.value = 0
        self.shared.game_time_left.value = FAKE_GAME_TIME
        self.shared.ui_state.value = 1  # waiting to start
        self.shared.enter_pressed.value = 0
        self._game_elapsed = 0.0
        self._wait_for_enter(self._run_game)

    def _run_game(self):
        self.shared.ui_state.value = 2  # running
        self._trial_timer.start()

    def _score_trial(self):
        points = random.choice([0, 1, 1, 2])  # bias towards scoring, occasional miss
        if points:
            self.shared.last_goal_points.value = points
            self.shared.session_total.value = int(self.shared.session_total.value) + points
        self._game_elapsed += FAKE_TRIAL_TIME
        self.shared.game_time_left.value = max(0.0, FAKE_GAME_TIME - self._game_elapsed)
        if self._game_elapsed >= FAKE_GAME_TIME:
            self._trial_timer.stop()
            self._end_game()

    def _end_game(self):
        total = int(self.shared.session_total.value)
        if total > int(self.shared.highscore.value):
            self.shared.highscore.value = total
        self.shared.ui_state.value = 3  # session over -> GameWindow handles the reveal delay
        self._wait_for_enter(self._start_prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject_id", nargs="?", default="PREVIEW", help="Subject id to preview as (default: PREVIEW)")
    parser.add_argument("--live", action="store_true",
                        help="Read the real data/results scoreboard instead of fake demo data")
    parser.add_argument("--new-player", action="store_true",
                        help="Preview the day-1 path: subject_id has no entry yet, so no scoreboard is shown")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    shared = game_ui.UIShared(
        current_score=mp.Value("i", 0),
        game_time_left=mp.Value("f", FAKE_GAME_TIME),
        trial_time_left=mp.Value("f", 0.0),
        last_goal_points=mp.Value("i", 0),
        session_total=mp.Value("i", 0),
        enter_pressed=mp.Value("i", 0),
        ui_state=mp.Value("i", 0),
        highscore=mp.Value("i", 0),
    )

    if args.live:
        backup_dir = None
    else:
        backup_dir = _make_fake_backup_dir(args.subject_id, include_current_player=not args.new_player)
    window = game_ui.GameWindow(shared, subject_id=args.subject_id, backup_dir=backup_dir)
    window.show()

    driver = FakeSessionDriver(shared)  # noqa: F841 - kept alive by reference for the app's lifetime

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
