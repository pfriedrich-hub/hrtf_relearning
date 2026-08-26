from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import json
import logging
import math
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from PyQt5 import QtSvg
    HAS_QTSVG = True
except Exception:
    HAS_QTSVG = False
import hrtf_relearning
ROOT = hrtf_relearning.PATH

# Subject.py is a lightweight, dependency-free module (no slab/pybinsim
# imports), so it's safe to import in this UI subprocess even though the
# rest of the training stack (slab, pybinsim, ...) may not be loaded here.
from hrtf_relearning.utils.paths import RESULTS_DIR as SUBJECT_RESULTS_DIR
from hrtf_relearning.utils import paths

# ────────────────────────────────────────────────────────────────
# Shared structure for multiprocessing.Values
# ────────────────────────────────────────────────────────────────
@dataclass
class UIShared:
    current_score: Any
    game_time_left: Any
    trial_time_left: Any
    last_goal_points: Any
    session_total: Any
    enter_pressed: Any
    ui_state: Any    # 0=idle, 1=waiting to start trial, 2=running, 3=session over/prompt
    highscore: Any
    quit_pressed: Any = None  # UI sets to 1 on ESC at the game-over prompt (optional)


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def find_coin_path() -> Optional[Path]:
    for p in (
        paths.DOCUMENTATION_DIR / "ui" / "mario-coin.svg",  # actual current location
        paths.IMG_DIR / "ui" / "mario-coin.svg",
        paths.UI_DIR / "mario-coin.svg",
        paths.UI_DIR / "mario-coin.png",
    ):
        if p.exists():
            return p
    return None


def read_scoreboard(results_dir: Path) -> List[Tuple[str, int]]:
    """Read (subject_id, highscore) pairs across participants.

    Sources from the small per-subject JSON backups Subject.write() already
    maintains (data/results/<ID>/<ID>.json), rather than the .pkl files
    directly — the pickles can hold slab objects (e.g. last_sequence) that
    aren't safely unpicklable from this dependency-light UI process, while
    the JSON backups are plain, cheap, and rewritten on every trial.

    Scans each subject folder for its <ID>/<ID>.json, so non-subject folders
    (the pilot archive results/pilot/, shared results/plot/, results/archive/)
    are ignored -- both by an explicit skip list and because they have no
    <folder>/<folder>.json. Pilot participants live flat under results/pilot/
    and are therefore never scoreboard entries. Unreadable/malformed files are
    skipped rather than raised, since these are written independently by other
    subjects' sessions.
    """
    rows: dict[str, int] = {}
    if not results_dir.exists():
        return []
    # non-participant folders under RESULTS_DIR -- never scoreboard entries
    skip_dirs = {"pilot", "plot", "archive", "__pycache__"}
    for sub in sorted(results_dir.iterdir()):
        if sub.name in skip_dirs:
            continue
        p = sub / f"{sub.name}.json"
        if not sub.is_dir() or not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = data.get("id")
        score = data.get("highscore")
        if not sid or score is None:
            continue
        try:
            score = int(score)
        except (TypeError, ValueError):
            continue
        if sid not in rows or score > rows[sid]:
            rows[sid] = score
    return rank_scores(rows)


def rank_scores(rows: dict) -> List[Tuple[str, int]]:
    """(id, score) sorted best first, ties broken by id.

    The tie-break is not cosmetic: without it two participants on the same
    score can swap places between one game and the next, which on a board
    the same person sees repeatedly looks like something happened when
    nothing did.
    """
    return sorted(rows.items(), key=lambda kv: (-kv[1], kv[0]))


# ────────────────────────────────────────────────────────────────
# Who appears on the board
# ────────────────────────────────────────────────────────────────
# The board is a WINDOW around the current participant, not the global top
# N. They see the few people just ahead of them -- close enough to chase --
# and a couple just behind, so it is clear they are not last. A leaderboard
# headed by a score far out of reach discourages more than it motivates.
#
# The window leans upwards: more above than below.
BOARD_ABOVE = 4
BOARD_BELOW = 2

# The chosen peers are frozen and persisted here, under the participant's
# own results folder (results/<id>/). Delete the file to re-pick them.
PEER_SET_FILENAME = "scoreboard_peers.json"


def peer_set_path(results_dir: Path, subject_id: str) -> Path:
    return results_dir / subject_id / PEER_SET_FILENAME


def load_peer_set(results_dir: Path, subject_id: str) -> Optional[List[str]]:
    """The peer ids already frozen for this participant, or None if the set
    has not been chosen yet. Unreadable/malformed files count as 'not
    chosen' rather than raising -- a scoreboard is never worth crashing the
    UI process over."""
    if not subject_id:
        return None
    try:
        data = json.loads(peer_set_path(results_dir, subject_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    peers = data.get("peers")
    if not isinstance(peers, list):
        return None
    return [x for x in peers if isinstance(x, str)]


def save_peer_set(results_dir: Path, subject_id: str, peers: List[str]) -> None:
    """Freeze this participant's peer set to disk.

    Written once, the first time they are shown a board, and read back on
    every later game and session. This is what keeps the cast of the
    scoreboard fixed: as the participant improves they climb PAST the same
    names rather than having unfamiliar ones appear above them, which would
    advertise that the board is picked around them.
    """
    if not subject_id:
        return
    path = peer_set_path(results_dir, subject_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id": subject_id,
                        "peers": list(peers),
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logging.exception("Could not persist the scoreboard peer set for %s; "
                          "the board may show different names next game.", subject_id)


def choose_peer_set(ranked: List[Tuple[str, int]], subject_id: str,
                    n_above: int = BOARD_ABOVE, n_below: int = BOARD_BELOW) -> List[str]:
    """Pick the participants either side of `subject_id` in `ranked`.

    Takes the n_above immediately above and n_below immediately below. If
    one side is short (the participant is near the top or the bottom of the
    field) the other side makes up the difference, so the board is always
    full where there are enough participants to fill it.
    """
    ids = [sid for sid, _ in ranked]
    if subject_id not in ids:
        return []
    i = ids.index(subject_id)
    above, below = ids[:i], ids[i + 1:]
    picked_above = above[-n_above:] if n_above else []
    picked_below = below[:n_below]
    short = n_above - len(picked_above)
    if short:
        picked_below = below[: n_below + short]
    short = n_below - len(picked_below)
    if short:
        picked_above = above[-(n_above + short):]
    return list(picked_above) + list(picked_below)


def board_rows(ranked: List[Tuple[str, int]], subject_id: str,
               peers: List[str]) -> List[Tuple[str, int]]:
    """The participant plus their frozen peers, best score first.

    Peers who are no longer in the data (folder removed, say) just drop
    out. Scores are always the CURRENT ones -- only the cast is frozen, not
    the standings.
    """
    scores = dict(ranked)
    if subject_id not in scores:
        return []
    keep = {subject_id: scores[subject_id]}
    for pid in peers:
        if pid in scores and pid != subject_id:
            keep[pid] = scores[pid]
    return rank_scores(keep)


_PIXEL_FONT_FAMILY: Optional[str] = None

# Candidate pixel/retro fonts bundled under data/ui/fonts/, selectable via
# the HRTF_PIXEL_FONT env var (see _ensure_pixel_font_loaded) so different
# options can be A/B tested without editing code:
#   HRTF_PIXEL_FONT=jersey10     python -m ...game_ui_preview     (default)
#   HRTF_PIXEL_FONT=dotgothic16  python -m ...game_ui_preview
#   HRTF_PIXEL_FONT=handjet      python -m ...game_ui_preview
#   HRTF_PIXEL_FONT=vt323        python -m ...game_ui_preview
#
# Press Start 2P and Pixelify Sans were tried and dropped (2P's numerals
# turn into oversized blocks at these sizes; Pixelify Sans just didn't
# read well). VT323 fixed the letters but its digits still looked off, so
# jersey10 — a font literally designed around bold, clean sports-jersey
# numerals — is the new default. dotgothic16 (dot-matrix terminal) and
# handjet (built-in "digital scoreboard" numeral style, though it needs
# the right OpenType stylistic set to show — may render as a plain face
# without it) are there to compare against.
_PIXEL_FONT_CHOICES = {
    "jersey10": "Jersey10-Regular.ttf",
    "vt323": "VT323-Regular.ttf",
    "dotgothic16": "DotGothic16-Regular.ttf",
    "handjet": "Handjet.ttf",
}
_DEFAULT_PIXEL_FONT_CHOICE = "jersey10"


def _ensure_pixel_font_loaded() -> Optional[str]:
    """Load the selected pixel font (data/ui/fonts/, see _PIXEL_FONT_CHOICES)
    once per process via QFontDatabase, so the retro look renders the same
    on any machine regardless of what's installed system-wide.

    Returns the resolved family name, or None if the file is missing/
    unloadable (caller falls back to a generic monospace stack then).
    """
    global _PIXEL_FONT_FAMILY
    if _PIXEL_FONT_FAMILY is not None:
        return _PIXEL_FONT_FAMILY or None
    _PIXEL_FONT_FAMILY = ""
    choice = os.environ.get("HRTF_PIXEL_FONT", _DEFAULT_PIXEL_FONT_CHOICE).strip().lower()
    filename = _PIXEL_FONT_CHOICES.get(choice, _PIXEL_FONT_CHOICES[_DEFAULT_PIXEL_FONT_CHOICE])
    path = paths.UI_DIR / "fonts" / filename
    if path.exists():
        try:
            font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
            families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _PIXEL_FONT_FAMILY = families[0]
        except Exception:
            pass
    return _PIXEL_FONT_FAMILY or None


def pixel_font_family() -> str:
    """CSS font-family value for retro/pixel-styled UI text."""
    fam = _ensure_pixel_font_loaded()
    if fam:
        return f"'{fam}'"
    return "'VT323', 'Courier New', monospace"


class CoinGraphic:
    def __init__(self, path: Optional[Path]):
        self.renderer = None
        self.pixmap = None
        if path and path.exists():
            if HAS_QTSVG and path.suffix.lower() == ".svg":
                r = QtSvg.QSvgRenderer(str(path))
                if r.isValid():
                    self.renderer = r
            else:
                pm = QtGui.QPixmap(str(path))
                if not pm.isNull():
                    self.pixmap = pm

    def valid(self) -> bool:
        return (self.renderer is not None) or (self.pixmap is not None and not self.pixmap.isNull())

    def paint(self, painter: QtGui.QPainter, rect: QtCore.QRect):
        if self.renderer is not None:
            self.renderer.render(painter, QtCore.QRectF(rect))
        elif self.pixmap is not None:
            painter.drawPixmap(rect, self.pixmap)


class CoinPopGraphic(QtWidgets.QWidget):
    """Mario-style coin: appears above the score, jumps higher, spins around
    its vertical axis the whole time it's visible, lingers, then vanishes.

    The "spin" is the classic 2D trick for a flat sprite: since we only have
    a single flat image (no separate edge-on frames), rotation around the
    vertical axis is faked by animating the horizontal scale between full
    width and a thin sliver (|cos(angle)|) while height stays fixed — that's
    exactly what a coin flipping face-on/edge-on/face-on looks like.
    """
    def __init__(self, anchor_label: QtWidgets.QLabel, parent: QtWidgets.QWidget, coin: CoinGraphic):
        super().__init__(parent)
        self.anchor = anchor_label
        self.coin = coin
        self._y_offset = 0
        self._spin_deg = 0.0
        self._visible = False
        self.start_offset = -70
        self.jump_height = 160
        self.jump_duration = 600
        self.linger_time = 300
        self.move = QtCore.QPropertyAnimation(self, b"yOffset", self)
        self.move.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.spin = QtCore.QPropertyAnimation(self, b"spinDeg", self)
        self.spin.setEasingCurve(QtCore.QEasingCurve.Linear)
        # One reusable, restartable timer for scheduling _vanish — not
        # QTimer.singleShot(), which creates a new independent timer on
        # every call. With a double coin (score=2, two _one_pop calls
        # 300ms apart) that meant the FIRST pop's vanish timer could still
        # fire on schedule and hide/cut off the SECOND pop's still-running
        # spin. Restarting one timer means only the most recent pop's
        # vanish ever fires — earlier ones get superseded, same as the
        # spin/jump animations already do via .stop() + restart.
        self._vanish_timer = QtCore.QTimer(self)
        self._vanish_timer.setSingleShot(True)
        self._vanish_timer.timeout.connect(self._vanish)

    def getYOffset(self) -> int: return int(self._y_offset)
    def setYOffset(self, v: int) -> None: self._y_offset = int(v); self.update()
    yOffset = QtCore.pyqtProperty(int, fget=getYOffset, fset=setYOffset)

    def getSpinDeg(self) -> float: return float(self._spin_deg)
    def setSpinDeg(self, v: float) -> None: self._spin_deg = float(v); self.update()
    spinDeg = QtCore.pyqtProperty(float, fget=getSpinDeg, fset=setSpinDeg)

    def pop(self, count: int, on_pop: Optional[callable] = None):
        if count >= 2:
            QtCore.QTimer.singleShot(0,   lambda: self._one_pop(on_pop))
            QtCore.QTimer.singleShot(300, lambda: self._one_pop(on_pop))
        elif count == 1:
            self._one_pop(on_pop)

    def _one_pop(self, on_pop: Optional[callable]):
        if not self.coin.valid() or self.anchor is None:
            return
        if callable(on_pop):
            on_pop()
        self._visible = True
        self._y_offset = self.start_offset
        self._spin_deg = 0.0
        self.show(); self.raise_()
        self.move.stop()
        self.move.setDuration(self.jump_duration)
        self.move.setStartValue(self.start_offset)
        self.move.setEndValue(self.start_offset - self.jump_height)
        self.move.start()
        total_visible = self.jump_duration + self.linger_time
        self.spin.stop()
        self.spin.setDuration(total_visible)
        self.spin.setStartValue(0.0)
        # paintEvent uses |cos(spinDeg)|, which repeats every 180 deg (not
        # 360) — face-on -> edge-on -> face-on is a full visual "flip" at
        # only half a rotation. So 1.5 visible flips = 1.5 * 180, not 1.5 * 360.
        self.spin.setEndValue(1.5 * 180.0)
        self.spin.start()
        self._vanish_timer.start(total_visible)

    def _vanish(self):
        self._visible = False
        self.hide()
        self._y_offset = 0
        self._spin_deg = 0.0
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        if not self._visible or not self.coin.valid():
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        c = self.anchor.mapTo(self.parentWidget(), self.anchor.rect().center())
        size = max(48, int(self.anchor.height() * 0.9))
        half = size // 2
        # Keep a thin sliver visible at the edge-on point rather than
        # scaling to zero width, so the flip reads as "coin", not a blink.
        scale_x = max(0.12, abs(math.cos(math.radians(self._spin_deg))))
        p.save()
        p.translate(c.x(), c.y() + self._y_offset)
        p.scale(scale_x, 1.0)
        rect = QtCore.QRect(-half, -half, size, size)
        self.coin.paint(p, rect)
        p.restore()


class SparkleBurst(QtWidgets.QWidget):
    """Short starburst around the score; used ~80 ms after SFX start for extra 'ding'."""
    def __init__(self, anchor: QtWidgets.QLabel, parent: QtWidgets.QWidget, duration_ms: int = 140):
        super().__init__(parent)
        self.anchor = anchor
        self._opacity = 0.0
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.anim = QtCore.QPropertyAnimation(self, b"opacity", self)
        self.anim.setDuration(duration_ms)
        self.anim.setStartValue(0.0)
        self.anim.setKeyValueAt(0.30, 1.0)
        self.anim.setEndValue(0.0)

    def getOpacity(self): return float(self._opacity)
    def setOpacity(self, v): self._opacity = float(v); self.update()
    opacity = QtCore.pyqtProperty(float, fget=getOpacity, fset=setOpacity)

    def trigger(self):
        self.anim.stop()
        self.setOpacity(0.0)
        self.show(); self.raise_()
        self.anim.start()

    def paintEvent(self, e):
        if self._opacity <= 0:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setOpacity(self._opacity)
        c = self.anchor.mapTo(self.parentWidget(), self.anchor.rect().center())
        base = max(40, int(self.anchor.height() * 0.55))
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setWidth(4)
        p.setPen(pen)
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            r = base if angle_deg % 90 == 0 else int(base * 0.7)
            dx = int(r * math.cos(math.radians(angle_deg)))
            dy = int(r * math.sin(math.radians(angle_deg)))
            p.drawLine(c.x(), c.y(), c.x() + dx, c.y() + dy)


# Palette matching the rest of the training GUI (sky-blue window
# background, dark-blue captions) instead of a standalone card/frame:
# names/scores in white directly on the window background, headers (title,
# subtitle) in the same dark blue used for captions elsewhere.
SB_HEADER = "#083c74"       # matches the caption color elsewhere ("HIGH SCORES" title)
SB_ROW = "#ffffff"          # names/scores
SB_HIGHLIGHT = "#ffe066"    # the current participant's row


class ScoreboardRow(QtWidgets.QWidget):
    """One 'RANK  ID .......... SCORE' line.

    Every dimension is expressed relative to `scale` so the whole table can
    be shrunk to fit the available screen height (see ScoreboardPanel /
    GameWindow._fit_scoreboard). The row height is fixed rather than left to
    the font metrics, which keeps the panel's total height predictable and
    therefore fittable in a single measure-then-scale pass.
    """
    BASE_HEIGHT = 40

    def __init__(self, subj_id: str, score: int, rank: int, highlight: bool,
                 scale: float = 1.0, parent=None):
        super().__init__(parent)

        def px(v: float) -> int:
            return max(1, int(round(v * scale)))

        self.setFixedHeight(px(self.BASE_HEIGHT))
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(px(10), 0, px(10), 0)
        # No uniform spacing here — the dotted leader needs to butt right up
        # against the score digits with zero gap, so spacing is added
        # explicitly only where it's wanted (rank->id, id->dots).
        row.setSpacing(0)
        pf = pixel_font_family()
        color = SB_HIGHLIGHT if highlight else SB_ROW
        size = px(30) if highlight else px(26)

        lbl_rank = QtWidgets.QLabel(f"{rank:02d}")
        lbl_rank.setFixedWidth(px(56))
        lbl_rank.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        lbl_id = QtWidgets.QLabel(subj_id.upper())
        lbl_id.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        dots = QtWidgets.QFrame()
        dots.setFrameShape(QtWidgets.QFrame.NoFrame)
        dots.setFixedHeight(2)
        dots.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        dots.setStyleSheet(f"border-bottom: {px(4)}px dotted {color}; background: transparent;")

        lbl_score = QtWidgets.QLabel(str(score))
        lbl_score.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl_score.setFixedWidth(px(140))
        lbl_score.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        row.addWidget(lbl_rank, 0)
        row.addSpacing(px(14))
        row.addWidget(lbl_id, 0)
        row.addSpacing(px(10))
        row.addWidget(dots, 1)      # dots butt directly against lbl_score below — no gap
        row.addWidget(lbl_score, 0)


class ScoreboardPanel(QtWidgets.QWidget):
    """Cross-participant SCOREBOARD table: no card/frame, sits directly on
    the window background like the rest of the training GUI — dark-blue
    header, white names/scores, current participant highlighted in yellow.

    Renders whatever rows it is handed, in the order given, numbered 01..N
    down the page. Those numbers are POSITIONS ON THIS BOARD, not global
    ranks: the rows are a window around the current participant (see
    GameWindow._board_for_player), and printing true ranks like "14" would
    give away that this is not the whole field.

    Everything is drawn at `scale`, set by GameWindow._fit_scoreboard once
    the rows are known, so the table shrinks to fit the screen instead of
    running off the bottom of it.

    The just-finished score is shown separately, above this panel (see
    GameWindow's reveal page) rather than as part of it, so it can be
    enlarged independently.
    """
    MAX_ROWS = 7
    BASE_WIDTH = 760

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 1.0
        self._ranked: List[Tuple[str, int]] = []
        self._current_id = ""

        self.outer = QtWidgets.QVBoxLayout(self)
        self.outer.setSpacing(0)

        self.title = QtWidgets.QLabel("SCOREBOARD")
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        self.outer.addWidget(self.title)

        # Kept as an object (rather than addSpacing) so the title/rows gap
        # can be rescaled along with everything else.
        self.title_gap = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.outer.addSpacerItem(self.title_gap)

        self.rows_layout = QtWidgets.QVBoxLayout()
        self.outer.addLayout(self.rows_layout)

        self._apply_scale()

    # ── scaling ──────────────────────────────────────────────────────
    def set_scale(self, scale: float) -> None:
        """Redraw the panel at `scale` (1.0 = full size)."""
        if abs(scale - self._scale) < 1e-3:
            return
        self._scale = float(scale)
        self._apply_scale()
        self._render()

    def _apply_scale(self) -> None:
        s = self._scale

        def px(v: float) -> int:
            return max(1, int(round(v * s)))

        self.setFixedWidth(px(self.BASE_WIDTH))
        self.outer.setContentsMargins(px(34), px(16), px(34), px(16))
        self.title.setStyleSheet(
            f"font: {px(46)}px {pixel_font_family()}; color: {SB_HEADER}; "
            f"letter-spacing: {px(3)}px; background: transparent; border: none;"
        )
        self.title_gap.changeSize(
            0, px(10), QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.rows_layout.setSpacing(px(6))
        self.outer.invalidate()
        self.updateGeometry()

    # ── content ──────────────────────────────────────────────────────
    def _clear(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def set_scores(self, ranked: List[Tuple[str, int]], current_id: str = ""):
        self._ranked = list(ranked)
        self._current_id = current_id
        self._render()

    def _render(self):
        self._clear()
        s = self._scale

        top = self._ranked[: self.MAX_ROWS]
        if not top:
            empty = QtWidgets.QLabel("NO SCORES YET")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setStyleSheet(
                f"font: {max(1, int(round(22 * s)))}px {pixel_font_family()}; "
                f"color: {SB_ROW}; background: transparent; border: none;"
            )
            self.rows_layout.addWidget(empty)
            return

        for rank, (sid, score) in enumerate(top, start=1):
            self.rows_layout.addWidget(
                ScoreboardRow(sid, score, rank, highlight=(sid == self._current_id), scale=s)
            )
        self.updateGeometry()


class GameWindow(QtWidgets.QMainWindow):
    SPARKLE_DELAY_MS = 80
    POLL_INTERVAL_MS = 50
    SCORE_REVEAL_DELAY_S = 2.5  # how long the bare score shows before the scoreboard + continue prompt appear

    # Fixed parts of the root layout, needed to work out how much vertical
    # room the game-over block may occupy (see _fit_scoreboard).
    ROOT_MARGIN_V = 32
    ROOT_SPACING = 24
    BUTTON_H = 140
    # Base sizes of the "THIS GAME <score>" block above the scoreboard.
    REVEAL_CAP_PX = 32
    REVEAL_SCORE_PX = 110
    MIN_REVEAL_SCALE = 0.45

    def __init__(self, shared: UIShared, subject_id: str = "", backup_dir: Optional[Path] = None):
        super().__init__()
        self.shared = shared
        self.subject_id = subject_id
        # dir scanned for per-subject <id>/<id>.json backups (RESULTS_DIR)
        self.backup_dir = backup_dir or SUBJECT_RESULTS_DIR
        self._session_over_since: Optional[float] = None
        self._reveal_ready = False       # past the SCORE_REVEAL_DELAY_S gate (button becomes available)
        self._show_scoreboard = False    # reveal happened AND the player made the top N
        self._scoreboard_cache: List[Tuple[str, int]] = []
        self.coin_asset = CoinGraphic(find_coin_path() or Path())
        self.coinpop: Optional[CoinPopGraphic] = None
        self.sparkle: Optional[SparkleBurst] = None

        self.setWindowTitle("HRTF Localization Training")
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#7EC8FF"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self.showFullScreen()

        cw = QtWidgets.QWidget(self); self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(40, self.ROOT_MARGIN_V, 40, self.ROOT_MARGIN_V)
        root.setSpacing(self.ROOT_SPACING)

        pf = pixel_font_family()  # used throughout for the retro/pixel look

        # Top row. Wrapped in a widget (rather than added as a bare layout)
        # so its height can be measured when fitting the game-over block.
        self.top_widget = QtWidgets.QWidget()
        top = QtWidgets.QHBoxLayout(self.top_widget)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(20)
        root.addWidget(self.top_widget, 0)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(8)
        self.lblHighCap = QtWidgets.QLabel("HIGH SCORE")
        self.lblHighCap.setStyleSheet(f"font: 32px {pf}; color: #083c74; letter-spacing: 2px;")
        high_row = QtWidgets.QHBoxLayout(); high_row.setSpacing(12)
        self.coin_icon_lbl = QtWidgets.QLabel(); self.coin_icon_lbl.setFixedSize(72, 72)
        self.lblHigh = QtWidgets.QLabel("0"); self.lblHigh.setStyleSheet(f"font: 108px {pf}; color: #003e9f;")
        high_row.addWidget(self.coin_icon_lbl, 0, QtCore.Qt.AlignVCenter)
        high_row.addWidget(self.lblHigh,       0, QtCore.Qt.AlignVCenter)
        left.addWidget(self.lblHighCap); left.addLayout(high_row)
        top.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout(); right.setSpacing(8)
        self.lblTimeCap = QtWidgets.QLabel("TIME REMAINING")
        self.lblTimeCap.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.lblTimeCap.setStyleSheet(f"font: 32px {pf}; color: #083c74; letter-spacing: 2px;")
        self.lblTime = QtWidgets.QLabel("00:00")
        self.lblTime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.lblTime.setStyleSheet(f"font: 108px {pf}; color: #003e9f;")
        right.addWidget(self.lblTimeCap); right.addWidget(self.lblTime)
        top.addLayout(right, 1)

        # Central content area: swaps between the big score number and (once
        # a game ends and the reveal delay elapses) the scoreboard panel.
        # Both pages share one flexible slot instead of being stacked on top
        # of each other, so the total window height stays bounded — the
        # "press enter" button keeps its slot regardless of how tall the
        # scoreboard is (which _fit_scoreboard then scales to fit that
        # slot) — and the scoreboard gets the whole
        # upper/center area to sit in rather than being squeezed into
        # leftover space above the button.
        self.center_stack = QtWidgets.QStackedLayout()

        score_page = QtWidgets.QWidget()
        score_page_layout = QtWidgets.QVBoxLayout(score_page)
        score_page_layout.setContentsMargins(0, 0, 0, 0)
        score_page_layout.addStretch(1)
        self.lblScore = QtWidgets.QLabel("0")
        self.lblScore.setAlignment(QtCore.Qt.AlignCenter)
        self.lblScore.setStyleSheet(f"font: 260px {pf}; color: #ffffff;")
        score_page_layout.addWidget(self.lblScore, 0, QtCore.Qt.AlignCenter)
        score_page_layout.addStretch(2)
        self.center_stack.addWidget(score_page)

        # Scoreboard page: the just-finished score (enlarged) sits above the
        # HIGH SCORES table, both pinned to the top of the flexible area
        # ("go higher") with leftover space collecting below. No scroll area —
        # the content is laid out directly so no scroll bar can ever appear;
        # a negative top margin pulls the block up into the header gap.
        scoreboard_page = QtWidgets.QWidget()
        scoreboard_page_layout = QtWidgets.QVBoxLayout(scoreboard_page)
        scoreboard_page_layout.setContentsMargins(0, 0, 0, 0)

        self.scoreboard_content = QtWidgets.QWidget()
        self.scoreboard_content.setStyleSheet("background: transparent;")
        self.scl = QtWidgets.QVBoxLayout(self.scoreboard_content)
        self.scl.setSpacing(4)

        self.lblRevealCap = QtWidgets.QLabel("THIS GAME")
        self.lblRevealCap.setAlignment(QtCore.Qt.AlignHCenter)
        self.scl.addWidget(self.lblRevealCap)

        self.lblRevealScore = QtWidgets.QLabel("0")
        self.lblRevealScore.setAlignment(QtCore.Qt.AlignHCenter)
        self.scl.addWidget(self.lblRevealScore)

        # Kept as an object so the gap scales with the rest of the block.
        self.reveal_gap = QtWidgets.QSpacerItem(
            0, 8, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.scl.addSpacerItem(self.reveal_gap)

        self.scoreboard = ScoreboardPanel()
        self.scl.addWidget(self.scoreboard, 0, QtCore.Qt.AlignHCenter)
        self.scl.addStretch(1)

        # Sizes/fonts of the block above come from _apply_reveal_scale so
        # there is exactly one place that defines them.
        self._apply_reveal_scale(1.0)

        scoreboard_page_layout.addWidget(self.scoreboard_content)
        self.center_stack.addWidget(scoreboard_page)

        center_holder = QtWidgets.QWidget()
        center_holder.setLayout(self.center_stack)
        root.addWidget(center_holder, 1)

        # Overlay (used for both start AND play-again)
        self.start_stack = QtWidgets.QStackedLayout()
        start_page = QtWidgets.QWidget()
        sp = QtWidgets.QVBoxLayout(start_page); sp.setContentsMargins(0, 0, 0, 0)
        self.overlay_btn = QtWidgets.QPushButton("PRESS ENTER TO START")
        self.overlay_btn.setFixedHeight(self.BUTTON_H)
        self.overlay_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.overlay_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,255,255,0.3);
                border: 2px solid rgba(255,255,255,0.6);
                border-radius: 24px;
                font: 34px {pf};
                color: #003366;
                padding: 0 40px;
            }}
            QPushButton:pressed {{
                background: #14b8a6;
                border-color: #10a191;
                color: #00120f;
            }}
        """)
        self.overlay_btn.clicked.connect(self._on_enter_pressed)
        sp.addWidget(self.overlay_btn, 0, QtCore.Qt.AlignHCenter)
        self.start_stack.addWidget(start_page)
        spacer = QtWidgets.QWidget(); spacer.setFixedHeight(self.BUTTON_H)
        self.start_stack.addWidget(spacer)
        holder = QtWidgets.QWidget(); holder.setLayout(self.start_stack)
        holder.setFixedHeight(self.BUTTON_H)
        root.addWidget(holder, 0, QtCore.Qt.AlignHCenter)

        # In-app shortcuts: only fire when the game window is the focused
        # application. Kept as a fallback for when the global listener below
        # can't start (e.g. pynput missing / no OS accessibility permission).
        for key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(self._on_enter_pressed)
        esc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        esc.setContext(QtCore.Qt.ApplicationShortcut)
        esc.activated.connect(self._on_esc_pressed)

        # OS-level global listener so Enter/Space also work when another
        # window (terminal, plot, etc.) is focused, matching Localization_AR.
        self._hotkey_listener = None
        self._last_hotkey_ts = 0.0
        self._start_global_hotkeys()

        QtCore.QTimer.singleShot(0, self._init_overlays)

        self._prev_state = -1
        self._last_session_total = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _init_overlays(self):
        cw = self.centralWidget()
        if self.coin_asset.valid():
            h = max(48, int(self.lblHigh.height()))
            self.coin_icon_lbl.setFixedSize(h, h)
            pm = QtGui.QPixmap(h, h); pm.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pm)
            self.coin_asset.paint(painter, QtCore.QRect(0, 0, h, h))
            painter.end()
            self.coin_icon_lbl.setPixmap(pm)
            self.coinpop = CoinPopGraphic(self.lblScore, cw, self.coin_asset)
            self.coinpop.setGeometry(cw.rect())
            self.coinpop.lower()
            self.lblScore.raise_()

        self.sparkle = SparkleBurst(self.lblScore, cw, duration_ms=140)
        self.sparkle.setGeometry(cw.rect())
        self.sparkle.lower()
        self.sparkle.stackUnder(self.lblScore)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        cw = self.centralWidget()
        if cw and self.coinpop:
            self.coinpop.setGeometry(cw.rect())
        if cw and self.sparkle:
            self.sparkle.setGeometry(cw.rect())

    # ── game-over block: scaling / fitting ───────────────────────────
    def _apply_reveal_scale(self, scale: float) -> None:
        """Size the whole game-over block (caption, this-game score, table)
        at `scale`. Single source of truth for those sizes."""
        pf = pixel_font_family()

        def px(v: float) -> int:
            return max(1, int(round(v * scale)))

        # Negative on purpose: pulls the block up into the header gap so it
        # sits high on the page. Not run through px(), which floors at 1.
        self.scl.setContentsMargins(0, int(round(-24 * scale)), 0, 0)
        self.scl.setSpacing(px(4))
        self.lblRevealCap.setStyleSheet(
            f"font: {px(self.REVEAL_CAP_PX)}px {pf}; color: #083c74; "
            f"letter-spacing: {px(2)}px;"
        )
        self.lblRevealScore.setStyleSheet(
            f"font: {px(self.REVEAL_SCORE_PX)}px {pf}; color: #ffffff;"
        )
        self.reveal_gap.changeSize(
            0, px(8), QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.scoreboard.set_scale(scale)
        self.scoreboard.layout().activate()
        self.scl.invalidate()
        self.scl.activate()

    def _fit_scoreboard(self) -> None:
        """Shrink the game-over block until it fits between the header row
        and the continue button.

        The block used to be a fixed pixel size, which only fitted on a
        full-height display: on anything shorter the bottom rows of the
        table were cut off at the edge of the screen and the "PRESS ENTER"
        button was pushed off it entirely, leaving no visible way to carry
        on. So: lay the block out at full size, measure what it wants,
        compare with what is actually free, and rescale by the ratio.

        Call this AFTER set_scores() -- the height depends on how many rows
        there are.
        """
        cw = self.centralWidget()
        if cw is None:
            return
        # Measuring only works on the visible page: Qt layouts report a
        # hidden widget as zero-sized, so with the score page still current
        # the block measured as if the table weren't there at all.
        self.center_stack.setCurrentIndex(1)
        avail = (cw.height()
                 - 2 * self.ROOT_MARGIN_V
                 - self.top_widget.sizeHint().height()
                 - 2 * self.ROOT_SPACING
                 - self.BUTTON_H)
        if avail <= 0:
            return
        self._apply_reveal_scale(1.0)
        need = self.scoreboard_content.sizeHint().height()
        if need <= avail or need <= 0:
            return
        self._apply_reveal_scale(max(self.MIN_REVEAL_SCALE, avail / float(need)))

    def _current_scoreboard(self, live_highscore: int) -> List[Tuple[str, int]]:
        """Ranked (id, highscore) rows with the current player's live score
        folded in.

        read_scoreboard() only sees what has been written to disk. The
        parent process persists this game's high score as the game ends, but
        merging the shared highscore value on top means the board is never
        one run behind even if that write is slow, or fails.
        """
        rows = dict(read_scoreboard(self.backup_dir))
        if self.subject_id and int(live_highscore) > rows.get(self.subject_id, 0):
            rows[self.subject_id] = int(live_highscore)
        return sorted(rows.items(), key=lambda kv: kv[1], reverse=True)

    def _board_for_player(self, ranked: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """The rows to put on the board: the participant plus a frozen set
        of peers from either side of them.

        The peer set is chosen ONCE — upward-biased around wherever the
        participant stood at the time — and then persisted under
        results/<id>/. Every later game and session reuses it.

        That persistence is the point, not an optimisation. If the window
        were re-picked each time, improving would make unfamiliar names
        appear above the participant and familiar ones drop off the bottom,
        which advertises that the board is assembled around them. With the
        cast fixed, what changes as they improve is their POSITION among
        the same people — which is the motivating part — while the names
        stay put. Their peers' scores are always the current ones, so being
        overtaken still happens.
        """
        peers = load_peer_set(self.backup_dir, self.subject_id)
        if peers is None:
            peers = choose_peer_set(ranked, self.subject_id)
            if peers:
                save_peer_set(self.backup_dir, self.subject_id, peers)
        return board_rows(ranked, self.subject_id, peers or [])

    def _on_enter_pressed(self):
        state = int(self.shared.ui_state.value)
        if state == 1:  # start prompt
            self.shared.enter_pressed.value = 1
        elif state == 3 and self._reveal_ready:  # play-again prompt (after the reveal delay)
            self.shared.enter_pressed.value = 1

    def _on_esc_pressed(self):
        """ESC quits the session, but ONLY at the game-over prompt (between
        games) — never mid-game or at a trial prompt, so a stray ESC can't
        abort a running game. The parent (Training_AR.play_session) polls
        quit_pressed at its between-game wait loops and shuts down cleanly
        (sensor disconnect, worker teardown)."""
        state = int(self.shared.ui_state.value)
        if (state == 3 and self._reveal_ready
                and getattr(self.shared, "quit_pressed", None) is not None):
            self.shared.quit_pressed.value = 1

    def _start_global_hotkeys(self):
        """Start an OS-level keyboard listener (pynput) so Enter/Space are
        caught even when the game window isn't the focused application.

        The callback runs on pynput's own thread; it only reads shared
        state and sets the enter_pressed multiprocessing.Value (both
        thread-safe), never touching Qt widgets directly, so it's safe to
        route straight into _on_enter_pressed. A short debounce swallows
        OS key-repeat while a key is held. Failures (pynput not installed,
        or macOS accessibility permission not granted) are non-fatal — the
        in-app QShortcuts above still cover the focused-window case.
        """
        try:
            from pynput import keyboard
        except Exception:
            logging.warning("pynput unavailable; Enter only works when the "
                            "training window is focused.")
            return

        trigger_keys = {keyboard.Key.enter, keyboard.Key.space}

        def on_press(key):
            if key not in trigger_keys and key != keyboard.Key.esc:
                return
            now = time.monotonic()
            if now - self._last_hotkey_ts < 0.3:  # debounce key-repeat
                return
            self._last_hotkey_ts = now
            if key == keyboard.Key.esc:
                self._on_esc_pressed()
            else:
                self._on_enter_pressed()

        try:
            self._hotkey_listener = keyboard.Listener(on_press=on_press)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception:
            logging.exception("Could not start global hotkey listener; Enter "
                              "only works when the training window is focused.")
            self._hotkey_listener = None

    def closeEvent(self, ev):
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
        super().closeEvent(ev)

    def _tick(self):
        session_total = int(self.shared.session_total.value)
        game_time = float(self.shared.game_time_left.value)
        highscore = int(self.shared.highscore.value)
        state = int(self.shared.ui_state.value)
        last_goal = int(self.shared.last_goal_points.value)

        # update top
        self.lblTime.setText(fmt_time(game_time))
        self.lblHigh.setText(str(highscore))

        # score
        self._last_session_total = session_total
        self.lblScore.setText(str(session_total))

        # overlay visibility + text depends on state (start vs play-again).
        # For state 3 (session over) we first show the bare score for
        # SCORE_REVEAL_DELAY_S, then reveal the continue prompt — and the
        # scoreboard too, as a window around wherever the participant
        # stands (see _board_for_player). It is skipped entirely when there
        # is nothing meaningful to show — a participant with no score yet,
        # or no one else to put beside them.
        if state == 3:
            if self._session_over_since is None:
                self._session_over_since = time.monotonic()
                self._reveal_ready = False
                self._show_scoreboard = False
                self._scoreboard_cache = []
            elapsed = time.monotonic() - self._session_over_since
            if not self._reveal_ready and elapsed >= self.SCORE_REVEAL_DELAY_S:
                self._reveal_ready = True
                # Read the board HERE, at reveal time, rather than the
                # moment the game ended: the parent persists this game's
                # high score as it ends, and read_scoreboard() sources those
                # per-subject backups. Reading late (plus the live-highscore
                # merge in _current_scoreboard) is what makes the standings
                # include the run that just finished.
                self._scoreboard_cache = self._board_for_player(
                    self._current_scoreboard(highscore))
                # A one-row board (nobody else recorded yet) says nothing;
                # show the plain score instead.
                self._show_scoreboard = len(self._scoreboard_cache) >= 2
                if self._show_scoreboard:
                    self.lblRevealScore.setText(str(session_total))
                    self.scoreboard.set_scores(self._scoreboard_cache, self.subject_id)
                    self._fit_scoreboard()
        else:
            self._session_over_since = None
            self._reveal_ready = False
            self._show_scoreboard = False

        show_prompt = (state == 1) or (state == 3 and self._reveal_ready)
        if show_prompt:
            self.start_stack.setCurrentIndex(0)
            if state == 1:
                self.overlay_btn.setText("PRESS ENTER TO START")
            else:
                self.overlay_btn.setText("GAME OVER — ENTER: PLAY AGAIN  ·  ESC: QUIT")
        else:
            self.start_stack.setCurrentIndex(1)
        self.center_stack.setCurrentIndex(1 if (state == 3 and self._show_scoreboard) else 0)

        # goal effects
        if last_goal in (1, 2):
            if self.coinpop:
                def bump():
                    self._last_session_total += 1
                    self.lblScore.setText(str(self._last_session_total))
                self.coinpop.pop(last_goal, on_pop=bump)
            if self.sparkle:
                QtCore.QTimer.singleShot(self.SPARKLE_DELAY_MS, self.sparkle.trigger)
            self.shared.last_goal_points.value = 0

        # keep coin icon size aligned
        h = max(48, int(self.lblHigh.height()))
        if self.coin_icon_lbl.width() != h:
            self.coin_icon_lbl.setFixedSize(h, h)
            pm = QtGui.QPixmap(h, h); pm.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pm)
            self.coin_asset.paint(painter, QtCore.QRect(0, 0, h, h))
            painter.end()
            self.coin_icon_lbl.setPixmap(pm)

        if state != self._prev_state:
            self._prev_state = state


def run_ui(shared: UIShared, subject_id: str = "", backup_dir: Optional[Path] = None):
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = GameWindow(shared, subject_id=subject_id, backup_dir=backup_dir)
    w.show()
    sys.exit(app.exec_())
