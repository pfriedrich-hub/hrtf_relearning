from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import json
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
from hrtf_relearning.experiment.misc.Subject import backup_dir as SUBJECT_BACKUP_DIR

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


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def find_coin_path() -> Optional[Path]:
    for p in (
        ROOT / "data" / "documentation" / "ui" / "mario-coin.svg",  # actual current location
        ROOT / "data" / "img" / "ui" / "mario-coin.svg",
        ROOT / "data" / "ui" / "mario-coin.svg",
        ROOT / "data" / "ui" / "mario-coin.png",
    ):
        if p.exists():
            return p
    return None


def read_scoreboard(backup_dir: Path) -> List[Tuple[str, int]]:
    """Read (subject_id, highscore) pairs across participants.

    Sources from the small per-subject JSON backups Subject.write() already
    maintains (data/results/backup/<ID>.json), rather than the .pkl files
    directly — the pickles can hold slab objects (e.g. last_sequence) that
    aren't safely unpicklable from this dependency-light UI process, while
    the JSON backups are plain, cheap, and rewritten on every trial.

    Unreadable/malformed files are skipped rather than raised, since these
    are written independently by other subjects' sessions.
    """
    rows: dict[str, int] = {}
    if not backup_dir.exists():
        return []
    for p in sorted(backup_dir.glob("*.json")):
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
    return sorted(rows.items(), key=lambda kv: kv[1], reverse=True)


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
    path = ROOT / "data" / "ui" / "fonts" / filename
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
    """One 'RANK  ID .......... SCORE' line."""
    def __init__(self, subj_id: str, score: int, rank: int, highlight: bool, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        # No uniform spacing here — the dotted leader needs to butt right up
        # against the score digits with zero gap, so spacing is added
        # explicitly only where it's wanted (rank->id, id->dots).
        row.setSpacing(0)
        pf = pixel_font_family()
        color = SB_HIGHLIGHT if highlight else SB_ROW
        size = 30 if highlight else 26

        lbl_rank = QtWidgets.QLabel(f"{rank:02d}")
        lbl_rank.setFixedWidth(56)
        lbl_rank.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        lbl_id = QtWidgets.QLabel(subj_id.upper())
        lbl_id.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        dots = QtWidgets.QFrame()
        dots.setFrameShape(QtWidgets.QFrame.NoFrame)
        dots.setFixedHeight(2)
        dots.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        dots.setStyleSheet(f"border-bottom: 4px dotted {color}; background: transparent;")

        lbl_score = QtWidgets.QLabel(str(score))
        lbl_score.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl_score.setFixedWidth(140)
        lbl_score.setStyleSheet(f"font: {size}px {pf}; color: {color}; background: transparent; border: none;")

        row.addWidget(lbl_rank, 0)
        row.addSpacing(14)
        row.addWidget(lbl_id, 0)
        row.addSpacing(10)
        row.addWidget(dots, 1)      # dots butt directly against lbl_score below — no gap
        row.addWidget(lbl_score, 0)


class ScoreboardPanel(QtWidgets.QWidget):
    """Cross-participant HIGH SCORES table: no card/frame, sits directly on
    the window background like the rest of the training GUI — dark-blue
    header, white names/scores, current participant highlighted in yellow.

    The just-finished score is shown separately, above this panel (see
    GameWindow's reveal page) rather than as part of it, so it can be
    enlarged independently.
    """
    MAX_ROWS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(760)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(34, 20, 34, 20)
        outer.setSpacing(0)

        pf = pixel_font_family()
        self.title = QtWidgets.QLabel("HIGH SCORES")
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        self.title.setStyleSheet(
            f"font: 54px {pf}; color: {SB_HEADER}; letter-spacing: 3px; background: transparent; border: none;"
        )
        outer.addWidget(self.title)

        outer.addSpacing(12)

        self.rows_layout = QtWidgets.QVBoxLayout()
        self.rows_layout.setSpacing(8)
        outer.addLayout(self.rows_layout)

    def _clear(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def set_scores(self, ranked: List[Tuple[str, int]], current_id: str = ""):
        self._clear()

        if not ranked:
            empty = QtWidgets.QLabel("NO SCORES YET")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setStyleSheet(
                f"font: 22px {pixel_font_family()}; color: {SB_ROW}; background: transparent; border: none;"
            )
            self.rows_layout.addWidget(empty)
            return

        top = ranked[: self.MAX_ROWS]
        top_ids = {sid for sid, _ in top}
        for rank, (sid, score) in enumerate(top, start=1):
            self.rows_layout.addWidget(ScoreboardRow(sid, score, rank, highlight=(sid == current_id)))

        # Always surface the current participant, even if outside the top N.
        if current_id and current_id not in top_ids:
            for rank, (sid, score) in enumerate(ranked, start=1):
                if sid == current_id:
                    sep = QtWidgets.QLabel("...")
                    sep.setAlignment(QtCore.Qt.AlignCenter)
                    sep.setStyleSheet(
                        f"font: 20px {pixel_font_family()}; color: rgba(255,255,255,0.55); "
                        f"background: transparent; border: none;"
                    )
                    self.rows_layout.addWidget(sep)
                    self.rows_layout.addWidget(ScoreboardRow(sid, score, rank, highlight=True))
                    break


class GameWindow(QtWidgets.QMainWindow):
    SPARKLE_DELAY_MS = 80
    POLL_INTERVAL_MS = 50
    SCORE_REVEAL_DELAY_S = 2.5  # how long the bare score shows before the scoreboard + continue prompt appear

    def __init__(self, shared: UIShared, subject_id: str = "", backup_dir: Optional[Path] = None):
        super().__init__()
        self.shared = shared
        self.subject_id = subject_id
        self.backup_dir = backup_dir or SUBJECT_BACKUP_DIR
        self._session_over_since: Optional[float] = None
        self._reveal_ready = False       # past the SCORE_REVEAL_DELAY_S gate (button becomes available)
        self._show_scoreboard = False    # reveal happened AND the player is actually listed on it
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
        root = QtWidgets.QVBoxLayout(cw); root.setContentsMargins(40, 32, 40, 32); root.setSpacing(24)

        pf = pixel_font_family()  # used throughout for the retro/pixel look

        # Top row
        top = QtWidgets.QHBoxLayout(); top.setSpacing(20); root.addLayout(top)
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
        # "press enter" button can never get pushed off screen regardless of
        # how tall the scoreboard is — and the scoreboard gets the whole
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
        # HIGH SCORES table, both pinned toward the top of the flexible area
        # ("go higher") with leftover space collecting below. Wrapped in a
        # QScrollArea as a hard safety net: if this content ever runs taller
        # than the available space (a smaller display, more rows, bigger
        # fonts...), it scrolls internally instead of pushing the "press
        # enter" button off screen — the exact failure mode hit earlier.
        scoreboard_page = QtWidgets.QWidget()
        scoreboard_page_layout = QtWidgets.QVBoxLayout(scoreboard_page)
        scoreboard_page_layout.setContentsMargins(0, 0, 0, 0)

        scoreboard_scroll = QtWidgets.QScrollArea()
        scoreboard_scroll.setWidgetResizable(True)
        scoreboard_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scoreboard_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scoreboard_scroll.setStyleSheet("background: transparent; border: none;")
        scoreboard_scroll.viewport().setStyleSheet("background: transparent;")

        scoreboard_content = QtWidgets.QWidget()
        scoreboard_content.setStyleSheet("background: transparent;")
        scl = QtWidgets.QVBoxLayout(scoreboard_content)
        scl.setContentsMargins(0, 8, 0, 0)
        scl.setSpacing(4)

        self.lblRevealCap = QtWidgets.QLabel("THIS GAME")
        self.lblRevealCap.setAlignment(QtCore.Qt.AlignHCenter)
        self.lblRevealCap.setStyleSheet(f"font: 32px {pf}; color: #083c74; letter-spacing: 2px;")
        scl.addWidget(self.lblRevealCap)

        self.lblRevealScore = QtWidgets.QLabel("0")
        self.lblRevealScore.setAlignment(QtCore.Qt.AlignHCenter)
        self.lblRevealScore.setStyleSheet(f"font: 120px {pf}; color: #ffffff;")
        scl.addWidget(self.lblRevealScore)

        scl.addSpacing(8)

        self.scoreboard = ScoreboardPanel()
        scl.addWidget(self.scoreboard, 0, QtCore.Qt.AlignHCenter)
        scl.addStretch(1)

        scoreboard_scroll.setWidget(scoreboard_content)
        scoreboard_page_layout.addWidget(scoreboard_scroll)
        self.center_stack.addWidget(scoreboard_page)

        center_holder = QtWidgets.QWidget()
        center_holder.setLayout(self.center_stack)
        root.addWidget(center_holder, 1)

        # Overlay (used for both start AND play-again)
        self.start_stack = QtWidgets.QStackedLayout()
        start_page = QtWidgets.QWidget()
        sp = QtWidgets.QVBoxLayout(start_page); sp.setContentsMargins(0, 0, 0, 0)
        self.overlay_btn = QtWidgets.QPushButton("PRESS ENTER TO START")
        self.overlay_btn.setFixedHeight(140)
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
        spacer = QtWidgets.QWidget(); spacer.setFixedHeight(140)
        self.start_stack.addWidget(spacer)
        holder = QtWidgets.QWidget(); holder.setLayout(self.start_stack)
        holder.setFixedHeight(140)
        root.addWidget(holder, 0, QtCore.Qt.AlignHCenter)

        for key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(self._on_enter_pressed)

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

    def _on_enter_pressed(self):
        state = int(self.shared.ui_state.value)
        if state == 1:  # start prompt
            self.shared.enter_pressed.value = 1
        elif state == 3 and self._reveal_ready:  # play-again prompt (after the reveal delay)
            self.shared.enter_pressed.value = 1

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
        # scoreboard too, but only if the current participant actually has
        # an entry on it. A brand-new subject with no recorded highscore
        # yet would otherwise see a leaderboard full of established
        # participants' scores on day one, which is discouraging rather
        # than motivating; showing just their own score in that case.
        if state == 3:
            if self._session_over_since is None:
                self._session_over_since = time.monotonic()
                self._reveal_ready = False
                self._show_scoreboard = False
                self._scoreboard_cache = read_scoreboard(self.backup_dir)
            elapsed = time.monotonic() - self._session_over_since
            if not self._reveal_ready and elapsed >= self.SCORE_REVEAL_DELAY_S:
                self._reveal_ready = True
                player_listed = any(sid == self.subject_id for sid, _ in self._scoreboard_cache)
                self._show_scoreboard = bool(self.subject_id) and player_listed
                if self._show_scoreboard:
                    self.lblRevealScore.setText(str(session_total))
                    self.scoreboard.set_scores(self._scoreboard_cache, self.subject_id)
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
                self.overlay_btn.setText("GAME OVER — PRESS ENTER")
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
