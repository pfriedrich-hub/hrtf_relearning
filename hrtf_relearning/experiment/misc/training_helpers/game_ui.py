from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import math

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from PyQt5 import QtSvg
    HAS_QTSVG = True
except Exception:
    HAS_QTSVG = False
import hrtf_relearning
ROOT = hrtf_relearning.PATH

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
        ROOT / "data" / "img" / "ui" / "mario-coin.svg",
        ROOT / "data" / "ui"  / "mario-coin.png",
    ):
        if p.exists():
            return p
    return None


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
    """Mario-style coin: appears just above score, jumps higher, lingers, then vanishes instantly."""
    def __init__(self, anchor_label: QtWidgets.QLabel, parent: QtWidgets.QWidget, coin: CoinGraphic):
        super().__init__(parent)
        self.anchor = anchor_label
        self.coin = coin
        self._y_offset = 0
        self._visible = False
        self.start_offset = -40
        self.jump_height = 120
        self.jump_duration = 600
        self.linger_time = 300
        self.move = QtCore.QPropertyAnimation(self, b"yOffset", self)
        self.move.setEasingCurve(QtCore.QEasingCurve.OutCubic)

    def getYOffset(self) -> int: return int(self._y_offset)
    def setYOffset(self, v: int) -> None: self._y_offset = int(v); self.update()
    yOffset = QtCore.pyqtProperty(int, fget=getYOffset, fset=setYOffset)

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
        self.show(); self.raise_()
        self.move.stop()
        self.move.setDuration(self.jump_duration)
        self.move.setStartValue(self.start_offset)
        self.move.setEndValue(self.start_offset - self.jump_height)
        self.move.start()
        QtCore.QTimer.singleShot(self.jump_duration + self.linger_time, self._vanish)

    def _vanish(self):
        self._visible = False
        self.hide()
        self._y_offset = 0
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        if not self._visible or not self.coin.valid():
            return
        p = QtGui.QPainter(self)
        c = self.anchor.mapTo(self.parentWidget(), self.anchor.rect().center())
        size = max(48, int(self.anchor.height() * 0.9))
        half = size // 2
        rect = QtCore.QRect(c.x() - half, c.y() - half + self._y_offset, size, size)
        self.coin.paint(p, rect)


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


class GameWindow(QtWidgets.QMainWindow):
    SPARKLE_DELAY_MS = 80
    POLL_INTERVAL_MS = 50

    def __init__(self, shared: UIShared, highscore_path: Optional[Path] = None):
        super().__init__()
        self.shared = shared
        self.highscore_path = highscore_path
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

        # Top row
        top = QtWidgets.QHBoxLayout(); top.setSpacing(20); root.addLayout(top)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(0)
        self.lblHighCap = QtWidgets.QLabel("High Score")
        self.lblHighCap.setStyleSheet("font: 600 28px 'Inter'; color: #083c74;")
        high_row = QtWidgets.QHBoxLayout(); high_row.setSpacing(12)
        self.coin_icon_lbl = QtWidgets.QLabel(); self.coin_icon_lbl.setFixedSize(72, 72)
        self.lblHigh = QtWidgets.QLabel("0"); self.lblHigh.setStyleSheet("font: 700 96px 'Inter'; color: #003e9f;")
        high_row.addWidget(self.coin_icon_lbl, 0, QtCore.Qt.AlignVCenter)
        high_row.addWidget(self.lblHigh,       0, QtCore.Qt.AlignVCenter)
        left.addWidget(self.lblHighCap); left.addLayout(high_row)
        top.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout(); right.setSpacing(0)
        self.lblTimeCap = QtWidgets.QLabel("Time Remaining")
        self.lblTimeCap.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.lblTimeCap.setStyleSheet("font: 600 28px 'Inter'; color: #083c74;")
        self.lblTime = QtWidgets.QLabel("00:00")
        self.lblTime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.lblTime.setStyleSheet("font: 700 96px 'Inter'; color: #003e9f;")
        right.addWidget(self.lblTimeCap); right.addWidget(self.lblTime)
        top.addLayout(right, 1)

        root.addStretch(1)

        # Center score
        score_holder = QtWidgets.QWidget()
        score_layout = QtWidgets.QVBoxLayout(score_holder)
        score_layout.setContentsMargins(0, 60, 0, 0)
        self.lblScore = QtWidgets.QLabel("0")
        self.lblScore.setAlignment(QtCore.Qt.AlignCenter)
        self.lblScore.setStyleSheet("font: 900 200px 'Inter'; color: #ffffff;")
        score_layout.addWidget(self.lblScore, 0, QtCore.Qt.AlignCenter)
        root.addWidget(score_holder)

        root.addStretch(2)

        # Overlay (used for both start AND play-again)
        self.start_stack = QtWidgets.QStackedLayout()
        start_page = QtWidgets.QWidget()
        sp = QtWidgets.QVBoxLayout(start_page); sp.setContentsMargins(0, 0, 0, 0)
        self.overlay_btn = QtWidgets.QPushButton("Press Enter to start")
        self.overlay_btn.setFixedHeight(140)
        self.overlay_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.overlay_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.3);
                border: 2px solid rgba(255,255,255,0.6);
                border-radius: 24px;
                font: 700 48px 'Inter';
                color: #003366;
            }
            QPushButton:pressed {
                background: #14b8a6;
                border-color: #10a191;
                color: #00120f;
            }
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
        if int(self.shared.ui_state.value) in (1, 3):  # start or play-again prompt
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

        # overlay visibility + text depends on state (start vs play-again)
        if state in (1, 3):
            self.start_stack.setCurrentIndex(0)
            if state == 1:
                self.overlay_btn.setText("Press Enter to start")
            else:
                self.overlay_btn.setText("Session over — Press Enter to play again")
        else:
            self.start_stack.setCurrentIndex(1)

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


def run_ui(shared: UIShared, highscore_path: Optional[Path] = None):
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = GameWindow(shared, highscore_path)
    w.show()
    sys.exit(app.exec_())
