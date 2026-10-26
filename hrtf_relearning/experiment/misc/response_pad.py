"""Participant response pad: a USB SNES-style gamepad read through Windows winmm.

Why a gamepad instead of Enter: a pynput keyboard listener sees every keyboard
on the machine and does not consume the key, so the participant's Enter also
lands in whatever window the experimenter has focused, and the experimenter's
own Enter/Space count as participant responses. Gamepad buttons never enter the
keyboard stream, and winmm's joyGetPosEx reads them regardless of window focus.

Usage
-----
    from hrtf_relearning.experiment.misc.response_pad import get_pad

    pad = get_pad()
    pad.wait("A", f"Look at the center and press {pad.label('A')}")   # blocking
    pad.on_press("START", start_game)          # callback, runs on the poll thread
    pad.remove("START", start_game)

Without a pad (no device found, or not on Windows) every wait()/on_press()
falls back to the Enter key via pynput, i.e. the old behaviour, so nothing
blocks a session on a machine without a pad (the freefield PC, a Mac).
In that mode every button name maps to Enter; label() returns "Enter".

Run this file directly to print button presses by name (pad test).
"""
import ctypes
import logging
import sys
import threading
import time

# Bit index in JOYINFOEX.dwButtons (= number shown in joy.cpl minus 1).
# Measured on the lab pad 2026-10-26. L/R (6/7) inferred, not labelled.
BUTTONS = {"X": 0, "A": 1, "B": 2, "Y": 3, "L": 6, "R": 7, "SELECT": 8, "START": 9}

POLL_INTERVAL = 0.005   # s; 200 Hz
DEBOUNCE = 0.03         # s; ignore re-presses of the same button within this window
JOY_RETURNALL = 0xFF
JOYERR_NOERROR = 0


class _JOYINFOEX(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in (
        "dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos",
        "dwVpos", "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2")]


class _WinmmReader:
    """Reads the button bitmask of one joystick id. read() -> int or None (unplugged)."""

    def __init__(self, joy_id=None):
        self._winmm = ctypes.WinDLL("winmm")
        self._info = _JOYINFOEX(dwSize=ctypes.sizeof(_JOYINFOEX), dwFlags=JOY_RETURNALL)
        if joy_id is None:
            ids = [i for i in range(16) if self._query(i) is not None]
            if not ids:
                raise RuntimeError("no game controller found")
            joy_id = ids[0]
        self.joy_id = joy_id

    def _query(self, joy_id):
        if self._winmm.joyGetPosEx(joy_id, ctypes.byref(self._info)) != JOYERR_NOERROR:
            return None
        return self._info.dwButtons

    def read(self):
        return self._query(self.joy_id)


class ResponsePad:
    """Background-polled gamepad with press callbacks and a blocking wait()."""

    def __init__(self, reader=None):
        self._reader = reader
        self._callbacks = {name: [] for name in BUTTONS}
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._last_press = {}

    @property
    def available(self):
        return self._reader is not None

    def label(self, button):
        """Name to show the participant for this button: the pad button's
        name, or "Enter" when running on the keyboard fallback."""
        return button.upper() if self.available else "Enter"

    # -- callbacks ---------------------------------------------------------
    def on_press(self, button, callback):
        """Call callback() on every press of button. Runs on the poll thread
        (or pynput's thread in fallback mode) — keep it short and thread-safe."""
        button = button.upper()
        if not self.available:
            _enter_fallback.add(callback)
            return
        with self._lock:
            self._callbacks[button].append(callback)
        self._ensure_running()

    def remove(self, button, callback):
        button = button.upper()
        if not self.available:
            _enter_fallback.discard(callback)
            return
        with self._lock:
            if callback in self._callbacks[button]:
                self._callbacks[button].remove(callback)

    # -- blocking ----------------------------------------------------------
    def wait(self, button="A", msg=None, timeout=None, tick=None):
        """Block until the next press of button. Returns False on timeout.
        tick: optional callable run every ~50 ms while waiting (e.g. to keep
        matplotlib figures responsive)."""
        if msg:
            print(msg)
        pressed = threading.Event()
        self.on_press(button, pressed.set)
        try:
            if tick is None:
                return pressed.wait(timeout)
            t_end = None if timeout is None else time.monotonic() + timeout
            while not pressed.is_set():
                tick()
                if t_end is not None and time.monotonic() >= t_end:
                    return False
                pressed.wait(0.05)
            return True
        finally:
            self.remove(button, pressed.set)

    # -- polling -----------------------------------------------------------
    def _ensure_running(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="response-pad", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _poll(self):
        prev = self._reader.read() or 0   # held buttons at start don't count as presses
        unplugged = False
        while not self._stop.is_set():
            state = self._reader.read()
            if state is None:
                if not unplugged:
                    logging.warning("response pad not readable (unplugged?) — waiting for it")
                    unplugged = True
                prev = 0
                time.sleep(0.2)
                continue
            if unplugged:
                logging.warning("response pad back")
                unplugged = False
                prev = state            # don't fire for buttons held while re-plugging
            new = state & ~prev
            prev = state
            if new:
                now = time.monotonic()
                for name, bit in BUTTONS.items():
                    if new & (1 << bit) and now - self._last_press.get(name, -1) >= DEBOUNCE:
                        self._last_press[name] = now
                        with self._lock:
                            callbacks = list(self._callbacks[name])
                        for cb in callbacks:
                            try:
                                cb()
                            except Exception:
                                logging.exception("response pad callback failed")
            time.sleep(POLL_INTERVAL)


class _EnterFallback:
    """Old behaviour: global pynput Enter listener, shared by all fallback callbacks."""

    def __init__(self):
        self._callbacks = set()
        self._lock = threading.Lock()
        self._listener = None

    def add(self, callback):
        with self._lock:
            self._callbacks.add(callback)
            if self._listener is None:
                from pynput import keyboard

                def on_press(key):
                    if key == keyboard.Key.enter:
                        with self._lock:
                            callbacks = list(self._callbacks)
                        for cb in callbacks:
                            cb()

                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.daemon = True
                self._listener.start()

    def discard(self, callback):
        with self._lock:
            self._callbacks.discard(callback)


_enter_fallback = _EnterFallback()
_pad = None


def get_pad():
    """Process-wide pad instance; falls back to Enter if no pad is found."""
    global _pad
    if _pad is None:
        reader = None
        if sys.platform == "win32":
            try:
                reader = _WinmmReader()
            except Exception as err:
                logging.info(f"response pad unavailable ({err}); using Enter instead")
        else:
            logging.info("response pad only supported on Windows; using Enter instead")
        _pad = ResponsePad(reader)
    return _pad


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pad = get_pad()
    print(f"pad available: {pad.available}"
          + (f" (joystick id {pad._reader.joy_id})" if pad.available else ""))
    for name in BUTTONS:
        pad.on_press(name, lambda n=name: print(f"{time.perf_counter():.3f}  {n}"))
    print("press buttons, Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
