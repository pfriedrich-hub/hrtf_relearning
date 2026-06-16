"""Shared helper utilities."""
from pynput import keyboard


def wait_for_enter(msg=None):
    """Block until Enter is pressed, even if the console window isn't focused.

    Uses a global keyboard listener (pynput) instead of `input()`, so the
    prompt is caught regardless of which window has focus (e.g. while a
    TDT/freefield or matplotlib window is active).
    """
    if msg:
        print(msg)

    def on_press(key):
        if key == keyboard.Key.enter:
            listener.stop()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
