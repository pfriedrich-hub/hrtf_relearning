"""Force the Windows system master volume to a fixed level.

Keeps headphone output level consistent across runs so the calibrated
presentation level stays valid. No-op on non-Windows platforms.

The COM call runs on its own thread. COM is initialised per *thread*, and
``import comtypes`` only does it for whichever thread imports it; once Qt or Tk
has initialised the main thread for its own purposes, pycaw's call there fails
with RPC_E_CHANGED_MODE ("Cannot change thread mode after it is set"). A fresh
thread has no apartment model yet, so it cannot collide.

Nothing here raises: the volume lock is a safeguard for the calibration, not a
reason to lose a session. On failure it logs why and returns False.
"""
import logging
import platform
import threading


def set_windows_volume(level=50):
    """Set the Windows master volume slider to ``level`` (0-100).

    The scalar passed to Windows matches the on-screen slider position, so
    ``level=50`` puts the slider at 50. Returns True only if the volume was set
    and read back at that level.
    """
    if platform.system() != "Windows":
        logging.info(f"set_windows_volume: skipped (not Windows, level={level}).")
        return False

    level = max(0, min(100, int(level)))
    result = {}

    def worker():
        try:
            from ctypes import cast, POINTER
            import comtypes
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            comtypes.CoInitialize()
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMute(0, None)
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            # Read back: on an endpoint whose volume is not host-controlled the
            # setter succeeds and changes nothing.
            result["level"] = volume.GetMasterVolumeLevelScalar() * 100.0
        except BaseException as err:
            result["error"] = err

    thread = threading.Thread(target=worker, name="set_windows_volume", daemon=True)
    thread.start()
    thread.join(timeout=10)

    actual = result.get("level")
    if actual is None:
        logging.warning(
            f"set_windows_volume: could NOT set the master volume "
            f"({result.get('error', 'timed out')!r}). Set the Windows slider to "
            f"{level}% by hand — the presentation level is only calibrated there."
        )
        return False
    if abs(actual - level) > 2:  # driver quantisation is fine, ignoring us is not
        logging.warning(
            f"set_windows_volume: asked for {level}%, endpoint reports "
            f"{actual:.1f}%. The slider is not host-controlled — check the "
            f"default playback device and set the level by hand."
        )
        return False

    logging.info(f"set_windows_volume: master volume set to {level}%.")
    return True


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    set_windows_volume(50)
