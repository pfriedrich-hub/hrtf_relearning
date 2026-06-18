"""Force the Windows system master volume to a fixed level.

Keeps headphone output level consistent across runs so the calibrated
presentation level stays valid. No-op on non-Windows platforms.
"""
import logging
import platform


def set_windows_volume(level=50):
    """Set the Windows master volume slider to ``level`` (0-100).

    Uses the Core Audio endpoint volume (pycaw). The scalar passed to Windows
    matches the on-screen slider position, so ``level=50`` puts the slider at 50.
    Returns True if the volume was set, False otherwise.
    """
    if platform.system() != "Windows":
        logging.info(f"set_windows_volume: skipped (not Windows, level={level}).")
        return False

    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError:
        logging.warning("set_windows_volume: pycaw not installed; run `pip install pycaw`.")
        return False

    level = max(0, min(100, int(level)))
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)
    volume.SetMasterVolumeLevelScalar(level / 100.0, None)
    logging.info(f"set_windows_volume: master volume set to {level}.")
    return True


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    set_windows_volume(50)
