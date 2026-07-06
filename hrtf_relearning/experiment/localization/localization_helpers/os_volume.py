"""
os_volume.py

Pin the Windows master output volume so pybinsim playback SPL is reproducible.

The AR/VR (pybinsim) loudness is matched to the dome loudspeakers BY EAR
(see match_ar_dome_loudness.py). That calibration -- currently gain 0.07 in the
AR localization tests -- is only valid at a fixed OS master volume, because
pybinsim's physical SPL scales with the Windows fader. Call ensure_windows_volume()
at the start of any protocol that runs the AR/VR localization test.
"""
import platform

OS_VOLUME = 0.50   # Windows master output level the loudness match was made at (0..1)


def ensure_windows_volume(target=OS_VOLUME):
    """
    Pin the Windows master output volume to `target` (0..1) and unmute it.

    Requires pycaw (`pip install pycaw comtypes`). On non-Windows or if pycaw is
    missing, prints a manual reminder instead of failing.
    """
    pct = target * 100
    if platform.system() != "Windows":
        print(f"!! Not Windows -- set the OS output volume to {pct:.0f}% manually.")
        return
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = cast(interface, POINTER(IAudioEndpointVolume))
        vol.SetMute(0, None)
        vol.SetMasterVolumeLevelScalar(float(target), None)
        scalar = vol.GetMasterVolumeLevelScalar()
        muted = vol.GetMute()
        assert abs(scalar - target) < 1e-3 and muted == 0, \
            f"Windows volume not at {pct:.0f}%/unmuted (scalar={scalar:.3f}, muted={muted})"
        print(f"Windows master volume = {scalar*100:.0f}%, muted={bool(muted)}  OK")
    except ImportError:
        print(f"!! pycaw not installed -- set Windows master volume to {pct:.0f}% (unmuted) MANUALLY. "
              "(`pip install pycaw comtypes` to automate this.)")
