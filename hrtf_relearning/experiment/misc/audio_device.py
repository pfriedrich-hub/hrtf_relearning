"""Verify the playback route before a session starts.

A session played through the laptop speakers instead of the USB interface is
wasted data, and nothing in the run itself makes that visible -- the tracker
works, the game works, the responses get logged. So check the route up front and
refuse to start when it is wrong.

Machine-local by design
-----------------------
The check is inert unless this machine asks for it. Put the required route in
``local_config.json`` (gitignored, see :mod:`hrtf_relearning.utils.local_config`)::

    {
      "torch_device": "cuda",
      "audio_output": "Sound Blaster",
      "audio_volume": 50
    }

With no ``audio_output`` key, :func:`preflight` logs a debug line and returns
None. That way the same commit runs on the rig, the VR PC and a laptop, and only
the machine that has a fixed interface enforces it.

Optional keys (env override in brackets):

==================  =========================================================
``audio_output``    substring of the required device name  [HRTF_AUDIO_OUTPUT]
``audio_volume``    master volume in percent, default 50  [HRTF_AUDIO_VOLUME]
``audio_strict``    false -> log an error and continue    [HRTF_AUDIO_STRICT]
``audio_autofix``   false -> never switch the device      [HRTF_AUDIO_AUTOFIX]
``soundvolumeview`` path to SoundVolumeView.exe        [HRTF_SOUNDVOLUMEVIEW]
==================  =========================================================

Switching the default device has no supported Windows API, so the autofix shells
out to NirSoft SoundVolumeView (portable, no install) when it can be found. With
no such binary the check still runs -- it just reports instead of fixing.

Run ``python -m hrtf_relearning.experiment.misc.audio_device`` to print what
Windows currently reports, including the exact device names to copy into
``local_config.json``.
"""
import csv
import io
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.utils import local_config, paths

logger = logging.getLogger(__name__)

#: Driver quantisation is fine; the endpoint ignoring us is not.
VOLUME_TOLERANCE = 2.0

#: Windows audio roles as SoundVolumeView numbers them.
_ROLE_CONSOLE = "1"
_ROLE_MULTIMEDIA = "2"

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


class AudioRouteError(RuntimeError):
    """The default playback device or its volume is not what this machine requires."""


# --------------------------------------------------------------- config helpers

def _as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    logger.warning("audio preflight: unreadable boolean %r — using %s.", value, default)
    return default


# ------------------------------------------------------------- reading the state

def _friendly_name(device):
    """Endpoint name for a pycaw device, across pycaw versions."""
    name = getattr(device, "FriendlyName", None)
    if name:
        return str(name)
    # Older pycaw hands back the raw IMMDevice pointer, which carries no name.
    try:
        from pycaw.pycaw import AudioUtilities

        device_id = device.GetId()
        for candidate in AudioUtilities.GetAllDevices():
            if getattr(candidate, "id", None) == device_id:
                return str(candidate.FriendlyName)
    except BaseException:  # pragma: no cover - depends on pycaw internals
        logger.debug("audio preflight: could not resolve endpoint name", exc_info=True)
    return None


def current_output():
    """
    What Windows is playing through right now.

    Returns a dict with ``name``, ``volume`` (percent), ``muted`` -- any of which
    may be None -- and ``error`` when the COM query failed.

    The COM call runs on its own thread for the same reason
    :mod:`~hrtf_relearning.experiment.misc.system_volume` does it: COM is
    initialised per thread, and once Qt or Tk has claimed the main thread pycaw's
    call there fails with RPC_E_CHANGED_MODE.
    """
    state = {"name": None, "volume": None, "muted": None}

    if platform.system() != "Windows":
        state["error"] = RuntimeError("not Windows")
        return state

    def worker():
        try:
            from ctypes import POINTER, cast

            import comtypes
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            comtypes.CoInitialize()
            device = AudioUtilities.GetSpeakers()
            state["name"] = _friendly_name(device)

            if hasattr(device, "EndpointVolume"):
                volume = device.EndpointVolume
            else:
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
            state["volume"] = volume.GetMasterVolumeLevelScalar() * 100.0
            state["muted"] = bool(volume.GetMute())
        except BaseException as err:
            state["error"] = err

    thread = threading.Thread(target=worker, name="audio_device_query", daemon=True)
    thread.start()
    thread.join(timeout=10)
    if thread.is_alive():
        state.setdefault("error", TimeoutError("COM query did not return in 10 s"))
    return state


# ------------------------------------------------------- switching the device

def soundvolumeview_path(explicit=None):
    """Path to SoundVolumeView.exe, or None. Config and env win over the repo copy."""
    candidates = [
        explicit,
        local_config.get("soundvolumeview"),
        paths.PATH.parent / "tools" / "SoundVolumeView.exe",
        paths.PATH / "tools" / "SoundVolumeView.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    found = shutil.which("SoundVolumeView.exe") or shutil.which("SoundVolumeView")
    return found or None


def list_render_devices(svv):
    """Active playback endpoints as SoundVolumeView sees them (list of dicts)."""
    with tempfile.TemporaryDirectory(prefix="svv_") as tmp:
        export = Path(tmp) / "devices.csv"
        try:
            subprocess.run([svv, "/scomma", str(export)], timeout=30, check=False,
                           creationflags=_CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as err:
            logger.warning("audio preflight: SoundVolumeView export failed (%s).", err)
            return []
        if not export.is_file():
            logger.warning("audio preflight: SoundVolumeView wrote no export.")
            return []
        # SoundVolumeView writes the system ANSI codepage, not UTF-8.
        encoding = "mbcs" if os.name == "nt" else "utf-8"
        text = export.read_text(encoding=encoding, errors="replace")

    rows = csv.DictReader(io.StringIO(text))
    return [row for row in rows
            if row.get("Type") == "Device"
            and row.get("Direction") == "Render"
            and row.get("Device State") == "Active"]


def _matching(devices, match):
    needle = match.casefold()
    return [d for d in devices
            if needle in f"{d.get('Name', '')} {d.get('Device Name', '')}".casefold()]


def set_default_output(match, svv=None):
    """
    Make the one active playback device matching ``match`` the default.

    Sets the Console and Multimedia roles -- the two the Windows volume slider
    follows. The separate Communications default is left alone.

    Returns True when the switch was issued, False when it could not be (no
    SoundVolumeView, no match, or an ambiguous match). Never raises.
    """
    svv = svv or soundvolumeview_path()
    if not svv:
        logger.info("audio preflight: no SoundVolumeView.exe found — cannot switch "
                    "the device automatically. Put it in <repo>/tools/ or set "
                    "\"soundvolumeview\" in local_config.json.")
        return False

    devices = list_render_devices(svv)
    hits = _matching(devices, match)
    if not hits:
        logger.warning("audio preflight: no active playback device matching %r.", match)
        return False
    if len(hits) > 1:
        logger.warning("audio preflight: %r matches %d devices (%s) — refusing to "
                       "guess. Narrow \"audio_output\" in local_config.json.",
                       match, len(hits),
                       ", ".join(f"{d.get('Name')} / {d.get('Device Name')}" for d in hits))
        return False

    target = hits[0]
    device_id = target.get("Command-Line Friendly ID") or target.get("Name")
    for role in (_ROLE_CONSOLE, _ROLE_MULTIMEDIA):
        try:
            subprocess.run([svv, "/SetDefault", device_id, role], timeout=15,
                           check=False, creationflags=_CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as err:
            logger.warning("audio preflight: /SetDefault failed (%s).", err)
            return False

    logger.info("audio preflight: switched default output to %r / %r.",
                target.get("Name"), target.get("Device Name"))
    return True


# ------------------------------------------------------------------- preflight

def describe():
    """Human-readable summary of the current route, for eyeballing on a new machine."""
    state = current_output()
    lines = ["Default playback device:"]
    if state.get("error") and state.get("name") is None:
        lines.append(f"  could not be read: {state['error']!r}")
    else:
        lines.append(f"  name   : {state['name']}")
        volume = state.get("volume")
        lines.append(f"  volume : {volume:.1f}%" if volume is not None else "  volume : unknown")
        lines.append(f"  muted  : {state.get('muted')}")

    required = local_config.get("audio_output")
    lines.append("")
    lines.append(f"local_config audio_output : {required!r}"
                 + ("" if required else "  (check disabled on this machine)"))
    lines.append(f"local_config audio_volume : {local_config.get('audio_volume', 50)}")

    svv = soundvolumeview_path()
    lines.append(f"SoundVolumeView           : {svv or 'not found (autofix unavailable)'}")
    if svv:
        lines.append("")
        lines.append("Active playback devices (copy a substring into audio_output):")
        for device in list_render_devices(svv):
            default = " <- default" if device.get("Default") == "Render" else ""
            lines.append(f"  {device.get('Name')!r} / {device.get('Device Name')!r}"
                         f"  {device.get('Volume Percent')}{default}")
    return "\n".join(lines)


def preflight(match=None, level=None, autofix=None, strict=None,
              tolerance=VOLUME_TOLERANCE):
    """
    Check -- and optionally fix -- the playback route before a session.

    Returns None when this machine does not ask for the check, True when the
    route is correct, False when it is wrong and ``strict`` is off. Raises
    :class:`AudioRouteError` when it is wrong and ``strict`` is on.

    Parameters
    ----------
    match : str, optional
        Substring of the required device name. Defaults to ``audio_output`` from
        local_config.json; when that is absent the whole check is skipped.
    level : int, optional
        Required master volume in percent. Defaults to ``audio_volume`` or 50.
    autofix : bool, optional
        Try to switch the default device before failing. Defaults to
        ``audio_autofix`` or True.
    strict : bool, optional
        Raise instead of returning False. Defaults to ``audio_strict`` or True.
    """
    match = match if match is not None else local_config.get("audio_output")
    if not match:
        logger.debug("audio preflight: no \"audio_output\" in local config — skipped.")
        return None

    if platform.system() != "Windows":
        logger.info("audio preflight: skipped (not Windows).")
        return None

    level = int(level if level is not None else local_config.get("audio_volume", 50))
    autofix = _as_bool(autofix if autofix is not None else local_config.get("audio_autofix"), True)
    strict = _as_bool(strict if strict is not None else local_config.get("audio_strict"), True)

    def fail(reason, hint=""):
        message = f"Audio route wrong: {reason}"
        if hint:
            message += f"\n{hint}"
        if strict:
            raise AudioRouteError(message)
        logger.error("audio preflight: %s", message)
        return False

    state = current_output()
    name = state.get("name")
    if name is None:
        return fail(
            f"could not read the default playback device ({state.get('error', 'unknown')!r}).",
            "Check that pycaw and comtypes are installed in this environment.")

    # --- right device? ---
    if match.casefold() not in name.casefold():
        if autofix and set_default_output(match):
            state = current_output()
            name = state.get("name") or ""
        if match.casefold() not in (name or "").casefold():
            svv = soundvolumeview_path()
            available = ""
            if svv:
                available = "Active playback devices:\n" + "\n".join(
                    f"  - {d.get('Name')} / {d.get('Device Name')}"
                    for d in list_render_devices(svv))
            return fail(
                f"default output is {name!r}, this machine requires a device "
                f"matching {match!r}.",
                "Plug the interface in and select it in the Windows sound "
                "settings, then start again.\n" + available)

    # --- right level? ---
    set_windows_volume(level)  # also unmutes; never raises
    state = current_output()
    volume, muted = state.get("volume"), state.get("muted")

    if volume is None:
        return fail(f"could not read the volume of {name!r} "
                    f"({state.get('error', 'unknown')!r}).")
    if muted:
        return fail(f"{name!r} is muted.", "Unmute it in the Windows mixer.")
    if abs(volume - level) > tolerance:
        return fail(
            f"{name!r} is at {volume:.1f}%, this machine requires {level}%.",
            "The slider is not host-controlled on this endpoint — set it by hand. "
            "The presentation level is only calibrated at that setting.")

    logger.info("audio preflight: %r at %.1f%% — route OK.", name, volume)
    return True


def preflight_or_exit(label="Session", **kwargs):
    """
    :func:`preflight`, but a wrong route ends the process instead of raising.

    Call this from a script's ``__main__`` -- before anything expensive -- so the
    operator gets the one line that matters rather than a traceback::

        audio_device.preflight_or_exit("Training")
    """
    try:
        return preflight(**kwargs)
    except AudioRouteError as err:
        raise SystemExit(f"[{label}] {err}")


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    print(describe())
