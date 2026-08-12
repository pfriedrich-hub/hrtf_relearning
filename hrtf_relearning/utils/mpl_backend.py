"""One matplotlib backend for the whole package, chosen once at import time.

Why this module exists
----------------------
Plots would not open, or opened and froze, on most of the Windows machines this
runs on. Four separate causes, all of them fixed here:

1. **PyCharm's SciView backend.** PyCharm injects ``MPLBACKEND=module://backend_interagg``
   (its "Show plots in tool window" setting) into the run configuration. That
   backend renders into the IDE tool window and regularly hangs or silently
   drops figures when the script also runs subprocesses or an event loop, which
   every protocol here does. It is overridden unconditionally below.

2. **Every module picked its own backend.** Roughly fifty files each called
   ``matplotlib.use(...)`` at import time with a mix of ``TkAgg``, ``Qt5Agg``
   and ``Agg``, so which one you got depended on import order. Switching
   backends after a figure exists silently does nothing, which is why plots
   opened in one script and not in the next. The choice now happens exactly
   once, here, before anything imports pyplot.

3. **A blocking ``plt.show()`` in the PyCharm Python Console.** This is what
   most of the "frozen plot" reports actually were, on either toolkit.
   ``plt.show()`` calls the toolkit's ``mainloop()``, which blocks the console
   prompt until the window is closed. Ctrl-C gives the prompt back but rips the
   event loop out from under a window that is still on screen, and from then on
   it does not repaint or respond::

       File "matplotlib/backends/_backend_tk.py", line 583, in start_main_loop
         first_manager.window.mainloop()
       KeyboardInterrupt

   No backend fixes this. :func:`_enable_console_gui` does, by giving the
   console's input hook the job of pumping the toolkit between prompts, after
   which interactive mode is safe and ``plt.show()`` stops blocking.

4. **QtAgg was reported not to give usable windows here**, while Tk is confirmed
   to open a live, movable window. Cause 3 accounts for at least part of that
   report and would hit Qt just as hard, so this is a preference rather than a
   diagnosis — if Qt ever turns out to be the better default, changing
   :data:`CANDIDATES` is the whole edit.

So: **TkAgg is the backend**. ``QtAgg`` stays as the fallback for a machine
whose interpreter was built without Tk (``tkinter`` cannot be pip-installed, so
that case cannot be fixed by a dependency), and ``Agg`` (file-only, never opens
a window) is the last resort so a script that only saves PNGs still runs on a
headless box.

The known cost of Tk: creating a Tk figure can fail *lazily* when a plot
function runs from inside ``Localization.run()`` — i.e. after the pybinsim
multiprocessing worker, the training subprocess and the pynput listener have
started. ``localization_analysis._safe_subplots`` catches that and falls back to
``Agg`` so the PNG is still written.

Resolution order (first hit wins), mirroring :mod:`hrtf_relearning.utils.local_config`:

1. environment variable ``HRTF_MPL_BACKEND``   -- one-off override
2. ``mpl_backend`` in ``local_config.json``    -- this machine
3. auto-detection over :data:`CANDIDATES`

Any matplotlib backend name works in 1. and 2.; ``"agg"``, ``"none"`` and
``"headless"`` all mean "never open a window".

Usage
-----
Nothing to call: ``import hrtf_relearning`` (which every module here does) runs
:func:`use_interactive` before anything imports ``pyplot``. Call it explicitly
only in a script that must not import the package first::

    from hrtf_relearning.utils.mpl_backend import use_interactive
    use_interactive()

Batch scripts that only write files should say so, rather than hardcoding Agg::

    from hrtf_relearning.utils.mpl_backend import use_headless
    use_headless()
"""
import importlib
import logging
import os
import sys

from hrtf_relearning.utils import local_config

logger = logging.getLogger(__name__)

#: Interactive backends to try, best first. See the module docstring for why Tk
#: is preferred over Qt.
CANDIDATES = ("TkAgg", "QtAgg")

#: Non-interactive backend: renders to file, never opens a window, always importable.
HEADLESS = "Agg"

#: local_config.json key / ``HRTF_MPL_BACKEND`` environment variable.
CONFIG_KEY = "mpl_backend"

#: Values that mean "do not open windows at all".
_HEADLESS_ALIASES = {"agg", "none", "off", "headless", "file", "false"}

#: IDE-injected backends we refuse to inherit (PyCharm SciView, see docstring).
_IDE_BACKENDS = ("backend_interagg", "backend_inline")

_resolved = None


def _drop_ide_backend_env():
    """Unset ``MPLBACKEND`` when PyCharm points it at its own tool-window backend.

    Has to happen before matplotlib is imported: matplotlib reads ``MPLBACKEND``
    once at import and PyCharm's hook keys off the same variable, so clearing it
    first is what actually stops SciView from taking over — a later
    ``use(..., force=True)`` swaps the canvas but leaves pydev's patched
    ``plt.show`` in place.
    """
    value = os.environ.get("MPLBACKEND", "")
    if any(name in value for name in _IDE_BACKENDS):
        logger.debug("ignoring IDE-injected MPLBACKEND=%s", value)
        del os.environ["MPLBACKEND"]


def _backend_module(backend):
    """Import path of the backend module for `backend` ('QtAgg' -> ...backend_qtagg)."""
    return "matplotlib.backends.backend_" + backend.lower()


def _apply(backend):
    """Switch matplotlib to `backend`; True on success, False if it is unusable.

    ``matplotlib.use()`` alone is not a test: with pyplot not yet imported it
    only writes ``rcParams['backend']`` and the missing-toolkit ImportError does
    not surface until the first figure is created — long after the point where
    we could still fall back. Importing the backend module first turns that into
    an error we can handle here.
    """
    import matplotlib

    try:
        if not backend.startswith("module://"):
            importlib.import_module(_backend_module(backend))
        matplotlib.use(backend, force=True)
    except Exception as err:  # ImportError, but Qt/Tk raise plenty of others
        logger.debug("matplotlib backend %r unusable: %s: %s",
                     backend, type(err).__name__, err)
        return False
    return True


def _has_display():
    """True unless we are on a Unix box with no X/Wayland session."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


#: matplotlib backend -> GUI toolkit name understood by the console input hooks.
_GUI_NAMES = {"tkagg": "tk", "qtagg": "qt5", "qt5agg": "qt5", "qt6agg": "qt6",
              "wxagg": "wx", "macosx": "osx"}


def _in_console():
    """True in an interactive REPL — PyCharm's Python Console, IPython, or plain python -i.

    Not true for a script run, where a blocking ``plt.show()`` at the end is the
    only thing keeping the window on screen until the process exits.
    """
    if "pydevconsole" in sys.modules:      # PyCharm Python Console (the `# %%` workflow)
        return True
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return bool(getattr(sys, "ps1", None))  # python -i


def _enable_console_gui(backend):
    """Hand the GUI event loop to the console so figures live next to the prompt.

    In a console, a bare ``plt.show()`` calls ``mainloop()`` and blocks the
    prompt until the window is closed; interrupting it with Ctrl-C returns the
    prompt but leaves the window on screen with nothing pumping its events, so
    it hangs — the "plot freezes" everyone hits. The cure is not a different
    backend: it is to let the console's input hook pump the toolkit between
    prompts. Then interactive mode is safe, ``plt.show()`` returns immediately
    and the figure stays live and redrawable while you keep typing.

    Interactive mode is only switched on if a hook was actually installed —
    ``ion()`` without one gives a window that never blocks *and* never responds,
    which is worse than blocking.
    """
    gui = _GUI_NAMES.get(backend.lower())
    if gui is None or not _in_console():
        return False

    installed = False
    try:
        # PyCharm ships these helpers and puts them on sys.path for its console;
        # its startup registers the return_control callback that enable_gui needs.
        from pydev_ipython.inputhook import enable_gui, get_inputhook
        enable_gui(gui)
        installed = get_inputhook() is not None
        if not installed:
            logger.debug("pydev accepted gui %r but installed no input hook", gui)
    except Exception as err:
        logger.debug("pydev input hook for %s unavailable: %s: %s",
                     gui, type(err).__name__, err)

    if not installed:
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic("matplotlib", gui)
            installed = True
        except Exception as err:
            logger.debug("IPython gui integration for %s unavailable: %s: %s",
                         gui, type(err).__name__, err)

    if not installed:
        logger.info(
            "Interactive console without GUI event-loop integration: plt.show() "
            "will block until you close the window. Close it rather than "
            "interrupting with Ctrl-C — an interrupted window stops responding.")
        return False

    import matplotlib.pyplot as plt
    plt.ion()
    logger.debug("console gui integration: %s (interactive mode on)", gui)
    return True


def _settle(backend, source):
    """Record `backend` as the resolved one and wire up console integration."""
    global _resolved
    _resolved = backend
    logger.debug("matplotlib backend %s (%s)", backend, source)
    _enable_console_gui(backend)
    return _resolved


def _requested():
    """Backend asked for by the environment or local_config.json, or None."""
    return local_config.get(CONFIG_KEY)


def use_interactive(force=False):
    """Select and apply the interactive backend for this machine.

    Called once from ``hrtf_relearning/__init__.py``; repeat calls are no-ops so
    it is safe at the top of any module. Pass ``force=True`` to re-resolve after
    something else changed the backend.

    Returns
    -------
    str
        The backend now in use — one of :data:`CANDIDATES`, :data:`HEADLESS`, or
        whatever was requested explicitly.
    """
    global _resolved
    if _resolved is not None and not force:
        return _resolved

    _drop_ide_backend_env()
    requested = _requested()
    if requested:
        if str(requested).lower() in _HEADLESS_ALIASES:
            return use_headless()
        if _apply(requested):
            return _settle(requested, "local config")
        logger.warning(
            "matplotlib backend %r requested (%s / HRTF_MPL_BACKEND) but not usable "
            "here — falling back to auto-detection.",
            requested, local_config.config_path() or "local_config.json")

    if not _has_display():
        logger.debug("no display — matplotlib backend %s", HEADLESS)
        return use_headless()

    for backend in CANDIDATES:
        if _apply(backend):
            return _settle(backend, "auto-detect")

    logger.warning(
        "No interactive matplotlib backend available (tried %s) — using %s, so "
        "figures are saved to file but no plot window opens. Usually this means "
        "the interpreter was built without tkinter; `pip install pyqt5` gets the "
        "fallback backend, or reinstall Python from python.org, which ships Tk.",
        ", ".join(CANDIDATES), HEADLESS)
    return use_headless()


def use_headless():
    """Force the file-only :data:`HEADLESS` backend (no window, savefig always works).

    For batch/QC scripts that only write figures, and as the recovery path when
    an interactive backend fails at figure-creation time.
    """
    global _resolved
    _apply(HEADLESS)
    _resolved = HEADLESS
    return _resolved


def current():
    """The backend matplotlib is actually using right now."""
    import matplotlib
    return matplotlib.get_backend()


def describe():
    """One-line summary of the backend situation, for diagnostics."""
    import matplotlib

    available = [b for b in CANDIDATES if _import_ok(b)]
    if _in_console():
        import matplotlib.pyplot as plt
        console = ("console, interactive mode on — plt.show() returns immediately"
                   if plt.isinteractive() else
                   "console, no event-loop integration — plt.show() blocks")
    else:
        console = "script — plt.show() blocks, as it should"
    return (f"matplotlib {matplotlib.__version__}, backend {current()!r} "
            f"(resolved: {_resolved}, requested: {_requested()!r}, "
            f"interactive backends available: {', '.join(available) or 'none'}; "
            f"{console})")


def _import_ok(backend):
    """True if `backend`'s toolkit is importable — probe only, does not switch."""
    try:
        importlib.import_module(_backend_module(backend))
    except Exception:
        return False
    return True
