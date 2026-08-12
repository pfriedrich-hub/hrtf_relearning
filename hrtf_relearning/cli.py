import sys
import runpy

COMMANDS = {
    "training": "hrtf_relearning.experiment.training.Training_AR",
    "localize-ar": "hrtf_relearning.experiment.localization.Localization_AR",
    "localize-vr": "hrtf_relearning.experiment.localization.Localization_VR",
    "analyze": "hrtf_relearning.experiment.analysis.localization.localization_analysis",
}

BUILTINS = {
    "backend": "report the matplotlib backend and open a test plot window",
}


def _backend():
    """`hrtf backend` — check on a new machine whether plot windows work."""
    import hrtf_relearning  # noqa: F401  (its import resolves the backend)
    from hrtf_relearning.utils import mpl_backend

    print(mpl_backend.describe())
    if mpl_backend.current().lower() == mpl_backend.HEADLESS.lower():
        print("\nNo interactive backend — figures can be saved but no window opens.")
        try:
            import tkinter  # noqa: F401
        except ImportError as err:
            print(f"tkinter is not importable ({err}). It ships with the interpreter "
                  "and cannot be pip-installed: reinstall Python from python.org, or "
                  "`pip install pyqt5` to use the Qt fallback instead.")
        else:
            print("Fix: pip install pyqt5")
        return 1

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 4, 9], marker="o")
    ax.set_title(f"backend: {mpl_backend.current()} — close this window")
    print("\nA plot window should be open now. Close it to finish.")
    plt.show()
    print("Window closed — plotting works on this machine.")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Available commands:")
        for k in COMMANDS:
            print(f"  {k}")
        for k, help_text in BUILTINS.items():
            print(f"  {k:<12} {help_text}")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "backend":
        sys.exit(_backend())

    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    # forward remaining args
    sys.argv = [cmd] + sys.argv[2:]

    runpy.run_module(COMMANDS[cmd], run_name="__main__")
