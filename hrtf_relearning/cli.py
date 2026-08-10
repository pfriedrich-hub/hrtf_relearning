import sys
import runpy

COMMANDS = {
    "training": "hrtf_relearning.experiment.training.Training_AR",
    "localize-ar": "hrtf_relearning.experiment.localization.Localization_AR",
    "localize-vr": "hrtf_relearning.experiment.localization.Localization_VR",
    "analyze": "hrtf_relearning.experiment.analysis.localization.localization_analysis",
}

def main():
    if len(sys.argv) < 2:
        print("Available commands:")
        for k in COMMANDS:
            print(f"  {k}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    # forward remaining args
    sys.argv = [cmd] + sys.argv[2:]

    runpy.run_module(COMMANDS[cmd], run_name="__main__")