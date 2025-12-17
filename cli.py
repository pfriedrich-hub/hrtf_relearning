import sys
import runpy

COMMANDS = {
    "training": "hrtf_relearning.experiment.training",
    "record-dome": "hrtf_relearning.experiment.record_dome",
    "analyze": "hrtf_relearning.experiment.analyze_localization",
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