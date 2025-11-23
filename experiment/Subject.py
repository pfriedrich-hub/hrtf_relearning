import json
import logging
from pathlib import Path

results_dir = Path.cwd() / "data" / "results"
results_dir.mkdir(parents=True, exist_ok=True)


class Subject:
    def __init__(self, id: str):
        self.id = id
        self.file_path = results_dir / f"{id}.json"
        self.localization = {}
        self.trials = []
        self.last_sequence = None

        if self.file_path.exists():
            logging.info("Loading subject data.")
            self._load()
        else:
            logging.info("Creating new subject.")

    def _load(self):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        self.id = data.get("id", self.id)
        self.localization = data.get("localization", {})
        self.trials = data.get("trials", [])
        self.last_sequence = data.get("last_sequence", None)
        self.file_path = results_dir / f"{self.id}.json"

    def save(self):
        data = {
            "id": self.id,
            "localization": self.localization,
            "trials": self.trials,
            "last_sequence": self.last_sequence,
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
