import json
import logging
from pathlib import Path
import pickle
import hrtf_relearning
results_dir = Path(hrtf_relearning.__file__).resolve().parent / "data" / "results"
results_dir.mkdir(parents=True, exist_ok=True)


class Subject:
    def __init__(self, id: str):
        self.file_path = results_dir / f"{id}.pkl"
        self.id = id
        if self.file_path.exists():
            self._load()
        else:
            logging.info("Creating new subject.")
            self.localization = {}
            self.trials = []
            self.last_sequence = None
            self.highscore = 0

    def _load(self):
        logging.info("Loading subject data.")
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        self.id = data.get("id", self.id)
        self.localization = data.get("localization", {})
        self.trials = data.get("trials", [])
        self.last_sequence = data.get("last_sequence", None)
        self.highscore = data.get("highscore", 0)

    def write(self):
        logging.debug("Writing subject data.")
        data = {
            "id": self.id,
            "localization": self.localization,
            "trials": self.trials,
            "last_sequence": self.last_sequence,
            "highscore": self.highscore,
        }
        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)