from pathlib import Path

class Participants:
    def __init__(self, id):
        self.id = id
        self.data_dir = Path.cwd() / 'data' / 'results' / self.id
        self.localization_data = []
        self.hrtf_data = []


class localization_results:
    def __init__(self, condition):
        trialsequence = []


