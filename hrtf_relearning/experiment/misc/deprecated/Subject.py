# matplotlib.use('tkagg')
from experiment.misc.deprecated.safe_pickle import load, dump
import logging
from pathlib import Path
results_dir = Path.cwd() / 'data' / 'results'
import sys
from pathlib import Path
# Add parent directory of 'experiment' to path
sys.path.insert(0, str(Path.cwd()))

class Subject:
    def __init__(self, id):
        self.file_path = results_dir / f'{id}.pkl'
        # check if subject exists in data folder and laod the data
        if self.file_path.exists():
            logging.info('Loading subject data.')
            with open(self.file_path, 'rb') as subj_file:
                subject = load(subj_file)
                self.__dict__ = subject.__dict__.copy()
                self.file_path = results_dir / f'{id}.pkl'  # overwrite Path to match current system for writing

        else:  # otherwise create a new subject object
            logging.info('Creating new subject data.')
            self.id = id
            self.localization = dict()
            self.trials = []

        if list(self.localization.keys()):  # get the last localization sequence
            self.last_sequence = self.localization[list(self.localization.keys())[-1]]
        else: self.last_sequence = None

    def write(self):
        if (self.file_path).exists():
            logging.debug('Updating subject file.')
        else: logging.info('Creating subject file.')
        with open(self.file_path, 'wb') as subj_file:
            dump(self, subj_file)  # highest protocol dumping -> numpy error while loading on mac

