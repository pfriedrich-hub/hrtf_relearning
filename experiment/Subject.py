from pathlib import Path
import pickle
import logging
results_dir = Path.cwd() / 'data' / 'results'

class Subject:
    def __init__(self, id):
        self.file_path = results_dir / f'{id}.pkl'

        # check if subject exists in data folder and laod the data
        if (self.file_path).exists():
            logging.info('Loading subject data.')
            with open(self.file_path, 'rb') as subj_file:
                subject = pickle.load(subj_file)
                self.__dict__ = subject.__dict__.copy()

        else:  # otherwise create a new subject object
            logging.info('Creating new subject data.')
            self.id = id
            self.localization = dict()

    def write(self):
        if (self.file_path).exists():
            logging.info('Updating subject file.')
        else: logging.info('Creating subject file.')
        with open(self.file_path, 'wb') as subj_file:
            pickle.dump(self, subj_file, pickle.HIGHEST_PROTOCOL)

