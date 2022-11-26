import slab
from pathlib import Path
data_dir = Path.cwd() / 'data'

def read_wav(path):
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = sorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path))
    return recordings