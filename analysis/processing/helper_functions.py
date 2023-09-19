import slab
from pathlib import Path
import numpy
data_dir = Path.cwd() / 'final_data'

def read_wav(path):
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = sorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path))
    return recordings, path_list

def hrtf_signal(level=80, duration=0.1, low_freq=1000, high_freq=17000, fs = 97656):
    ramp_duration = duration / 20
    slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
    signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq,
                              kind='linear')
    signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
    return signal

def read_source_txt(path):
    for file_path in path.rglob('*.txt'):
        sources = numpy.loadtxt(file_path)
    return sources
