import slab
import numpy


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
                              kind='quadratic')
    signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
    return signal

def read_source_txt(path):
    for file_path in path.rglob('*.txt'):
        sources = numpy.loadtxt(file_path)
    return sources


"""
# create hrtf by microphone origin transfer function measured from the dome central arc with an in-the-ear mic
signal = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'mean_central_arc_rec.wav')
signal = slab.Sound(data=numpy.mean(signal.data, axis=1), samplerate=97656)
hrtf = slab.HRTF.estimate_hrtf(recordings, signal, sources)

result: increased artifacts (compared to clean chirp as reference signal)
"""

"""
# ole_test create hrtf by quadratic sweep signal instead of linear to detrend dtfs
# result: doesnt do much to vsi / spectral strength
subject = 'vk'
condition = 'Ears Free'

data_dir = Path.cwd() / 'data' / 'experiment' / 'master'/ subject / condition
signal = hrtf_signal()
recordings = read_wav(data_dir)[0]
sources = read_source_txt(data_dir)
hrtf = slab.HRTF.estimate_hrtf(recordings, signal, sources)

"""