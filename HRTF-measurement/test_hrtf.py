import slab
import numpy as np
import matplotlib
# matplotlib.use('MacOSX')
# from matplotlib import pyplot as plt
from pathlib import Path
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate

# write sofa
def read_wav(path):
    from natsort import natsorted
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = natsorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path).data)
    return slab.Sound(data=recordings, samplerate=fs)

slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=1.0, level=90, from_frequency=200, to_frequency=16000, samplerate=fs)
recs = read_wav(path=data_dir / 'in-ear_recordings' / 'KEMAR')
sources = np.loadtxt(data_dir / 'in-ear_recordings' / 'KEMAR' /'sources_kemar_test.txt')
# sources = sources[:, 1:]
recorded_hrtf = slab.HRTF.estimate_hrtf(recs, signal, sources)
recorded_hrtf.write_sofa(filename=data_dir / 'hrtfs' / 'kemar')

# read back
hrtf = slab.HRTF(str(data_dir) + '/hrtfs/KEMAR.sofa')
# get azimuths
azs = np.unique(hrtf.sources[:, 0])

cone_src = hrtf.sources[hrtf.cone_sources(17.5)]
cone_idx = hrtf.cone_sources(1)
    hrtf.apply(cone_idx)


# on headphones
for source in range(hrtf.n_sources):
    print(source)
    snd = hrtf.apply(source, chirp)
    snd.play()

# play from cones of confusion and get hit rate for front-back confusion
for i in
    for source in range(hrtf.cone_sources(i)):
        print(source)
        snd = hrtf.apply(source, chirp)
        snd.play()

# move sound (use slab transition) around using hrtfs


