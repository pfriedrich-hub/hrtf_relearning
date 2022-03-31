import slab
from pathlib import Path
import numpy
rec_path = Path.cwd() /'data' / 'in-ear_recordings' / 'pilot'
source_file = rec_path / 'in-ear_paul_sources.txt'  # csv containing sound source coordinates
sources = numpy.loadtxt(source_file, skiprows=1, usecols=(1, 2, 3), delimiter=",")
probe_len = 0.5  # length of the sound probe in seconds
fs = 48828  # sampling rate

chirp = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz

hrtf = slab.HRTF(data=str(Path.cwd() / 'data' / 'hrtfs' / 'test_hrtf.sofa'))

# on headphones
for source in range(hrtf.n_sources):
    print(source)
    snd = hrtf.apply(source, chirp)
    snd.play()

# play from cones of confusion and get hit rate for front-back confusion
for source in range(hrtf.cone_sources):
    print(source)
    snd = hrtf.apply(source, chirp)
    snd.play()

# move sound (use slab transition) around using hrtfs
