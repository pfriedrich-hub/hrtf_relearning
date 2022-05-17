import slab
import numpy
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from pathlib import Path
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# compare hrtfs
# get hrtfs with similar source coordinates
filename = 'kemar_fflab.sofa'
filename = 'jp.sofa'
filename = 'mit_kemar_large_pinna.sofa'


hrtf = slab.HRTF(data_dir / 'hrtfs' / filename)
azs = numpy.unique(hrtf.sources[:, 0])
az = 0
for az in azs:
    sources = hrtf.cone_sources(az, coords='interaural')
    hrtf.plot_tf(sources, xlim=(0, 25e3))
    plt.title('cone at azimuth: %f' % az)

kemar = slab.HRTF.kemar()
hrtf = slab.HRTF(data_dir / 'hrtfs' / str(filename + '.sofa'))
# kemar = slab.HRTF(str(data_dir) + '/hrtfs/examples/mit_kemar_large_pinna.sofa')
# compare waterfall
cs1 = hrtf.cone_sources(cone=35, coord_system='interaural', full_cone=False)
cs2 = kemar.cone_sources(cone=35, coord_system='polar', full_cone=False)

hrtf.plot_tf(cs1, n_bins=400)
kemar.plot_tf(cs2, n_bins=500)

hrtf.sources[37] # 35, 12.5, 1.4
kemar.sources[339] # 35, 10, 1.4
chrp = slab.Sound.chirp()
hrtf.apply(37, chrp).spectrum()
kemar.apply(339, chrp).spectrum()

# write sofa
def read_wav(path):
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = sorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path))
    return recordings

slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=0.05, level=90, from_frequency=0, to_frequency=18000, samplerate=fs)
recs = read_wav(path=data_dir / 'in-ear_recordings' / 'kemar_fflab')
sources = numpy.loadtxt(data_dir / 'in-ear_recordings' / 'kemar_fflab' /'sources_kemar_fflab.txt')
hrtf1 = slab.HRTF.estimate_hrtf(recs, signal, sources)
hrtf1.write_sofa(filename=data_dir / 'hrtfs' / 'kemar_fflab.sofa')

# move sound (use slab transition) around using hrtfs


# load some coin sounds
coin64 = slab.Sound('/Users/paulfriedrich/Projects/hrtf_relearning/data/Mario64_Coin.wav')
coin = coin64.resample(fs)
coin = slab.Sound('/Users/paulfriedrich/Projects/hrtf_relearning/data/Mario_Coin.wav')
coins = hrtf.apply(src[0], coin)
coins.play()


