import slab
import numpy
import matplotlib
import random
# matplotlib.use('MacOSX')
import freefield
from matplotlib import pyplot as plt
from pathlib import Path
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# compare hrtfs
# get hrtfs with similar source coordinates
hrtf = slab.HRTF(str(data_dir) + '/hrtfs/kemar.sofa')
kemar = slab.HRTF.kemar()
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


sound = slab.Sound.pinknoise(duration=2.0)
def front_back_test(sofa_file='kemar', sound=sound, freefield=False):
    if freefield:
        freefield.set_logger('warning')
        freefield.initialize(setup='dome', default="loctest_headphones")
        print('\n.\n.\n.\n.\n.Press GREEN button for sounds from the FRONT \n and RED for sounds from the BACK.')
    hrtf = slab.HRTF(str(data_dir) + '/hrtfs/%s.sofa' % sofa_file)
    # get cones for every azimuth
    azs = numpy.unique(hrtf.sources[:, 0])
    cone_src = []
    for az in azs:
        src = hrtf.cone_sources(az, coord_system='interaural', full_cone=True)
        cone_src.append(src)
    # play from random sources on a single random cone and test listeners front back discrimination
    hits = []
    for c in range(5):
        cone = random.choice(cone_src)
        print(numpy.unique(hrtf.sources[cone][:, 0]))
        for s in range(10):
            source = random.choice(cone)
            #  print hrtf.sources[source]
            if numpy.deg2rad(hrtf.sources[source][0]) < 1:
                direction = 'front'
            else: direction = 'back'
            #  print(direction)
            dir_snd = hrtf.apply(source, sound)
            if freefield:
                freefield.write('playbuf', dir_snd)
                freefield.write('playbuflen', dir_snd.n_samples)
                freefield.play('RP2')
                choice=(freefield.wait_for_button())
            else:
                sound.play()
                choice = (inumpyut('Press ''F'' for sounds from the FRONT / ''B'' for sounds from the BACK.'))
            if ((choice == 'F' or choice == '0') and direction == 'front') or \
                ((choice == 'B' or choice == '1') and direction == 'back'):
                hits.append(1)
            else: hits.append(0)
    return hits.count(1) / len(hits)


slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=1.0, level=90, from_frequency=0, to_frequency=18000, samplerate=fs)
recs = read_wav(path=data_dir / 'in-ear_recordings' / 'KEMAR')
sources = numpy.loadtxt(data_dir / 'in-ear_recordings' / 'KEMAR' /'sources_kemar_test.txt')
recorded_hrtf = slab.HRTF.estimate_hrtf(recs, signal, sources)
recorded_hrtf.write_sofa(filename=data_dir / 'hrtfs' / 'kemar.sofa')

# move sound (use slab transition) around using hrtfs

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

# load some coin sounds
coin64 = slab.Sound('/Users/paulfriedrich/Projects/hrtf_relearning/data/Mario64_Coin.wav')
coin = coin64.resample(fs)
coin = slab.Sound('/Users/paulfriedrich/Projects/hrtf_relearning/data/Mario_Coin.wav')
coins = hrtf.apply(src[0], coin)
coins.play()


