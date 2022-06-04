import slab
import numpy
from pathlib import Path
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
import freefield
import helper_functions
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

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
hrtf = slab.HRTF(data_dir / 'hrtfs' / filename)
# kemar = slab.HRTF(str(data_dir) + '/hrtfs/examples/mit_kemar_large_pinna.sofa')
# compare waterfall
cs1 = hrtf.cone_sources(cone=0, coords='interaural', full_cone=False)
hrtf.plot_sources(cs1, coords='interaural')
cs2 = kemar.cone_sources(cone=0, coords='polar', full_cone=False)
hrtf.plot_tf(cs1, n_bins=800, kind='waterfall')
kemar.plot_tf(cs2, n_bins=800, kind='surface')
hrtf.sources[37] # 35, 12.5, 1.4
kemar.sources[339] # 35, 10, 1.4
chrp = slab.Sound.chirp()
hrtf.apply(37, chrp).spectrum()
kemar.apply(339, chrp).spectrum()

# write sofa
filename = 'kemar_fflab'
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=0.1, level=70, from_frequency=200, to_frequency=20000, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=0.001)
recs = helper_functions.read_wav(path=data_dir / 'in-ear_recordings' / filename)
sources = numpy.loadtxt(data_dir / 'in-ear_recordings' / 'kemar_fflab' / str('sources_' + filename + '.txt'))
hrtf = slab.HRTF.estimate_hrtf(recs, signal, sources)
hrtf.write_sofa(filename=data_dir / 'hrtfs' / str(filename + '.sofa'))

# move sound (use slab transition) around using hrtfs

#------ plot waveforms / spectra for each column -----#
# get speaker id's for each column in the dome
freefield.initialize('dome', default='play_rec')  # initialize setup
azimuthal_angles = numpy.array([-52.5, -35, -17.5, 0, 17.5, 35, 52.5])
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
speaker_list = []
for az in azimuthal_angles:
    speaker_list.append((speaker_table[speaker_table[:, 1] == az][:, 0]).astype('int'))
for speaker_column in speaker_list:
    speakers = freefield.pick_speakers(speaker_column)
    fig, axis = plt.subplots(len(speaker_column), 1, sharex=True)
    for i, speaker_id in enumerate(speaker_column):
        # recs[speaker_id].waveform(axis=axis[i])
        recs[speaker_id].spectrum(low_cutoff=1000, high_cutoff=18000, axis=axis[i])
        fig.suptitle('column at %.1f° azimuth' % speakers[i].azimuth, fontsize=12)
        axis[i].set_title('%.1f elevation' % (speakers[i].elevation), fontsize=8)



# load some coin sounds
coin64 = slab.Sound(data_dir / 'sounds' / 'Mario64_Coin.wav')
coin = coin64.resample(fs)
coin = slab.Sound('/Users/paulfriedrich/Projects/hrtf_relearning/data/Mario_Coin.wav')
coins = hrtf.apply(cs1[0], coin)
coins.play()

# linear chirp
chirp = slab.Sound.chirp(duration=0.5, from_frequency=20, to_frequency=20000, kind='linear')
chirp = chirp.ramp(duration=0.02, when='both')
chirp.waveform()
chirp.spectrum()

# scipy chirp
t = np.linspace(0, 10, 1500)
w = chirp(t, f0=6, f1=1, t1=10, method='linear')
plt.plot(t, w)
plt.title("Linear Chirp, f(0)=6, f(10)=1")
plt.xlabel('t (sec)')
plt.show()
fs = 48828
T = 0.5
t = np.arange(0, int(T*fs)) / fs

def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = spectrogram(w, fs=fs, nperseg=256, nfft=576)
    plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r', shading='gouraud')
    plt.title(title)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()


# matplotlib spectrun
rec =slab.Sound.read('/Users/paulfriedrich/Projects/hrtf_relearning/data/in-ear_recordings/kemar_fflab/in-ear_kemar_fflab_src_id70_az0_el0.wav')
fs = 48828
t = rec.times
s = rec.channel(0).data[:, 0]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axs.set_title("Log. Magnitude Spectrum")
axs.magnitude_spectrum(s, Fs=fs, scale='dB', color='C1')
