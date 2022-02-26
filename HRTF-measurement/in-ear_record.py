# record form in-ear microphones
# example - from terminal/shell:
# python in-ear_record.py --id paul_hrtf

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import slab
import freefield
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--id", type=str,
# 	default="paul_hrtf",
# 	help="enter subject id")
# args = vars(ap.parse_args())
# id = args["id"]
# print('record from %s speakers, subj_id: %i'%(id, 9))

# get speakers and locations(az,ele) to play from
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
table = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
# todo record from whole dome
speakers = table[20:27]  # for now only use positive az (half dome)

fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
probe_len = 0.5  # length of the sound probe in seconds

#tone = slab.Sound.whitenoise(duration=probe_len) # chirp?
chirp = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz

def dome_rec(speakers, subject=id, n_reps=50):
    # initialize setup
    freefield.initialize('dome', default='play_birec')
    freefield.load_equalization()

    # equalize speaker transfer functions
    # freefield.equalize_speakers(speakers=speaker_coordinates)
    # rec_raw, rec_lvl, rec_full = freefield.test_equalization(speakers=speaker_speaker_coordinates)
    # play , record  and average n_reps times, from all speakers in the list + save as .wav
    freefield.set_logger('WARNING')
    recordings = np.zeros([len(speakers), int(probe_len*fs), 2])  # array to store recordings as data arrays
    avg_rec_list = []  # list to hold recordings as slab binaural objects
    for i, source_location in enumerate(speakers):
        print(source_location)
        speaker = freefield.pick_speakers(tuple((source_location[1], source_location[2])))
        # get avg of 20 recordings from each sound source location
        recs = []
        for r in range(n_reps):
            rec = freefield.play_and_record(speaker, chirp, compensate_delay=True,
                                            compensate_attenuation=False, equalize=True)
            recs.append(rec.data)
        recs = np.asarray(recs)
        avgrec = slab.Binaural(data=recs.mean(axis=0))
        avgrec.write(os.getcwd() + '/data/in-ear_recordings/in-ear_%s_%s_%s.wav'
                     % (subject, str(source_location[1]), str(source_location[2])))
        recordings[i] = avgrec.data # as np.array
        avg_rec_list.append(avgrec)
    freefield.set_logger('INFO')
    return avg_rec_list

def read_wav(speakers, subject):
    slabobj_rec = []
    recordings = np.zeros([len(speakers), int(probe_len*fs), 2])  # array to store recordings
    for i, source_location in enumerate(speakers):
        rec = slab.Binaural(slab.Sound.read(os.getcwd() + '/data/in-ear_recordings/in-ear_%s_%s_%s.wav'
                                           % (subject, str(source_location[1]), str(source_location[2]))))
        slabobj_rec.append(rec) # append slab objects to list
        recordings[i] = rec.data
    return recordings

# safe HRTFs in MRN array  shape(measurements, receivers, number_datapoints)
def HRTF_estimate(signal, recordings):
    x = signal.data[:, 0]
    N = int(len(x)/2)+1
    hrtf = np.zeros([len(recordings), 2, N], dtype=complex)
    for i, recfile in enumerate(recordings):
        yr = recfile[:, 1]
        yl = recfile[:, 0]

        # input
        xfft = np.fft.rfft(x, axis=0)  # compute discrete fourier transform
        # output
        yr_fft = np.fft.rfft(yr, axis=0)  # compute discrete fourier transform
        yl_fft = np.fft.rfft(yl, axis=0)  # compute discrete fourier transform

        # transfer function: h = y / x
        tf_r = yr_fft / xfft
        tf_l = yl_fft / xfft
        hrtf[i, 0] = tf_r
        hrtf[i, 1] = tf_l
    return hrtf


if __name__ == "__main__":
    dome_rec(speakers, )



"""
# MATLAB PSD and CSD -> HRTF 
import os
import math
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.mlab import psd, csd
import slab
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)

# generate signals
chirp = slab.Sound.chirp(duration=0.5, level=90)
snd = slab.Binaural(data=slab.Sound.read(os.getcwd() + '/data/in-ear_recordings/in-ear_paul_hrtf_0.0_0.0.wav'))
x = chirp.data[:, 0]
y = snd.left.data[:, 0]

fft_window = 512  # window size for welch approach, default = 256
overlap = int(fft_window / 2) # overlap of the hanning windows
zero_pad = 2 ** (math.ceil(math.log(len(x), 2))) # frequency resolution (zero pad signal)

pxx_x, freqs_x = psd(x, Fs=fs, NFFT=fft_window, noverlap=overlap, pad_to=zero_pad)
pxx_y, freqs_y = psd(y, Fs=fs, NFFT=fft_window, noverlap=overlap, pad_to=zero_pad)
pxx_yx, freqs = csd(y, x, Fs=fs, NFFT=fft_window, noverlap=overlap, pad_to=zero_pad)
hrtf = np.abs(pxx_yx / pxx_x)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0, 0].set_xlim(10**2, 2*10**3)
# ax[0, 0].set_ylim()
ax[0, 0].semilogx(freqs_x, np.log(pxx_x), label='signal PSD')
ax[0, 1].semilogx(freqs_y, np.log(pxx_y), label='in-ear PSD')
ax[1, 0].semilogx(freqs, np.log(hrtf), label='HRTF')
ax[1, 1].semilogx(freqs, np.log(hrtf*pxx_x), label='signal PSD * HRTF')
ax[1, 1].semilogx(freqs, np.log(pxx_y), label='in-ear PSD')


ax.set_title('nfft window: ' +str(fft_window) + ' zero_pad: ' + str(zero_pad))
ax.legend()


# original signal
plt.plot(np.arange(0, len(x)/fs, 1/fs), x)







# mpl PSD
# fft / psd parameters for mpl.psd (welch's method)
fft_window = 256  # window size for welch approach, default = 256
overlap = int(fft_window / 2) # overlap of the hanning windows
zero_pad = 2 ** (math.ceil(math.log(len(x), 2))) # frequency resolution (zero pad signal)
# estimate psd
pxx_m, freqs_m = psd(x, Fs=fs, NFFT=fft_window, noverlap=overlap, pad_to=zero_pad)

# numpy fft - PSD
freqs_n = np.fft.rfftfreq(len(x), d=1 / fs)  # frequency array to plot DFT across
rfft = np.fft.rfft(x, axis=0)  # compute discrete fourier transform
rfft = np.abs(rfft)  # euclidian distance of complex output array
rfft = rfft / len(freqs_n)  # rescale so magnitude is independent of signal length
pxx_n = rfft ** 2  # square to estimate PSD


# plot mpl and np psd estimation
fig, ax = plt.subplots(1,1)
ax.semilogx(freqs_n, np.log(pxx_n), label='numpy PSD')  # plot on logarithmic scale
ax.semilogx(freqs_m, np.log(pxx_m), label='mpl PSD')
ax.set_title('nfft window: ' +str(fft_window) + ' zero_pad: ' + str(zero_pad))
ax.legend()

# HRTF: csd(y,x)/psd(x) 
pxx_csd, freqs = csd(y, x, NFFT=fft_window, Fs=fs, pad_to=zero_pad)
hrtf = np.abs(pxx_csd / pxx_m)
#plot hrtf transformed signal psd (mpl)
ax.plot(freqs, np.log(hrtf * pxx_m), label='mpl hrtf transformed signal')




# todo calculate PSD with numpy
## Pyx is calculated (after appropriate windowing and normalization) as the average of fft(y) * conj(fft(x))
## over the individual windows. Similarly, the PSD Pxx is calculated as averaging fft(x) * conj(fft(x))

# NFFT 
# various ways to calculate nfft: (transform length)
# The length is typically specified as a power of 2
# double the number of time points (zero padding) -> double frequency resolution of fft

# nfft depends on the spectral resolution you want to achieve.
# Given a sampling rate fS and a minimum resolution fResMin,
# can simply compute the number of points nFFT by
# nFFT = ceil(fS/fResMin)
fResMin = 2 # Hz
nfft = math.ceil(fs/fResMin)

# marcs matlab approach:
Npoints = len(x)
nfft = 2 ** (math.ceil(math.log(Npoints, 2)))


# WINDOWING 


# numpy fft for complex TFs
# input
xfreqs = np.fft.rfftfreq(len(x), d=1 / fs)  # frequency array to plot DFT across
xfft = np.fft.rfft(x, axis=0)  # compute discrete fourier transform

# output
yfreqs = np.fft.rfftfreq(len(y), d=1 / fs)  # frequency array to plot DFT across
yfft = np.fft.rfft(y, axis=0)  # compute discrete fourier transform

# transfer function: h = y / x
TF = yfft / xfft

# separate real and imaginary part
TF_r = np.real(TF)  # power and phase information in
TF_i = np.imag(TF)  # complex array

to_output = TF * xfft  # works!

# go to PSD
signal = TF
signal = np.abs(signal)  # euclidian distance of complex output array
signal = signal / len(xfreqs)  # rescale so magnitude is independent of signal length
signal = signal ** 2  # square to estimate PSD

# plot
fig, ax = plt.subplots(1, 1)
ax.semilogx(xfreqs, np.log(signal), label='numpy PSD')  # plot on logarithmic scale


ax.plot(freqs_n, xfft_i)
ax.plot(freqs_n, xfft_r)
ax.semilogx(freqs_n, (xfft_r), label='numpy PSD')  # plot on logarithmic scale
ax.semilogx(freqs_n, (xfft_i), label='numpy PSD')  # plot on logarithmic scale
"""