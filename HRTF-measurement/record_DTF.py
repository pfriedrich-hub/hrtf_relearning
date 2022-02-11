import numpy as np
import math
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.mlab import psd, csd
from pathlib import Path
import slab
import freefield

table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
probe_len = 0.5  # length of the sound probe in seconds
# get speakers and locations(az,ele) to play from
table = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
# todo record from whole dome
speakers = table[20:27]  # for now only use positive az (half dome)
#tone = slab.Sound.whitenoise(duration=probe_len) # chirp?
chirp = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz

def dome_rec_wav(speakers, subject='dummy_head', n_reps=50):
    # initialize setup
    freefield.initialize('dome', default='play_birec')
    freefield.load_equalization()

    # equalize speaker transfer functions
    # freefield.equalize_speakers(speakers=speaker_coordinates)
    # rec_raw, rec_lvl, rec_full = freefield.test_equalization(speakers=speaker_speaker_coordinates)
    # play , record  and average n_reps times, from all speakers in the list + save as .wav
    freefield.set_logger('WARNING')
    recordings = np.zeros([len(speakers), int(probe_len*fs), 2])  # array to store recordings as data arrays
    slab_obj_list = []  # list to hold recordings as slab binaural objects
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
        slab_obj_list.append(avgrec)
    freefield.set_logger('INFO')
    return slab_obj_list

def read_wav(speakers, subject):
    recordings = np.zeros([len(speakers), int(probe_len*fs), 2])  # array to store recordings
    for i, source_location in enumerate(speakers):
        recordings[i] = slab.Binaural.read(os.getcwd() + '/data/in-ear_recordings/in-ear_%s_%s_%s.wav'
                                           % (subject, str(source_location[1]), str(source_location[2]))).data
    return recordings

def HRTF_estimate(recordings):
    Npoints = len(recordings[0])
    nfft = 2 ** (math.ceil(math.log(Npoints, 2)))

    NumUniquePts = math.ceil((nfft+1)/2)

    HRTF = np.zeros([2, len(recordings), NumUniquePts])
    for i, recfile in enumerate(recordings):
        x = chirp.data[:Npoints]
        yr = recfile[:, 1]
        yl = recfile[:, 0]
        HRTF[0, i] = csd(yr, x, NFFT=nfft, Fs=fs)[0] / psd(x, NFFT=nfft, Fs=fs)[0]
        HRTF[1, i] = csd(yl, x, NFFT=nfft, Fs=fs)[0] / psd(x, NFFT=nfft, Fs=fs)[0]

    return HRTF

def tfe(x, y, *args, **kwargs):
   """estimate transfer function from x to y, see csd for calling convention"""
   return csd(y, x, *args, **kwargs) / psd(x, *args, **kwargs)

# psd examples
# chirp
chirp = slab.Sound.chirp(duration=0.5, level=90)

# various ways to calculate nfft: (transform length)
# The length is typically specified as a power of 2 or a value
# that can be factored into a product of small prime numbers.
# double the number of time points (zero padding) -> double frequency resolution of fft

# 1. think about NFFT as a way to "increase" the spectral resolution,
# use NFFT = 10*N; to increase it by x10
N = len(chirp.data)
nfft = 10*N

# 2. nfft depends on the spectral resolution you want to achieve.
# Given a sampling rate fS and a minimum resolution fResMin,
# can simply compute the number of points nFFT by
# nFFT = ceil(fS/fResMin)

fResMin = 2 # Hz
nfft = math.ceil(fs/fResMin)

# 3. marcs matlab approach:
Npoints = len(chirp.data[:,0])
nfft = 2 ** (math.ceil(math.log(Npoints, 2)))

# 4. set arbitrary nffp
nfft = 1024
# NumUniquePts = math.ceil((nfft + 1) / 2)

# try different nfft parameters
x = chirp.data[:, 0]
for nfft in range(len(chirp.data), len(chirp.data)*2, int(len(chirp.data)/5)):
    psdx = psd(x, NFFT=nfft, Fs=fs, window=np.hamming(len(x)))
    _, ax = plt.subplots()
    ax.plot(psdx[1], np.log(psdx[0]))
    ax.set_title('nfft: ' + str(nfft))




dt = 1/fs
t = np.arange(0, len(chirp.data)/fs, dt)
fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(t, chirp.data)
ax1.psd(chirp.data[:,0], nfft, 1 / fs)

# recording
dt = 1/fs
t = np.arange(0, len(recordings[0, :, 0])/fs, 1/fs)
fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(t, recordings[0, :, 0])
ax1.psd(recordings[0, :, 0], nfft, 1 / fs)
plt.show()

# csd examples
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.psd(chirp.data, nfft, fs)
ax1.psd(recordings[0, :, 0], nfft, fs)
ax2.csd(chirp.data, recordings[0, :, 0], nfft, fs)

# hrtf example: csd(y,x)/psd(x)
x = chirp.data
y = recordings[0, :, 0]

psdx = psd(x, NFFT=nfft, Fs=fs)
psdy = psd(y, NFFT=nfft, Fs=fs)
csdyx = csd(y, x, NFFT=nfft, Fs=fs)
## csdxy = csd(x, y, NFFT=nfft, Fs=fs) # why is csdyx==csdxy?
hrtf = np.abs(csdyx[0] / psdx[0])

fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2, 2, sharex=True)
ax0.plot(psdx[1], np.log(psdx[0]))
ax1.plot(psdy[1], np.log(psdy[0]))
ax2.plot(csdyx[1], np.log(csdyx[0]))
ax3.plot(csdyx[1], np.log(hrtf))

ax0.set_title('PSD (x)')
ax0.set_ylabel('dB/Hz')
ax1.set_title('PSD (y)')
ax2.set_title('CSD (yx)')
ax2.set_ylabel('dB/Hz')
ax2.set_xlabel('Frequency')
ax2.set_yticks([-25, -20, -15, -10])
ax3.set_yticks([-10, -5, -0, 5])
ax3.set_title('HRTF CSD(yx)/PSD(x)')
ax3.set_xlabel('Frequency')


test1 = psdx[0]*hrtf# should return ear recordings!

plt.plot(psdx[1], np.log(test1))
plt.plot(psdx[1], np.log(psdy[0]))


test2 = psdy[0]/hrtf # should return original signal?

plt.plot(psdx[1], np.log(test2))
plt.plot(psdy)

