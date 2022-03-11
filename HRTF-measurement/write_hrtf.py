# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   write_hrtf.py
#   read in-the-ear recordings (.wav), compute HRTF via numpy and store in file (.sofa)
#   after Andrés Pérez-López - Eurecat / UPF
#   24/08/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from netCDF4 import Dataset
import time
import numpy
import os
import scipy
import slab
import freefield
from pathlib import Path
import argparse
from matplotlib import pyplot as plt

filename = 'paul_hrtf'
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--id", type=str,
# 	default="paul_hrtf",
# 	help="enter subject id")
# args = vars(ap.parse_args())
# subject = args["id"]
# print('record from %s speakers, subj_id: %i' %(id, 9))

#---------Load Recordings-----------#
rec_path = Path.cwd() /'data' / 'in-ear_recordings'
source_file = rec_path / 'in-ear_paul_sources.txt'  # csv containing sound source coordinates
sources = numpy.loadtxt(source_file, skiprows=1, usecols=(1, 2, 3), delimiter=",")
probe_len = 0.5  # length of the sound probe in seconds
fs = 48828  # sampling rate

def read_wav(path):
    recordings = []  # list to hold slab.Binaural objects
    for wav_file in path.rglob('*.wav'):
        recordings.append(slab.Sound.read(wav_file).data)
    return slab.Sound(data=recordings)

def filter_rec(recordings, frequency=(200, 18000), kind='bp'):
    filt = slab.Filter.band(frequency=frequency, kind=kind)  # bandpass filter
    return filt.apply(recordings)

# read recorded .wav files and return multichannel slab.Sound object
recordings = read_wav(rec_path)

#---------filter recordings-----------#
recordings = filter_rec(recordings)

# generate probe signal
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz

# estimate hrtf
hrtf = slab.HRTF.estimate_hrtf(signal=signal, recordings=recordings, sources=sources)

# compute HRTFs and safe them in MRN array - Measurements, Receivers, N_frequencies
def write(signal, recordings, sources, filename):
    if len(sources) != recordings.n_channels / 2:
        raise ValueError('Number of sound sources must be equal to number of recordings.')
    m = int(recordings.n_channels / 2)  # number of measurements
    n = int(recordings.n_samples / 2 + 1)  # samples - frequencies in the transfer function
    # n = int(recordings.samplerate / 2)
    r = 2  # number of receivers (HRTFs measured for 2 ears)
    e = 1  # number of emitters (1 speaker per measurement)
    i = 1  # always 1
    c = 3  # number of dimensions in space (elevation, azimuth, radius)
    sig = signal.data[:, 0]
    rec_data = numpy.empty([m, r, recordings.n_samples], dtype=float)  # store Sound.data
    hrtf_data = numpy.empty([m, r, n], dtype=complex)  # store complex fft output [Measurements, Receivers, N_datapoints]
    if not signal.samplerate == recordings.samplerate:
        signal = signal.resample(recordings.samplerate)
    if not signal.n_samples == recordings.n_samples:
        sig_freq_bins = numpy.fft.rfftfreq(signal.n_samples, d=1 / signal.samplerate)
        rec_freq_bins = numpy.fft.rfftfreq(recordings.n_samples, d=1 / recordings.samplerate)
        sig_fft = numpy.interp(rec_freq_bins, sig_freq_bins, numpy.fft.rfft(sig))
    else:
        sig_fft = numpy.fft.rfft(sig)
    for z in range(0, m * 2, 2):
        source_idx = int(z / 2)  # indices to array with sofa dimensions (m, r, n)
        channel_idx = [z, z + 1]  # pick data of left and right channels from Slab object
        rec_data[source_idx] = recordings.data.T[channel_idx, :]
        hrtf_data[source_idx] = [numpy.fft.rfft(rec_data[source_idx, 0]),
                                 numpy.fft.rfft(rec_data[source_idx, 1])]
        hrtf_data[source_idx] = hrtf_data[source_idx] / sig_fft

    filepath = Path.cwd() / 'data' / 'hrtfs' / (filename + '.sofa')
    if Path(filepath).is_file():
        Path(filepath).unlink()
    sofa = Dataset(filepath, 'w', format='NETCDF4')

    # ----------Required Dimensions----------#
    sofa.createDimension('M', m)
    sofa.createDimension('N', n)
    sofa.createDimension('E', e)
    sofa.createDimension('R', r)
    sofa.createDimension('I', i)
    sofa.createDimension('C', c)
    # ----------Required Attributes----------#
    sofa.DataType = 'TF'
    sofa.RoomType = 'free field'
    sofa.Conventions, sofa.Version = 'SOFA', '2.0'
    sofa.SOFAConventions, sofa.SOFAConventionsVersion = 'SimpleFreeFieldHRTF', '2.0'
    sofa.APIName, sofa.APIVersion = 'pysofaconventions', '0.1'
    sofa.AuthorContact, sofa.License = 'paul.friedr1@gmail.com', 'PublicLicence'
    sofa.ListenerShortName, sofa.Organization = 'pf01', 'Eurecat - UPF'
    sofa.DateCreated, sofa.DateModified = time.ctime(time.time()), time.ctime(time.time())
    sofa.Title, sofa.DatabaseName = 'testpysofaconventions', 'UniLeipzig Freefield'

    # ----------Required Variables----------#
    listenerPositionVar = sofa.createVariable('ListenerPosition', 'f8', ('I', 'C'))
    listenerPositionVar.Units = 'metre'
    listenerPositionVar.Type = 'cartesian'
    listenerPositionVar[:] = numpy.zeros(c)
    receiverPositionVar = sofa.createVariable('ReceiverPosition', 'f8', ('R', 'C', 'I'))
    receiverPositionVar.Units = 'metre'
    receiverPositionVar.Type = 'cartesian'
    receiverPositionVar[:] = numpy.zeros((r, c, i))
    sourcePositionVar = sofa.createVariable('SourcePosition', 'f8', ('M', 'C'))
    sourcePositionVar.Units = 'degree, degree, metre'
    sourcePositionVar.Type = 'spherical'
    sourcePositionVar[:] = sources  # array of speaker positions
    emitterPositionVar = sofa.createVariable('EmitterPosition', 'f8', ('E', 'C', 'I'))
    emitterPositionVar.Units = 'metre'
    emitterPositionVar.Type = 'cartesian'
    emitterPositionVar[:] = numpy.zeros((e, c, i))
    listenerUpVar = sofa.createVariable('ListenerUp', 'f8', ('I', 'C'))
    listenerUpVar.Units = 'metre'
    listenerUpVar.Type = 'cartesian'
    listenerUpVar[:] = numpy.asarray([0, 0, 1])
    listenerViewVar = sofa.createVariable('ListenerView', 'f8', ('I', 'C'))
    listenerViewVar.Units = 'metre'
    listenerViewVar.Type = 'cartesian'
    listenerViewVar[:] = numpy.asarray([0, 1, 0])
    dataRealVar = sofa.createVariable('Data.Real', 'f8', ('M', 'R', 'N'))  # data
    dataRealVar[:] = numpy.real(hrtf_data)
    dataImagVar = sofa.createVariable('Data.Imag', 'f8', ('M', 'R', 'N'))
    dataImagVar[:] = numpy.imag(hrtf_data)
    NVar = sofa.createVariable('N', 'f8', ('N'))
    NVar.LongName = 'frequency'
    NVar.Units = 'hertz'
    NVar[:] = n
    samplingRateVar = sofa.createVariable('Data.SamplingRate', 'f8', ('I'))
    samplingRateVar.Units = 'hertz'
    samplingRateVar[:] = recordings.samplerate
    sofa.close()


 """   import h5netcdf
    m = int(recordings.n_channels / 2)  # number of measurements
    n = int(recordings.n_samples / 2 + 1)  # samples - frequencies in the transfer function
    # n = int(recordings.samplerate / 2)
    r = 2  # number of receivers (HRTFs measured for 2 ears)
    e = 1  # number of emitters (1 speaker per measurement)
    i = 1  # always 1
    c = 3  # number of dimensions in space (elevation, azimuth, radius)
    f = h5netcdf.File(filename + '.sofa', 'w')
    f.dimensions = {'M': m, 'N': n, 'E': e, 'R': r, 'I': i, 'C': c}
    f.attrs['DataType'] = 'TF'
    f.attrs['RoomType'] = 'free field'
    f.attrs['Conventions'], f.attrs['Version'] = 'SOFA', '2.0'
    f.attrs['SOFAConventions'], f.attrs['SOFAConventionsVersion'] = 'SimpleFreeFieldHRTF', '2.0'
    f.attrs['APIName'], f.attrs['APIVersion'] = 'pysofaconventions', '0.1'
    f.attrs['AuthorContact'], f.attrs['License'] = 'paul.friedr1@gmail.com', 'PublicLicence'
    f.attrs['ListenerShortName'], f.attrs['Organization'] = 'pf01', 'Eurecat - UPF'
    f.attrs['DateCreated'], f.attrs['DateModified'] = time.ctime(time.time()), time.ctime(time.time())
    f.attrs['Title'], f.attrs['DatabaseName'] = 'testpysofaconventions', 'UniLeipzig Freefield'

    listenerPositionVar = f.create_variable('ListenerPosition', ('I', 'C'), 'f8')
    listenerPositionVar.Units = 'metre'
    listenerPositionVar.Type = 'cartesian'
    listenerPositionVar[:] = numpy.zeros(c)
    receiverPositionVar = f.create_variable('ReceiverPosition', ('R', 'C', 'I'), 'f8')
    receiverPositionVar.Units = 'metre'
    receiverPositionVar.Type = 'cartesian'
    receiverPositionVar[:] = numpy.zeros((r, c, i))
    sourcePositionVar = f.create_variable('SourcePosition', ('M', 'C'), 'f8')
    sourcePositionVar.Units = 'degree, degree, metre'
    sourcePositionVar.Type = 'spherical'
    sourcePositionVar[:] = sources  # array of speaker positions
    emitterPositionVar = f.create_variable('EmitterPosition', ('E', 'C', 'I'), 'f8')
    emitterPositionVar.Units = 'metre'
    emitterPositionVar.Type = 'cartesian'
    emitterPositionVar[:] = numpy.zeros((e, c, i))
    listenerUpVar = f.create_variable('ListenerUp', ('I', 'C'), 'f8')
    listenerUpVar.Units = 'metre'
    listenerUpVar.Type = 'cartesian'
    listenerUpVar[:] = numpy.asarray([0, 0, 1])
    listenerViewVar = f.create_variable('ListenerView', ('I', 'C'), 'f8')
    listenerViewVar.Units = 'metre'
    listenerViewVar.Type = 'cartesian'
    listenerViewVar[:] = numpy.asarray([0, 1, 0])
    dataRealVar = f.create_variable('Data.Real', ('M', 'R', 'N'), 'f8')  # data
    dataRealVar[:] = numpy.real(hrtf_data)
    dataImagVar = f.create_variable('Data.Imag', ('M', 'R', 'N'), 'f8')
    dataImagVar[:] = numpy.imag(hrtf_data)
    NVar = f.create_variable('N', ('N'), 'f8')
    NVar.LongName = 'frequency'
    NVar.Units = 'hertz'
    NVar[:] = n"""

 """   
 """


"""
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
    return hrtf"""



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

import os
import math
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.mlab import psd, csd
import slab
fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)
source_file = Path.cwd() /'data' / 'in-ear_recordings' / 'in-ear_paul_sources.txt'  # csv containing sound source coordinates
sources = np.loadtxt(source_file, skiprows=1, usecols=(1, 2, 3), delimiter=",")


### reproduce waterfall plot
# generate signals
fig, axis = plt.subplots()
vlines = np.arange(0, len(sources[:,1])) * 20
for idx, ele in enumerate(sources[:, 1]):
    chirp = slab.Binaural.chirp(duration=0.5, level=90)
    az = 0.0
    # ele = -12.5
    snd = slab.Binaural.read(Path.cwd() / 'data' / 'in-ear_recordings' / ('in-ear_%s_%s_%s.wav' \
                                                                       %('paul_hrtf', az,  ele)))
    x = chirp.data
    y = snd.data
    # input
    xfft_l = np.fft.rfft(x[:,0], axis=0)  # compute discrete fourier transform
    xfft_r = np.fft.rfft(x[:,1], axis=0)
    # output
    yfft_l = np.fft.rfft(y[:,0], axis=0)  # compute discrete fourier transform
    yfft_r = np.fft.rfft(y[:,1], axis=0)
    # transfer function: h = y / x
    TF_r = yfft_r / xfft_r
    TF_l = yfft_l / xfft_l
    tf = TF_l
    tf = np.abs(tf)  # euclidean distance of complex output array
    tf = 20 * np.log10(tf)
    # from FIlter.tf - because its already a filter we don't want to cut its frequencies in half by fft
    frequencies = np.fft.rfftfreq(len(tf) * 2 - 1, d=1 / fs)

    #smooth tf
    n_bins = 500
    if not n_bins == len(frequencies):  # interpolate if necessary
        w_interp = np.linspace(0, frequencies[-1], n_bins)
        h_interp = np.zeros((n_bins))
        h_interp = np.interp(w_interp, frequencies, tf)
        tf = h_interp
        frequencies = w_interp
# waterfall
    axis.plot(frequencies, tf + vlines[idx],
              linewidth=0.75, color='0.0', alpha=0.7)

# check psd of tf
slab.Sound(data=TF_l).spectrum()

# todo check for power difference of output and TF*input
o=TF_l*xfft_l
oo=np.fft.irfft(0)
slab.Sound(data=oo).waveform()
slab.Sound(data=snd.data[:,0]).waveform()
slab.Sound(data=oo).spectrum()
slab.Sound(data=snd.data[:,0]).spectrum()


# go to PSD (do that before applying filter)
signal = filtered
signal = np.abs(signal)  # euclidian distance of complex output array
signal = signal / len(freqs)  # rescale so magnitude is independent of signal length
signal = signal ** 2  # square to estimate PSD
signal = np.log(signal)
# plot
fig, ax = plt.subplots(1, 1)
ax.semilogx(freqs, signal, label='numpy PSD')  # plot on logarithmic scale

# plot butterworth filter response
plt.figure()
b, a = scipy.signal.butter(3, 500, 'low', fs=fs)# analog=True)
w, h = scipy.signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()"""