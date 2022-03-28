import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from pathlib import Path
import os
import scipy.io
import math

fs = 48000
# time axis for time traces
signal = np.random.rand(fs*2)  # 2 seconds signal
time = np.arange(0, len(signal) / fs, 1 / fs)  # start, stop, step size
time = np.linspace(0, len(signal) / fs, len(signal))  # start, stop, number of steps
plt.plot(time, signal)

#-------- CHAPTER XI --------#
# making waves
y = np.arange(-1, 1, 1/fs)  # time points to plot data across
A = 1  # amplitude
f = 2  # frequency
phi = np.sin(0)  # phase angle offset 0-360
x = A * np.sin(2 * np.pi * f * y + phi)

# add some sine waves together
freqs = [2, 4, 8]  # different frequencies
sine_waves = []
for freq in freqs:
    x = A * np.sin(2 * np.pi * freq * y + phi)
    sine_waves.append(x)
x = np.sum(sine_waves, axis=0)

# plot
plt.figure()
plt.plot(y, x)

# manual Fourier Transformation
data = x
# data = np.random.rand(10)  # random numbers
N = len(data)        # length of sequence
fs = 200        # sampling rate in Hz
nyquist = fs / 2    # Nyquist frequency -- the highest frequency you can measure in the data
# initialize Fourier output matrix
fourier = np.zeros(N, dtype=complex)
# These are the actual frequencies in Hz that will be returned by the
# Fourier transform. The number of unique frequencies we can measure is
# exactly 1/2 of the number of data points in the time series (plus DC).
frequencies = np.linspace(0, nyquist, int(N/2)+1)
time = np.arange(0, N)/N
# Fourier transform is dot-product between sine wave and data at each frequency
for fi in range(N):
    sine_wave = np.exp(-1j * (2 * np.pi * fi * time))  # complex sine wave
    fourier[fi] = np.sum(sine_wave * data)
fourier = fourier / N  # rescale by signal length
fig, ax = plt.subplots(1, 1)
ax.plot(frequencies, np.abs(fourier)[:int(N/2)+1]**2)

# plot complex sine wave
for fi in range(1, 100):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    sine_wave = np.exp(-1j * (2 * np.pi * fi * time))  # complex sine wave
    ax.plot(time, sine_wave.real, sine_wave.imag)
    ax.set_title('%i Hz sine wave' %fi)

# simple convolution
srate = 1000
time = np.arange(-.5, .5 - 1 / srate, 1 / srate)
f = 20  # sine wave frequency
fg = 15  # gaussian frequency
# compute sine wave
s = np.sin(2 * np.pi * f * time)
# compute Gaussian
g = np.exp((- time ** 2) / (2 * (4 / (2 * np.pi * fg) ** 2))) / fg  # amplitude scaled by fg
# convolve sine wave by gaussian
c = np.convolve(s, g, 'same')
# plot
fig, ax = plt.subplots(3, 1)
ax[0].plot(time, s)
ax[0].set_title('Sine wave (signal)')
ax[0].set_ylim(-1.1, 1.1)
ax[1].plot(time, g)
ax[1].set_title('Gaussian (kernel)')
ax[2].plot(time, c)
ax[2].set_title('result of convolution')


# -------- CHAPTER XII -------- #
# creating Morlet wavelets
# to make a morlet wavelet, create a gaussian and a sine wave and multiply them point by point
srate = 500  # sampling rate in Hz
f = 10  # frequency of the sine wave and of gaussian in Hz = center/peak frequency of resulting wavelet
time = np.arange(-1, 1, 1 / srate)  # time, from -1 to 1 second in steps of 1/sampling-rate

sine_wave = np.exp(2 * np.pi * 1j * f * time)  # complex wavelet
# make a Gaussian
n = 6  # number of cycles - trade-off between temporal and frequency precision
s = n / (2 * np.pi * f)  # standard deviation of gaussian
a = 1  # amplitude of gaussian
gaussian_win = a * np.exp(-time**2/(2*s**2))
# and together they make a wavelet! - non-complex - convolve with a signal to create a bandpass filter
wavelet = sine_wave * gaussian_win

# plot
fig, ax = plt.subplots(3, 1)
ax[0].plot(time, sine_wave)
ax[0].set_title('Sine wave (signal)')
ax[0].set_ylim(-1.1, 1.1)
ax[1].plot(time, gaussian_win)
ax[1].set_title('Gaussian window')
ax[2].plot(time, wavelet)
ax[2].set_title('resulting wavelet')

# -------- CHAPTER XIII -------- #
# Complex Morlet Wavelets - extracting power and phase
# parameters...
srate = 500  # sampling rate in Hz
f = 10  # center frequency of wavelet in Hz
time = np.arange(-1, 1, 1/srate)  # time, from -1 to 1 second in steps of 1/sampling-rate
s = 6/(2*np.pi*f) # std of gaussian
# and together they make a wavelet
wavelet = np.exp(2*np.pi*1j*f*time) * np.exp(-time**2/(2*s**2))

# plot 3d complex morlet wavelet
fig, ax = plt.subplots(1, 1)
ax = plt.axes(projection='3d')
ax.plot(time, wavelet.real, wavelet.imag)
ax.set_title('%i Hz complex morlet wavelet' % f)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('real amplitude')
ax.set_zlabel('imag amplitude')

# convolve EEG signal with a complex morlet wavelet
# load sample eeg data - brain vision
DIR = Path(os.getcwd())
folder_path = DIR / 'signal_processing' / 'sample_data'
for header_file in folder_path.glob('*.vhdr'):
    eeg = mne.io.read_raw_brainvision(header_file, preload=True)
fs = eeg.info['sfreq']
eeg_data = eeg._data[0, 15500:17500] # 4 seconds of eeg data channel 0
eeg_time = np.arange(0, len(eeg_data) / fs, 1 / fs)  # start, stop, step size
# load sample eeg data - eeglab - does not work yet
mat = scipy.io.loadmat(folder_path / 'sampleEEGdata.mat')

# create wavelet
frequency = 6  # in Hz, as usual
time = np.arange(-1, 1, 1/fs)
n = 4  # number of cycles of gaussian taper
s = (n / (2 * np.pi * frequency))  # note that s is squared here rather than in the next line
wavelet = np.exp(2 * 1j * np.pi * frequency * time) * np.exp(-time ** 2 / (2 * s ** 2) / frequency)

# FFT parameters
n_wavelet = len(wavelet)
n_data = len(eeg_data)
n_convolution = n_wavelet + n_data - 1
half_of_wavelet_size = math.ceil((n_wavelet - 1) / 2)
# FFT of wavelet and EEG data
fft_wavelet = np.fft.fft(wavelet, n_convolution)
fft_data = np.fft.fft(eeg_data, n_convolution)  # FCz, trial 1

# convolve and get inverse  of fft
convolution_result_fft = np.fft.ifft(fft_wavelet * fft_data, n_convolution) * np.sqrt(s) # scale by root of cycles
# cut off edges
convolution_result_fft = convolution_result_fft[half_of_wavelet_size + 1:n_convolution - half_of_wavelet_size]
# plot for comparison
fig, ax = plt.subplots(3, 1)
ax[0].plot(eeg_time, convolution_result_fft.real)
ax[0].set_title('Projection onto real axis is filtered signal at %i Hz.'%frequency)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Voltage (\muV)')
ax[1].plot(eeg_time, np.abs(convolution_result_fft) ** 2)
ax[1].set_title('Magnitude of projection vector squared is power at %i Hz.'%frequency)
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Voltage (\muV)')
ax[2].plot(eeg_time, np.angle(convolution_result_fft))
ax[2].set_title('Angle of vector is phase angle time series at %i Hz.'%frequency)
ax[2].set_xlabel('Time (ms)')
ax[2].set_ylabel('Phase angle (rad.)')

# some notes on convolution parameters:
# NUMBER OF CYCLES on the gaussian taper - defines width of gaussian - in turn defines width of wavelet
# trade-off between temporal and frequency precision
# a larger number of cycles results in better frequency precision at the cost of worse temporal precision
# number of cycles can also change as a function on wavelet frequency
# in general, you should use at least 3 cycles and at most 14 cycles, for >7 cycles, check if wavelet tapers to zero

# WAVELET LENGTH - should be long enough so that lowest frequency wavelets taper to zero

# one final note: data should be stationary during the period in which the wavelet is nonzero

# full blown wavelet convolution!
# get eeg data
DIR = Path(os.getcwd())
folder_path = DIR / 'signal_processing' / 'sample_data'
for header_file in folder_path.glob('*.vhdr'):
    eeg = mne.io.read_raw_brainvision(header_file, preload=True)
fs = eeg.info['sfreq']
eeg_len = int(fs * 4)
eeg_data = eeg._data[5, 0:eeg_len]  # one minute of eeg data
eeg_time = np.arange(0, len(eeg_data) / fs, 1 / fs)  # start, stop, step size

# define frequency range
min_freq = 2
max_freq = 80
num_frex = 30
# define wavelet parameters
time = np.arange(-1, 1, 1 / fs)
frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
# number of cycles of morlet wavelet as function of frequency (more cycles with increasing wavelet frequency)
s = np.logspace(np.log10(3), np.log10(10), num_frex) / (2 * np.pi * frex)
# define convolution parameters
n_wavelet = len(time)
n_data = len(eeg_data)
n_convolution = n_wavelet + n_data - 1
n_conv_pow2 = 2 ** (math.ceil(math.log(n_convolution, 2)))
half_of_wavelet_size = int((n_wavelet - 1) / 2)
# get FFT of data
eeg_fft = np.fft.fft(eeg_data, n_conv_pow2)
# initialize
eeg_power = np.zeros((num_frex, n_data))  # frequencies X time
base_idx = [0, 500]  # eeg data used for baseline normalization
# loop through frequencies and compute synchronization
for fi in range(num_frex):
    wavelet_fft = \
    np.fft.fft(np.sqrt(1 / (s[fi] * np.sqrt(np.pi)))  #todo empirical scaling factor for varying wavelet cycles
               # complex sine at different frequencies
               * np.exp(2 * 1j * np.pi * frex[fi] * time)
               # gaussian with s cycles
               * np.exp(-time ** 2 / (2 * (s[fi] ** 2))), n_conv_pow2)  # more cycles with increasing wavelet frequency
    # convolution
    eeg_conv = np.fft.ifft(wavelet_fft * eeg_fft)  # convolve
    eeg_conv = eeg_conv[:n_convolution]  # cut result to length of n_convolution
    eeg_conv = eeg_conv[half_of_wavelet_size + 1: n_convolution - half_of_wavelet_size]
    # calculate power
    temp_power = (np.abs(eeg_conv) ** 2)
    # baseline normalization
    eeg_power[fi] = 10 * np.log10(temp_power[fi] / np.mean(temp_power[base_idx[0]:base_idx[1]]))

# plot
x, y = np.meshgrid(eeg_time, frex)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
c_ax = ax.contour(x, y, eeg_power, linewidths=0.3, colors="k", norm=colors.Normalize())
c_ax = ax.contourf(x, y, eeg_power, norm=colors.Normalize(), cmap=plt.cm.jet)
cbar = fig.colorbar(c_ax)
plt.show()


# -------- CHAPTER XIV -------- #
# Hilbert transform
# used to extract complex information from a real-valued signal (alternative to convolution with complex wavelet)
# can be applied to bandpass filtered signal to obtain band specific power and phase over time
## the FFT-based hilbert transform
# generate random signal
signal_len = 21
signal = np.random.rand(signal_len)
# take FFT of signal
signal_fft = np.fft.fft(signal)
# create copy of fourier transform that has been multiplied by complex operator (i);
# this turns Mcos(2pift) into iMcos(2pift)
complex_copy = 1j * signal_fft

# find indices of positive and negative frequencies
# zero and nyquist frequency should stay untouched
posF = np.arange(1, int(np.floor(signal_len/2)))
negF = np.arange(int(np.ceil(signal_len/2)), signal_len)
# rotate Fourier coefficients
# (note 1: this works by computing the iAsin(2pft) component, i.e., the phase quadrature)
# (note 2: positive frequencies are rotated counter-clockwise negative frequencies are rotated clockwise)
signal_fft[posF] = signal_fft[posF] + -1j * complex_copy[posF]
signal_fft[negF] = signal_fft[negF] +  1j * complex_copy[negF]
# The next two lines are an alternative and slightly faster method. 
# The book explains why this is equivalent to the previous two lines.
# f(posF) = f(posF)*2
# f(negF) = f(negF)*0
# take inverse FFT
hilbert_x = np.fft.ifft(signal_fft)
# compare with scipy function hilbert
from scipy.signal import hilbert
hilbert_s = hilbert(signal)

# plot results
plt.figure()
plt.plot(np.abs(hilbert_s), label='scipy')
plt.plot(np.abs(hilbert_x), label='manual')
plt.legend()
plt.title('magnitude of Hilbert transform')




