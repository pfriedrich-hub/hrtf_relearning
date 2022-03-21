import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import os

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
# load sample eeg data
DIR = Path(os.getcwd())
folder_path = DIR / 'signal_processing' / 'sample_data'
for header_file in folder_path.glob('*.vhdr'):
    eeg = mne.io.read_raw_brainvision(header_file, preload=True)
fs = eeg.info['sfreq']
eeg_data = eeg._data[0, 15000:15500] # one second of eeg data channel 0
eeg_time = np.arange(0, len(eeg_data) / fs, 1 / fs)  # start, stop, step size

# create wavelet
frequency = 6 # in Hz, as usual
time = np.arange(-1, 1, 1/fs)
s = (4 / (2 * np.pi * frequency)) ** 2 # note that s is squared here rather than in the next line...
wavelet = np.exp(2 * 1j * np.pi * frequency * time) * np.exp(-time ** 2 / (2 * s) / frequency)

# FFT parameters
n_wavelet = len(wavelet)
n_data = len(eeg_data)
n_convolution = n_wavelet + n_data - 1
half_of_wavelet_size = int((n_wavelet - 1) / 2)

# FFT of wavelet and EEG data
fft_wavelet = np.fft.fft(wavelet, n_convolution)
# fft_data = np.fft.fft(np.squeeze(EEG.data(47,:,1)), n_convolution)  # FCz, trial 1
fft_data = np.fft.fft(eeg_data, n_convolution)  # FCz, trial 1

# convolve and get inverse  of fft
convolution_result_fft = np.fft.ifft(fft_wavelet * fft_data, n_convolution) * np.sqrt(s)

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
