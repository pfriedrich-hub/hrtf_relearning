import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fs = 48000

# time axis for time traces
signal = np.random.rand(fs*2)  # 2 seconds signal
time = np.arange(0, len(signal) / fs, 1 / fs)
time = np.linspace(0, len(signal) / fs, len(signal))
plt.plot(time, signal)

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

# manual FFT
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
fi = 10
fig = plt.figure()
ax = plt.axes(projection='3d')
sine_wave = np.exp(-1j * (2 * np.pi * fi * time))  # complex sine wave
ax.plot(time, sine_wave.real, sine_wave.imag)
ax.set_title('%i Hz sine wave'%fi)

