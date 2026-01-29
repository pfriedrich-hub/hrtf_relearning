import slab
import numpy
import copy
import scipy
import logging

def hrtf2hrir(hrtf):
    """
    takes a slab.HRTF of type TF and converts it to IR
    """
    hrir = copy.deepcopy(hrtf)
    if hrtf.datatype == 'FIR':
        logging.info(f'Datatype already {hrtf.datatype}. Returning HRIR.')
        return hrir
    for src_idx, filt in enumerate(hrir.data):
        ir = []
        for ch_idx in range(filt.n_channels):
            # frequency sampling function
            # magnitude = filt.channel(ch_idx).data  # get raw power spectrum
            ir.append(fsamp(magnitude = filt.channel(ch_idx).data))
            # slab filter method
            # frequency, gain = filt.tf(channels=ch_idx, show=False)  # get magnitude in dB?
            # ir.append(slab.Filter.band(kind='hp', frequency=frequency.tolist(), gain=gain[:, 0].tolist(),
            #                            samplerate=filt.samplerate, length=filt.n_samples, fir='IR'))
        hrir[src_idx] = slab.Filter(data=ir, samplerate=hrtf.samplerate, fir='IR')
        # shift to make causal
        # (path differences between the origin and the ear are usually
        # smaller than 30 cm but numerical HRIRs show stringer pre-ringing)
        # hrir = np.roll(hrir, n_shift, axis=-1)
    hrir.datatype = 'FIR'
    return hrir

def fsamp(magnitude):
    """
    Use the frequency sampling method to design an FIR filter to match the frequency
    response of a given magnitude spectrum;
    The desired response is sampled on a frequency grid { H(w) = abs[H(w)] * exp[i * phi(w)] }
    This discretized response is then transformed to the time domain by inverse Fourier transform (IFFT).
    The resulting impulse response is windowed to smooth out the resulting response and to reduce the impulse response
     to the desired filter length 𝑁.

    Arguments:
        magnitude (numpy.array): The magnitude spectrum to fit an FIR Filter to
    Returns:
        h (numpy.array): The impulse response filter.
    """
    magnitude = magnitude.flatten()
    M = len(magnitude)
    if magnitude.min() < 0:    # shift filter to avoid negative gain
        magnitude -= magnitude.min()
    if (M % 2) == 0:
        n = 2*(M-1)  # FFT length
        w = numpy.arange(0, M) / (M-1) * numpy.pi  # frequency grid
        D = magnitude * numpy.exp(-1j * w * (n-1) / 2)  # desired complex frequency response
        D = numpy.concatenate((D, numpy.conj(D[:0:-1])))  # extend to 2 * Nyquist by adding negative frequencies
    else:
        n = 2*M-1
        w = numpy.arange(0, M) * 2*numpy.pi / (2 * M-1)
        D = magnitude * numpy.exp(-1j * w * (n-1) / 2)
        D = numpy.concatenate((D, numpy.conj(D[:0:-1])))

    h = numpy.real(numpy.fft.ifft(D))   # impulse response via IFFT
    # h = numpy.fft.irfft(D)  # go without adding negative frequencies. similar result

    # windowing (Hamming)
    I = int((n-M) / 2 + 1)
    ww = scipy.signal.windows.hamming(M)
    h = h[I:I+M] * ww
    return h

def tf2ir(tf):
    """
    Takes a slab.Filter of type TF as input and converts it to FIR
    """
    #todo
    for src_idx, filt in enumerate(tf.data):
        ir = []
        for ch_idx in range(tf.n_channels):
            # frequency sampling function
            # magnitude = filt.channel(ch_idx).data  # get raw power spectrum
            ir.append(fsamp(magnitude = tf.channel(ch_idx).data))
            # slab filter method
            # frequency, gain = filt.tf(channels=ch_idx, show=False)  # get magnitude in dB?
            # ir.append(slab.Filter.band(kind='hp', frequency=frequency.tolist(), gain=gain[:, 0].tolist(),
            #                            samplerate=filt.samplerate, length=filt.n_samples, fir='IR'))
        slab.Filter(data=ir, samplerate=tf.samplerate, fir='IR')
        # shift to make causal
        # (path differences between the origin and the ear are usually
        # smaller than 30 cm but numerical HRIRs show stringer pre-ringing)
        # hrir = np.roll(hrir, n_shift, axis=-1)
    ir.datatype = 'FIR'
    return ir




"""
    # workaround for when only an amplitude spectrum is provided (no complex valued fourier coefs available):
    # maybe see Kramers-Kronig or Bode

    # try: multiply flat frequency spectrum with the amplitude spectrum of the TF
        # try: create flat spectrum from scratch
    # zero_hz_bin = numpy.ones(1)
    # frequency_spectrum = numpy.ones(hrtf[0].n_taps-1) * 7.854726969857107e-16 - 1j
    # frequency_spectrum = numpy.hstack((zero_hz_bin, frequency_spectrum))
        # try: create flat spectrum from time domain signal constituted of sine waves
    fs = hrtf[0].samplerate
    nyquist = int(fs / 2)
    freqs = numpy.arange(0, nyquist)
    t = numpy.linspace(0, 1, fs)  # time axis
    sine_waves = []
    for freq in freqs:
        sine_waves.append(numpy.sin(2 * numpy.pi * freq * t))
    sig = numpy.sum(sine_waves, axis=0)

    # try:


    # try: obtain complex transfer function by dividing filtered signal by input signal in the frequency domain)
    fs = hrtf.samplerate
    freqs = hrtf[0].frequencies
    sig = scipy.signal.chirp(t=numpy.linspace(0, 1, fs), f0=0, t1=1, f1=hrtf.samplerate/2, method='quadratic')


    # compute fft of signal (flat frequency spectrum)
    fft = numpy.fft.rfft(sig)
    fft = fft / len(fft)  # rescale by signal length (do not multiply by 2, since this is already taken care of by irfft)
    sig_freqs = numpy.fft.rfftfreq(len(sig), d=1/fs)

    # interpolate and filter
    fft_interp = numpy.interp(freqs, sig_freqs, fft)
    filtered = fft_interp * hrtf[0].data[:, 0]

    # get complex tf by dividing filtered signal by input signal in the frequency domain
    tf = filtered / fft_interp  # of course this returns the original hrtf filter


    # plots
    data = tf
    plt.figure()
    plt.plots(freqs, numpy.imag(data))
    plt.plots(freqs, numpy.real(data))
    plt.plots(freqs, numpy.abs(data))

    # try: reconstruct phase- from powerspectrum
    ir = hrtf[0].data[:, 0]
    fs = hrtf[0].samplerate
    frequencies = numpy.fft.rfftfreq(n=len(ir), d=1/fs)
    frequency_spectrum = numpy.fft.rfft(ir)
    power_spectrum = numpy.abs(frequency_spectrum)
    phase_spectrum = numpy.imag(frequency_spectrum)
    amplitude_spectrum = numpy.real(frequency_spectrum)

    # ln H(w) = ln abs(H(w)) + i*phi(w)
    numpy.log(frequency_spectrum) == numpy.log(power_spectrum) + 1j * phase_spectrum

    # h(t) = F-1(abs(Hw)**(i*phi(w))
    ifft = numpy.fft.irfft(power_spectrum * numpy.e ** (1j * phase_spectrum))
    ifft = numpy.fft.irfft(frequency_spectrum)



    data = frequency_spectrum
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plots(frequencies, numpy.log(power_spectrum), phase_spectrum)


    freqs = numpy.arange(1, hrtf.samplerate/2, 1)
    time = numpy.linspace(0, len(signal) / fs, len(signal))  # start, stop, number of steps
    nyquist = hrtf.samplerate/2
    N = hrtf[0].n_taps
    fourier = numpy.zeros(N, dtype=complex)
    for fi in range(freqs):
        sine_wave = numpy.exp(-1j * (2 * numpy.pi * fi * time))  # complex sine wave
        fourier[fi] = numpy.sum(sine_wave * x)
    fourier = fourier / N  # rescale by signal length


    fs = 44000
    freqs = numpy.arange(1, 10, 1)
    y = numpy.arange(0, 1, 1 / fs)
    A = 1  # amplitude
    sine_waves = []
    for freq in freqs:
        x = numpy.sin(2 * numpy.pi * freq * y)
        sine_waves.append(x)
    x = numpy.sum(sine_waves, axis=0)
    # x = x / len(sine_waves)
    plt.figure()
    plt.plots(y, x)

    fc = numpy.fft.rfft(x, axis=0)

    fc = fc /len(y) * 2
    plt.figure()
    plt.plots(numpy.real(fc))
    plt.plots(numpy.imag(fc))
    plt.plots(numpy.abs(fc))


    # step 2: obtain complex transfer function (divide filtered signal by input signal in the frequency domain)





    # todo add option for complex valued TFs
    input = copy.deepcopy(hrtf)
    tf_data = numpy.zeros((hrtf.n_sources, hrtf[0].n_samples, 2))
    for src_idx, tf in enumerate(input.data):
        tf_data[src_idx] = tf.data

    # ifft (take complex conjugate because sign conventions differ)
    ir_data = numpy.fft.irfft(numpy.conj(tf_data), axis=1)

    # shift to make causal
    # (path differences between the origin and the ear are usually
    # smaller than 30 cm but numerical HRIRs show stringer pre-ringing)
    # hrir = np.roll(hrir, n_shift, axis=-1)

    for src_idx, ir_data in enumerate(hrir):
        input[src_idx] = slab.Filter(data=ir_data, samplerate=hrtf.samplerate, fir='IR')
    input.datatype = 'IR'
"""