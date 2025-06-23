import numpy
import scipy.signal.windows
from matplotlib import pyplot as plt


mag = numpy.concatenate((numpy.ones((200)),.75*numpy.ones((200)),.5*numpy.ones((200)),-1*numpy.ones((200)),numpy.zeros((200))))
N = 200

def fsamp(mag, N):
    """
    Use the frequency sampling method to design an IR filter to match the frequency
    response of a given magnitude spectrum;
    The desired response is sampled on a dense equidistant frequency grid { H(w) = abs[H(w)] * exp[i * phi(w)] }
    This discretized response is then transformed to the time domain by inverse Fourier transform (IFFT).
    The resulting impulse response is windowed to smooth out the resulting response and to reduce the impulse response
     to the desired filter length 𝑁.
    """
    M = len(mag)
    if mag.min < 0:    # shift filter to avoid negative gain
        mag -= mag.min()
    if (N % 2) == 0:
        n = 2*(M-1)  # FFT length
        w = numpy.arange(0, M) / (M-1) * numpy.pi  # frequency grid
        D = mag * numpy.exp(-1j * w * (n-1) / 2)  # desired complex frequency response
        D = numpy.concatenate((D, numpy.conj(D[:0:-1])))  # extend to 2 * Nyquist (add negative frequencies)
    else:
        n = 2*M-1
        w = numpy.arange(0, M) * 2 * numpy.pi / (2 * M-1)
        D = mag * numpy.exp(-1j * w * (n-1) / 2)
        D = numpy.concatenate((D, numpy.conj(D[:0:-1])))

    h = numpy.real(numpy.fft.ifft(D))   # impulse response via IFFT
    # h = numpy.fft.irfft(D) # not the same, but almost

    # windowing (Hamming)
    I = int((n-N) / 2 + 1)
    ww = scipy.signal.windows.hamming(N)
    h = h[I:I+N] * ww
    return h