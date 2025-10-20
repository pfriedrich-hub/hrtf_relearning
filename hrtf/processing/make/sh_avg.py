import numpy
from scipy.signal import sosfiltfilt, butter


"""
from fabian:
import pyfar as pf # pyfar v0.7.2
import numpy as np

sg = pf.samplings.sph_gaussian(sh_order=7)
azimuth = np.sort(np.unique(np.round(sg.azimuth / np.pi * 180, 1)))
print(azimuth)
"""

def to_log_mag(hrtf_complex):
    return numpy.log(numpy.maximum(numpy.abs(hrtf_complex), 1e-12))

def subject_dtf(logmag_dir_by_dir):  # shape: [D, F]
    ref = logmag_dir_by_dir.mean(axis=0, keepdims=True)
    return logmag_dir_by_dir - ref

def smooth_erb_axis(x, fs_hz, freqs_hz):
    # simple 1D Butterworth along frequency bins (pretend uniform after ERB remap)
    sos = butter(2, 0.08, output='sos')  # tune
    return sosfiltfilt(sos, x, axis=-1)

def slope_signs(dtf_logmag, threshold=0.8):
    # dtf_logmag: [F] for one ear+direction
    d = numpy.diff(dtf_logmag, n=1, axis=-1)
    s = numpy.zeros_like(d)
    s[d >  threshold] =  1
    s[d < -threshold] = -1
    return s  # length F-1

def jeffreys_p(count_pos, count_neg, count_zero, eps=1e-9):
    N = count_pos + count_neg + count_zero
    a_pos = 0.5 + count_pos
    a_neg = 0.5 + count_neg
    den   = a_pos + a_neg + (0.5 + count_zero)
    return a_pos/den, a_neg/den

def reconstruct_from_edge_probs(p_pos, p_neg, base_gain=0.0, lam=0.1):
    # Build derivative target d_hat favoring the dominant sign
    pref = p_pos - p_neg  # in [-1,1]
    d_hat = pref  # scale later to desired spectral contrast
    # Smooth (TV-like) by solving (I - lam*L) x' = d_hat where L is Laplacian
    F1 = d_hat.size
    xprime = d_hat.copy()
    for _ in range(50):
        xprime[1:-1] = (d_hat[1:-1] + lam*(xprime[0:-2] + xprime[2:])) / (1 + 2*lam)
    # Integrate to log-mag and zero-mean
    x = numpy.concatenate([[base_gain], base_gain + numpy.cumsum(xprime)])
    x -= x.mean()
    return x  # log-mag DTF

