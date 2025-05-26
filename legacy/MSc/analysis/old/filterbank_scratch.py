import slab
from pathlib import Path
import numpy
import scipy
import copy
data_dir = Path.cwd() / 'final_data'


def triangular_filterbank(length=5000, bandwidth=0.0286, low_cutoff=4000,
                          high_cutoff=16000, pass_bands=True, samplerate=97656):
    """

    """
    if not high_cutoff:
        high_cutoff = samplerate / 2
    freq_bins = numpy.fft.rfftfreq(length, d=1 / samplerate)
    n_freqs = len(freq_bins)
    center_freqs, bandwidth, erb_spacing = slab.Filter._center_freqs(
        low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, pass_bands=pass_bands)
    n_filters = len(center_freqs)
    filts = numpy.zeros((n_freqs, n_filters))
    freqs_erb = slab.Filter._freq2erb(freq_bins)
    for i in range(n_filters):
        l = center_freqs[i] - erb_spacing
        h = center_freqs[i] + erb_spacing
        avg = center_freqs[i]  # center of filter
        # width = erb_spacing * 2  # width of filter
        # filts[(freqs_erb > l) & (freqs_erb < h), i] = numpy.cos(
        #     (freqs_erb[(freqs_erb > l) & (freqs_erb < h)] - avg) / width * numpy.pi)
        filts[(freqs_erb > l) & (freqs_erb < avg), i] = numpy.linspace(0, 1,
                                                    len(freqs_erb[(freqs_erb > l) & (freqs_erb < avg)]))
        filts[(freqs_erb < h) & (freqs_erb > avg), i] = numpy.linspace(1, 0,
                                                    len(freqs_erb[(freqs_erb < h) & (freqs_erb > avg)]))
    return slab.Filter(data=filts, samplerate=samplerate, fir=False)


def scepstral_filter_recordings(recordings, high_cutoff=1500):
    samplerate = recordings[0].samplerate
    n_samples = recordings[0].n_samples
    rec_out = copy.deepcopy(recordings)
    filt = slab.Filter.band(kind='lp', frequency=high_cutoff, samplerate=samplerate,
                            length=n_samples, fir=True)
    for rec_idx, recording in enumerate(rec_out):
        # fft
        recording_fft = numpy.fft.rfft(recording.data, axis=0)
        # log scale and filter
        recording_fft = 20 * numpy.log10(recording_fft)
        to_filter = slab.Sound(recording_fft, samplerate=samplerate)  # numpy abs?
        filtered = filt.apply(to_filter)
        recording_fft = 10 ** (filtered.data / 20)
        # plot
        # rec_freq_bins = numpy.fft.rfftfreq(n_samples, d=1 / samplerate)
        # plt.figure()
        # plt.semilogy(rec_freq_bins, recording_fft)
        # plt.xlim(4000, 16000)
        # inverse fft
        recording.data = numpy.fft.irfft(recording_fft, axis=0)
        recordings[rec_idx] = recording
    return rec_out

# todo
def apply_filterbank_dtf(hrtf, type=None, bandwidth=0.0286, low_cutoff=4000, high_cutoff=16000):
    if not type:
        print('type must be triangular or cosine')
    samplerate = hrtf[0].samplerate
    n_samples = hrtf[0].n_samples
    if not low_cutoff:
        low_cutoff == 0
    if not high_cutoff:
        high_cutoff = samplerate / 2
    if type == 'cosine'.lower():
        f_bank = slab.Filter.cos_filterbank(length=n_samples, bandwidth=bandwidth, low_cutoff=low_cutoff,
                                            high_cutoff=high_cutoff, pass_bands=True, samplerate=samplerate)
    elif type == 'triangular'.lower():
        f_bank = triangular_filterbank(length=(hrtf[0].n_samples * 2) - 1, bandwidth=bandwidth, low_cutoff=low_cutoff,
                          high_cutoff=high_cutoff, pass_bands=True, samplerate=samplerate)

        f_bank = triangular_filterbank(length=(hrtf[0].n_samples * 2) - 1, bandwidth=0.0286, low_cutoff=low_cutoff,
                          high_cutoff=high_cutoff, pass_bands=True, samplerate=samplerate)
    for dtf in hrtf:
        for chan_idx in range(dtf.n_channels):
            tf = dtf[:, chan_idx]
            tf = 20 * numpy.log10(tf)
            subbands = numpy.empty((n_samples, f_bank.n_filters))
            for filter in range(f_bank.n_filters):
                subbands[:, filter] = tf * f_bank[:, filter]
            filtered = subbands.sum(axis=1)
    return hrtf

def apply_filterbank_wav(recordings, type=None, bandwidth=1/3, low_cutoff=None, high_cutoff=None):
    """
    args:
    recordings: list of slab.Sound objects
    """
    if not type:
        print('type must be triangular or cosine')
    samplerate = recordings[0].samplerate
    n_samples = recordings[0].n_samples
    if not low_cutoff:
        low_cutoff == 0
    if not high_cutoff:
        high_cutoff = samplerate / 2
    if type == 'cosine'.lower():
        filter = slab.Filter.cos_filterbank(length=n_samples, bandwidth=bandwidth, low_cutoff=low_cutoff,
                                            high_cutoff=high_cutoff, pass_bands=True, samplerate=samplerate)
    elif type == 'triangular'.lower():
        filter = triangular_filterbank(length=n_samples, bandwidth=bandwidth, low_cutoff=low_cutoff,
                          high_cutoff=high_cutoff, pass_bands=True, samplerate=samplerate)
    for rec_idx, recording in enumerate(recordings):
        filt_left = filter.apply(recording.channel(0))
        filt_right = filter.apply(recording.channel(1))
        filt_left = slab.Filter.collapse_subbands(filt_left, filter)
        filt_right = slab.Filter.collapse_subbands(filt_right, filter)
        recordings[rec_idx] = slab.Sound(data=(filt_left, filt_right))
    return recordings
