import slab
from pathlib import Path
import numpy
import copy
data_dir = Path.cwd() / 'final_data'

def read_wav(path):
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = sorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path))
    return recordings, path_list

def hrtf_signal(level=80, duration=0.1, low_freq=1000, high_freq=17000, fs = 97656):
    ramp_duration = duration / 20
    slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
    signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq,
                              kind='linear')
    signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
    return signal

def read_source_txt(path):
    for file_path in path.rglob('*.txt'):
        sources = numpy.loadtxt(file_path)
    return sources

def apply_filterbank(recordings, type=None, bandwidth=1/3, low_cutoff=None, high_cutoff=None):
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


def triangular_filterbank(length=5000, bandwidth=1 / 3, low_cutoff=0,
                          high_cutoff=None, pass_bands=False, samplerate=97656):
    """
    Generate a set of Fourier filters. Following the organization of the
    cochlea, the width of the filter increases in proportion to it's center frequency. This increase is defined
    by Moore & Glasberg's formula for the equivalent rectangular bandwidth (ERB) of auditory filters. This
    functions is used for example to divide a sound into bands for equalization.

    Attributes:
        length (int): The number of bins in each filter, determines the frequency resolution.
        bandwidth (float): Width of the sub-filters in octaves. The smaller the bandwidth, the more filters
            will be generated.
        low_cutoff (int | float): The lower limit of frequency range in Hz.
        high_cutoff (int | float): The upper limit of frequency range in Hz. If None, use the Nyquist frequency.
        pass_bands (bool): Whether to include a half cosine at the filter bank's lower and upper edge frequency.
            If True, allows reconstruction of original bandwidth when collapsing subbands.
        samplerate (int | None): the samplerate of the sound that the filter shall be applied to.
            If None, use the default samplerate.s
    Examples::

        sig = slab.Sound.pinknoise(samplerate=44100)
        fbank = slab.Filter.cos_filterbank(length=sig.n_samples, bandwidth=1/10, low_cutoff=100,
                                      samplerate=sig.samplerate)
        fbank.tf()
        # apply the filter bank to the data. The filtered sound will contain as many channels as there are
        # filters in the bank. Every channel is a copy of the original sound with one filter applied.
        # In this context, the channels are the signals sub-bands:
        sig_filt = fbank.apply(sig)
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
        width = erb_spacing * 2  # width of filter
        filts[(freqs_erb > l) & (freqs_erb < h), i] = numpy.cos(
            (freqs_erb[(freqs_erb > l) & (freqs_erb < h)] - avg) / width * numpy.pi)
        filts[(freqs_erb > l) & (freqs_erb < avg), i] = numpy.linspace(0, 1,
                                                    len(freqs_erb[(freqs_erb > l) & (freqs_erb < avg)]))
        filts[(freqs_erb < h) & (freqs_erb > avg), i] = numpy.linspace(1, 0,
                                                    len(freqs_erb[(freqs_erb < h) & (freqs_erb > avg)]))
    return slab.Filter(data=filts, samplerate=samplerate, fir=False)
