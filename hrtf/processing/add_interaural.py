
import slab
import numpy
import copy
import scipy

def add_ils(tf, azimuth, template_hrtf=None, band_stop=(6e3, 11e3)):
    """
    Get a low-res version of a DTF (at 0° az and 0° elevation) from a recorded HRTF to externalize a synthetic HRTF
    Takes DTFs of the left hemisphere and mirror for the right hemisphere.
    """
    print('Adding interaural level spectrum for each azimuth.')
    n_bins = tf.n_samples
    if not template_hrtf:
        template_hrtf = slab.HRTF.kemar()  # load Kemar as default
    template_dtf_idx = template_hrtf.get_source_idx(azimuth=azimuth, elevation=0)[0]
    w, h = template_hrtf[template_dtf_idx].tf(n_bins=12, show=False)  # get low-res frequency response of HRIR
    h = scipy.signal.resample(h, n_bins, axis=0)  # resample to HRTF samplerate
    h = 10 ** (h / 20) # convert dB to linear values
    tf.data += h
    return tf


def add_ild(hrtf, template_hrtf=None, band_stop=(6e3, 11e3)):
        """
        Get a low-res version of a DTF (at 0° az and 0° elevation) from a recorded HRTF to externalize a synthetic HRTF
        """
        print('Adding interaural level spectrum for each azimuth.')
        n_bins = hrtf[0].n_samples
        if not template_hrtf:
            template_hrtf = slab.HRTF.kemar()  # load KEMAR as default
        for azimuth in numpy.unique(hrtf.sources.vertical_polar[:, 0]):
            # todo workaround: convert to psychophys. convention to use with kemar
            azimuth_converted = copy.deepcopy(azimuth)
            if azimuth > 180:
                azimuth_converted = azimuth - 360
            # get IR data of the template dtf (at spec. az and 0° elevation)
            template_dtf_idx = template_hrtf.get_source_idx(azimuth_converted, 0)[0]
            ir_data = template_hrtf.data[template_dtf_idx].data
            # get low-res version of HRTF spectrum
            w, tf_data_l = numpy.abs(scipy.signal.freqz(ir_data[:, 0], worN=12, fs=hrtf.samplerate))
            _, tf_data_r = numpy.abs(scipy.signal.freqz(ir_data[:, 1], worN=12, fs=hrtf.samplerate))
            tf_data = numpy.vstack((tf_data_l, tf_data_r))
            tf_data[:, 0] = 1  # avoids low-freq attenuation in KEMAR HRTF
            tf_data = scipy.signal.resample(tf_data, n_bins, axis=1)  # resample to hrtf samplerate
            w = numpy.linspace(0, w.max(), n_bins)  # frequency bins
            # avoid interference of interaural level spectrum with band stop region
            tf_data[:, numpy.where((w > band_stop[0]) & (w < band_stop[1]))] = numpy.finfo(float).eps
            # ramp edges of bandstop region
            envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2  # squared sine window
            multiplier = envelope(numpy.linspace(0.0, 1.0, 20))
            tf_data[:, numpy.where(w < band_stop[0])[0][-20:]] *= numpy.flip(multiplier)
            tf_data[:, numpy.where(w > band_stop[1])[0][:20]] *= multiplier
            hrtf_idx = hrtf.get_source_idx(azimuth=azimuth_converted).tolist()
            for id in hrtf_idx:
                # todo workaround: mirror az to use with kemar
                hrtf[id].data[:, 0] += tf_data[1]
                hrtf[id].data[:, 1] += tf_data[0]
        return hrtf

def add_itd(hrir):
    """
    Add interaural time difference to HRIR (only works on IR)
    """
    print('Adding interaural time differences for each azimuth.')
    out = copy.deepcopy(hrir)
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fir_coefs = hrir[source_idx].data
        azimuth = ((coordinates[0] + 180) % 360) - 180  # convert to (-180, 180)
        itd = slab.Binaural.azimuth_to_itd(azimuth=azimuth, head_radius=8.75)  # head radius in cm
        # print(f'az: {azimuth}, itd: {itd}')
        delay_n_samples = numpy.abs(int(itd * hrir.samplerate))
        if itd < 0:  # add left delay
            fir_coefs_left = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 0]))[:-delay_n_samples]
            fir_coefs_right = fir_coefs[:, 1]
        elif itd > 0:  # add right delay
            fir_coefs_left = fir_coefs[:, 0]
            fir_coefs_right = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 1]))[:-delay_n_samples]
        elif itd == 0:
            fir_coefs_left = fir_coefs[:, 0]
            fir_coefs_right = fir_coefs[:, 1]
        out[source_idx].data = numpy.array((fir_coefs_left, fir_coefs_right)).T
    return out

