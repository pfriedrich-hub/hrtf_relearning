
import slab
import numpy

def add_ild(hrtf):
    """
    Add interaural level difference to HRTF (works on TF data)
    """
    w = hrtf[0].frequencies
    for source_idx in range(hrtf.n_sources):
        coordinates = hrtf.sources.vertical_polar[source_idx]
        ils_db = slab.Binaural.azimuth_to_ild(coordinates[0], frequency=2000)
        # Convert dB gains to linear scale
        ils = 10 ** (ils_db / 20)
        h = hrtf[source_idx].data


        # H = ...  # existing frequency response (complex)
        # G_dB = ...  # desired gain in dB
        # G = 10 ** (G_dB / 20)
        #
        # H_mod = H * G

    return hrtf

def add_itd(hrir):
    """
    Add interaural time difference to HRIR (only works on IR)
    """
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fir_coefs = hrir[source_idx].data
        itd = slab.Binaural.azimuth_to_itd(azimuth=coordinates[0], head_radius=11)  # head radius in cm
        delay_n_samples = int(itd * hrir.samplerate)
        if itd >= 0:  # add left delay
            fir_coefs_left = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 0]))
            fir_coefs_right = numpy.hstack((fir_coefs[:, 1], numpy.zeros(delay_n_samples)))
        elif itd < 0:  # add right delay
            fir_coefs_left = numpy.hstack((fir_coefs[:, 0], numpy.zeros(delay_n_samples)))
            fir_coefs_right = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 1]))
        hrir[source_idx].data = numpy.array((fir_coefs_left, fir_coefs_right)).T
    return hrir