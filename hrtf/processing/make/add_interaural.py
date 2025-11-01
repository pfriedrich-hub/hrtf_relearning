import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import slab
import numpy
import copy
import scipy
import logging

def add_itd(hrir):
    """
    Add interaural time difference to HRIR (only works on IR)
    """
    logging.info('Adding interaural time differences.')
    out = copy.deepcopy(hrir)
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fir_coefs = hrir[source_idx].data
        azimuth = ((coordinates[0] + 180) % 360) - 180  # convert to (-180, 180)
        itd = slab.Binaural.azimuth_to_itd(azimuth=azimuth, head_radius=8.75)  # head radius in cm
        # print(f'az: {azimuth}, itd: {itd}')
        delay_n_samples = numpy.abs(int(itd * hrir.samplerate))
        if itd < 0:  # add right delay
            fir_coefs_left = fir_coefs[:, 0]
            fir_coefs_right = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 1]))[:-delay_n_samples]
        elif itd > 0:  # add left delay
            fir_coefs_left = numpy.hstack((numpy.zeros(delay_n_samples), fir_coefs[:, 0]))[:-delay_n_samples]
            fir_coefs_right = fir_coefs[:, 1]
        elif itd == 0:
            fir_coefs_left = fir_coefs[:, 0]
            fir_coefs_right = fir_coefs[:, 1]
        out[source_idx].data = numpy.array((fir_coefs_left, fir_coefs_right)).T
    return out

def add_ild(hrir):
    """
    Add interaural level differences to HRIR (only works on IR)
    """
    logging.info('Adding interaural level differences.')
    out = copy.deepcopy(hrir)
    ils = slab.Binaural.make_interaural_level_spectrum()
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        azimuth = ((coordinates[0] + 180) % 360) - 180  # convert to counterclockwise (-180, 180) for use with kemar
        ils_idx = numpy.argmin(abs(ils['azimuths']-azimuth))
        ild_db = numpy.mean(ils['level_diffs'][:, :, ils_idx], axis=1)
        # ild_db -= ild_db.mean()  # zero-mean → preserves ILD, keeps overall level stable
        gains = 10.0 ** (ild_db / 20.0)  # amplitude gains
        out[source_idx].data *= gains
    return out
#
# plt.figure()
# plt.plot(out[source_idx].data[:,0], label='left')
# plt.plot(out[source_idx].data[:,1], label='right')
# plt.legend()
# plt.title(azimuth)

# work in progress
# def add_ils(hrtf, template_hrtf=None, band_stop=(6e3, 11e3)):
#     """
#     Get a low-res version of a DTF (at 0° az and 0° elevation) from a recorded HRTF to externalize a synthetic HRTF
#     Takes DTFs of the left hemisphere and mirror for the right hemisphere.
#     """
#     print('Adding interaural level spectrum for each azimuth.')
#     n_bins = hrtf[0].n_samples
#     # if not template_hrtf:
#     #     template_hrtf = slab.HRTF.kemar()  # load Kemar as default
#
#     azimuths = numpy.unique(hrtf.sources.vertical_polar[:,0])
#
#     template_dtf_idx = template_hrtf.get_source_idx(azimuth=azimuth, elevation=0)[0]
#     w, h = template_hrtf[template_dtf_idx].tf(n_bins=12, show=False)  # get low-res frequency response of HRIR
#     h = scipy.signal.resample(h, n_bins, axis=0)  # resample to HRTF samplerate
#     h = 10 ** (h / 20) # convert dB to linear values
#
#     out = copy.deepcopy(hrtf)
#     for source_idx in range(hrtf.n_sources):
#         coordinates = hrtf.sources.vertical_polar[source_idx]
#         template_dtf_idx = template_hrtf.get_source_idx(azimuth=coordinates[0], elevation=0)[0]
#         w, h = template_hrtf[template_dtf_idx].tf(n_bins=12, show=False)  # get low-res frequency response of HRIR
#         h = scipy.signal.resample(h, n_bins, axis=0)  # resample to HRTF samplerate
#         h = 10 ** (h / 20)  # convert dB to linear values
#         tf_coefs = hrtf[source_idx].data
#         tf_coefs.data += h
#         out[source_idx].data = tf_coefs
#
#     return hrtf

# deprecated
# def add_ild(hrtf, template_hrtf=None, band_stop=(6e3, 11e3)):
#         """
#         Get a low-res version of a DTF (at 0° az and 0° elevation) from a recorded HRTF to externalize a synthetic HRTF
#         """
#         print('Adding interaural level spectrum for each azimuth.')
#         n_bins = hrtf[0].n_samples
#         if not template_hrtf:
#             template_hrtf = slab.HRTF.kemar()  # load KEMAR as default
#         for azimuth in numpy.unique(hrtf.sources.vertical_polar[:, 0]):
#             azimuth_converted = copy.deepcopy(azimuth)
#             if azimuth > 180:
#                 azimuth_converted = azimuth - 360
#             # get IR data of the template dtf (at spec. az and 0° elevation)
#             template_dtf_idx = template_hrtf.get_source_idx(azimuth_converted, 0)[0]
#             ir_data = template_hrtf.data[template_dtf_idx].data
#             # get low-res version of HRTF spectrum
#             w, tf_data_l = numpy.abs(scipy.signal.freqz(ir_data[:, 0], worN=12, fs=hrtf.samplerate))
#             _, tf_data_r = numpy.abs(scipy.signal.freqz(ir_data[:, 1], worN=12, fs=hrtf.samplerate))
#             tf_data = numpy.vstack((tf_data_l, tf_data_r))
#             tf_data[:, 0] = 1  # avoids low-freq attenuation in KEMAR HRTF
#             tf_data = scipy.signal.resample(tf_data, n_bins, axis=1)  # resample to hrtf samplerate
#             w = numpy.linspace(0, w.max(), n_bins)  # frequency bins
#             # avoid interference of interaural level spectrum with band stop region
#             tf_data[:, numpy.where((w > band_stop[0]) & (w < band_stop[1]))] = numpy.finfo(float).eps
#             # ramp edges of bandstop region
#             envelope = lambda t: numpy.sin(numpy.pi * t / 2) ** 2  # squared sine window
#             multiplier = envelope(numpy.linspace(0.0, 1.0, 20))
#             tf_data[:, numpy.where(w < band_stop[0])[0][-20:]] *= numpy.flip(multiplier)
#             tf_data[:, numpy.where(w > band_stop[1])[0][:20]] *= multiplier
#             hrtf_idx = hrtf.get_source_idx(azimuth=azimuth_converted).tolist()
#             for id in hrtf_idx:
#                 hrtf[id].data[:, 0] += tf_data[1]
#                 hrtf[id].data[:, 1] += tf_data[0]
#         return hrtf
#

