import slab
import numpy
from pathlib import Path
import copy

def get_hrtf_list(database_name='aachen_database'):
    database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / database_name
    return [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]

def make_avg_hrtf():
    """
    Construct a new HRTF from a list of existing HRTFs.
    Computes the probability of peaks and notches for each source and frequency bin across HRTFs in the list.

    Args:
        database_name (string): Name the folder containing the HRTF database.
                                HRTFs in the folder must have uniform source space and filters.
        threshold (tuple): upper and lower threshold in dB for feature detection
    """
    hrtf_list = get_hrtf_list()
    hrtf = hrtf_list[0]
    w, _ = hrtf[0].tf(show=False)
    in_freq = w > 4000
    data = []
    thr = []
    for i, hrtf in enumerate(hrtf_list):
        tfs = hrtf.tfs_from_sources(sources=range(len(hrtf.sources.vertical_polar)), n_bins=None, ear='both')
        tfs = tfs[:, in_freq]
        thr.append((numpy.percentile(tfs, 25), numpy.percentile(tfs, 75)))
        print(numpy.percentile(tfs, 25), numpy.percentile(tfs, 75))
        data.append(hrtf.tfs_from_sources(sources=range(len(hrtf.sources.vertical_polar)), n_bins=None, ear='both'))

    data = numpy.asarray(data)
    threshold = numpy.percentile(data, 25), numpy.percentile(data, 75)
    #todo find a better measure than thresholding,
    # for example take the jnd spectral notch width and depth in dB
    print(f"Thresholds; lower: {threshold[0]}, upper: {threshold[1]}")
    # compute probability map of spectral peaks and notches across frequencies and sources
        print(f"Retrieving TF data from database: {i / len(hrtf_list) * 100:.2f} %")
        dtfs = []
        for filter in hrtf:
            w, h = filter.tf(n_bins=None, show=False)
            dtfs.append(h)
        # thresh = [numpy.percentile(tfs, 25), numpy.percentile(tfs, 75)]  # local threshold
        data.append(dtfs)
    del hrtf_list
    data = numpy.array(data)
    freq_mask = numpy.where(w > 4000)  # disregard torso shadow for cue threshold estimation
    if threshold is None:
        threshold = numpy.percentile(data[:, :, freq_mask], 25), numpy.percentile(data[:, :, freq_mask], 75)  # global threshold
        print(f"Thresholds; lower: {threshold[0]}, upper: {threshold[1]}")
    notch_map = (data < threshold[0]) * -1 # binary map of notches across hrtfs, sources, frequencies
    peak_map = (data > threshold[1]) # binary map of peaks across hrtfs, sources, frequencies
    p_map = numpy.mean(notch_map + peak_map, axis=0)
    # rescale to dB


    hrtf = copy.deepcopy(hrtf_list[0])  # new HRTF
    for i, tf in enumerate(p_map):  # reconstruct filters from feature probability
        tf = (tf + 1) * 0.5 * (threshold[1] - threshold[0]) + threshold[0]  # rescale probabilities between thresholds
        hrtf[i].data = slab.Filter(data=tf, samplerate=hrtf_list[0].samplerate, fir='TF')
    # return numpy.sum(numpy.array(feature_maps), axis=0) / len(feature_maps), frequencies[freq_idx]  # average
    feature_map = numpy.sum(notch_map + peak_map, axis=0) / len(data)
    # rescale probabilities to thresholds
    dtfs = (feature_map + 1) * 0.5 * (threshold[1] - threshold[0]) + threshold[0]
    for i, tf in enumerate(dtfs):  # reconstruct filters from feature probability
        hrtf[i].data = slab.Filter(data=tf, samplerate=hrtf.samplerate, fir='TF')

# return numpy.sum(numpy.array(feature_maps), axis=0) / len(feature_maps), frequencies[freq_idx]  # average

def feature_p(hrtf_list, source_idx, thresholds=None, bandwidth=(1000,18000), ear='both', show=False):
    threshold_list = []
    feature_maps = []
    for i, hrtf in enumerate(hrtf_list):
        frequencies = hrtf[0].tf(show=False)[0]
        freq_idx = numpy.where(numpy.logical_and(frequencies>bandwidth[0], frequencies<bandwidth[1]))
        tf_data = hrtf.tfs_from_sources(sources=source_idx, n_bins=None, ear=ear)
        if ear == 'both':   # average
            tf_data = numpy.average(tf_data, axis=2)
        tf_data = numpy.squeeze(tf_data[:, freq_idx], axis=1)  # crop freq bins
        if not thresholds:
            # for each hrtf, define thresholds at the lower and upper quartiles at 0 azimtuh to mark spectral features:
            src_idx_0_az = hrtf.get_source_idx(azimuth=0, elevation=(-90, 90))
            tf_0_az = hrtf.tfs_from_sources(sources=src_idx_0_az, n_bins=None, ear='both')
            thresh = [numpy.percentile(tf_0_az, 25), numpy.percentile(tf_0_az, 75)]

            # # get mean of RMS differences across all combinations of DTFs measured with free ears (Trapeau, Schönwiesner 2015)
            # # instead, get mean across min / max for each elevation - a bit extreme
            # # instead, get upper and lower 5%
            # # thresholds = (numpy.mean(numpy.min(tf_data, axis=1)), numpy.mean(numpy.max(tf_data, axis=1)))
            # l_thresh = [numpy.percentile(tf_data[:,:,0], 25), numpy.percentile(tf_data[:,:,0], 75)]
            # r_thresh = [numpy.percentile(tf_data[:,:,1], 25), numpy.percentile(tf_data[:,:,1], 75)]
            # if ear == 'left':
            #     threshold_list.extend([l_thresh])
            # elif ear == 'right':
            #     threshold_list.extend([r_thresh])
            # elif ear == 'both':
            #     threshold_list.extend([r_thresh, l_thresh])
            # else: print('Error. ear must be "left" "right" or "both"')
        else:
            thresh = thresholds
        # get freq bins and elevations for which gain was below threshold
        # l_notch_map = (tf_data[:,:,0] < l_thresh[0]) * -1
        # l_peak_map = (tf_data[:,:,0] > l_thresh[1])
        # l_filtered = l_notch_map + l_peak_map
        # r_notch_map = (tf_data[:,:,1] < r_thresh[0]) * -1
        # r_peak_map = (tf_data[:,:,1] > r_thresh[1])
        # r_filtered = r_notch_map + r_peak_map
        # if ear == 'left':
        #     feature_maps.extend([l_filtered])
        # elif ear == 'right':
        #     feature_maps.extend([r_filtered])
        # elif ear == 'both':
        #     feature_maps.extend([r_filtered, l_filtered])
        notch_map = (tf_data < thresh[0]) * -1
        peak_map = (tf_data > thresh[1])
        filtered = notch_map + peak_map
        feature_maps.extend([filtered])
    return numpy.sum(numpy.array(feature_maps), axis=0) / len(feature_maps), frequencies[freq_idx]  # average

