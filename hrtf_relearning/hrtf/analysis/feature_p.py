import numpy

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

