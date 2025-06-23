import numpy

def feature_p(hrtf_list, source_idx, thresholds=None, bandwidth=(1000,18000), show=False):
    threshold_list = []
    feature_maps = []
    for i, hrtf in enumerate(hrtf_list):
        frequencies = hrtf[0].tf(show=False)[0]
        freq_idx = numpy.where(numpy.logical_and(frequencies>bandwidth[0], frequencies<bandwidth[1]))
        tf_data = hrtf.tfs_from_sources(sources=source_idx, n_bins=None, ear='both')
        tf_data = numpy.squeeze(tf_data[:, freq_idx], axis=1)  # crop freq bins
        if not thresholds:
            # get mean of RMS differences across all combinations of DTFs measured with free ears (Trapeau, Schönwiesner 2015)
            # instead, get mean across min / max for each elevation - a bit extreme
            # instead, get upper and lower 5%
            # thresholds = (numpy.mean(numpy.min(tf_data, axis=1)), numpy.mean(numpy.max(tf_data, axis=1)))
            l_thresh = [numpy.percentile(tf_data[:,:,0], 25), numpy.percentile(tf_data[:,:,0], 75)]
            r_thresh = [numpy.percentile(tf_data[:,:,1], 25), numpy.percentile(tf_data[:,:,1], 75)]
            threshold_list.extend([r_thresh, l_thresh])
        else:
            thresh = thresholds
        # get freq bins and elevations for which gain was below threshold
        l_notch_map = (tf_data[:,:,0] < l_thresh[0]) * -1
        l_peak_map = (tf_data[:,:,0] > l_thresh[1])
        l_filtered = l_notch_map + l_peak_map
        r_notch_map = (tf_data[:,:,1] < r_thresh[0]) * -1
        r_peak_map = (tf_data[:,:,1] > r_thresh[1])
        r_filtered = r_notch_map + r_peak_map
        feature_maps.extend([l_filtered, r_filtered])
    return numpy.sum(numpy.array(feature_maps), axis=0) / len(feature_maps), frequencies[freq_idx]  # average


"""
database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'aachen_database'
hrtf_list = [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*'))]

hrtf = slab.HRTF(list(database_path.glob('*'))[0])

# get source idx for sources in range -40 / +40 elevation
sourceidx = numpy.array(hrtf.cone_sources(0))  # get vertical midline sources within +/- 40° Elevation
sourceidx = sourceidx[numpy.logical_and(hrtf.sources.vertical_polar[sourceidx, 1] > -41,
                                    hrtf.sources.vertical_polar[sourceidx, 1] < 41)]
elevations = hrtf.sources.vertical_polar[sourceidx, 1]


# plot features for sources in range 0 / +50 azimuth across elevations
for ele in elevations:
    sourceidx = numpy.where(hrtf.sources.vertical_polar[:, 1] == ele)[0]
    sourceidx = sourceidx[numpy.logical_and(hrtf.sources.vertical_polar[sourceidx, 0] > 0,
                                        hrtf.sources.vertical_polar[sourceidx, 0] < 50)]
    azimuths = hrtf.sources.vertical_polar[sourceidx, 0]

    map, frequencies = feature_p(hrtf_list, sourceidx, thresholds=None, bandwidth=(1000,18000))

    # plot average
    fig, axis = plt.subplots()
    contour = axis.contourf(frequencies, azimuths, map)
    plt.title(ele)
    
"""