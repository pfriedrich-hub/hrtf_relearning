import numpy
import copy
import slab

def hrtf_average(hrtf_list, average_ears=False):
    input = copy.deepcopy(hrtf_list)
    dtf_data = numpy.zeros((hrtf_list[0].n_sources, len(hrtf_list), hrtf_list[0][0].n_samples, 2))
    for hrtf_idx, hrtf in enumerate(input):
        for src_idx, tf in enumerate(hrtf.data):
            dtf_data[src_idx, hrtf_idx] = tf.data
    if average_ears:
        dtf_data = dtf_data.mean(axis=(1,3))
        for src_idx, tf_data in enumerate(dtf_data):
            hrtf[src_idx] = slab.Filter(data=tf_data, samplerate=hrtf_list[0].samplerate, fir=False)
    else:
        dtf_data = numpy.mean(dtf_data, axis=1)
        for src_idx, tf_data in enumerate(dtf_data):
            hrtf[src_idx].data = tf_data
    return hrtf