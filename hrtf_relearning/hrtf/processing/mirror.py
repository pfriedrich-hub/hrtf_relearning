import copy
import numpy

def mirror_hrtf(hrtf):
    hrtf_out = copy.deepcopy(hrtf)
    for idx, (tf, source) in enumerate(zip(hrtf_out, hrtf_out.sources.vertical_polar)):
        tf.data = tf.data[:, [1, 0]] # swap left and right channels
        source[0] = numpy.mod(360 - source[0], 360)  # mirror azimuth
    return hrtf_out
