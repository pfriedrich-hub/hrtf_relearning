import copy

def mirror_hrtf(hrtf):
    hrtf_out = copy.deepcopy(hrtf)
    for tf in hrtf_out:
        tf.data = tf.data[:, [1, 0]]
    return hrtf_out
