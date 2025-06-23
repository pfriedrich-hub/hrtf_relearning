import numpy
import copy

def flatten_dtf(hrir, ear='left'):
    """
    Flatten the TF of a channel in the HRIR (only works on IR)
    """
    logging.info(f'Flattening dtfs of the {ear} ear.')
    out = copy.deepcopy(hrir)
    if ear == 'left':
        ear_idx = 0
    elif ear == 'right':
        ear_idx = 1
    for source_idx in range(hrir.n_sources):
        flat_ir = numpy.zeros_like(hrir[0].data[:, 0])  # flat ir
        # flat_ir = fsamp(numpy.ones_like(hrir[0].data[:,0]))  # 3 sample wide peak
        ir_onset_idx = numpy.argmax(out[source_idx].data[:, ear_idx])  # onset time of original hrir
        ir_onset_gain = out[source_idx].data[ir_onset_idx, ear_idx]    # onset gain of original ir
        flat_ir[ir_onset_idx] = ir_onset_gain
        out[source_idx].data[:, ear_idx] = flat_ir
    return out