import numpy
import copy
import logging
from hrtf.analysis.plot import plot

def flatten_dtf(hrir, ear):
    """
    Flatten the TF of a channel in the HRIR (only works on IR)
    """
    out = copy.deepcopy(hrir)
    if ear == 'left':
        ear_idx = 1 # keep the left and flatten the right ear
    elif ear == 'right':
        ear_idx = 0
    else: return out
    logging.debug(f'Flattening DTFs for the {ear} ear.')
    for source_idx in range(hrir.n_sources):
        flat_ir = numpy.zeros_like(hrir[0].data[:, 0])  # flat ir
        ir = out[source_idx].data[:, ear_idx]
        ir_diff = numpy.diff(ir)  # differential
        peak_indices = numpy.where((ir_diff[:-1] > 0) & (ir_diff[1:] < 0))[0] + 1  # find peaks
        threshold = numpy.max(numpy.abs(ir)) * 0.5  # 50% of max absolute value
        ir_onset_idx = peak_indices[numpy.abs(ir[peak_indices]) > threshold][0]  # Get the earliest large peak
        # ir_onset_idx = numpy.argmax(numpy.abs(out[source_idx].data[:, ear_idx]))  # timing of max peak in the HRIR
        ir_onset_gain = numpy.max(numpy.abs(ir))    # onset gain of original ir
        flat_ir[ir_onset_idx] = ir_onset_gain
        out[source_idx].data[:, ear_idx] = flat_ir
    plot(out, title=f'{out.name} flattened')
    return out