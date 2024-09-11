import slab
import numpy
import copy

# from pathlib import Path
# hrtf = slab.HRTF(Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'hrtf_1.sofa')

def tf2ir(hrtf):
    if not hrtf.datatype == 'TF':
        raise ValueError('Input datatype must be TF.')

    # todo add option for complex valued TFs
    input = copy.deepcopy(hrtf)
    dtf_data = numpy.zeros((hrtf.n_sources, hrtf[0].n_samples, 2))
    for src_idx, tf in enumerate(input.data):
        dtf_data[src_idx] = tf.data

    # ifft (take complex conjugate because sign conventions differ)
    hrir = numpy.fft.irfft(numpy.conj(dtf_data))

    # shift to make causal
    # (path differences between the origin and the ear are usually
    # smaller than 30 cm but numerical HRIRs show stringer pre-ringing)
    # hrir = np.roll(hrir, n_shift, axis=-1)

    for src_idx, tf_data in enumerate(dtf_data):
        input[src_idx] = slab.Filter(data=hrir, samplerate=hrtf.samplerate, fir=True)
    return hrtf