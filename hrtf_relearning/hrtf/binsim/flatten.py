import numpy
import copy
import logging
from hrtf_relearning.hrtf.analysis.plot_ir import plot

def flatten_dtf(hrir, ear='left', method='energy', window_ms=None, onset_thresh=0.5, keep_polarity=True):
    """
    Flatten one ear of an HRIR to a single-sample impulse while preserving ITD and broadband ILD.

    This replaces all samples of the chosen ear's impulse responses with a single delta impulse
    at the detected onset (ITD preserved). The impulse amplitude is scaled to match the
    original ear’s broadband energy or RMS level (ILD preserved). Intended for use with
    anechoic HRIRs, where the full energy represents the direct sound.

    Parameters
    ----------
    hrir : slab.HRTF
        Input HRIR object (in the time domain).
    flatten_ear : {'left', 'right'}, default='left'
        Which ear to flatten (the other ear is left unchanged).
    method : {'energy', 'rms'}, default='energy'
        Defines how to set the single-sample amplitude:
        - 'energy': match total L2 energy of the original IR.
        - 'rms': match RMS × sqrt(N), equivalent to energy for equal-length signals.
    window_ms : float or None, default=None
        Optional analysis window after onset (ms) for amplitude matching.
        For anechoic HRIRs, use None to include the full IR.
    onset_thresh : float, default=0.5
        Relative threshold (0–1) for onset detection as fraction of max |IR|.
    keep_polarity : bool, default=True
        If True, preserves the sign (polarity) of the onset peak.

    Returns
    -------
    out : slab.HRTF
        A copy of the HRIR object with one ear flattened.
    """

    out = copy.deepcopy(hrir)
    fs = hrir.samplerate
    n_src = hrir.n_sources
    ear_idx = 1 if ear == 'left' else 0

    for s in range(n_src):
        irL, irR = out[s].data[:, 0], out[s].data[:, 1]
        ir = irL if ear_idx == 0 else irR
        N = ir.size
        # Onset detection (earliest strong local max; fallback: abs max)
        abs_ir = numpy.abs(ir)
        if abs_ir.max() == 0:
            out[s].data[:, ear_idx] = 0.0
            continue
        thr = onset_thresh * abs_ir.max()
        diff = numpy.diff(ir)
        peaks = numpy.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
        strong = peaks[abs_ir[peaks] >= thr]
        onset = int(strong[0]) if strong.size else int(numpy.argmax(abs_ir))
        # Energy/RMS (full IR by default for anechoic)
        if window_ms is None:
            seg = ir
        else:
            w = max(1, int(round(window_ms * 1e-3 * fs)))
            seg = ir[onset:onset+w] if onset+w <= N else ir[onset:]
        if method == 'energy':
            A = float(numpy.linalg.norm(seg, ord=2))
        elif method == 'rms':
            rms = float(numpy.sqrt(numpy.mean(seg**2)))
            A = rms * numpy.sqrt(seg.size)
        else:
            raise ValueError("method must be 'energy' or 'rms'.")
        # Single-sample impulse
        flat = numpy.zeros_like(ir)
        tap = A if not keep_polarity else numpy.sign(ir[onset]) * A
        flat[onset] = tap
        out[s].data[:, ear_idx] = flat
    return out


def flatten_dtf_old(hrir, ear):
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
