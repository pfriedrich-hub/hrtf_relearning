import pyfar
import numpy
import slab
import logging

def rec2ir(recording, reference):
    """
    Compute an impulse response (IR) from a recorded playback of a known reference signal.

    The function performs deconvolution of a recorded sweep or test signal with its
    corresponding reference signal to obtain the system's impulse response. It also
    applies regularization, time alignment, windowing, and low/high-frequency extrapolation
    to produce a clean, time-limited IR suitable for further HRTF or room-response analysis.

    Steps
    -----
        1. Ensure both signals share the same sampling rate (resample reference if needed).
        2. Convert both signals to `pyfar.Signal` objects and DC-remove them.
        3. Compute the inverse (regularized) spectrum of the reference signal.
        4. Deconvolve the recording with this inverted spectrum to obtain a preliminary IR.
        5. Identify the earliest main peak across channels as temporal reference.
        6. Apply a rectangular time window around this peak to remove reflections.
        7. Zero-pad the IR back to original length.
        8. Split the response at 400 Hz using a 4th-order crossover:
        - low-frequency component: delay-aligned and extrapolated
        - high-frequency component: taken from windowed IR
        9. Combine low- and high-frequency components and crop to 256 samples.
        10. Return as a `slab.Filter` FIR filter.

    Parameters
    ----------
        recording : slab.Binaural | slab.Filter | pyfar.Signal
        Recorded signal of the playback (e.g., measured sweep or system response).
        reference : slab.Filter | pyfar.Signal
        Original excitation signal used for playback (e.g., sweep).

    Returns
    -------
        slab.Filter
        FIR filter containing the cleaned, windowed impulse response.
        The returned object has shape (n_samples, 2) for binaural data.

    Notes
    -----
        - The function assumes two-channel (binaural) data but works for mono as well.
        - The time window around the main peak is currently fixed to (n0−50, n0+100).
        - Regularization limits are hard-coded to 20–18 750 Hz.
        - The output IR is cropped to 256 samples; adjust as needed for longer tails.
    """
    fs = recording.samplerate
    if recording.samplerate != reference.samplerate:
        logging.warning('sampling rate mismatch. resampling reference')
        reference = reference.resample(recording.samplerate)
    reference = pyfar.Signal(reference.data.T, fs)
    recording = pyfar.Signal(recording.data.T, fs)
    recording.time -= numpy.mean(recording.time, axis=1, keepdims=True)  # remove 0 hz (DC) component
    reference.time -= numpy.mean(reference.time, axis=1, keepdims=True)
    reference_inverted = pyfar.dsp.regularized_spectrum_inversion(reference, freq_range=(20, 18750))
    # 18000? over this frequency the magnitude in reference spectrum falls
    ir = recording * reference_inverted
    # apply time window to remove reflections
    # use the earlier peak (left or right) as reference in time
    n0 = min(int(numpy.argmax(numpy.abs(ir.time[0]))), int(numpy.argmax(numpy.abs(ir.time[1]))))
    ir_windowed = pyfar.dsp.time_window(ir, (max(0, n0 - 50), min(n0 + 100, len(ir.time[0]) - 1)),
                                          window='boxcar', unit="samples", crop="window")  # (170,335)
    # todo see if IR can be expressed in fewer samples
    ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir.n_samples - ir_windowed.n_samples)
    # *(10**(-1))
    ir_low = pyfar.dsp.filter.crossover(pyfar.signals.impulse(ir_windowed.n_samples), 4, 400)[0]
    ir_low.sampling_rate = fs
    ir_high = pyfar.dsp.filter.crossover(ir_windowed, 4, 400)[1]
    # error if too noisy, overwrite function and use exactly the peakpoint?
    ir_low_delayed = pyfar.dsp.fractional_time_shift(ir_low, pyfar.dsp.find_impulse_response_delay(ir_windowed))
    ir_extrapolated = ir_low_delayed + ir_high
    ir_final = pyfar.dsp.time_window(ir_extrapolated, (0, 255), 'boxcar',  # interval: (n_start,n_start + n_samples)
        crop='window')
    return slab.Filter(data=ir_final.time, samplerate=fs, fir='IR')


def plot(ir_dict, kind='tf'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for id, ir in ir_dict.items():

        if kind=='tf':
            _ = ir.tf(show=True, axis=ax)
        elif kind=='ir'.upper():
            ax.plot(ir.data)
        line = ax.lines[-1]
        line.set_label(id)
    plt.legend()

