import numpy
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import slab
import hrtf_relearning
hrir_folder = hrtf_relearning.PATH / 'data' / 'hrtf' / 'sofa'

hrir_name = 'SW'


def _compute_etc_linear_from_binaural_ir(data, channel="both"):
    """
    Compute linear ETC (not in dB) from a binaural IR.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, 2).
    channel : {"left", "right", "both"}
        Ear selection.

    Returns
    -------
    etc_linear : numpy.ndarray
        Linear ETC.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Expected IR data with shape (n_samples, 2).")

    envelope_left = numpy.abs(hilbert(data[:, 0]))
    envelope_right = numpy.abs(hilbert(data[:, 1]))

    if channel == "left":
        envelope = envelope_left
    elif channel == "right":
        envelope = envelope_right
    elif channel == "both":
        envelope = 0.5 * (envelope_left + envelope_right)
    else:
        raise ValueError("channel must be 'left', 'right', or 'both'.")

    etc_linear = envelope ** 2
    return etc_linear


def _estimate_onset_index_from_etc_db(etc_db, threshold_db=20.0):
    """
    Estimate onset as first sample above -threshold_db relative to peak.
    """
    above = numpy.where(etc_db >= -abs(threshold_db))[0]
    if len(above) == 0:
        return int(numpy.argmax(etc_db))
    return int(above[0])


def plot_average_etc_across_sources(
    hrir,
    source_idx,
    channel="both",
    align_onset=True,
    onset_threshold_db=20.0,
    average="median",
    percentile_band=(25, 75),
    time_range_ms=(0.0, 10.0),
    normalize_each=True,
    show_individual=False,
    show=True,
):
    """
    Plot an average ETC across multiple loudspeaker/source positions.

    Parameters
    ----------
    hrir
        HRTF/HRIR object. Must support indexing like hrir[idx].data and have
        hrir[idx].samplerate or hrir.samplerate.
    source_idx : array-like
        Indices of source positions to include.
    channel : {"left", "right", "both"}
        Which ear(s) to analyze.
    align_onset : bool
        If True, align individual ETCs to their estimated direct-sound onset
        before averaging.
    onset_threshold_db : float
        Threshold below peak for onset detection.
    average : {"mean", "median"}
        Type of average across sources.
    percentile_band : tuple[float, float]
        Percentile interval to visualize across positions.
    time_range_ms : tuple[float, float]
        Time axis range to display, relative to aligned onset if align_onset=True.
    normalize_each : bool
        If True, normalize each ETC individually to 0 dB peak before averaging.
        This is usually useful for comparing temporal structure.
    show_individual : bool
        If True, overlay individual ETCs in light gray.
    show : bool
        If True, create the plot.

    Returns
    -------
    result : dict
        Contains average ETC, spread, time axis, and all aligned ETCs in dB.
    """
    source_idx = numpy.asarray(source_idx)
    if source_idx.ndim != 1:
        raise ValueError("source_idx must be a 1D array-like of indices.")

    first_ir = hrir[int(source_idx[0])]
    if hasattr(first_ir, "samplerate"):
        samplerate = first_ir.samplerate
    elif hasattr(hrir, "samplerate"):
        samplerate = hrir.samplerate
    else:
        raise ValueError("Could not determine samplerate from hrir object.")

    # Collect ETCs and onset indices
    etc_linear_list = []
    onset_indices = []

    for idx in source_idx:
        ir_data = numpy.asarray(hrir[int(idx)].data, dtype=float)
        etc_linear = _compute_etc_linear_from_binaural_ir(ir_data, channel=channel)

        if normalize_each:
            etc_linear = etc_linear / numpy.max(numpy.maximum(etc_linear, 1e-30))

        etc_db = 10.0 * numpy.log10(numpy.maximum(etc_linear, 1e-30))
        onset_idx = _estimate_onset_index_from_etc_db(
            etc_db,
            threshold_db=onset_threshold_db,
        )

        etc_linear_list.append(etc_linear)
        onset_indices.append(onset_idx)

    etc_linear_array = numpy.asarray(etc_linear_list)
    onset_indices = numpy.asarray(onset_indices, dtype=int)

    # Align to onset by shifting so all onsets land at a common reference sample
    if align_onset:
        ref_onset = int(numpy.min(onset_indices))
        aligned = numpy.full_like(etc_linear_array, 1e-30)

        for n in range(len(source_idx)):
            shift = onset_indices[n] - ref_onset
            if shift >= 0:
                aligned[n, :etc_linear_array.shape[1] - shift] = etc_linear_array[n, shift:]
            else:
                aligned[n, -shift:] = etc_linear_array[n, :etc_linear_array.shape[1] + shift]

        etc_linear_array = aligned
        time_axis_ms = (numpy.arange(etc_linear_array.shape[1]) - ref_onset) / samplerate * 1e3
    else:
        time_axis_ms = numpy.arange(etc_linear_array.shape[1]) / samplerate * 1e3

    # Aggregate in linear domain
    if average == "mean":
        etc_linear_avg = numpy.mean(etc_linear_array, axis=0)
    elif average == "median":
        etc_linear_avg = numpy.median(etc_linear_array, axis=0)
    else:
        raise ValueError("average must be 'mean' or 'median'.")

    p_lo, p_hi = percentile_band
    etc_linear_lo = numpy.percentile(etc_linear_array, p_lo, axis=0)
    etc_linear_hi = numpy.percentile(etc_linear_array, p_hi, axis=0)

    # Convert to dB
    etc_db_all = 10.0 * numpy.log10(numpy.maximum(etc_linear_array, 1e-30))
    etc_db_avg = 10.0 * numpy.log10(numpy.maximum(etc_linear_avg, 1e-30))
    etc_db_lo = 10.0 * numpy.log10(numpy.maximum(etc_linear_lo, 1e-30))
    etc_db_hi = 10.0 * numpy.log10(numpy.maximum(etc_linear_hi, 1e-30))

    if show:
        fig, axis = plt.subplots(figsize=(10, 5))

        if show_individual:
            for n in range(etc_db_all.shape[0]):
                axis.plot(time_axis_ms, etc_db_all[n], linewidth=0.8, alpha=0.25)

        axis.fill_between(
            time_axis_ms,
            etc_db_lo,
            etc_db_hi,
            alpha=0.25,
            label=f"{p_lo}–{p_hi} percentile",
        )
        axis.plot(time_axis_ms, etc_db_avg, linewidth=2.0, label=f"{average} ETC")

        axis.axvline(0.0, linestyle="--", linewidth=1.0, label="aligned onset")

        axis.set_xlim(time_range_ms)
        axis.set_xlabel("Time relative to onset (ms)" if align_onset else "Time (ms)")
        axis.set_ylabel("Level (dB re individual peak)" if normalize_each else "Level (dB)")
        axis.set_title(f"Average ETC across {len(source_idx)} source positions ({channel})")
        axis.grid(True)
        axis.legend()
        plt.tight_layout()
        plt.show()

    return {
        "time_axis_ms": time_axis_ms,
        "etc_db_avg": etc_db_avg,
        "etc_db_lo": etc_db_lo,
        "etc_db_hi": etc_db_hi,
        "etc_db_all": etc_db_all,
        "onset_indices": onset_indices,
        "samplerate": samplerate,
    }

hrir = slab.HRTF(hrir_folder / (str(hrir_name) + '.sofa'))
result = plot_average_etc_across_sources(
    hrir=hrir,
    source_idx=hrir.cone_sources(0),
    channel="both",
    align_onset=True,
    onset_threshold_db=20.0,
    average="median",
    percentile_band=(25, 75),
    time_range_ms=(-0.5, 8.0),
    normalize_each=True,
    show_individual=True,
    show=True,
)