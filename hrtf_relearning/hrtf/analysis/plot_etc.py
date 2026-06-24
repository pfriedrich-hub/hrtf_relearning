
import numpy
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks


def plot_etc_across_elevations(
    hrtf,
    ear="left",
    azimuth=0.0,
    use_energy=True,
    normalize="global",
    time_window_ms=(0.0, 5.0),
    floor_db=-60.0,
    ax=None,
    show=True,
):
    """Plot energy-time curves across elevation as a 2-D image.

    For a fixed cone of confusion (constant azimuth) the vertical-midline
    sources are selected, the Hilbert-envelope ETC is computed per elevation,
    and the result is rendered as an image: time on the x-axis, elevation on the
    y-axis, level (dB) as colour. Makes it easy to see how the direct sound and
    early reflections shift with source elevation.

    Parameters
    ----------
    hrtf : slab.HRTF
        Impulse-response HRTF (datatype 'FIR'). Each filter's ``data`` has shape
        (n_taps, n_channels) at ``hrtf[idx].samplerate``.
    ear : {"left", "right"}
        Which channel to inspect (left = 0, right = 1).
    azimuth : float
        Azimuth of the cone of sources to plot (``hrtf.cone_sources(azimuth)``).
    use_energy : bool
        If True, ETC = 10*log10(envelope**2), else 20*log10(envelope).
    normalize : {"global", "per_elevation", None}
        Normalisation of the dB scale: peak across the whole image, peak per
        elevation row, or none.
    time_window_ms : tuple[float, float]
        Time range shown on the x-axis, in milliseconds.
    floor_db : float
        Lower clip applied to the dB image (also the colour-scale minimum).
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. A new figure/axis is created if omitted.
    show : bool
        If True, call plt.show().

    Returns
    -------
    result : dict
        Contains ``times_ms``, ``elevations``, the 2-D ``etc_db`` image
        (n_elevations, n_times), the matplotlib ``ax`` and the colorbar mesh.
    """
    chan = {"left": 0, "right": 1}[ear]

    # vertical-midline sources for this cone of confusion, sorted by elevation
    src_idx = numpy.asarray(hrtf.cone_sources(azimuth))
    elevations = hrtf.sources.vertical_polar[src_idx, 1]
    order = numpy.argsort(elevations)
    src_idx = src_idx[order]
    elevations = elevations[order]

    samplerate = hrtf[int(src_idx[0])].samplerate

    # one ETC row per elevation
    etc_rows = []
    for idx in src_idx:
        data = numpy.asarray(hrtf[int(idx)].data)
        ir = data[:, chan]
        envelope = numpy.abs(hilbert(ir))
        if use_energy:
            etc_db = 10.0 * numpy.log10(numpy.maximum(envelope ** 2, 1e-30))
        else:
            etc_db = 20.0 * numpy.log10(numpy.maximum(envelope, 1e-15))
        if normalize == "per_elevation":
            etc_db -= numpy.max(etc_db)
        etc_rows.append(etc_db)

    etc_db = numpy.asarray(etc_rows)                     # (n_elevations, n_samples)
    if normalize == "global":
        etc_db -= numpy.max(etc_db)
    etc_db = numpy.clip(etc_db, floor_db, None)

    times_ms = numpy.arange(etc_db.shape[1]) / samplerate * 1e3

    # restrict to requested time window
    t_mask = (times_ms >= time_window_ms[0]) & (times_ms <= time_window_ms[1])
    times_ms = times_ms[t_mask]
    etc_db = etc_db[:, t_mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(times_ms, elevations, etc_db, cmap="hot", shading="auto")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(f"Energy-time curve across elevation ({ear}, az={azimuth:g}°)")
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label("Level (dB re peak)" if normalize else "Level (dB)")

    if show:
        plt.tight_layout()
        plt.show()

    return {
        "times_ms": times_ms,
        "elevations": elevations,
        "etc_db": etc_db,
        "ax": ax,
        "mesh": mesh,
    }


# Single-IR ETC inspector (pyfar). See plot_etc_across_elevations for the
# elevation-resolved view across a whole cone of sources.
def inspect_energy_time_curve_pyfar(
    ir,
    channel="both",
    use_energy=True,
    normalize_to_peak=True,
    direct_threshold_db=20.0,
    min_peak_distance_ms=0.2,
    search_window_ms=(0.0, 20.0),
    show=True,
):
    """
    Inspect the energy-time curve (ETC) of a pyfar impulse response and detect
    candidate early reflections.

    Parameters
    ----------
    ir : pyfar.Signal
        Time-domain impulse response signal.
    channel : {"left", "right", "both"}
        Which channel to inspect. For mono signals, this is ignored.
    use_energy : bool
        If True, compute ETC as 10*log10(envelope^2), otherwise as
        20*log10(envelope). Both have the same shape in dB.
    normalize_to_peak : bool
        If True, normalize ETC so its maximum is 0 dB.
    direct_threshold_db : float
        Threshold below the peak for estimating the direct sound onset.
    min_peak_distance_ms : float
        Minimum spacing between detected peaks.
    search_window_ms : tuple[float, float]
        Time window for peak search in milliseconds.
    show : bool
        If True, plot the ETC.

    Returns
    -------
    result : dict
        Contains ETC data, direct sound index, and detected reflection peaks.
    """
    data = numpy.asarray(ir.time)
    samplerate = ir.sampling_rate

    # pyfar usually stores channels first: (n_channels, n_samples)
    if data.ndim == 1:
        data_to_use = data
        channel_label = "mono"
    elif data.ndim == 2:
        if data.shape[0] == 1:
            data_to_use = data[0]
            channel_label = "mono"
        elif data.shape[0] >= 2:
            if channel == "left":
                data_to_use = data[0]
                channel_label = "left"
            elif channel == "right":
                data_to_use = data[1]
                channel_label = "right"
            elif channel == "both":
                envelope_left = numpy.abs(hilbert(data[0]))
                envelope_right = numpy.abs(hilbert(data[1]))
                envelope = 0.5 * (envelope_left + envelope_right)

                if use_energy:
                    etc_linear = envelope**2
                    etc_db = 10.0 * numpy.log10(numpy.maximum(etc_linear, 1e-30))
                else:
                    etc_db = 20.0 * numpy.log10(numpy.maximum(envelope, 1e-15))

                if normalize_to_peak:
                    etc_db -= numpy.max(etc_db)

                times_s = numpy.arange(len(etc_db)) / samplerate
                channel_label = "both"
            else:
                raise ValueError("channel must be 'left', 'right', or 'both'")
        else:
            raise ValueError("Signal has invalid shape.")
    else:
        raise ValueError("Only 1D or 2D pyfar signals are supported.")

    if data.ndim == 1 or (data.ndim == 2 and channel != "both"):
        envelope = numpy.abs(hilbert(data_to_use))
        if use_energy:
            etc_linear = envelope**2
            etc_db = 10.0 * numpy.log10(numpy.maximum(etc_linear, 1e-30))
        else:
            etc_db = 20.0 * numpy.log10(numpy.maximum(envelope, 1e-15))

        if normalize_to_peak:
            etc_db -= numpy.max(etc_db)

        times_s = numpy.arange(len(etc_db)) / samplerate

    # Direct sound onset: first sample above threshold relative to peak
    above = numpy.where(etc_db >= -abs(direct_threshold_db))[0]
    if len(above) == 0:
        direct_index = int(numpy.argmax(etc_db))
    else:
        direct_index = int(above[0])

    # Search for peaks in requested window
    start_idx = max(0, int(search_window_ms[0] * 1e-3 * samplerate))
    stop_idx = min(len(etc_db), int(search_window_ms[1] * 1e-3 * samplerate))
    distance_samples = max(1, int(min_peak_distance_ms * 1e-3 * samplerate))

    peaks, properties = find_peaks(
        etc_db[start_idx:stop_idx],
        distance=distance_samples,
    )
    peaks = peaks + start_idx

    # Keep only peaks after the direct sound onset
    reflection_indices = peaks[peaks > direct_index]
    reflection_times_s = times_s[reflection_indices]
    reflection_levels_db = etc_db[reflection_indices]
    reflection_delays_ms = (reflection_indices - direct_index) / samplerate * 1e3

    result = {
        "times_s": times_s,
        "etc_db": etc_db,
        "direct_index": direct_index,
        "direct_time_ms": times_s[direct_index] * 1e3,
        "reflection_indices": reflection_indices,
        "reflection_times_s": reflection_times_s,
        "reflection_times_ms": reflection_times_s * 1e3,
        "reflection_delays_ms": reflection_delays_ms,
        "reflection_levels_db": reflection_levels_db,
        "channel": channel_label,
    }

    if show:
        plt.figure(figsize=(10, 5))
        plt.plot(times_s * 1e3, etc_db, linewidth=1.2, label=f"ETC ({channel_label})")
        plt.axvline(times_s[direct_index] * 1e3, linestyle="--", label="direct sound")

        if len(reflection_indices) > 0:
            plt.plot(
                reflection_times_s * 1e3,
                reflection_levels_db,
                "x",
                label="candidate reflections",
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Level (dB re peak)")
        plt.title("Energy-Time Curve")
        plt.xlim(search_window_ms)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result

"""
result = inspect_energy_time_curve_pyfar(
    ir=ir,
    channel="both",
    direct_threshold_db=10.0,
    min_peak_distance_ms=0.15,
    search_window_ms=(0.0, 15.0),
    show=True,
)

inspect_energy_time_curve_pyfar(ir, channel="left")
inspect_energy_time_curve_pyfar(ir, channel="right")
inspect_energy_time_curve_pyfar(ir, channel="both")
"""