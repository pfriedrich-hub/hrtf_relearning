
import numpy
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks


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
    ir=my_ir,
    channel="both",
    direct_threshold_db=20.0,
    min_peak_distance_ms=0.15,
    search_window_ms=(0.0, 15.0),
    show=True,
)

inspect_energy_time_curve_pyfar(ir, channel="left")
inspect_energy_time_curve_pyfar(ir, channel="right")
inspect_energy_time_curve_pyfar(ir, channel="both")
"""