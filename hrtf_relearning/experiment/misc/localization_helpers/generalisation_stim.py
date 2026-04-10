import numpy
import slab


def erb_space(f_min, f_max, n_bands):
    """
    Approximate ERB-rate spacing using the Glasberg/Moore style ERB-rate formula.
    Returns band edge frequencies in Hz.
    """
    def hz_to_erbrate(f_hz):
        f_khz = f_hz / 1000.0
        return 21.4 * numpy.log10(4.37 * f_khz + 1.0)

    def erbrate_to_hz(erb):
        return (10 ** (erb / 21.4) - 1.0) / 4.37 * 1000.0

    erb_min = hz_to_erbrate(f_min)
    erb_max = hz_to_erbrate(f_max)
    erb_edges = numpy.linspace(erb_min, erb_max, n_bands + 1)
    return erbrate_to_hz(erb_edges)


def make_band_envelope(freqs, band_edges, band_gains_db, smooth_bins=8):
    """
    Create a piecewise-constant spectral envelope on the FFT frequency axis,
    then smooth it slightly to avoid sharp discontinuities.
    """
    env_db = numpy.zeros_like(freqs)

    for i in range(len(band_gains_db)):
        idx = numpy.logical_and(freqs >= band_edges[i], freqs < band_edges[i + 1])
        env_db[idx] = band_gains_db[i]

    # include last edge
    env_db[freqs >= band_edges[-2]] = band_gains_db[-1]

    # smooth in frequency domain
    if smooth_bins > 1:
        kernel = numpy.hanning(smooth_bins)
        kernel = kernel / kernel.sum()
        env_db = numpy.convolve(env_db, kernel, mode="same")

    env_lin = 10 ** (env_db / 20.0)
    return env_lin


def apply_random_spectral_shape(
    stim,
    band_edges,
    band_gains_db,
    f_min=500,
    f_max=16000,
    smooth_bins=8
):
    """
    Apply a random spectral envelope to a mono slab.Sound stimulus.
    """
    data = stim.data.squeeze()
    n_samples = len(data)

    spectrum = numpy.fft.rfft(data)
    freqs = numpy.fft.rfftfreq(n_samples, d=1 / stim.samplerate)

    env = numpy.ones_like(freqs)
    band_env = make_band_envelope(freqs, band_edges, band_gains_db, smooth_bins=smooth_bins)

    valid = numpy.logical_and(freqs >= f_min, freqs <= f_max)
    env[valid] = band_env[valid]

    shaped = numpy.fft.irfft(spectrum * env, n=n_samples)
    shaped = shaped / numpy.max(numpy.abs(shaped)) * 0.99

    out = slab.Sound(shaped, samplerate=stim.samplerate)
    out.level = stim.level
    return out


def apply_temporal_segments(stim, segment_gains_db, segment_edges_ms, ramp_ms=5):
    """
    Apply piecewise level changes across time segments.
    Keeps the stimulus noise-like but introduces mild temporal variation.
    """
    data = stim.data.copy().squeeze()
    sr = stim.samplerate
    n = len(data)

    segment_edges = (numpy.asarray(segment_edges_ms) * sr / 1000.0).astype(int)
    gain = numpy.ones(n)

    for i in range(len(segment_gains_db)):
        start = segment_edges[i]
        end = segment_edges[i + 1]
        gain[start:end] *= 10 ** (segment_gains_db[i] / 20.0)

    ramp_len = int(ramp_ms / 1000.0 * sr)
    if ramp_len > 1:
        kernel = numpy.hanning(ramp_len * 2 + 1)
        kernel = kernel / kernel.sum()
        gain = numpy.convolve(gain, kernel, mode="same")

    data = data * gain
    data = data / numpy.max(numpy.abs(data)) * 0.99

    out = slab.Sound(data, samplerate=sr)
    out.level = stim.level
    return out


def make_generalisation_bank(
    n_stimuli,
    duration=0.225,
    level=80,
    f_min=500,
    f_max=16000,
    n_bands=10,
    spectral_depth_db=6,
    use_temporal_variation=True,
    random_state=None
):
    """
    Generate a bank of broadband, noise-like localisation stimuli with
    balanced random spectral patterns and optional mild temporal variation.
    """
    rng = numpy.random.default_rng(random_state)

    band_edges = erb_space(f_min, f_max, n_bands)

    # Balanced spectral patterns:
    # each band is high in half the stimuli and low in half the stimuli
    band_matrix = numpy.zeros((n_stimuli, n_bands))
    for b in range(n_bands):
        vals = numpy.array([spectral_depth_db] * (n_stimuli // 2) +
                           [-spectral_depth_db] * (n_stimuli - n_stimuli // 2))
        rng.shuffle(vals)
        band_matrix[:, b] = vals

    stimuli = []
    metadata = []

    for i in range(n_stimuli):
        stim = slab.Sound.whitenoise(duration=duration, level=level).ramp(
            when="both", duration=0.01
        )

        stim = apply_random_spectral_shape(
            stim=stim,
            band_edges=band_edges,
            band_gains_db=band_matrix[i],
            f_min=f_min,
            f_max=f_max,
            smooth_bins=10
        )

        if use_temporal_variation:
            # 3 equal segments with balanced random gains
            segment_edges_ms = numpy.array([0, 75, 150, 225])
            temporal_pattern = numpy.array([-3, 0, 3], dtype=float)
            rng.shuffle(temporal_pattern)
            stim = apply_temporal_segments(
                stim,
                segment_gains_db=temporal_pattern,
                segment_edges_ms=segment_edges_ms,
                ramp_ms=5
            )
        else:
            temporal_pattern = None

        stim = stim.ramp(when="both", duration=0.01)
        stimuli.append(stim)
        metadata.append({
            "spectral_gains_db": band_matrix[i].copy(),
            "temporal_gains_db": temporal_pattern
        })

    return stimuli, metadata, band_edges

stimuli, metadata, band_edges = make_generalisation_bank(
    n_stimuli=40,
    duration=0.225,
    level=80,
    f_min=500,
    f_max=16000,
    n_bands=10,
    spectral_depth_db=6,
    use_temporal_variation=True,
    random_state=42
)

