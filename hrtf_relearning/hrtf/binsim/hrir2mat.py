import copy
import json
from fractions import Fraction

import numpy
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.signal import fftconvolve, resample_poly
import slab
import logging
import pyfar
from pathlib import Path
import hrtf_relearning
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import load_hp_filter
from hrtf_relearning.utils import paths

logger = logging.getLogger(__name__)

ROOT = Path(hrtf_relearning.__file__).resolve().parent
wav_path = paths.BINSIM_DIR
sofa_path = paths.SOFA_DIR
sound_path = paths.SOUNDS_DIR
rec_path = paths.REC_DIR

def resample_sounds(target_samplerate, target_directory):
    logging.info('Resampling sound files.')
    # soundfile needs an integer rate, and slab carries whatever it is given
    # straight through resample() into write(). A float target only shows up
    # when a resample actually happens, so coerce here rather than relying on
    # the caller.
    target_samplerate = int(target_samplerate)
    for file in sound_path.glob('*.wav'): # resample sound files
        sound = slab.Sound.read(file)
        if not sound.samplerate == target_samplerate:
            sound = sound.resample(target_samplerate)
        sound.write(target_directory / file.name, normalise=True)


def resample_hrir(hrir, samplerate):
    """Return a copy of ``hrir`` resampled to ``samplerate``.

    The recordings stay on disk at the rate the rig measured them at; this is
    the single point where a database is moved to the rate the playback device
    actually runs. Doing it here rather than leaving it to the audio backend
    matters because a mismatch is not reported: WASAPI shared mode converts
    silently to the endpoint's mix format, and exclusive mode simply refuses to
    open the stream.

    ``slab.HRTF`` has no resample of its own, but it carries nothing beyond
    ``data`` / ``datatype`` / ``samplerate`` / ``sources`` / ``listener``, so
    resampling each Filter on a deepcopy keeps source positions, listener
    geometry and ``name`` intact without going through the constructor.

    Deliberately NOT slab's ``Filter.resample``: that wraps
    ``scipy.signal.resample``, whose periodicity assumption is a poor fit for an
    impulse response and drifts the pinna notches -- measured over KEMAR at
    48828 -> 48000 it moved them by a median 7.2 Hz, against 0.2 Hz for the
    polyphase route. Small either way, but it is a systematic bias in the exact
    cue these experiments manipulate, and the exact 4000/4069 ratio only costs
    about six seconds over a 710-source database.

    Resampling preserves duration, not tap count -- 512 taps at 48828 Hz come
    back as 504 at 48000. ``block_size`` is derived from ``n_taps``, so filters
    that shrink are zero-padded back to their original length and the block
    size stays what it was. Filters that grow are left alone: padding is free,
    but trimming would drop the tail of the IR.
    """
    # Keep this an int. It becomes hrir.samplerate, which is handed straight to
    # resample_sounds and ends up in slab.Sound.write -> soundfile, and that
    # rejects a float rate outright ("an integer is required").
    samplerate = int(samplerate)

    if hrir.samplerate == samplerate:
        return hrir

    logger.info("Resampling HRIR %s: %g -> %g Hz",
                getattr(hrir, "name", "?"), hrir.samplerate, samplerate)

    # limit_denominator keeps the polyphase kernel finite; 10000 is loose enough
    # that the rig rates in use resolve exactly (48000/48828 -> 4000/4069).
    ratio = Fraction(samplerate / hrir.samplerate).limit_denominator(10000)

    n_taps_in = hrir[0].n_taps
    out = copy.deepcopy(hrir)

    resampled = []
    for filt in hrir.data:
        new = copy.deepcopy(filt)
        new.data = resample_poly(filt.data, ratio.numerator, ratio.denominator,
                                 axis=0)
        new.samplerate = samplerate
        resampled.append(new)

    out.data = resampled
    out.samplerate = samplerate

    if out[0].n_taps < n_taps_in:
        out.data = [filt.resize(n_taps_in) for filt in out.data]

    return out

# ---- level normalisation ---- #

# Broadband gain the rendered chain should have: the RMS that a unit-RMS source
# picks up on its way to the ear drum (DS convolved with the headphone filter),
# taken as the MEDIAN over source directions.
#
# Why this is needed: DTF-derived HRIRs carry no meaningful absolute scale --
# free-field equalisation, the diffuse-field division and the headphone
# inversion each move the level by several dB, differently per subject. Measured
# across the first four databases the chain gain spanned 6.5 to 15.2 dB, i.e.
# the SAME loc_settings['gain'] produced an 8.7 dB louder presentation for one
# subject than for another, and the loudest ones drove pyBinSim past +-1
# ("Clipping occured: Adjust loudnessFactor!" -- pybinsim/application.py).
#
# The reference is the level of the first databases (JS, GLK) that the AR/dome
# loudness match was made against, so the matched gain stays valid; re-verify it
# once with match_ar_dome_loudness.py.
REFERENCE_LEVEL = 2.3

# Runtime loudness (loc_settings['gain'] / '/pyBinSimLoudness') that the
# headroom check assumes. Only used for the QC warning, not for the scaling.
REFERENCE_GAIN = 0.2

# Warn when the predicted output peak gets this close to full scale.
PEAK_LIMIT = 0.9


def _chain_ir(ds_ir, hp_ir):
    """Direct-sound IR in series with the headphone filter, per ear."""
    if hp_ir is None:
        return ds_ir
    return numpy.stack(
        [fftconvolve(ds_ir[:, c], hp_ir[:, c]) for c in range(2)], axis=1
    )


def chain_levels(hrir, hp_ir=None):
    """
    Broadband gain of the DS (-> HP) chain, per source direction.

    Returns
    -------
    numpy.ndarray
        Array of shape [n_sources]: the output RMS a unit-RMS white-noise source
        would produce, averaged over the two ears. This is the L2 norm of the
        chain impulse response, so it is independent of the stimulus.
    """
    return numpy.array([
        numpy.sqrt(numpy.mean(numpy.sum(_chain_ir(hrir[idx].data, hp_ir) ** 2, axis=0)))
        for idx in range(hrir.n_sources)
    ])


def normalization_gain(hrir, hp_ir=None, reference=REFERENCE_LEVEL):
    """
    Single scalar that brings the whole filter set to the reference level.

    ONE gain for the entire HRTF -- applied to every direction and to the reverb
    tail alike -- so all amplitude ratios survive untouched: left/right within a
    direction (ILD), direction-to-direction level differences, and the
    direct-to-reverberant ratio. Only the absolute scale changes.

    Returns
    -------
    (float, numpy.ndarray)
        The gain, and the per-direction chain levels it was derived from.
    """
    levels = chain_levels(hrir, hp_ir)
    level = float(numpy.median(levels))
    if not level > 0:
        raise ValueError(f"Median chain level is {level}. Check the HRIR and HP filter.")

    gain = reference / level
    logger.info(
        "Level normalisation | median chain gain %.2f dB -> reference %.2f dB "
        "(scalar %.4f, %+.2f dB)",
        20 * numpy.log10(level), 20 * numpy.log10(reference), gain, 20 * numpy.log10(gain),
    )
    return gain, levels


def predicted_peak(hrir, lr_ir, hp_ir, stimulus, norm_gain=1.0,
                   loudness=REFERENCE_GAIN, levels=None, n_check=20):
    """
    Peak output pyBinSim would produce, for the loudest directions.

    Mirrors the render chain in pybinsim.application.audio_callback: the source
    is scaled by ``loudness``, convolved with DS and LR, summed, then convolved
    with the headphone filter. Values above 1.0 clip.

    Only the ``n_check`` loudest directions are checked, which is where the peak
    lives, so the build does not pay for all directions.
    """
    stimulus = numpy.asarray(stimulus, dtype=float)
    if stimulus.ndim > 1:
        stimulus = stimulus[:, 0]
    source = stimulus * loudness * norm_gain

    if levels is None:
        levels = chain_levels(hrir, hp_ir)
    loudest = numpy.argsort(levels)[::-1][:n_check]

    peak = 0.0
    for idx in loudest:
        ds = hrir[idx].data
        out = numpy.stack([fftconvolve(source, ds[:, c]) for c in range(2)], axis=1)
        if lr_ir is not None:
            rev = numpy.stack(
                [fftconvolve(source, lr_ir[:, c] * norm_gain) for c in range(2)], axis=1)
            n = max(out.shape[0], rev.shape[0])
            summed = numpy.zeros((n, 2))
            summed[:out.shape[0]] = out
            summed[:rev.shape[0]] += rev
            out = summed
        if hp_ir is not None:
            out = numpy.stack(
                [fftconvolve(out[:, c], hp_ir[:, c]) for c in range(2)], axis=1)
        peak = max(peak, float(numpy.abs(out).max()))
    return peak


def write_level_info(hrir, mat_path, norm_gain, levels, peak=None,
                     reference=REFERENCE_LEVEL, loudness=REFERENCE_GAIN):
    """
    Write the normalisation provenance next to the MAT database.

    It cannot live inside the MAT: pyBinSim's FilterStorage.parse_and_load_matfile
    walks EVERY top-level variable in the file and raises "Filter indentifier
    wrong or missing" on anything that is not a filter struct.
    """
    info_path = Path(mat_path).with_name(f"{hrir.name}_level.json")
    info = {
        "name": hrir.name,
        "reference_level": reference,
        "median_chain_level": float(numpy.median(levels)),
        "chain_level_range": [float(levels.min()), float(levels.max())],
        "norm_gain": float(norm_gain),
        "norm_gain_db": float(20 * numpy.log10(norm_gain)),
        "reference_loudness": loudness,
        "predicted_peak": None if peak is None else float(peak),
    }
    info_path.write_text(json.dumps(info, indent=2))
    return info_path


# ---- mat writers ---- #

def write_filters(hrir, lr_ir, hp_ir, mat_path, norm_gain=1.0):
    """
    Write a pyBinSim-compatible MAT database matching FilterStorage.parse_and_load_matfile()

    Fields expected by pyBinSim (per row):
      - type
      - listenerOrientation (1x3)
      - listenerPosition (1x3)
      - sourceOrientation (1x3)
      - sourcePosition (1x3)   (az, el, r) in your convention
      - custom (1x3)
      - ir  (nSamples x 2)

    norm_gain scales the DS and LR filters (never HP, which is the measured
    equalisation and sits in series with them). One scalar for the whole set, so
    ILD, direction-to-direction level differences and the DRR are all preserved.
    See normalization_gain().
    """
    norm_gain = float(norm_gain)

    # structured dtype with all required fields
    dtype = numpy.dtype([
        ("type", "U2"),
        ("listenerOrientation", "O"),
        ("listenerPosition", "O"),
        ("sourceOrientation", "O"),
        ("sourcePosition", "O"),
        ("custom", "O"),
        ("filter", "O"),
    ])

    rows = []

    zeros3 = numpy.zeros(3, dtype=numpy.float32)

    # ---------- DS rows ----------
    for src_idx in range(hrir.n_sources):
        az, el, _ = hrir.sources.vertical_polar[src_idx]

        rows.append((
            "DS",
            zeros3.copy(),                 # listenerOrientation
            zeros3.copy(),                 # listenerPosition
            zeros3.copy(),                 # sourceOrientation
            numpy.array([az, el, 0.0], dtype=numpy.float32),  # sourcePosition
            zeros3.copy(),                 # custom
            (hrir[src_idx].data * norm_gain).astype(numpy.float32),  # ir
        ))

    # ---------- LR row ----------
    if lr_ir is not None:
        rows.append((
            "LR",
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            (lr_ir * norm_gain).astype(numpy.float32),
        ))

    # ---------- HP row ----------
    if hp_ir is not None:
        rows.append((
            "HP",
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            zeros3.copy(),
            hp_ir.astype(numpy.float32),
        ))

    filters = numpy.zeros((1, len(rows)), dtype=dtype)
    for i, r in enumerate(rows):
        filters[0, i]["type"] = r[0]
        filters[0, i]["listenerOrientation"] = r[1]
        filters[0, i]["listenerPosition"] = r[2]
        filters[0, i]["sourceOrientation"] = r[3]
        filters[0, i]["sourcePosition"] = r[4]
        filters[0, i]["custom"] = r[5]
        filters[0, i]["filter"] = r[6]

    savemat(mat_path, {"filters": filters}, do_compression=True)

def write_filter_list(hrir):
    pose = "0 0 0  0 0 0  0 0 0  0 0 0  0 0 0"
    fname = wav_path / hrir.name / f"filter_list_{hrir.name}.txt"

    with open(fname, "w") as f:
        f.write(f"DS {pose}\n")
        f.write(f"LR {pose}\n")
        f.write("HP\n")

    return fname

def compute_lr_ir(
    hrir,
    drr: float = 20.0,
    block_size: int = 256,
    tail_duration: float = 0.1,
    predelay_ms: float = 3.0,
    onset_threshold_db: float = 10.0,
    show: bool = False,
) -> numpy.ndarray:
    """
    Compute a binaural late-reverb impulse response without writing a WAV file.

    The function:
    1. loads a binaural reverb tail from ``reverb.wav``,
    2. crops it to a fixed duration aligned to ``block_size``,
    3. applies a short onset ramp to avoid clicks,
    4. delays the tail so it starts shortly after the average direct HRIR onset,
    5. sets its level relative to the mean direct HRIR level using ``drr``.

    Parameters
    ----------
    hrir
        HRTF/HRIR object with attributes such as ``samplerate``, ``n_sources``,
        ``name``, and item access ``hrir[idx].data`` returning an array of shape
        ``[n_samples, 2]``.
    drr : float, default=20.0
        Desired direct-to-reverberant ratio in dB. The reverb tail level is set
        to ``mean_direct_level - drr``.
    block_size : int, default=256
        Output length is cropped to an integer multiple of this block size.
    tail_duration : float, default=0.3
        Duration of the reverb tail in seconds before block alignment.
    predelay_ms : float, default=1.5
        Extra delay added after the estimated average HRIR onset. Small values
        are appropriate here to reduce masking of direct spectral cues without
        creating a perceptually oversized room.
    onset_threshold_db : float, default=20.0
        Threshold below the peak binaural energy used for onset detection.

    Returns
    -------
    numpy.ndarray
        Binaural late-reverb IR of shape ``[n_samples, 2]`` and dtype
        ``numpy.float32``.

    Notes
    -----
    This function assumes that the HRIRs are already short/windowed and mostly
    contain direct sound. Under that assumption, HRIR RMS is a reasonable proxy
    for direct level.
    """
    # Load the stored binaural reverb tail.
    reverb_path = wav_path / hrir.name / "sounds" / "reverb.wav"
    reverb_sound = slab.Sound(reverb_path)

    # The tail is staged into the database by resample_sounds, which only runs
    # on a first build (or with overwrite=True), while this function is called
    # on every build. Changing the render rate without a rebuild therefore
    # leaves the whole sounds/ directory at the previous rate, and the crop and
    # predelay below -- both derived from hrir.samplerate -- would be applied to
    # a tail that does not match. Fail rather than resample here: a stale
    # reverb.wav means the stimuli pyBinSim streams at runtime are stale too,
    # and fixing only the tail would hide that.
    if reverb_sound.samplerate != hrir.samplerate:
        raise ValueError(
            f"reverb.wav is at {reverb_sound.samplerate:g} Hz but the HRIR is at "
            f"{hrir.samplerate:g} Hz ({reverb_path}). The database's sounds/ "
            f"directory was staged for a different render rate -- rebuild with "
            f"overwrite=True so resample_sounds runs again."
        )

    reverb = reverb_sound.data

    # Ensure the reverb has shape [n_samples, 2].
    # If the file is mono, duplicate it to both ears.
    if reverb.ndim == 1:
        reverb = numpy.column_stack([reverb, reverb])

    if reverb.ndim != 2 or reverb.shape[1] != 2:
        raise ValueError(
            f"Expected reverb.wav to have shape [n_samples, 2], got {reverb.shape}."
        )

    # Crop the reverb to the requested tail duration, rounded down to a whole block.
    cropped_len = int(
        (int(hrir.samplerate * tail_duration) // int(block_size)) * int(block_size)
    )
    if cropped_len <= 0:
        raise ValueError("cropped_len is zero. Check tail_duration and block_size.")

    reverb = reverb[:cropped_len]

    # If the file is shorter than the requested length, pad with zeros.
    if reverb.shape[0] < cropped_len:
        pad = numpy.zeros((cropped_len - reverb.shape[0], 2), dtype=reverb.dtype)
        reverb = numpy.vstack([reverb, pad])

    # Apply a short onset ramp to avoid clicks at the beginning of the tail.
    reverb = slab.Sound(reverb).ramp(duration=0.005, when="onset").data

    def estimate_onset(ir: numpy.ndarray, threshold_db: float = 20.0) -> int:
        """
        Estimate the binaural onset sample of one HRIR using an energy threshold.

        Parameters
        ----------
        ir : numpy.ndarray
            HRIR array of shape [n_samples, 2].
        threshold_db : float, default=20.0
            Onset threshold below the peak binaural energy.

        Returns
        -------
        int
            Estimated onset sample index.
        """
        # Sum squared energy across ears to get one temporal energy curve.
        energy = numpy.sum(ir ** 2, axis=1)

        peak = numpy.max(energy)
        if peak <= 0:
            return 0

        threshold = peak * 10 ** (-threshold_db / 10.0)
        above = numpy.where(energy >= threshold)[0]

        return int(above[0]) if len(above) else 0

    # Estimate one onset per source, then average across all HRIRs.
    mean_ir_onset = int(numpy.mean([
        estimate_onset(hrir[idx].data, threshold_db=onset_threshold_db)
        for idx in range(hrir.n_sources)
    ]))

    # Add a small extra delay so the late tail does not start directly on top of
    # the direct HRIR region.
    predelay_samples = int(round(predelay_ms * hrir.samplerate / 1000.0))
    reverb_start = mean_ir_onset + predelay_samples

    # Shift the reverb later in time while keeping its total length unchanged.
    # Anything shifted beyond the end is discarded.
    reverb = numpy.concatenate(
        (numpy.zeros((reverb_start, 2), dtype=reverb.dtype), reverb[:-reverb_start]),
        axis=0
    )

    # Estimate the mean direct HRIR level across sources.
    # Since your HRIRs are short/windowed, this is effectively the direct level.
    mean_ir_level = numpy.mean([
        20.0 * numpy.log10(
            max(numpy.sqrt(numpy.mean(hrir[idx].data ** 2)), 1e-12)
        )
        for idx in range(hrir.n_sources)
    ])

    # Set the reverb level relative to the direct HRIR level.
    reverb_level = 20.0 * numpy.log10(
            max(numpy.sqrt(numpy.mean(reverb ** 2)), 1e-12))

    # Target reverb level from desired DRR:
    # DRR = direct_level - reverb_level
    target_reverb_level = mean_ir_level - drr

    # Convert level difference to linear gain.
    gain_db = target_reverb_level - reverb_level
    gain = 10.0 ** (gain_db / 20.0)

    # Apply gain.
    reverb = reverb * gain

    if show:
        # Pick a representative source (midline fallback if needed)
        try:
            src_idx = hrir.get_source_idx((0, 0))[0]
        except Exception:
            src_idx = 0

        direct = hrir[src_idx].data  # [n_samples, 2]

        # --- length alignment ---
        n = max(direct.shape[0], reverb.shape[0])

        direct_pad = numpy.zeros((n, 2))
        reverb_pad = numpy.zeros((n, 2))

        direct_pad[:direct.shape[0]] = direct
        reverb_pad[:reverb.shape[0]] = reverb

        ir_sum = direct_pad + reverb_pad

        # --- convert to pyfar ---
        sig = pyfar.Signal(
            ir_sum.T,  # pyfar expects [n_channels, n_samples]
            sampling_rate=hrir.samplerate
        )

        # --- plot ---
        pyfar.plot.time_freq(
            sig,
            dB_time=True,
            dB_freq=True,
            freq_scale="log",
        )

    return reverb.astype(numpy.float32)


def compute_hp_ir(hrir, hp, subject_id, block_size=256):
    """
    Load and crop headphone filter, return IR array [nSamples, 2].

    Parameters
    ----------
    hrir : slab.HRTF
        HRIR object (used for samplerate and block_size).
    hp : str
        Headphone model ID (e.g. 'MYSPHERE', 'DT990').
    subject_id : str
        Subject identifier.  The filter is loaded from
        ``rec/{subject_id}/{hp}_equalization.npz`` (wav fallback supported).
    block_size : int
        Output length is rounded down to a multiple of this.
    """
    hp_sig = load_hp_filter(rec_path / subject_id / f"{hp}_equalization.npz", 'pyfar')

    # The saved filter carries the rate it was measured at, which is not
    # necessarily the rate this database is being built for. n_samp_out below
    # is derived from hrir.samplerate, so without this the window length would
    # refer to one rate while the data it crops sits at another -- and the
    # filter would then be convolved against signals at a rate it was not
    # designed for. This is the only rate that does not follow the HRIR
    # automatically.
    if hp_sig.sampling_rate != hrir.samplerate:
        logger.info("Resampling HP filter %s: %g -> %g Hz",
                    hp, hp_sig.sampling_rate, hrir.samplerate)

        # Driving resample_poly directly rather than going through
        # pyfar.dsp.resample, for the same reason resample_hrir does: pyfar
        # warns that a rate not divisible by 10 (48828 is not) can hang
        # scipy's resample_poly, and its frac_limit workaround perturbs the
        # realised rate. The exact ratio avoids both.
        ratio = Fraction(int(hrir.samplerate)
                         / hp_sig.sampling_rate).limit_denominator(10000)
        n_samp_in = hp_sig.n_samples
        hp_sig = pyfar.Signal(
            resample_poly(hp_sig.time, ratio.numerator, ratio.denominator,
                          axis=-1),
            int(hrir.samplerate), fft_norm=hp_sig.fft_norm)

        # Downsampling shortens the filter (1024 -> 1007 going 48828 -> 48000)
        # while n_samp_out below stays a multiple of block_size, so the window
        # would end up longer than the signal it crops. Pad back to the stored
        # length: the filter is boxcar-cropped to n_samp_out anyway and its
        # tail has already decayed, so this only restores the length the
        # windowing step has always assumed.
        if hp_sig.n_samples < n_samp_in:
            hp_sig = pyfar.dsp.pad_zeros(hp_sig, n_samp_in - hp_sig.n_samples)

    n_samp_out = int(
        (int(hrir.samplerate * 0.02) // int(block_size))
        * int(block_size)
    )
    if n_samp_out == 0:
        n_samp_out = block_size

    hp_sig = pyfar.dsp.time_window(
        hp_sig,
        [0, n_samp_out - 1],
        shape="right",
        window='boxcar',
        crop='window'
    )

    return hp_sig.time.astype(numpy.float32).T

# ---- wav writers ------ #

def write_ds_filter_wav(hrir):
    # zero pad and write IR to wav and coordinates to filter_list.txt
    logging.info(f'Writing HRIR filters to wav and filter_list for {hrir.name}')
    scaling_factor = min(1.0, 0.95 / numpy.max([hrir[idx].data for idx in range(hrir.n_sources)]))  # scaling factor
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fname = wav_path / hrir.name / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        fir_coefs = hrir[source_idx].data
        fir_coefs *= scaling_factor
        directional_ir = (slab.Sound(data=fir_coefs, samplerate=hrir.samplerate))
        directional_ir.write(filename=fname)  # write IR to wav
        with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:  # write to filter_list.txt
            file.write(f'DS'
                       f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                       f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                       f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                       f' {coordinates[0]} {coordinates[1]} 0'  # Value 10 - 12: source position[x, y, z]
                       f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                       f' {fname}\n')

def write_lr_filter_wav(hrir, drr=20):
    logging.info(f'Writing reverb filter to wav (DRR = {drr} dB)')
    fname_out = wav_path / hrir.name / 'sounds' / 'scaled_reverb.wav'  # output file name (level adjusted)
    reverb = slab.Sound(wav_path / hrir.name / 'sounds' / 'reverb.wav').data  # load reverb
    # crop to 100 ms and multiple of block size (hrir taps / 2)
    cropped_len = int((int(hrir.samplerate * 0.1) // int(hrir[0].n_taps / 2)) * int(hrir[0].n_taps / 2))
    reverb = reverb[:cropped_len]
    # ramp up reverb tail starting at the max impulse response
    reverb = slab.Sound(reverb).ramp(duration=0.005, when='onset').data  # ramp reverb onset
    mean_ir_onset = int(numpy.mean(  # average onset time of the direct IR
                [numpy.where(hrir[idx].data == (hrir[idx].data).max())[0][0] for idx in range(hrir.n_sources)]))
    reverb = numpy.concatenate((numpy.zeros((mean_ir_onset, 2)), reverb[:-mean_ir_onset]), axis=0)
    # adjust reverb level to DRR
    mean_ir_level = numpy.mean([20.0 * numpy.log10(numpy.maximum(
        numpy.sqrt(numpy.mean(numpy.square(hrir[idx].data))), 1e-12))
                for idx in range(hrir.n_sources)]) # get mean ir level of the impulse response in dB to apply DRR
    reverb = slab.Sound(data=reverb)
    reverb.level = mean_ir_level - drr
    #  write reverb IR and filter list entry
    reverb.write(fname_out, normalise=False)
    with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:
        file.write(f'LR'
                   f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                   f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 10 - 12: source position[x, y, z]
                   f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                   f' {fname_out}\n')
    # plot_reverb(hrir, reverb)

def write_hp_filter_wav(hrir, fname):
    """
    Crop HP Filter to around 5 ms and write filter list entry
    """
    fname_out = wav_path / hrir.name / 'sounds' / 'HP_filter.wav'  # output file name (length adjusted)
    hp_filt = pyfar.io.read_audio( wav_path / hrir.name / 'sounds' / fname)  # load reverb
    # crop close to 5 ms and multiple of block size (hrir taps / 2)
    n_samp_out = int((int(hrir.samplerate * 0.005) // int(hrir[0].n_taps / 2)) * int(hrir[0].n_taps / 2))
    hp_filt = pyfar.dsp.time_window(hp_filt, [0, n_samp_out - 1], shape="right", window='boxcar', crop='window')
    # pyfar.plot.time(hp_filt)
    logging.info(f'Writing headphone filter {fname} to wav ({hp_filt.signal_length*1000:.2f} ms.)')
    pyfar.io.write_audio(hp_filt, str(fname_out))
    with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:  # write filename to filter list
        file.write(f'HP {fname_out}\n')
