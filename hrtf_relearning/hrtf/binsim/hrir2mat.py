import numpy
from matplotlib import pyplot as plt
from scipy.io import savemat
import slab
import logging
import pyfar
from pathlib import Path
import hrtf_relearning
ROOT = Path(hrtf_relearning.__file__).resolve().parent
wav_path = ROOT / 'data' / 'hrtf' / 'binsim'
sofa_path = ROOT / 'data' / 'hrtf' / 'sofa'
sound_path = ROOT / 'data' / 'sounds'
rec_path = ROOT / 'data' / 'hrtf' / 'rec'

def resample_sounds(target_samplerate, target_directory):
    logging.info('Resampling sound files.')
    for file in sound_path.glob('*.wav'): # resample sound files
        sound = slab.Sound.read(file)
        if not sound.samplerate == target_samplerate:
            sound = sound.resample(target_samplerate)
        sound.write(target_directory / file.name, normalise=True)

# ---- mat writers ---- #

def write_filters(hrir, lr_ir, hp_ir, mat_path):
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
    """

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
            hrir[src_idx].data.astype(numpy.float32),         # ir
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
            lr_ir.astype(numpy.float32),
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
    reverb = slab.Sound(
        wav_path / hrir.name / "sounds" / "reverb.wav"
    ).data

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


def compute_hp_ir(hrir, hp, block_size=256):
    """
    Load and crop headphone filter, return IR array [nSamples, 2]
    """
    fname = f"{hp}_equalization.wav"
    hp = pyfar.io.read_audio(
        rec_path / hrir.name[:2] / fname  # todo
    )

    n_samp_out = int(
        (int(hrir.samplerate * 0.02) // int(block_size))
        * int(block_size)
    )
    if n_samp_out == 0:
        n_samp_out = block_size  # todo

    hp = pyfar.dsp.time_window(
        hp,
        [0, n_samp_out - 1],
        shape="right",
        window='boxcar',
        crop='window'
    )

    return hp.time.astype(numpy.float32).T

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
