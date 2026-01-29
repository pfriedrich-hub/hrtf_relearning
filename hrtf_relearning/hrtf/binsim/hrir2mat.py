import numpy
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

def compute_lr_ir(hrir,drr=20, block_size=256):
    """
    Compute late reverb IR (binaural) without writing WAV
    Returns numpy array [nSamples, 2]
    """
    reverb = slab.Sound(
        wav_path / hrir.name / 'sounds' / 'reverb.wav'
    ).data

    cropped_len = int(
        (int(hrir.samplerate * 0.3) // int(block_size))
        * int(block_size)
    )
    reverb = reverb[:cropped_len]

    reverb = slab.Sound(reverb).ramp(
        duration=0.005, when='onset'
    ).data

    mean_ir_onset = int(numpy.mean([
        numpy.argmax(hrir[idx].data)
        for idx in range(hrir.n_sources)
    ]))

    reverb = numpy.concatenate(
        (numpy.zeros((mean_ir_onset, 2)), reverb[:-mean_ir_onset]),
        axis=0
    )

    mean_ir_level = numpy.mean([
        20 * numpy.log10(
            max(numpy.sqrt(numpy.mean(hrir[idx].data ** 2)), 1e-12)
        )
        for idx in range(hrir.n_sources)
    ])

    reverb = slab.Sound(reverb)
    reverb.level = mean_ir_level - drr

    return reverb.data.astype(numpy.float32)

def compute_hp_ir(hrir, hp, block_size=256):
    """
    Load and crop headphone filter, return IR array [nSamples, 2]
    """
    fname = f"{hp}_equalization.wav"
    hp = pyfar.io.read_audio(
        wav_path / hrir.name / 'sounds' / fname
    )

    n_samp_out = int(
        (int(hrir.samplerate * 0.02) // int(block_size))
        * int(block_size)
    )

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
