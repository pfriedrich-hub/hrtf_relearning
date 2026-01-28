import matplotlib
matplotlib.use("tkagg")

import logging
from pathlib import Path
from typing import Optional

import slab
import hrtf_relearning

from hrtf_relearning.hrtf.binsim.tf2ir import hrtf2hrir
from hrtf_relearning.hrtf.binsim.flatten import flatten_dtf
from hrtf_relearning.hrtf.binsim.hrir2wav import (
    resample_sounds,
    compute_lr_ir,
    compute_hp_ir,
    write_filters,
    write_filter_list,
)

logger = logging.getLogger(__name__)

data_dir = hrtf_relearning.PATH / "data" / "hrtf"
wav_path = data_dir / "binsim"

# ---------------------------------------------------------------------
# Settings writer (UNCHANGED semantics, only logs + docs)
# ---------------------------------------------------------------------

def write_settings(
    hrir,
    lr_ir,
    hp_ir,
    mat_path: Path,
    *,
    reverb: bool = True,
    hp_filter: bool = True,
    convolution: str = "cpu",
    storage: str = "cpu",
    block_size = 256
):
    """
    Write pyBinSim settings files for MAT-based filters.

    Parameters
    ----------
    hrir : slab.HRTF
        HRIR object (for samplerate and taps).
    lr_ir : numpy.ndarray
        Late reverb IR array written into the MAT database (shape: [n_samples, 2]).
    hp_ir : numpy.ndarray
        Headphone IR array written into the MAT database (shape: [n_samples, 2]).
    mat_path : Path
        Path to MAT database written by write_filters().
    reverb, hp_filter, convolution, storage
        Runtime configuration flags written into the settings files.
    """
    base_path = wav_path / hrir.name
    # block_size = int(hrir[0].n_taps / 2)

    ds_filter_size = hrir[0].n_samples
    late_filter_size = int(lr_ir.shape[0])
    hp_filter_size = int(hp_ir.shape[0])

    # pybinsim requires these sizes to be multiples of blockSize
    if late_filter_size % block_size != 0:
        raise ValueError(
            f"late_filterSize ({late_filter_size}) must be a multiple of blockSize ({block_size}). "
            f"Got remainder {late_filter_size % block_size}."
        )
    if hp_filter_size % block_size != 0:
        raise ValueError(
            f"headphone_filterSize ({hp_filter_size}) must be a multiple of blockSize ({block_size}). "
            f"Got remainder {hp_filter_size % block_size}."
        )

    logger.info(
        "Writing settings | HRTF=%s reverb=%s hp_filter=%s conv=%s storage=%s",
        hrir.name, reverb, hp_filter, convolution, storage
    )
    logger.debug(
        "Sizes | block=%d DS=%d LR=%d HP=%d | fs=%d | mat=%s",
        block_size, ds_filter_size, late_filter_size, hp_filter_size, int(hrir.samplerate), mat_path
    )

    # ---------- TRAINING ----------
    train_fname = base_path / f"{hrir.name}_training_settings.txt"
    with open(train_fname, "w") as f:
        f.write(
            f"soundfile {base_path / 'sounds' / 'noise_pulse.wav'}\n"
            f"blockSize {block_size}\n"
            f"ds_filterSize {ds_filter_size}\n"
            f"early_filterSize {ds_filter_size}\n"
            f"late_filterSize {late_filter_size}\n"
            f"headphone_filterSize {hp_filter_size}\n"
            f"filterSource[mat/wav] mat\n"
            f"filterList {base_path / f'filter_list_{hrir.name}.txt'}\n"
            f"filterDatabase {mat_path}\n"
            f"maxChannels 1\n"
            f"samplingRate {int(hrir.samplerate)}\n"
            f"enableCrossfading True\n"
            f"loudnessFactor 0\n"
            f"loopSound False\n"
            f"torchConvolution[cpu/cuda] {convolution}\n"
            f"torchStorage[cpu/cuda] {storage}\n"
            f"pauseConvolution False\n"
            f"pauseAudioPlayback False\n"
            f"useHeadphoneFilter {hp_filter}\n"
            f"ds_convolverActive True\n"
            f"early_convolverActive False\n"
            f"late_convolverActive {reverb}\n"
            f"recv_type osc\n"
            f"recv_protocol udp\n"
            f"recv_ip 127.0.0.1\n"
            f"recv_port 10000\n"
        )

    # ---------- TEST ----------
    test_fname = base_path / f"{hrir.name}_test_settings.txt"
    with open(test_fname, "w") as f:
        f.write(
            f"soundfile {base_path / 'sounds' / 'localization.wav'}\n"
            f"blockSize {block_size}\n"
            f"ds_filterSize {ds_filter_size}\n"
            f"early_filterSize {ds_filter_size}\n"
            f"late_filterSize {late_filter_size}\n"
            f"headphone_filterSize {hp_filter_size}\n"
            f"filterSource[mat/wav] mat\n"
            f"filterList {base_path / f'filter_list_{hrir.name}.txt'}\n"
            f"filterDatabase {mat_path}\n"
            f"maxChannels 1\n"
            f"samplingRate {int(hrir.samplerate)}\n"
            f"enableCrossfading True\n"
            f"loudnessFactor 0\n"
            f"loopSound False\n"
            f"torchConvolution[cpu/cuda] {convolution}\n"
            f"torchStorage[cpu/cuda] {storage}\n"
            f"pauseConvolution False\n"
            f"pauseAudioPlayback False\n"
            f"useHeadphoneFilter {hp_filter}\n"
            f"ds_convolverActive True\n"
            f"early_convolverActive False\n"
            f"late_convolverActive {reverb}\n"
            f"recv_type osc\n"
            f"recv_protocol udp\n"
            f"recv_ip 127.0.0.1\n"
            f"recv_port 10000\n"
        )

    logger.info("Settings updated: %s / %s", train_fname.name, test_fname.name)



# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def hrtf2binsim(hrir_settings, overwrite: bool = True):
    """
    Convert a SOFA HRTF to a pyBinSim-compatible MAT database and write settings.

    DS filters are written only if the database does not exist or overwrite=True.
    LR / HP filters and settings are updated on every call.
    """
    sofa_name = hrir_settings["name"]
    ear = hrir_settings["ear"]
    reverb = hrir_settings["reverb"]
    drr = hrir_settings["drr"]
    hp_filter = hrir_settings["hp_filter"]
    hp = hrir_settings["hp"]
    convolution = hrir_settings["convolution"]
    storage = hrir_settings["storage"]

    logger.info(
        "hrtf2binsim | HRTF=%s ear=%s drr=%.1f hp_file=%s",
        sofa_name, ear or "binaural", drr, hp,
    )

    hrir = slab.HRTF(data_dir / "sofa" / f"{sofa_name}.sofa")
    hrir.name = sofa_name
    slab.set_default_samplerate(hrir.samplerate)

    block_size = int(hrir[0].n_taps * 2)  # *2 prevents glitches

    if hrir.datatype != "FIR":
        logger.info("Converting HRTF → HRIR (FIR)")
        hrir = hrtf2hrir(hrir)

    if ear:
        logger.info("Flattening DTF at ear: %s", ear)
        hrir = flatten_dtf(hrir, ear)
        hrir.name += f"_{ear}"

    base_dir = data_dir / "binsim" / hrir.name
    mat_path = base_dir / f"{hrir.name}_filters.mat"

    first_build = (not base_dir.exists()) or overwrite

    if first_build:
        logger.info("Resampling sound files (overwrite=%s)", overwrite)

        (base_dir / "sounds").mkdir(exist_ok=True, parents=True)
        (base_dir / "plot").mkdir(exist_ok=True)

        resample_sounds(
            target_samplerate=hrir.samplerate,
            target_directory=base_dir / "sounds",
        )

    # ALWAYS recompute LR + HP
    logger.info("Writing DS / LR / HP filters | DRR=%.1f HP=%s", drr, hp)

    lr_ir = compute_lr_ir(hrir, drr=drr, block_size=block_size)
    hp_ir = compute_hp_ir(hrir, hp=hp, block_size=block_size)

    write_filters(hrir, lr_ir, hp_ir, mat_path)
    write_filter_list(hrir)

    write_settings(
        hrir,
        lr_ir,
        hp_ir,
        mat_path,
        reverb=reverb,
        hp_filter=hp_filter,
        convolution=convolution,
        storage=storage,
        block_size=block_size,
    )

    return hrir
