import matplotlib
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()

import logging
from pathlib import Path

import slab
import hrtf_relearning

from hrtf_relearning.hrtf.processing.mirror import mirror_hrtf
from hrtf_relearning.hrtf.processing.tf2ir import hrtf2hrir
from hrtf_relearning.hrtf.processing.flatten import flatten_dtf
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf, DEFAULT_N_KEEP
from hrtf_relearning.hrtf.processing.native import native_dtf
from hrtf_relearning.hrtf.binsim.hrir2mat import (
    resample_sounds,
    resample_hrir,
    compute_lr_ir,
    compute_hp_ir,
    write_filters,
    write_filter_list,
    normalization_gain,
    max_safe_gain,
    frontal_index,
    write_level_info,
    REFERENCE_GAIN,
    PEAK_LIMIT,
)
from hrtf_relearning.utils import paths
from hrtf_relearning.utils.local_config import torch_device

logger = logging.getLogger(__name__)

data_dir = paths.HRTF_DIR
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

    # pybinsim requires these sizes to be multiples of blockSize
    if lr_ir is not None:
        late_filter_size = int(lr_ir.shape[0])
        if late_filter_size % block_size != 0:
            raise ValueError(
                f"late_filterSize ({late_filter_size}) must be a multiple of blockSize ({block_size}). "
                f"Got remainder {late_filter_size % block_size}."
            )
    else:
        late_filter_size = 0

    if hp_ir is not None:
        hp_filter_size = int(hp_ir.shape[0])
        if hp_filter_size % block_size != 0:
            raise ValueError(
                f"headphone_filterSize ({hp_filter_size}) must be a multiple of blockSize ({block_size}). "
                f"Got remainder {hp_filter_size % block_size}."
            )
    else:
        hp_filter_size = 0

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

def hrtf2binsim(hrir_settings, overwrite: bool = True, build: bool = True):
    """
    Convert a SOFA HRTF to a pyBinSim-compatible MAT database and write settings.

    DS filters are written only if the database does not exist or overwrite=True.
    LR / HP filters and settings are updated on every call.

    If build=False, only the in-memory HRIR object is reconstructed and returned;
    all disk side effects (sound resampling, LR/HP IR computation, MAT + settings
    writing) are skipped. Use this in spawned worker processes that re-import the
    training module and only need the HRIR object, not a fresh database rebuild.
    """
    sofa_name = hrir_settings.get("name", None)
    # subject_id is the base name before any modifier suffix (e.g. 'JP' from 'JP_notch_left')
    # so that JP, JP_left, JP_notch, JP_notch_left all share the same hp filter
    subject_id = hrir_settings.get("subject_id", sofa_name.split("_")[0])
    ear = hrir_settings.get("ear", None)
    # What happens to the OTHER (non-listening) ear in a monaural condition:
    #   'flat'     -> single delta at the onset (flatten_dtf); ITD + broadband
    #                 ILD kept, all spectral shape gone. Historical default.
    #   'envelope' -> its own coarse cepstral envelope (envelope_dtf), fine
    #                 detail removed; same ITD/ILD, but the ear still sounds
    #                 like an ear, which supports externalization.
    #   'native'   -> its own UNMODIFIED DTF, spliced back in from the native
    #                 SOFA (native_dtf), so only the listening ear carries the
    #                 modification. NOT a monaural condition: the other ear
    #                 keeps a full veridical elevation cue. See
    #                 hrtf.processing.native for what that costs.
    # Ignored when ear is None (binaural).
    other_ear = hrir_settings.get("other_ear", "flat")
    env_n_keep = int(hrir_settings.get("env_n_keep", DEFAULT_N_KEEP))
    # source of the untouched channel for other_ear='native'
    native_sofa = hrir_settings.get("native_sofa", subject_id)
    mirror = hrir_settings.get("mirror", False)
    reverb = hrir_settings.get("reverb", True)
    drr = hrir_settings.get("drr", 20)
    hp_filter = hrir_settings.get("hp_filter", True)
    hp = hrir_settings.get("hp", 'DT990')
    # Rate the database is rendered at. None keeps the rate the HRTF was
    # recorded at, which is what the TDT rig produces (48828) and what every
    # build did before this option existed. Set it to the playback device's
    # native rate (48000 for a USB interface) so the conversion happens here,
    # once and on record, instead of silently in the Windows mixer at runtime.
    # The recordings on disk are never touched -- this applies to the loaded
    # copy only.
    target_samplerate = hrir_settings.get("target_samplerate", None)
    # cpu/cuda is a property of the machine, not of the experiment. A
    # gitignored local_config.json ("torch_device") or HRTF_TORCH_DEVICE
    # overrides whatever the protocol script hardcoded; without either, the
    # requested value is used but 'cuda' still degrades to 'cpu' where CUDA is
    # unavailable. See hrtf_relearning.utils.local_config.
    convolution = torch_device(hrir_settings.get("convolution"))
    storage = torch_device(hrir_settings.get("storage"))

    logger.info(
        "hrtf2binsim | HRTF=%s ear=%s other_ear=%s drr=%.1f hp_file=%s",
        sofa_name, ear or "binaural", other_ear if ear else "-", drr, hp,
    )

    hrir = slab.HRTF(data_dir / "sofa" / subject_id / f"{sofa_name}.sofa")
    hrir.name = sofa_name
    slab.set_default_samplerate(hrir.samplerate)

    block_size = int(hrir[0].n_taps * 2)  # *2 prevents glitches

    if hrir.datatype != "FIR":  # pyBinSim only supports FIR filters, so convert if necessary
        logger.info("Converting HRTF → HRIR (FIR)")
        hrir = hrtf2hrir(hrir)

    if ear:   # monaural: reduce the other ear's DTF
        # NB the name suffix must differ per mode — it is the binsim database
        # folder AND the run label, so 'flat' and 'envelope' versions of the
        # same SOFA must never share a directory or a filter list.
        other = "right" if ear == "left" else "left"
        if other_ear == "flat":
            logger.info("Flattening DTF for %s ear", other)
            hrir = flatten_dtf(hrir, ear)
            hrir.name += f"_{ear}"
        elif other_ear == "envelope":
            # elevation_average defaults to False HERE, unlike envelope_dtf
            # itself. This is the render-time path, which operates on an
            # already-expanded 475-direction set; it is what every build before
            # 2026-08 used and it is pinned so those subjects stay reproducible.
            # The current pipeline applies the monaural reduction to the az=0
            # arc at SOFA build time instead (hrtf.processing.midline), where
            # averaging over elevation removes the cue outright, and reaches
            # this branch with ear=None.
            elevation_average = bool(hrir_settings.get("env_elevation_average", False))
            logger.info("Envelope-only DTF (n_keep=%d, elevation_average=%s) for %s ear",
                        env_n_keep, elevation_average, other)
            hrir = envelope_dtf(hrir, ear, n_keep=env_n_keep,
                                elevation_average=elevation_average)
            hrir.name += f"_{ear}_env{env_n_keep}"
        elif other_ear == "native":
            if native_sofa == sofa_name:
                logger.warning(
                    "other_ear='native' with native_sofa == the loaded SOFA (%s): "
                    "nothing to restore, this is a plain binaural condition", sofa_name)
            else:
                logger.info("Restoring native DTF from %s.sofa for %s ear",
                            native_sofa, other)
                native = slab.HRTF(data_dir / "sofa" / subject_id / f"{native_sofa}.sofa")
                if native.datatype != "FIR":
                    native = hrtf2hrir(native)
                hrir = native_dtf(hrir, native, ear)
            hrir.name += f"_{ear}_nat"
        else:
            raise ValueError(
                f"other_ear must be 'flat', 'envelope' or 'native', "
                f"got {other_ear!r}")

    if mirror:  # mirror left and right by swapping channels and sources (swap spectral cues)
        logger.info("Mirroring HRIR left ↔ right")
        hrir = mirror_hrtf(hrir)
        hrir.name += "_mirrored"

    # Resample last: the DTF manipulations above splice this HRIR against other
    # recordings (native_dtf loads a second SOFA), so they have to run while
    # everything is still at the rate it was measured at. Everything below
    # derives from hrir.samplerate -- the resampled stimuli, the LR/HP filter
    # lengths, the predelay, and the samplingRate written into the settings --
    # so it has to happen before any of them. Placed ahead of the build=False
    # return so spawned workers see the same object the database was built from.
    if target_samplerate:
        hrir = resample_hrir(hrir, target_samplerate)

    base_dir = data_dir / "binsim" / hrir.name
    mat_path = base_dir / f"{hrir.name}_filters.mat"

    if not build:
        # Cheap path: only the in-memory HRIR object is needed (e.g. in spawned
        # worker processes). Skip all disk side effects.
        logger.info("hrtf2binsim | build=False, returning HRIR object only")
        return hrir

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

    if reverb:
        lr_ir = compute_lr_ir(hrir, drr=drr, block_size=block_size)
    else:
        lr_ir = None
    if hp_filter:
        hp_ir = compute_hp_ir(hrir, hp=hp, subject_id=subject_id, block_size=block_size)
    else:
        hp_ir = None

    # One scalar for the whole set, so the rendered level is the same for every
    # subject at a given loc_settings['gain'] and pyBinSim keeps its headroom.
    # All amplitude ratios (ILD, direction-to-direction, DRR) are preserved.
    norm_gain, levels = normalization_gain(hrir, hp_ir)

    write_filters(hrir, lr_ir, hp_ir, mat_path, norm_gain=norm_gain)
    write_filter_list(hrir)

    # Headroom QC. The output is a max safe gain per stimulus rather than a peak
    # at one assumed gain: the chain is linear in the runtime loudness, so a
    # limit cannot go stale the way a quoted peak does when a protocol changes
    # its level. Scripts vet their own gain against it via hrir2mat.check_gain.
    #
    # Direction sets differ by how the sound is triggered. The stimuli sweep
    # EVERY direction, because the training game updates the DS filter from
    # target-minus-pose while the listener turns, so relative azimuth spans the
    # whole grid -- and the peak runs 1.6x above the median direction here,
    # since normalization_gain targets the median. The SFX play only once the
    # listener is on the target, i.e. at relative (0, 0), so bounding them over
    # all directions would report a limit no run can hit.
    #
    # localization.wav is rewritten per trial by the loc test; the copy on disk
    # is a representative realisation, peak-normalised like every other wav.
    sweep_all = ("noise_pulse.wav", "localization.wav")
    frontal_only = ("coin.wav", "coins.wav", "hi_score.wav", "buzzer.wav",
                    "beep.wav", "c_chord_guitar.wav", "reverb.wav")
    frontal = frontal_index(hrir)

    headroom = {}
    for fname in sweep_all + frontal_only:
        stim_path = base_dir / "sounds" / fname
        if not stim_path.exists():
            continue
        directions = None if fname in sweep_all else [frontal]
        limit, idx, unit_peak = max_safe_gain(
            hrir, lr_ir, hp_ir, slab.Sound.read(stim_path).data,
            norm_gain=norm_gain, directions=directions,
        )
        az, ele = hrir.sources.vertical_polar[idx, :2]
        headroom[fname] = {
            "max_safe_gain": float(limit),
            "unit_gain_peak": float(unit_peak),
            "direction": [float(az), float(ele)],
            "swept": directions is None,
        }
        logger.debug("Headroom | %-18s max safe gain %.3f (az %.1f el %.1f)",
                     fname, limit, az, ele)

    if headroom:
        worst = min(headroom, key=lambda k: headroom[k]["max_safe_gain"])
        limit = headroom[worst]["max_safe_gain"]
        az, ele = headroom[worst]["direction"]
        if limit < REFERENCE_GAIN:
            logger.warning(
                "Headroom | max safe gain %.3f, set by %s at az %.1f el %.1f. "
                "Protocols running above this WILL clip (loudest in use: %.2f). "
                "Every script should call hrir2mat.check_gain with its own gain.",
                limit, worst, az, ele, REFERENCE_GAIN,
            )
        else:
            logger.info("Headroom | max safe gain %.3f (set by %s at az %.1f "
                        "el %.1f), clear at every gain in use.",
                        limit, worst, az, ele)

    write_level_info(hrir, mat_path, norm_gain, levels, headroom=headroom)

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
