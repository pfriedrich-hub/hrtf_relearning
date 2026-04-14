"""
record_hrir_mesm.py — Top-level HRIR acquisition pipeline for the MESM system.

Mirrors the structure of hrtf/record/record_hrir.py but uses the
simultaneous multi-speaker exponential sweep method.

Future extensions (rotating platform)
--------------------------------------
When the rotating platform is added, this function will be called once per
platform position. The outer loop (positions) and the inner call
(record_hrir_mesm) stay cleanly separated.

Pipeline
--------
1. Load or run reference measurement  →  ReferenceParams
2. Compute MESM timing parameters     →  MESMParams
3. Record (or load) MESM measurement  →  MESMRecording
4. Deconvolve + window                →  ImpulseResponses (per speaker)
5. Equalize against reference IRs
6. Low-frequency extrapolation
7. (Optional) Azimuth expansion
8. Export to slab.HRTF / SOFA
"""
from __future__ import annotations

import logging
from pathlib import Path

import slab
import freefield
import hrtf_relearning

from .sweep import compute_mesm_params, MESMParams
from .recordings import (
    MESMRecording,
    ReferenceParams,
    initialize,
    record_reference,
    record_mesm,
)
from .processing import compute_ir_mesm

# Reuse post-processing from the existing pipeline
from hrtf_relearning.hrtf.record.processing import (
    equalize,
    lowfreq_extrapolate,
    expand_azimuths_with_binaural_cues,
)

base_dir = hrtf_relearning.PATH / "data" / "hrtf"

# ---------------------------------------------------------------------------
# Default parameters — adjust before each session
# ---------------------------------------------------------------------------

subject_id   = "subject_XX"
reference_id = "ref_XX"
fs           = 96000          # 96 kHz preferred; drop to 48000 if DSP limited
n_speakers   = 7
f1           = 20.0           # Hz — full range with balanced XLR system
f2           = 20_000.0       # Hz
T_prime      = 1.0            # sweep duration in seconds
n_repetitions = 1             # average over N repetitions for SNR improvement
head_radius  = 0.0875         # metres
n_samples_out = 512           # final HRIR length after processing
hp_freq      = 20.0           # Hz — only needed if low-freq noise persists
overwrite    = False
expand_az    = True
show         = True

freefield.set_logger("info")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def record_hrir_mesm(
    subject_id: str,
    reference_id: str,
    *,
    n_speakers: int = 7,
    fs: int = 96000,
    f1: float = 20.0,
    f2: float = 20_000.0,
    T_prime: float = 1.0,
    n_repetitions: int = 1,
    position: float | None = None,
    distance: float = 1.4,
    head_radius: float = 0.0875,
    n_samples_out: int = 512,
    hp_freq: float = 20.0,
    expand_az: bool = True,
    overwrite: bool = False,
    show: bool = True,
    base_dir: Path | str | None = None,
) -> slab.HRTF:
    """
    Full MESM HRIR acquisition + processing pipeline for one subject /
    one platform position.

    Parameters
    ----------
    subject_id : str
        Identifier written into filenames and metadata.
    reference_id : str
        Identifier for the reference measurement directory.
    n_speakers : int
        Number of simultaneously driven loudspeakers.
    fs : int
        Sample rate in Hz.
    f1, f2 : float
        Exponential sweep frequency bounds in Hz.
    T_prime : float
        Sweep duration in seconds (longer → higher SNR, longer measurement).
    n_repetitions : int
        Number of MESM repetitions to average.
    position : float, optional
        Turntable angle in degrees at the time of recording. Stored in the
        MESMRecording for bookkeeping. When the rotating platform is in use
        the outer loop passes this value for each platform step.
    distance : float
        Speaker-to-microphone distance in metres. Used for delay calculation.
    head_radius : float
        Used for spherical-head low-frequency extrapolation.
    n_samples_out : int
        Final HRIR filter length in samples.
    hp_freq : float
        High-pass cutoff for equalization in Hz.
    expand_az : bool
        Whether to expand measured azimuths symmetrically.
    overwrite : bool
        Re-record even if data already exists on disk.
    show : bool
        Plot the resulting HRTF.
    base_dir : Path-like, optional
        Root directory for data storage.

    Returns
    -------
    slab.HRTF
    """
    logging.info(f"Starting MESM HRIR pipeline — subject '{subject_id}'")

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------
    if base_dir is None:
        base_dir = hrtf_relearning.PATH / "data" / "hrtf"
    else:
        base_dir = Path(base_dir)

    subj_dir = base_dir / "rec_mesm" / subject_id
    ref_dir  = base_dir / "rec_mesm" / "reference" / reference_id

    # -----------------------------------------------------------------
    # 1) Reference measurement → timing parameters
    # -----------------------------------------------------------------
    ref_params = _load_or_record_reference(
        ref_dir=ref_dir,
        n_speakers=n_speakers,
        fs=fs,
        f1=f1,
        f2=f2,
        T_prime=T_prime,
        overwrite=overwrite,
    )
    logging.info(f"Reference: L1={ref_params.L1*1e3:.1f} ms, "
                 f"L2={ref_params.L2*1e3:.1f} ms, K={ref_params.K}")

    # -----------------------------------------------------------------
    # 2) Compute MESM timing parameters
    # -----------------------------------------------------------------
    params = compute_mesm_params(
        n_speakers=n_speakers,
        fs=fs,
        L1=ref_params.L1,
        K=ref_params.K,
        T_prime=T_prime,
        f1=f1,
        f2=f2,
        L2=ref_params.L2,
    )
    logging.info(params.summary())

    # -----------------------------------------------------------------
    # 3) Subject MESM recording
    # -----------------------------------------------------------------
    npz_file = subj_dir / "mesm_recording.npz"
    if overwrite or not npz_file.exists():
        logging.info("Recording subject ear pressure (MESM) ...")
        subj_dir.mkdir(parents=True, exist_ok=True)
        subject_rec = record_mesm(
            params=params,
            position=position,
            n_repetitions=n_repetitions,
            distance=distance,
            subject_id=subject_id,
        )
        subject_rec.to_npz(subj_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject MESM recording from disk ...")
        subject_rec = MESMRecording.from_npz(subj_dir)


    # -----------------------------------------------------------------
    # 4) Reference MESM recording (for equalization)
    # -----------------------------------------------------------------
    ref_npz = ref_dir / "mesm_recording.npz"
    if ref_npz.exists():
        logging.info("Loading reference MESM recording ...")
        reference_rec = MESMRecording.from_npz(ref_dir)
    else:
        logging.warning(
            "No reference MESM recording found — equalization will be skipped. "
            "Run a reference measurement (microphone at ear-canal entrance, "
            "no subject) and save to %s", ref_dir
        )
        reference_rec = None

    # -----------------------------------------------------------------
    # 5) Deconvolution
    # -----------------------------------------------------------------
    logging.info("Deconvolving subject recording ...")
    subject_ir = compute_ir_mesm(subject_rec)

    if reference_rec is not None:
        logging.info("Deconvolving reference recording ...")
        reference_ir = compute_ir_mesm(reference_rec)
    else:
        reference_ir = None

    # -----------------------------------------------------------------
    # 6) Equalization
    # -----------------------------------------------------------------
    if reference_ir is not None:
        logging.info("Applying equalization ...")
        hrir_equalized = equalize(
            measured=subject_ir,
            reference=reference_ir,
            n_samples_out=n_samples_out,
            inversion_range_hz=(hp_freq, 18e3),
            onset_threshold_db=10,
        )
    else:
        hrir_equalized = subject_ir

    # -----------------------------------------------------------------
    # 7) Low-frequency extrapolation
    # -----------------------------------------------------------------
    logging.info("Low-frequency extrapolation ...")
    hrir_extrapol = lowfreq_extrapolate(
        hrir_equalized,
        f_extrap=800.0,
        f_target=150.0,
        head_radius=head_radius,
    )

    # -----------------------------------------------------------------
    # 8) Azimuth expansion
    # -----------------------------------------------------------------
    if expand_az:
        logging.info("Expanding azimuths ...")
        hrir_final = expand_azimuths_with_binaural_cues(
            hrir_extrapol,
            az_range=(-50, 50),
            head_radius=head_radius,
            show=False,
        )
    else:
        hrir_final = hrir_extrapol

    # -----------------------------------------------------------------
    # 9) Export
    # -----------------------------------------------------------------
    hrtf = hrir_final.to_slab_hrtf(datatype="FIR")
    out_file = base_dir / "sofa" / f"{subject_id}_mesm.sofa"
    if overwrite or not out_file.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing HRTF to {out_file}")
        hrtf.write_sofa(str(out_file))

    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        hrtf.plot_tf(hrtf.cone_sources(0), axis=axes, ear="both")
        plt.show()

    logging.info("MESM HRIR pipeline finished.")
    return hrtf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_record_reference(
    ref_dir: Path,
    n_speakers: int,
    fs: int,
    f1: float,
    f2: float,
    T_prime: float,
    overwrite: bool,
) -> ReferenceParams:
    """Load cached ReferenceParams or run a fresh reference measurement."""
    cache = ref_dir / "reference_params.npz"
    if not overwrite and cache.exists():
        logging.info(f"Loading reference params from {cache}")
        npz = __import__("numpy").load(cache)
        return ReferenceParams(
            L1=float(npz["L1"]),
            L2=float(npz["L2"]),
            K=int(npz["K"]),
            fs=int(npz["fs"]),
            n_speakers=int(npz["n_speakers"]),
        )

    logging.info("Running reference measurement ...")
    ref_dir.mkdir(parents=True, exist_ok=True)
    params = record_reference(
        n_speakers=n_speakers,
        fs=fs,
        f1=f1,
        f2=f2,
        T=T_prime,
    )
    # Cache to disk
    __import__("numpy").savez(
        cache,
        L1=params.L1,
        L2=params.L2,
        K=params.K,
        fs=params.fs,
        n_speakers=params.n_speakers,
    )
    return params


