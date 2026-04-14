"""
processing.py — DSP layer for the MESM HRIR pipeline.

Responsibilities
----------------
- Deconvolve the raw binaural MESM recording into a HIR series.
- Window out each speaker's linear HRIR from the HIR series.
- Delegate equalization, low-freq extrapolation, and azimuth expansion
  to the existing hrtf.record.processing module (no duplication).

No hardware. No I/O.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyfar

from .recordings import MESMRecording
from .sweep import inverse_sweep

# Reuse post-processing steps from the existing pipeline
from hrtf_relearning.hrtf.record.processing import (
    ImpulseResponses,
    equalize,
    lowfreq_extrapolate,
    expand_azimuths_with_binaural_cues,
)


# ---------------------------------------------------------------------------
# Step 1 — Deconvolution
# ---------------------------------------------------------------------------

def deconvolve(recording: MESMRecording) -> tuple[np.ndarray, np.ndarray]:
    """
    Convolve the binaural recording with the inverse sweep to obtain the
    full HIR series for both ears.

        s(t) = y(t) * x'(t)                                    (Eq. 2)

    The result is a long signal in which each speaker's linear HRIR appears
    at time offset onset_samples[i] and higher-order HIRs fill the gaps.

    Parameters
    ----------
    recording : MESMRecording

    Returns
    -------
    s_l, s_r : np.ndarray
        Full deconvolved HIR series for left and right ear.
        Length ≈ T_total_samples + T_prime_samples − 1 (linear convolution).
    """
    p = recording.params
    inv = inverse_sweep(
        sweep=_reconstruct_sweep(p),
        f1=p.f1,
        f2=p.f2,
    )

    # Use pyfar for regularized convolution (consistent with existing pipeline)
    inv_sig = pyfar.Signal(inv, p.fs)

    s_l = _convolve_1d(recording.left,  inv, p.fs)
    s_r = _convolve_1d(recording.right, inv, p.fs)

    return s_l, s_r


def _reconstruct_sweep(params) -> np.ndarray:
    from .sweep import exponential_sweep
    return exponential_sweep(
        T_samples=params.T_prime_samples,
        fs=params.fs,
        f1=params.f1,
        f2=params.f2,
    )


def _convolve_1d(signal: np.ndarray, kernel: np.ndarray, fs: int) -> np.ndarray:
    """
    Deconvolve via frequency-domain multiplication (linear convolution).
    Uses pyfar regularized spectrum inversion for consistency with existing code.
    """
    sig_pyfar = pyfar.Signal(signal, fs)
    ker_pyfar = pyfar.Signal(kernel, fs)
    # Direct multiplication in frequency domain (kernel is already the inverse)
    result = pyfar.dsp.convolve(sig_pyfar, ker_pyfar, mode="linear")
    return result.time.squeeze()


# ---------------------------------------------------------------------------
# Step 2 — Windowing: extract per-speaker HRIRs
# ---------------------------------------------------------------------------

def extract_hrirs(
    s_l: np.ndarray,
    s_r: np.ndarray,
    recording: MESMRecording,
    window_length_samples: int | None = None,
    fade_samples: int = 32,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Window out each speaker's linear HRIR from the deconvolved HIR series.

    Each speaker i's linear HRIR is centred at sample onset_samples[i] in
    the deconvolved output. A rectangular window of length L1_samples is
    applied, with optional Tukey (raised-cosine) fade edges.

    Parameters
    ----------
    s_l, s_r : np.ndarray
        Full deconvolved HIR series (output of deconvolve()).
    recording : MESMRecording
    window_length_samples : int, optional
        Length of the extraction window in samples.
        Defaults to L1_samples (from params).
    fade_samples : int
        Number of samples for the raised-cosine fade-in/out. 0 = rectangular.

    Returns
    -------
    hrirs : dict mapping speaker index → (ir_l, ir_r)
        ir_l, ir_r : np.ndarray, shape (window_length_samples,)
    """
    p = recording.params
    L1_samples = int(round(p.L1 * p.fs))
    win_len = window_length_samples or L1_samples

    hrirs = {}
    for i, onset in enumerate(p.onset_samples):
        center = onset   # linear IR starts at onset in the deconvolved signal
        start  = center
        stop   = start + win_len

        # Bounds check
        if stop > len(s_l):
            logging.warning(
                f"Speaker {i}: window [{start}:{stop}] exceeds deconvolved "
                f"signal length {len(s_l)}. Truncating."
            )
            stop = len(s_l)

        ir_l = s_l[start:stop].copy()
        ir_r = s_r[start:stop].copy()

        # Pad to win_len if truncated
        if len(ir_l) < win_len:
            ir_l = np.pad(ir_l, (0, win_len - len(ir_l)))
            ir_r = np.pad(ir_r, (0, win_len - len(ir_r)))

        # Fade edges
        if fade_samples > 0:
            win = _tukey_fade(win_len, fade_samples)
            ir_l *= win
            ir_r *= win

        hrirs[i] = (ir_l, ir_r)

    return hrirs


def _tukey_fade(n: int, fade: int) -> np.ndarray:
    """Rectangular window with half-Hann fade-in and fade-out."""
    win = np.ones(n)
    ramp = np.hanning(2 * fade)
    win[:fade]  = ramp[:fade]
    win[-fade:] = ramp[fade:]
    return win


# ---------------------------------------------------------------------------
# Step 3 — Wrap into ImpulseResponses (compatible with existing pipeline)
# ---------------------------------------------------------------------------

def hrirs_to_impulse_responses(
    hrirs: dict[int, tuple[np.ndarray, np.ndarray]],
    recording: MESMRecording,
) -> ImpulseResponses:
    """
    Pack the windowed per-speaker IRs into an ImpulseResponses object,
    using the same key format as the existing pipeline ('idx_az_el').

    The speaker_table in recording maps speaker index → (azimuth, elevation).
    If speaker_table is empty, keys default to '{i}_0.0_0.0'.

    Parameters
    ----------
    hrirs : dict
        Output of extract_hrirs().
    recording : MESMRecording

    Returns
    -------
    ImpulseResponses

    TODO
    ----
    - Populate recording.speaker_table from the new speaker table file before
      calling this function so that correct az/el appear in the SOFA output.
    """
    p = recording.params
    ir_dict = {}

    for i, (ir_l, ir_r) in hrirs.items():
        spk = recording.speaker_table.get(i, {})
        az  = spk.get("azimuth",   0.0)
        el  = spk.get("elevation", 0.0)
        key = f"{i}_{az:.2f}_{el:.2f}"

        # Stack as pyfar Signal: shape (2, n_samples) → left, right
        data = np.stack([ir_l, ir_r], axis=0)
        ir_dict[key] = pyfar.Signal(data, p.fs)

    params = {
        "fs"          : p.fs,
        "n_speakers"  : p.n_speakers,
        "method"      : "MESM",
        "datetime"    : recording.meta.get("datetime", ""),
        "subject_id"  : recording.meta.get("subject_id", ""),
        "signal"      : {
            "kind"          : "logarithmic",
            "from_frequency": p.f1,
            "to_frequency"  : p.f2,
            "duration"      : p.T_prime,
            "samplerate"    : p.fs,
        },
    }
    return ImpulseResponses(data=ir_dict, params=params)


# ---------------------------------------------------------------------------
# Convenience: full deconvolution pipeline in one call
# ---------------------------------------------------------------------------

def compute_ir_mesm(
    recording: MESMRecording,
    window_length_samples: int | None = None,
    fade_samples: int = 32,
) -> ImpulseResponses:
    """
    Full MESM deconvolution pipeline:
        1. Deconvolve recording with inverse sweep.
        2. Window out per-speaker HRIRs.
        3. Return ImpulseResponses compatible with the existing pipeline.

    Parameters
    ----------
    recording : MESMRecording
    window_length_samples : int, optional
        Window length for IR extraction. Defaults to L1_samples.
    fade_samples : int
        Tukey-fade edge length in samples.

    Returns
    -------
    ImpulseResponses
    """
    logging.info("MESM deconvolution: computing HIR series ...")
    s_l, s_r = deconvolve(recording)

    logging.info("MESM deconvolution: windowing per-speaker HRIRs ...")
    hrirs = extract_hrirs(
        s_l, s_r, recording,
        window_length_samples=window_length_samples,
        fade_samples=fade_samples,
    )

    logging.info("MESM deconvolution: packing into ImpulseResponses ...")
    return hrirs_to_impulse_responses(hrirs, recording)
