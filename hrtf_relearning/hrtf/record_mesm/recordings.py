"""
recordings.py — Hardware interaction layer for MESM HRIR recording.

Responsibilities
----------------
- Reference measurement (single ES per speaker, sequential) to extract L1, K.
- MESM measurement: write all N sweep buffers simultaneously, trigger,
  read back the binaural recording.
- Serialisation of raw recordings to .npz.

Hardware layout
---------------
  RX8   : playback — 7 BufOut components (tags sweep_0 … sweep_6),
           each fed into a StereoScale (SFL=1, SFR=-1) → DAC pair.
           Control tags: n_samples (buffer length).
  RP2   : recording — 2 BufIn components (tags rec_l, rec_r).
           Control tag: rec_len (recording length in samples).

The zBusA trigger fires all components simultaneously.

TODO (once RCX is finalised)
-----------------------------
- Confirm exact tag names from RPvdsEX (sweep_0 … sweep_6, n_samples,
  rec_l, rec_r, rec_len). Update TAGS dict below.
- Confirm processor names (RX8 device string, RP2 device string) and
  update PROC_PLAY / PROC_REC below.
- Confirm completion-detection strategy (polling tag vs. fixed wait).
- Add speaker-equalization support once calibration data is available
  for the new 7-speaker dome.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import slab
import freefield

from .sweep import MESMParams, build_speaker_buffers, exponential_sweep, inverse_sweep

# ---------------------------------------------------------------------------
# Hardware tag names — update when RCX is finalised
# ---------------------------------------------------------------------------
TAGS = {
    "sweep"   : "sweep_{i}",   # format with speaker index
    "n_samples": "n_samples",  # BufOut length tag on RX8
    "rec_l"   : "rec_l",       # left-ear BufIn tag on RP2
    "rec_r"   : "rec_r",       # right-ear BufIn tag on RP2
    "rec_len" : "rec_len",     # recording length tag on RP2
    "rec_idx" : "rec_idx",     # running sample counter on RP2 (for completion poll)
}

PROC_PLAY = "RX8"   # TODO: confirm device string
PROC_REC  = "RP2"   # TODO: confirm device string


# ---------------------------------------------------------------------------
# Reference measurement result
# ---------------------------------------------------------------------------

@dataclass
class ReferenceParams:
    """
    Parameters extracted from the reference (single-sweep) measurement.
    These are used to compute the MESM timing via sweep.compute_mesm_params().

    Attributes
    ----------
    L1 : float
        Longest linear IR length across all speakers, in seconds.
    L2 : float
        Longest second-order HIR length across all speakers, in seconds.
    K : int
        Highest harmonic order with amplitude above threshold.
    fs : int
        Sample rate used during measurement.
    n_speakers : int
        Number of speakers measured.
    """
    L1: float
    L2: float
    K: int
    fs: int
    n_speakers: int


# ---------------------------------------------------------------------------
# Raw MESM recording container
# ---------------------------------------------------------------------------

class MESMRecording:
    """
    Container for one raw MESM binaural recording and its associated params.

    Attributes
    ----------
    left : np.ndarray, shape (T_total_samples,)
        Left-ear pressure recording.
    right : np.ndarray, shape (T_total_samples,)
        Right-ear pressure recording.
    params : MESMParams
        Timing/sweep parameters used for this recording.
    speaker_table : dict
        Mapping speaker index → (azimuth, elevation) in degrees.
        To be populated from the speaker table file.
    meta : dict
        Arbitrary metadata (datetime, subject_id, …).
    """

    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        params: MESMParams,
        speaker_table: dict | None = None,
        meta: dict | None = None,
    ):
        self.left  = np.asarray(left,  dtype=np.float64)
        self.right = np.asarray(right, dtype=np.float64)
        self.params = params
        self.speaker_table = speaker_table or {}
        self.meta  = meta or {}

    # ------------------------------------------------------------------ I/O

    def to_npz(self, path: Path, overwrite: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        fname = path / "mesm_recording.npz"
        if fname.exists() and not overwrite:
            logging.info(f"{fname} exists — skipping (use overwrite=True).")
            return
        np.savez(
            fname,
            left=self.left,
            right=self.right,
            # MESMParams fields
            f1=self.params.f1,
            f2=self.params.f2,
            fs=self.params.fs,
            T_prime=self.params.T_prime,
            T_total=self.params.T_total,
            c=self.params.c,
            tau_K=self.params.tau_K,
            delta=self.params.delta,
            onset_times_s=np.array(self.params.onset_times_s),
            onset_samples=np.array(self.params.onset_samples),
            T_prime_samples=self.params.T_prime_samples,
            T_total_samples=self.params.T_total_samples,
            n_speakers=self.params.n_speakers,
            K=self.params.K,
            L1=self.params.L1,
            L2=self.params.L2,
        )
        logging.info(f"MESM recording saved to {fname}")

    @classmethod
    def from_npz(cls, path: Path) -> "MESMRecording":
        from .sweep import MESMParams
        path = Path(path)
        npz = np.load(path / "mesm_recording.npz")
        params = MESMParams(
            f1=float(npz["f1"]),
            f2=float(npz["f2"]),
            fs=int(npz["fs"]),
            T_prime=float(npz["T_prime"]),
            T_total=float(npz["T_total"]),
            c=float(npz["c"]),
            tau_K=float(npz["tau_K"]),
            delta=float(npz["delta"]),
            onset_times_s=npz["onset_times_s"].tolist(),
            onset_samples=npz["onset_samples"].tolist(),
            T_prime_samples=int(npz["T_prime_samples"]),
            T_total_samples=int(npz["T_total_samples"]),
            n_speakers=int(npz["n_speakers"]),
            K=int(npz["K"]),
            L1=float(npz["L1"]),
            L2=float(npz["L2"]),
        )
        return cls(left=npz["left"], right=npz["right"], params=params)


# ---------------------------------------------------------------------------
# Reference measurement
# ---------------------------------------------------------------------------

def record_reference(
    n_speakers: int,
    fs: int,
    f1: float = 20.0,
    f2: float = 20_000.0,
    T: float = 1.0,
    ramp_ms: float = 0.5,
    K_threshold_db: float = 70.0,
) -> ReferenceParams:
    """
    Sequential single-sweep measurement to extract L1, L2, K.

    Plays one sweep per speaker one at a time (no overlapping), records the
    binaural response, deconvolves to get IRs, and extracts the timing
    parameters required by compute_mesm_params().

    Parameters
    ----------
    n_speakers : int
        Number of speakers to measure.
    fs : int
        Sample rate in Hz.
    f1, f2 : float
        Sweep frequency bounds.
    T : float
        Sweep duration in seconds. 1–2 s is typical.
    ramp_ms : float
        Fade-in/out ramp duration in ms.
    K_threshold_db : float
        Harmonics below this many dB relative to the linear IR peak are
        ignored when determining K.

    Returns
    -------
    ReferenceParams

    TODO
    ----
    - Implement hardware interaction (write sweep to RX8, trigger, read RP2).
      For now raises NotImplementedError.
    - Implement IR extraction and L1/L2/K estimation from deconvolved signal.
    """
    raise NotImplementedError(
        "record_reference() is not yet implemented. "
        "Implement the single-speaker sweep playback + recording loop here, "
        "then call _extract_reference_params() on the deconvolved IRs."
    )


def _extract_reference_params(
    ir_series: np.ndarray,
    fs: int,
    params_sweep: dict,
    threshold_db: float = 70.0,
) -> ReferenceParams:
    """
    Extract L1, L2, K from a deconvolved HIR series (reference measurement).

    Parameters
    ----------
    ir_series : np.ndarray
        Full deconvolved signal s(t) = y(t) * x'(t) for one speaker channel.
        The linear IR is the rightmost peak (k=1); higher-order HIRs appear
        further to the left.
    fs : int
        Sample rate.
    params_sweep : dict
        Must contain 'f1', 'f2', 'T_prime' (seconds).
    threshold_db : float
        Harmonics below this level relative to linear IR peak are discarded.

    Returns
    -------
    ReferenceParams (partial — L1 and L2 from one channel; caller aggregates
    across speakers and ears).

    TODO
    ----
    - Implement peak detection logic using tau_k positions to locate each HIR.
    - Estimate L1 as the length of the linear IR above the noise floor.
    - Estimate K as the highest harmonic above threshold_db.
    """
    raise NotImplementedError("_extract_reference_params() not yet implemented.")


# ---------------------------------------------------------------------------
# MESM measurement
# ---------------------------------------------------------------------------

def record_mesm(
    params: MESMParams,
    n_repetitions: int = 1,
    subject_id: str | None = None,
) -> MESMRecording:
    """
    Run one MESM measurement: write all N sweep buffers, trigger, record.

    Steps
    -----
    1. Build zero-padded sweep buffers (one per speaker).
    2. Write each buffer to its RCX tag (sweep_0 … sweep_{N-1}).
    3. Set n_samples and rec_len tags.
    4. Fire zBusA trigger.
    5. Wait for recording to complete.
    6. Read back rec_l, rec_r from RP2.
    7. Optionally average over n_repetitions.

    Parameters
    ----------
    params : MESMParams
        Output of sweep.compute_mesm_params().
    n_repetitions : int
        Number of repeated measurements to average (improves SNR by
        √n_repetitions). Default 1.
    subject_id : str, optional
        Written into meta dict.

    Returns
    -------
    MESMRecording

    TODO
    ----
    - Confirm tag names and processor strings (TAGS / PROC_PLAY / PROC_REC).
    - Add speaker equalization (apply calibration filter to each sweep buffer
      before writing, once calibration data is available for the new dome).
    - Add head-tracker logging if real-time head orientation is needed.
    """
    logging.info(f"Starting MESM measurement: {params.n_speakers} speakers, "
                 f"T={params.T_prime:.2f}s, T_total={params.T_total:.2f}s")

    buffers = build_speaker_buffers(params)

    recs_l, recs_r = [], []
    for rep in range(n_repetitions):
        logging.info(f"  Repetition {rep + 1}/{n_repetitions}")

        # --- write sweep buffers to RX8 ---
        freefield.write(tag=TAGS["n_samples"], value=params.T_total_samples,
                        processors=PROC_PLAY)
        for i, buf in enumerate(buffers):
            tag = TAGS["sweep"].format(i=i)
            freefield.write(tag=tag, value=buf.astype(np.float32),
                            processors=PROC_PLAY)

        # --- configure RP2 recording length ---
        freefield.write(tag=TAGS["rec_len"], value=params.T_total_samples,
                        processors=PROC_REC)

        # --- trigger (sample-accurate across both devices via zBus) ---
        freefield.play(kind="zBusA")

        # --- wait for recording to complete ---
        _wait_for_recording(params.T_total_samples, params.fs)

        # --- read back binaural recording ---
        rec_l = freefield.read(TAGS["rec_l"], proc=PROC_REC,
                               n_samples=params.T_total_samples)
        rec_r = freefield.read(TAGS["rec_r"], proc=PROC_REC,
                               n_samples=params.T_total_samples)
        recs_l.append(np.asarray(rec_l, dtype=np.float64))
        recs_r.append(np.asarray(rec_r, dtype=np.float64))

    # Average repetitions
    left  = np.mean(recs_l, axis=0)
    right = np.mean(recs_r, axis=0)

    meta = {
        "subject_id"   : subject_id,
        "n_repetitions": n_repetitions,
        "datetime"     : datetime.now().isoformat(),
    }
    logging.info("MESM measurement complete.")
    return MESMRecording(left=left, right=right, params=params, meta=meta)


def _wait_for_recording(n_samples: int, fs: int, poll_interval: float = 0.05) -> None:
    """
    Block until the RP2 has finished filling its recording buffer.

    Strategy: poll rec_idx tag on RP2.  Fall back to a fixed sleep if the
    tag is not available.

    TODO: confirm that rec_idx is implemented in the RCX circuit.
    """
    expected_duration = n_samples / fs
    deadline = time.time() + expected_duration + 2.0   # 2 s safety margin

    while time.time() < deadline:
        try:
            idx = freefield.read(TAGS["rec_idx"], proc=PROC_REC, n_samples=1)
            if int(idx) >= n_samples:
                return
        except Exception:
            # Tag not available yet — fall back to timed wait
            time.sleep(expected_duration)
            return
        time.sleep(poll_interval)

    logging.warning("_wait_for_recording: timed out waiting for RP2.")
