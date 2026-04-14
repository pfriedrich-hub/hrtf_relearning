"""
recordings.py — Hardware interaction layer for MESM HRIR recording.

Responsibilities
----------------
- Initialize the custom RCX circuits on RX8 (playback) and RP2 (recording).
- Reference measurement: sequential single-sweep per speaker to extract L1, K.
- MESM measurement: write all N sweep buffers, trigger once, read back.
- Serialisation of raw recordings to .npz.

Hardware layout
---------------
  RX8  : playback — 7 BufOut components, each wired to a dedicated DAC pair
         via StereoScale (SFL=1, SFR=-1).
         Tags written from Python:
             sweep_0 … sweep_6   — float32 buffer, length n_samples
             n_samples           — scalar, playback buffer length

  RP2  : recording — 2 BufIn components (left + right ear mic).
         Tags written from Python:
             rec_len             — scalar, recording buffer length
         Tags read from Python:
             rec_l               — float32 buffer, left ear
             rec_r               — float32 buffer, right ear
             rec_idx             — scalar, running sample counter (completion poll)

  zBusA trigger fires RX8 and RP2 simultaneously (sample-accurate).

Design notes
------------
- We use freefield.PROCESSORS (the global Processors instance) directly via
  its .write() / .read() / .trigger() methods rather than going through the
  high-level freefield.play_and_record(), which assumes a different tag
  layout (playbuflen / chan / data / datal / datar) and the dome speaker table.
- Speaker equalization is omitted: the new speakers are not in the freefield
  speaker table and are expected to be calibrated externally (or not at all
  in a first pass).
- Spatial position (elevation / rotation angle from the turntable) is a plain
  float passed directly to MESMRecording — no speaker table lookup needed.
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
# RCX tag names — keep in sync with the RPvdsEX circuit
# ---------------------------------------------------------------------------
TAGS = {
    "sweep"  : "sweep_{i}",  # one per speaker, format with index
    "n_samp" : "n_samples",  # BufOut buffer length (RX8)
    "rec_l"  : "rec_l",      # left-ear BufIn (RP2)
    "rec_r"  : "rec_r",      # right-ear BufIn (RP2)
    "rec_len": "rec_len",    # BufIn buffer length (RP2)
    "rec_idx": "rec_idx",    # running sample counter for completion poll (RP2)
}

# Processor names as passed to PROCESSORS.initialize()
PROC_RX8 = "RX8"
PROC_RP2 = "RP2"

# D/A and A/D delays in samples (from freefield.get_recording_delay)
_DA_DELAY = {"RX8": 24, "RP2": 30}
_AD_DELAY = {"RX8": 47, "RP2": 65}
_EMPIRICAL_S = 0.0014   # matches local freefield fork: int(.0014 * fs)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize(rcx_rx8: Path, rcx_rp2: Path, connection: str = "GB") -> None:
    """
    Load the MESM RCX circuits onto RX8 and RP2 and arm the zBus.

    Parameters
    ----------
    rcx_rx8 : Path
        Path to the RPvdsEX circuit for the RX8 (7-channel playback).
    rcx_rp2 : Path
        Path to the RPvdsEX circuit for the RP2 (binaural recording).
    connection : str
        TDT connection type — 'GB' (optical) or 'USB'.
    """
    freefield.PROCESSORS.initialize(
        device=[
            [PROC_RX8, "RX8", str(rcx_rx8)],
            [PROC_RP2, "RP2", str(rcx_rp2)],
        ],
        zbus=True,
        connection=connection,
    )
    logging.info("MESM hardware initialized.")


# ---------------------------------------------------------------------------
# Recording delay
# ---------------------------------------------------------------------------

def get_recording_delay(fs: int, distance: float = 1.4) -> int:
    """
    Total delay (samples) between trigger and a valid recording onset.

    Matches the logic of freefield.get_recording_delay() + the empirical
    correction added in freefield.play_and_record():

        n = n_acoustic + n_DA(RX8) + n_AD(RP2) + int(0.0014 * fs)

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    distance : float
        Speaker-to-microphone distance in metres. Default 1.4 m.

    Returns
    -------
    n_delay : int
    """
    n_acoustic  = int(distance / 343.0 * fs)
    n_da        = _DA_DELAY[PROC_RX8]
    n_ad        = _AD_DELAY[PROC_RP2]
    n_empirical = int(_EMPIRICAL_S * fs)
    n_delay = n_acoustic + n_da + n_ad + n_empirical
    logging.debug(
        f"Recording delay: acoustic={n_acoustic}, DA={n_da}, AD={n_ad}, "
        f"empirical={n_empirical} → total={n_delay} samples ({n_delay/fs*1e3:.2f} ms)"
    )
    return n_delay


# ---------------------------------------------------------------------------
# Reference measurement result
# ---------------------------------------------------------------------------

@dataclass
class ReferenceParams:
    """
    Parameters extracted from the reference (single-sweep) measurement.

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
    Container for one raw MESM binaural recording.

    Attributes
    ----------
    left, right : np.ndarray, shape (T_total_samples,)
        Delay-compensated binaural recording.
    params : MESMParams
        Sweep / timing parameters.
    position : float | None
        Turntable position in degrees at the time of recording (elevation or
        azimuth depending on platform orientation). None if not applicable.
    meta : dict
        Arbitrary metadata (datetime, subject_id, n_repetitions, …).
    """

    def __init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        params: MESMParams,
        position: float | None = None,
        meta: dict | None = None,
    ):
        self.left     = np.asarray(left,  dtype=np.float64)
        self.right    = np.asarray(right, dtype=np.float64)
        self.params   = params
        self.position = position
        self.meta     = meta or {}

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
            position=np.array(self.position if self.position is not None else np.nan),
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
        pos = float(npz["position"])
        return cls(
            left=npz["left"],
            right=npz["right"],
            params=params,
            position=None if np.isnan(pos) else pos,
        )


# ---------------------------------------------------------------------------
# Core play-and-record (custom, no speaker table)
# ---------------------------------------------------------------------------

def _play_and_record(
    buffers: list[np.ndarray],
    params: MESMParams,
    n_delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Write sweep buffers to RX8, trigger, record from RP2, return trimmed arrays.

    Parameters
    ----------
    buffers : list of np.ndarray
        Zero-padded sweep buffers, one per speaker (length T_total_samples).
    params : MESMParams
    n_delay : int
        Recording delay in samples (from get_recording_delay).

    Returns
    -------
    rec_l, rec_r : np.ndarray, shape (T_total_samples,)
        Delay-compensated binaural recording.
    """
    rec_n_samples = params.T_total_samples + n_delay

    # --- write playback buffers and length to RX8 ---
    freefield.PROCESSORS.write(
        tag=TAGS["n_samp"], value=params.T_total_samples, processors=PROC_RX8
    )
    for i, buf in enumerate(buffers):
        freefield.PROCESSORS.write(
            tag=TAGS["sweep"].format(i=i),
            value=buf.astype(np.float32),
            processors=PROC_RX8,
        )

    # --- set extended recording length on RP2 ---
    freefield.PROCESSORS.write(
        tag=TAGS["rec_len"], value=rec_n_samples, processors=PROC_RP2
    )

    # --- fire simultaneous trigger via zBus ---
    freefield.PROCESSORS.trigger(kind="zBusA")

    # --- wait for RP2 buffer to fill ---
    _wait_for_recording(rec_n_samples, params.fs)

    # --- read back and strip leading delay samples ---
    rec_l = freefield.PROCESSORS.read(TAGS["rec_l"], PROC_RP2, rec_n_samples)
    rec_r = freefield.PROCESSORS.read(TAGS["rec_r"], PROC_RP2, rec_n_samples)
    return (
        np.asarray(rec_l, dtype=np.float64)[n_delay:],
        np.asarray(rec_r, dtype=np.float64)[n_delay:],
    )


def _wait_for_recording(
    rec_n_samples: int, fs: int, poll_interval: float = 0.05
) -> None:
    """
    Block until the RP2 recording buffer is full.

    Polls rec_idx tag; falls back to a timed wait if the tag is unavailable.

    Parameters
    ----------
    rec_n_samples : int
        Total buffer length written to RP2 (T_total_samples + n_delay).
    fs : int
        Sample rate, used to compute expected wall-clock duration.
    """
    expected = rec_n_samples / fs
    deadline = time.time() + expected + 2.0  # 2 s safety margin

    while time.time() < deadline:
        try:
            idx = freefield.PROCESSORS.read(TAGS["rec_idx"], PROC_RP2, n_samples=1)
            if int(idx) >= rec_n_samples:
                return
        except Exception:
            time.sleep(expected)
            return
        time.sleep(poll_interval)

    logging.warning("_wait_for_recording: timed out.")


# ---------------------------------------------------------------------------
# Reference measurement
# ---------------------------------------------------------------------------

def record_reference(
    n_speakers: int,
    fs: int,
    f1: float = 20.0,
    f2: float = 20_000.0,
    T: float = 1.0,
    distance: float = 1.4,
) -> ReferenceParams:
    """
    Sequential single-sweep measurement to determine L1, L2, K.

    Plays one sweep per speaker in turn (no overlapping) and deconvolves
    each recording to extract the timing parameters needed by
    compute_mesm_params().

    Parameters
    ----------
    n_speakers : int
    fs : int
    f1, f2 : float
        Sweep frequency bounds in Hz.
    T : float
        Sweep duration in seconds.
    distance : float
        Speaker-to-microphone distance in metres.

    Returns
    -------
    ReferenceParams

    TODO
    ----
    - Implement single-speaker sweep write + trigger + read loop.
    - Implement _extract_reference_params() to get L1, L2, K from the
      deconvolved HIR series of each speaker.
    """
    raise NotImplementedError(
        "record_reference() not yet implemented. "
        "Write one sweep per speaker sequentially using _play_and_record(), "
        "deconvolve each result, then call _extract_reference_params()."
    )


def _extract_reference_params(
    ir_series: np.ndarray,
    fs: int,
    f1: float,
    f2: float,
    T_prime: float,
    threshold_db: float = 70.0,
) -> dict:
    """
    Extract L1, L2, K from one channel of a deconvolved HIR series.

    Parameters
    ----------
    ir_series : np.ndarray
        s(t) = y(t) * x'(t) for one speaker / one ear.
    fs : int
    f1, f2, T_prime : float
        Sweep parameters (needed to compute tau_k positions).
    threshold_db : float
        Harmonics below this level relative to the linear IR peak are ignored.

    Returns
    -------
    dict with keys 'L1', 'L2', 'K' (all in seconds / dimensionless).

    TODO
    ----
    - Locate linear IR peak (rightmost large peak in ir_series).
    - Estimate L1 from IR decay to noise floor.
    - Walk leftward through tau_k positions to find significant harmonics.
    - Return max K across all speakers / both ears.
    """
    raise NotImplementedError("_extract_reference_params() not yet implemented.")


# ---------------------------------------------------------------------------
# MESM measurement
# ---------------------------------------------------------------------------

def record_mesm(
    params: MESMParams,
    position: float | None = None,
    n_repetitions: int = 1,
    distance: float = 1.4,
    subject_id: str | None = None,
) -> MESMRecording:
    """
    Run one MESM measurement: write all N sweep buffers, trigger, record.

    Parameters
    ----------
    params : MESMParams
        Output of sweep.compute_mesm_params().
    position : float, optional
        Current turntable position in degrees. Stored in MESMRecording for
        bookkeeping; not used for routing or calibration.
    n_repetitions : int
        Number of repeated measurements to average for SNR improvement.
    distance : float
        Speaker-to-microphone distance in metres (for delay calculation).
    subject_id : str, optional
        Stored in metadata.

    Returns
    -------
    MESMRecording
    """
    logging.info(
        f"MESM measurement: {params.n_speakers} speakers, "
        f"T={params.T_prime:.2f} s, T_total={params.T_total:.2f} s"
        + (f", position={position}°" if position is not None else "")
    )

    n_delay = get_recording_delay(fs=params.fs, distance=distance)
    logging.info(f"  Recording delay: {n_delay} samples ({n_delay/params.fs*1e3:.2f} ms)")

    buffers = build_speaker_buffers(params)

    recs_l, recs_r = [], []
    for rep in range(n_repetitions):
        logging.info(f"  Repetition {rep + 1}/{n_repetitions}")
        rec_l, rec_r = _play_and_record(buffers, params, n_delay)
        recs_l.append(rec_l)
        recs_r.append(rec_r)

    left  = np.mean(recs_l, axis=0)
    right = np.mean(recs_r, axis=0)

    meta = {
        "subject_id"   : subject_id,
        "n_repetitions": n_repetitions,
        "datetime"     : datetime.now().isoformat(),
    }
    logging.info("MESM measurement complete.")
    return MESMRecording(
        left=left, right=right,
        params=params, position=position, meta=meta,
    )
