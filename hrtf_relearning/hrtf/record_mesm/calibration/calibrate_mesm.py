"""
calibrate_mesm.py — Loudspeaker level + spectral calibration for the MESM rig.

Mirrors hrtf/record/calibration/calibrate_dome_pyfar.py, but drives the seven
new MESM loudspeakers directly through the custom RCX tags (sweep_i / n_samples
on the RX8, rec_l / rec_r / rec_len on the RP2) instead of the freefield dome
speaker table — these speakers are not in that table.

What it produces
----------------
A pickle mapping each speaker index to a level offset (dB) and an inverse FIR
filter::

    {speaker_idx: {"level": float, "filter": slab.Filter}}

This is the same structure the dome pipeline uses, so the same apply logic
(`sig.level += cal["level"]; sig = cal["filter"].apply(sig)`) can be reused when
building the per-speaker MESM sweep buffers.

Method
------
1. PASS A (raw): play a log sweep on each speaker in turn, record on the RP2,
   average N repetitions.
2. Offline design: compute relative level offsets (flatten loudness across
   speakers) and a regularised inverse FIR per speaker.
3. PASS B (verify): replay with the calibration applied and inspect the
   flattened spectra.
4. Save the calibration to disk.

Hardware note
-------------
Per-speaker playback reuses the MESM hardware layer (`_play_and_record`): every
trigger writes all seven sweep buffers, but only the speaker under test carries
the sweep — the rest are silent.

Calibration is recorded with a flat freefield probe microphone (not the in-ear
mics), so the inverse filters genuinely flatten each speaker's level and
magnitude response. Set MIC to the RP2 channel the probe mic is wired to. This
calibration is then loaded at record time and applied to the sweep buffers when
recording both the reference and the subject DTFs through the in-ear mics.

Usage
-----
Edit the CONFIG block, place the calibration mic, then::

    python -m hrtf_relearning.hrtf.record_mesm.calibration.calibrate_mesm
"""
from __future__ import annotations

import logging
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import slab
import pyfar
import freefield

import hrtf_relearning
from hrtf_relearning.hrtf.record_mesm.recordings import (
    initialize,
    get_recording_delay,
    _play_and_record,
)

# ------------------------ CONFIG ------------------------

ROOT = hrtf_relearning.PATH

RCX_RX8 = ROOT / "hrtf" / "record_mesm" / "rcx" / "record_hrtf_rx8.rcx"
RCX_RP2 = ROOT / "hrtf" / "record_mesm" / "rcx" / "record_hrtf_rp2.rcx"

FS            = 48828          # match the recording session sample rate
N_SPEAKERS    = 7
SPEAKER_IDX   = list(range(N_SPEAKERS))   # which speakers to calibrate

LOW_FREQ      = 20.0          # Hz — inversion lower bound
HIGH_FREQ     = FS / 2        # Hz — inversion upper bound
SWEEP_DUR     = 0.5          # s — calibration sweep duration
LEVEL         = 50           # dB — excitation level
TAIL_S        = 0.10         # s — recording tail after the sweep (IR capture)
DISTANCE      = 1.4          # m — speaker-to-mic distance (delay compensation)

N_REPEATS     = 5
BETA          = 0.1          # inverse-filter regularisation weight
FILTER_LENGTH = 1024         # taps in the saved inverse FIR
MIC           = "left"       # RP2 channel the freefield probe mic is wired to: "left" | "right"

SHOW          = True

OUTPUT_FILE   = ROOT / "hrtf" / "record_mesm" / "calibration" / "calibration_mesm.pkl"


# ------------------------ SETUP ------------------------

def initialize_mesm() -> None:
    initialize(rcx_rx8=RCX_RX8, rcx_rp2=RCX_RP2)
    freefield.set_logger("info")
    slab.set_default_samplerate(FS)


def make_excitation() -> slab.Sound:
    sig = slab.Sound.chirp(
        duration=SWEEP_DUR,
        level=LEVEL,
        from_frequency=LOW_FREQ,
        to_frequency=FS / 2,
        kind="logarithmic",
    )
    return sig.ramp(duration=0.02)


# ------------------------ RECORD ONCE ------------------------

def _select_mic(rec_l: np.ndarray, rec_r: np.ndarray) -> np.ndarray:
    if MIC == "left":
        return rec_l
    if MIC == "right":
        return rec_r
    if MIC == "mean":
        return 0.5 * (rec_l + rec_r)
    raise ValueError(f"MIC must be 'left', 'right' or 'mean', got {MIC!r}")


def record_speaker(speaker_idx: int, excitation: slab.Sound) -> slab.Sound:
    """
    Play `excitation` on a single MESM speaker and return the averaged recording.

    All seven RX8 sweep buffers are written each trigger; only `speaker_idx`
    carries the sweep, the rest are silent. Reuses the MESM `_play_and_record`
    hardware path (delay-compensated read-back).
    """
    sweep = np.asarray(excitation.data).squeeze().astype(np.float64)
    n_total = len(sweep) + int(round(TAIL_S * FS))

    # lightweight stand-in for MESMParams: _play_and_record only reads these two
    params = SimpleNamespace(T_total_samples=n_total, fs=FS)
    n_delay = get_recording_delay(fs=FS, distance=DISTANCE)

    recs = []
    for _ in range(N_REPEATS):
        buffers = [np.zeros(n_total, dtype=np.float64) for _ in range(N_SPEAKERS)]
        buffers[speaker_idx][: len(sweep)] = sweep
        rec_l, rec_r = _play_and_record(buffers, params, n_delay)
        recs.append(slab.Sound(_select_mic(rec_l, rec_r), samplerate=FS))

    recs, shifts = align_recordings(recs)
    return slab.Sound(np.mean([r.data for r in recs], axis=0), samplerate=FS)


def record_all_speakers(
    excitation: slab.Sound,
    equalize: bool = False,
    calibration: dict | None = None,
) -> dict[int, slab.Sound]:
    recordings = {}
    for idx in SPEAKER_IDX:
        logging.info(f"Recording speaker {idx}")
        sig = excitation
        if equalize:
            if calibration is None:
                raise ValueError("Calibration required when equalize=True")
            sig = deepcopy(excitation)
            sig.level += calibration[idx]["level"]
            sig = calibration[idx]["filter"].apply(sig)
        recordings[idx] = record_speaker(idx, sig)
    return recordings


def align_recordings(recordings: list, max_shift: int = 3, ref_index: int = 0):
    """Align monaural recordings by small-shift correlation before averaging."""
    if len(recordings) == 0:
        raise ValueError("Empty recordings list")

    fs = recordings[0].samplerate
    n_samples = recordings[0].n_samples
    n_rec = len(recordings)

    def _mono(rec):
        x = np.asarray(rec.data)
        return x[:, 0] if x.ndim == 2 else x

    x = np.stack([_mono(r) for r in recordings], axis=0)
    ref = x[ref_index]

    shifts = np.zeros(n_rec, dtype=int)
    aligned = np.empty_like(x)
    candidates = np.arange(-max_shift, max_shift + 1)

    for i in range(n_rec):
        y = x[i]
        scores = []
        for s in candidates:
            if s < 0:
                score = np.dot(ref[-s:], y[: n_samples + s])
            elif s > 0:
                score = np.dot(ref[: n_samples - s], y[s:])
            else:
                score = np.dot(ref, y)
            scores.append(score)
        best = candidates[int(np.argmax(scores))]
        shifts[i] = best
        sig = pyfar.Signal(y[None, :], fs)
        aligned[i] = pyfar.dsp.time_shift(sig, -best, unit="samples").time[0]

    if (abs(shifts) > 2).any():
        logging.warning(f"Time shifts > 2 samples when averaging:\n{shifts}")

    return [slab.Sound(a, samplerate=fs) for a in aligned], shifts


# ------------------------ LEVEL EQUALIZATION ------------------------

def compute_level_equalization(recordings: dict[int, slab.Sound]) -> dict[int, float]:
    """Relative level offsets that flatten loudness across speakers."""
    levels = {idx: rec.level for idx, rec in recordings.items()}
    ref = np.mean(list(levels.values()))
    return {idx: ref - lvl for idx, lvl in levels.items()}


def update_levels_from_recordings(recordings, calibration, reference="mean"):
    """Fold residual level errors from a verification pass back into calibration."""
    levels = {idx: rec.level for idx, rec in recordings.items()}
    if reference == "mean":
        ref = np.mean(list(levels.values()))
    elif reference == "median":
        ref = np.median(list(levels.values()))
    else:
        raise ValueError("reference must be 'mean' or 'median'")
    residuals = {}
    for idx, lvl in levels.items():
        delta = ref - lvl
        calibration[idx]["level"] += delta
        residuals[idx] = delta
    return residuals


# ------------------------ FILTER CONSTRUCTION ------------------------

def build_inverse_filter(recording: slab.Sound, excitation: slab.Sound) -> slab.Filter:
    speaker_raw = pyfar.Signal(recording.data.T, recording.samplerate)
    sig = pyfar.Signal(excitation.data.T, excitation.samplerate)

    # The MESM recording is longer than the excitation (sweep + IR tail); pad the
    # excitation so the frequency-domain deconvolution operands match in length.
    if sig.n_samples < speaker_raw.n_samples:
        sig = pyfar.dsp.pad_zeros(sig, speaker_raw.n_samples - sig.n_samples)
    elif sig.n_samples > speaker_raw.n_samples:
        speaker_raw = pyfar.dsp.pad_zeros(
            speaker_raw, sig.n_samples - speaker_raw.n_samples)

    # speaker transfer function = recording deconvolved by the excitation
    sig_inv = pyfar.dsp.regularized_spectrum_inversion(
        sig, frequency_range=(LOW_FREQ, HIGH_FREQ))
    speaker_ir = speaker_raw * sig_inv

    speaker_ir.time /= np.max(np.abs(speaker_ir.time))  # normalise

    # force causality (centre the IR)
    onset = pyfar.dsp.find_impulse_response_start(speaker_ir, threshold=10)
    shift_s = 0.01 - onset / speaker_ir.sampling_rate
    speaker_ir = pyfar.dsp.time_shift(speaker_ir, shift_s, unit="s")

    # regularisation profile: relax inversion outside the usable band
    reg = pyfar.dsp.filter.low_shelf(
        pyfar.signals.impulse(speaker_ir.n_samples), 60, 20, 2)
    reg = pyfar.dsp.filter.high_shelf(reg, 6000, 20, 2) * 0.1
    inv = pyfar.dsp.regularized_spectrum_inversion(
        speaker_ir,
        frequency_range=(LOW_FREQ, HIGH_FREQ),
        regu_final=reg.freq * BETA)

    eq = pyfar.dsp.minimum_phase(inv, truncate=False)
    eq = pyfar.dsp.time_window(
        eq, [0, FILTER_LENGTH - 1], shape="right", window="boxcar", crop="window")

    return slab.Filter(eq.time.T, samplerate=FS, fir="IR")


# ------------------------ MAIN ------------------------

def main() -> dict:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    from hrtf_relearning.hrtf.record_mesm.calibration.diagnostics import plot_calibration_result, calibration_metrics

    initialize_mesm()
    excitation = make_excitation()

    # PASS A — raw recordings (no EQ)
    raw_recordings = record_all_speakers(excitation, equalize=False)

    # OFFLINE DESIGN
    level_eq = compute_level_equalization(raw_recordings)
    #todo level spread at 2 db still too high, run multiple level calibrations?
    calibration = {}
    for idx in SPEAKER_IDX:
        calibration[idx] = {
            "level": level_eq[idx],
            "filter": build_inverse_filter(raw_recordings[idx], excitation),
        }

    # PASS B — verification (explicit EQ)
    verified = record_all_speakers(excitation, equalize=True, calibration=calibration)

    # DIAGNOSTICS — quantify + visualise what the calibration achieved
    calibration_metrics(raw_recordings, verified, fs=FS)
    plot_calibration_result(raw_recordings, verified, calibration, fs=FS, show=SHOW)

    # SAVE
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(calibration, f)
    logging.info(f"Calibration verified and saved to {OUTPUT_FILE}")
    return calibration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # main()
