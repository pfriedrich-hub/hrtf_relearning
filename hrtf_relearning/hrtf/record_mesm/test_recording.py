"""
test_recording.py — Hardware test for the MESM recording system.

Run this script directly to verify:
  1. RCX circuits load and processors initialise correctly
  2. Sweep buffers are written to the RX8 without errors
  3. The RP2 records and returns sensible data
  4. Deconvolution produces recognisable IRs at the expected time offsets

Usage
-----
Edit the configuration block below (RCX paths, fs, L1_ms, K), then run:

    python -m hrtf_relearning.hrtf.record_mesm.test_recording

or from a Python shell in the project root:

    from hrtf_relearning.hrtf.record_mesm.test_recording import run_test
    run_test()

The reference measurement step is deliberately skipped here — L1 and K are
provided manually so the test can run before record_reference() is implemented.
"""
from __future__ import annotations

import logging
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import hrtf_relearning
from hrtf_relearning.hrtf.record_mesm.recordings import initialize, record_mesm, get_recording_delay
from hrtf_relearning.hrtf.record_mesm.sweep import compute_mesm_params, build_speaker_buffers
from hrtf_relearning.hrtf.record_mesm.processing import deconvolve, extract_hrirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Configuration — edit before each test session
# ---------------------------------------------------------------------------

RCX_RX8  = hrtf_relearning.PATH / "data" / "rcx" / "record_hrtf_rx8.rcx"
RCX_RP2  = hrtf_relearning.PATH / "data" / "rcx" / "record_hrtf_rp2.rcx"

FS           = 48000    # start at 48 kHz; switch to 96000 once confirmed working
N_SPEAKERS   = 7
F1           = 20.0     # Hz
F2           = 20_000.0 # Hz
T_PRIME      = 1.0      # sweep duration in seconds
DISTANCE     = 1.4      # speaker-to-mic distance in metres

# Manual L1 / K — replace with output of record_reference() once implemented.
# L1: conservative estimate for a semi-anechoic room (50 ms).
# K:  5 is typical for electroacoustic systems.
L1_S  = 0.050   # seconds
L2_S  = 0.010   # seconds (informational only)
K     = 5

SAVE_DIR     = hrtf_relearning.PATH / "data" / "hrtf" / "rec_mesm" / "test"
N_REPS       = 1        # increase for SNR check once basic recording works


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test(plot: bool = True) -> None:

    # ------------------------------------------------------------------
    # 1. Initialise hardware
    # ------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("MESM hardware test")
    logging.info("=" * 60)

    if not RCX_RX8.exists():
        raise FileNotFoundError(
            f"RX8 circuit not found: {RCX_RX8}\n"
            "Build the circuit in RPvdsEx and save it there first."
        )
    if not RCX_RP2.exists():
        raise FileNotFoundError(
            f"RP2 circuit not found: {RCX_RP2}\n"
            "Build the circuit in RPvdsEx and save it there first."
        )

    logging.info(f"Loading RX8 circuit: {RCX_RX8}")
    logging.info(f"Loading RP2 circuit: {RCX_RP2}")
    initialize(rcx_rx8=RCX_RX8, rcx_rp2=RCX_RP2)

    # ------------------------------------------------------------------
    # 2. Compute MESM parameters
    # ------------------------------------------------------------------
    params = compute_mesm_params(
        n_speakers=N_SPEAKERS,
        fs=FS,
        L1=L1_S,
        K=K,
        T_prime=T_PRIME,
        f1=F1,
        f2=F2,
        L2=L2_S,
    )
    logging.info("\n" + params.summary())

    n_delay = get_recording_delay(fs=FS, distance=DISTANCE)
    logging.info(
        f"\nRecording delay: {n_delay} samples  ({n_delay / FS * 1e3:.2f} ms)"
        f"\nTotal buffer written to RP2: {params.T_total_samples + n_delay} samples"
        f"  ({(params.T_total_samples + n_delay) / FS:.3f} s)"
    )

    # ------------------------------------------------------------------
    # 3. Preview sweep buffers (no hardware needed)
    # ------------------------------------------------------------------
    if plot:
        _plot_buffers(params)
        input("\nCheck sweep buffer plot — press Enter to continue to recording ...")

    # ------------------------------------------------------------------
    # 4. Record
    # ------------------------------------------------------------------
    logging.info("\nStarting recording ...")
    recording = record_mesm(
        params=params,
        position=None,
        n_repetitions=N_REPS,
        distance=DISTANCE,
        subject_id="test",
    )
    logging.info("Recording complete.")

    # ------------------------------------------------------------------
    # 5. Save raw recording
    # ------------------------------------------------------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    recording.to_npz(SAVE_DIR, overwrite=True)
    logging.info(f"Raw recording saved to {SAVE_DIR}")

    # ------------------------------------------------------------------
    # 6. Plot raw recording
    # ------------------------------------------------------------------
    if plot:
        _plot_recording(recording)

    # ------------------------------------------------------------------
    # 7. Deconvolve and plot HIR series
    # ------------------------------------------------------------------
    logging.info("Deconvolving ...")
    s_l, s_r = deconvolve(recording)

    if plot:
        _plot_hir_series(s_l, s_r, params)

    # ------------------------------------------------------------------
    # 8. Window individual IRs and plot
    # ------------------------------------------------------------------
    hrirs = extract_hrirs(s_l, s_r, recording)

    if plot:
        _plot_hrirs(hrirs, params)
        plt.show()

    logging.info("Test complete.")
    return recording, s_l, s_r, hrirs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_buffers(params) -> None:
    """Show the 7 sweep buffers (time domain) before any hardware is touched."""
    buffers = build_speaker_buffers(params)
    t = np.arange(params.T_total_samples) / params.fs * 1e3  # ms

    fig, axes = plt.subplots(N_SPEAKERS, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Sweep buffers written to RX8  (preview, no hardware)")
    for i, (buf, ax) in enumerate(zip(buffers, axes)):
        ax.plot(t, buf, lw=0.5)
        ax.set_ylabel(f"spk {i}", fontsize=8)
        ax.axvline(params.onset_times_s[i] * 1e3, color="r", lw=0.8, ls="--")
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()


def _plot_recording(recording) -> None:
    """Plot the raw binaural recording (both ears)."""
    t = np.arange(len(recording.left)) / recording.params.fs * 1e3

    fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle("Raw binaural recording (delay-compensated)")
    ax_l.plot(t, recording.left,  lw=0.4)
    ax_l.set_ylabel("Left")
    ax_r.plot(t, recording.right, lw=0.4, color="C1")
    ax_r.set_ylabel("Right")
    ax_r.set_xlabel("Time (ms)")
    plt.tight_layout()


def _plot_hir_series(s_l, s_r, params) -> None:
    """Plot the deconvolved HIR series with expected IR onset markers."""
    t = (np.arange(len(s_l)) / params.fs - params.T_total_samples / params.fs) * 1e3

    fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Deconvolved HIR series — linear IRs expected at onset markers (▲)")

    for ax, sig, label in [(ax_l, s_l, "Left"), (ax_r, s_r, "Right")]:
        ax.plot(t, 20 * np.log10(np.abs(sig) + 1e-12), lw=0.4)
        ax.set_ylabel(f"{label}  (dBFS)")
        ax.set_ylim(-100, 5)
        for i, onset in enumerate(params.onset_samples):
            t_onset = (onset / params.fs - params.T_total_samples / params.fs) * 1e3
            ax.axvline(t_onset, color=f"C{i}", lw=0.8, ls="--",
                       label=f"spk {i}")
    ax_l.legend(fontsize=7, ncol=7)
    ax_r.set_xlabel("Time (ms)")
    plt.tight_layout()


def _plot_hrirs(hrirs, params) -> None:
    """Plot each extracted per-speaker HRIR (left ear)."""
    n = len(hrirs)
    fig, axes = plt.subplots(1, n, figsize=(14, 3), sharey=True)
    fig.suptitle("Extracted HRIRs — left ear")
    if n == 1:
        axes = [axes]
    for i, (ir_l, ir_r) in hrirs.items():
        t = np.arange(len(ir_l)) / params.fs * 1e3
        axes[i].plot(t, ir_l, lw=0.8)
        axes[i].set_title(f"Speaker {i}", fontsize=9)
        axes[i].set_xlabel("ms")
    axes[0].set_ylabel("Amplitude")
    plt.tight_layout()


if __name__ == "__main__":
    run_test()
