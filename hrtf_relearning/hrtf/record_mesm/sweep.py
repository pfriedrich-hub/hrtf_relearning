"""
sweep.py — Exponential sweep generation and MESM parameter computation.

References
----------
Majdak, Balazs, Laback (2007). "Multiple Exponential Sweep Method for Fast
Measurement of Head-Related Transfer Functions." J. Audio Eng. Soc. 55(7/8).

Equations cited as (Eq. N) refer to that paper.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataclass holding all derived MESM parameters
# ---------------------------------------------------------------------------

@dataclass
class MESMParams:
    """
    All timing and sweep parameters for one MESM measurement session.

    Attributes
    ----------
    f1, f2 : float
        Sweep frequency bounds in Hz.
    fs : int
        Sample rate in Hz.
    T_prime : float
        Adjusted sweep duration in seconds (Eq. 15 / 18). May equal T if
        the overlapping-only mode is used and no SNR gain is sought.
    T_total : float
        Total measurement duration in seconds (Eq. 11).
    c : float
        Sweep rate constant ln(f2/f1).
    tau_K : float
        Time offset of the K-th harmonic from the linear IR, in seconds (Eq. 4).
    delta : float
        Required onset gap between consecutive speakers, in seconds.
        delta = tau_K + L1.
    onset_times_s : list[float]
        Absolute onset time (seconds) for each speaker's sweep, length N.
    onset_samples : list[int]
        onset_times_s expressed in samples.
    T_prime_samples : int
        Sweep duration in samples.
    T_total_samples : int
        Total buffer length in samples.
    n_speakers : int
        Number of simultaneously excited speakers.
    K : int
        Highest harmonic order used to compute tau_K.
    L1 : float
        Linear IR length used for parameter calculation, in seconds.
    L2 : float
        Second-order HIR length, in seconds.
    """
    f1: float
    f2: float
    fs: int
    T_prime: float
    T_total: float
    c: float
    tau_K: float
    delta: float
    onset_times_s: list[float]
    onset_samples: list[int]
    T_prime_samples: int
    T_total_samples: int
    n_speakers: int
    K: int
    L1: float
    L2: float

    def summary(self) -> str:
        lines = [
            "MESM Parameters",
            "---------------",
            f"  Speakers        : {self.n_speakers}",
            f"  Sample rate     : {self.fs} Hz",
            f"  f1 / f2         : {self.f1} / {self.f2} Hz",
            f"  Sweep duration  : {self.T_prime:.3f} s  ({self.T_prime_samples} samples)",
            f"  τ_K (K={self.K})     : {self.tau_K*1e3:.1f} ms",
            f"  L1              : {self.L1*1e3:.1f} ms",
            f"  Onset gap Δ     : {self.delta*1e3:.1f} ms",
            f"  Total duration  : {self.T_total:.3f} s  ({self.T_total_samples} samples)",
            "  Onset times     :",
        ]
        for i, t in enumerate(self.onset_times_s):
            lines.append(f"    Speaker {i:2d}   : {t*1e3:.1f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter computation
# ---------------------------------------------------------------------------

def compute_mesm_params(
    n_speakers: int,
    fs: int,
    L1: float,
    K: int,
    T_prime: float,
    f1: float = 20.0,
    f2: float = 20_000.0,
    L2: float = 0.0,
) -> MESMParams:
    """
    Compute all MESM timing parameters for the pure-overlapping (η=1) design.

    In this design every speaker is its own group. Sweeps overlap in the
    recorded signal but their linear IRs are cleanly separated in the
    deconvolved output because consecutive onset times are spaced by

        Δ = τ_K + L1                                            (derived from Eq. 10)

    where τ_K = T'/c · ln(K) is the time offset of the K-th harmonic from
    the linear IR (Eq. 4).

    Parameters
    ----------
    n_speakers : int
        Number of simultaneously excited speakers (N in the paper).
    fs : int
        Sample rate in Hz.
    L1 : float
        Length of the longest linear IR across all speakers, in seconds.
        Obtained from a reference single-sweep measurement.
    K : int
        Highest harmonic order present in the system response. Obtained
        from the reference measurement.
    T_prime : float
        Desired sweep duration in seconds. Longer → higher SNR (+3 dB per
        doubling). Must satisfy T' > L1·c/ln(2) for clean IR separation
        (automatically checked here).
    f1 : float
        Lower sweep frequency in Hz. Default 20 Hz.
    f2 : float
        Upper sweep frequency in Hz. Default 20 000 Hz.
    L2 : float
        Second-order HIR length in seconds (informational, not used in
        overlapping-mode timing). Default 0.

    Returns
    -------
    MESMParams
    """
    c = np.log(f2 / f1)

    # Time offset of K-th harmonic from linear IR (Eq. 4)
    tau_K = (T_prime / c) * np.log(K)

    # Minimum gap to prevent K-th harmonic of sweep i from overlapping
    # with the linear IR of sweep i-1 (Eq. 10 rearranged for η=1)
    delta = tau_K + L1

    # Total measurement duration (Eq. 11, η=1 → N groups)
    # T_OV = T' + (N-1)*delta + L1
    T_total = T_prime + (n_speakers - 1) * delta + L1

    # Onset times: speaker i starts at i * delta  (Eq. 12, η=1)
    onset_times_s = [i * delta for i in range(n_speakers)]

    # Convert to samples
    T_prime_samples = int(round(T_prime * fs))
    T_total_samples = int(round(T_total * fs))
    onset_samples   = [int(round(t * fs)) for t in onset_times_s]

    return MESMParams(
        f1=f1,
        f2=f2,
        fs=fs,
        T_prime=T_prime,
        T_total=T_total,
        c=c,
        tau_K=tau_K,
        delta=delta,
        onset_times_s=onset_times_s,
        onset_samples=onset_samples,
        T_prime_samples=T_prime_samples,
        T_total_samples=T_total_samples,
        n_speakers=n_speakers,
        K=K,
        L1=L1,
        L2=L2,
    )


# ---------------------------------------------------------------------------
# Sweep generation
# ---------------------------------------------------------------------------

def exponential_sweep(
    T_samples: int,
    fs: int,
    f1: float,
    f2: float,
    ramp_samples: int = 0,
) -> np.ndarray:
    """
    Generate a single exponential (logarithmic) sweep.

        x(t) = sin( ω1 · T/c · (e^(c·t/T) − 1) )             (Eq. 1)

    Parameters
    ----------
    T_samples : int
        Sweep duration in samples.
    fs : int
        Sample rate in Hz.
    f1, f2 : float
        Start and end frequencies in Hz.
    ramp_samples : int
        Length of a half-Hann fade-in/fade-out ramp (samples). 0 = no ramp.
        A short ramp (~1 ms) avoids transient clicks on the speaker.

    Returns
    -------
    sweep : np.ndarray, shape (T_samples,), float64
        Normalised to peak amplitude 1.0.
    """
    t = np.arange(T_samples) / fs
    T = T_samples / fs
    c = np.log(f2 / f1)
    omega1 = 2 * np.pi * f1
    sweep = np.sin(omega1 * T / c * (np.exp(c * t / T) - 1.0))

    if ramp_samples > 0:
        ramp = np.hanning(2 * ramp_samples)
        sweep[:ramp_samples]  *= ramp[:ramp_samples]
        sweep[-ramp_samples:] *= ramp[ramp_samples:]

    return sweep


def inverse_sweep(sweep: np.ndarray, f1: float, f2: float) -> np.ndarray:
    """
    Compute the inverse (time-reversed, amplitude-weighted) sweep.

    The inverse sweep x'(t) is derived by time-reversing x(t) and applying
    a spectral amplitude weighting so that x(t) * x'(t) = δ(t) (Eq. 3):

        X'(ω) = X(-ω) / |X(-ω)|²

    In practice this is equivalent to time-reversing the sweep and multiplying
    by an envelope that compensates the 6 dB/octave amplitude increase of the
    ES spectrum (Müller & Massarani 2001).

    Parameters
    ----------
    sweep : np.ndarray
        Output of `exponential_sweep()`.
    f1, f2 : float
        Sweep frequency bounds, same as used to generate *sweep*.

    Returns
    -------
    inv : np.ndarray, same shape as *sweep*, float64
    """
    N = len(sweep)
    c = np.log(f2 / f1)

    # Time-reverse
    inv = sweep[::-1].copy()

    # Amplitude envelope: compensate 6 dB/oct rise of ES spectrum.
    # The envelope is e^(-c·t/T) evaluated on the reversed time axis.
    t = np.arange(N) / N          # normalised 0..1
    envelope = np.exp(-c * t)     # decays from 1 to f1/f2

    # Normalise so peak of the deconvolution is unity
    envelope /= envelope.max()
    inv *= envelope

    return inv


def build_speaker_buffers(
    params: MESMParams,
    ramp_samples: int = 48,
) -> list[np.ndarray]:
    """
    Build the N zero-padded sweep buffers ready to write to the RCX.

    Each buffer has length `params.T_total_samples`. Speaker i's sweep
    starts at `params.onset_samples[i]`.

    Parameters
    ----------
    params : MESMParams
        Output of `compute_mesm_params()`.
    ramp_samples : int
        Passed to `exponential_sweep()`. Default ~0.5 ms at 96 kHz.

    Returns
    -------
    buffers : list of np.ndarray, length N, each shape (T_total_samples,)
    """
    sweep = exponential_sweep(
        T_samples=params.T_prime_samples,
        fs=params.fs,
        f1=params.f1,
        f2=params.f2,
        ramp_samples=ramp_samples,
    )

    buffers = []
    for onset in params.onset_samples:
        buf = np.zeros(params.T_total_samples)
        end = onset + params.T_prime_samples
        if end > params.T_total_samples:
            raise ValueError(
                f"Sweep extends beyond buffer: onset={onset}, "
                f"T_prime={params.T_prime_samples}, T_total={params.T_total_samples}"
            )
        buf[onset:end] = sweep
        buffers.append(buf)

    return buffers
