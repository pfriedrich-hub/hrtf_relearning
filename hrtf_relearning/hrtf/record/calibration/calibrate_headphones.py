"""
Headphone Equalization Script
-----------------------------

This script measures or loads a raw headphone impulse response (HpIR), and
computes an equalization filter that compensates the headphone transfer
function. The resulting inverse filter can be used for HRTF recordings or
psychophysical experiments requiring flat headphone playback.

Steps:
1. Initialize hardware and generate the test signal (logarithmic chirp).
2. Record or load the raw headphone response.
3. Convert to pyfar.Signal.
4. Compute the headphone transfer function.
5. Build a regularized minimum-phase inverse filter.
6. Optionally plot TF/IR and intermediate steps.
7. Return or save the equalization filter in a dictionary.

Dependencies:
- freefield
- slab
- pyfar
- matplotlib
"""

import matplotlib
from pyfar import Signal
from slab import Filter

matplotlib.use("tkagg")
from matplotlib import pyplot as plt
import numpy
import pyfar
import slab
import freefield
import warnings
import soundfile as sf
from pathlib import Path
import pickle
import logging
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)
freefield.set_logger("info")
import hrtf_relearning
from hrtf_relearning.utils import paths
ROOT = hrtf_relearning.PATH

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
SUB_ID = 'KEMAR'
HP_ID = 'MYSPHERE'  # MYSPHERE, DT990


fs = 48828
slab.set_default_samplerate(fs)

# sweep parameters
LEVEL = 70
N_REC = 3
LOW_CUTOFF = 20
HIGH_CUTOFF = fs/2
CHIRP_DURATION = 1.0
RAMP_DURATION = .001  # 1ms ramp

# regularization parameters
BETA = 0.1

N_OUT = 256
# -------------------------------------------------------------------------
# SIGNAL GENERATION
# -------------------------------------------------------------------------

def generate_chirp():
    """
    Generate a logarithmic binaural test chirp for measuring headphone TF.

    Returns
    -------
    signal : slab.Binaural
        Two-channel chirp from LOW_CUTOFF to HIGH_CUTOFF Hz.
    """
    signal =  slab.Binaural.chirp(
        duration=CHIRP_DURATION,
        level=LEVEL,
        from_frequency=LOW_CUTOFF,
        to_frequency=HIGH_CUTOFF,
        kind="logarithmic"
    )
    signal = signal.ramp('both', RAMP_DURATION)
    return signal

# -------------------------------------------------------------------------
# MEASUREMENT OR LOADING
# -------------------------------------------------------------------------

def measure_hp_raw(signal, repeats=1):
    """
    Measure raw headphone impulse response using freefield.

    Parameters
    ----------
    signal : slab.Sound
        Test signal to be played.
    repeats : int
        Number of repeated measurements to average.

    Returns
    -------
    slab.Binaural
        Averaged binaural recording.
    """

    recs = []
    for _ in range(repeats):
        rec = freefield.play_and_record_headphones(
            speaker="both",
            sound=signal,
            compensate_delay=True,
            distance=0,
            equalize=False,
            recording_samplerate=fs, #* 2,
        )
        freefield.wait_to_finish_playing()
        recs.append(rec)
    return slab.Sound(data=numpy.mean(recs, axis=0))


# -------------------------------------------------------------------------
# EQUALIZATION FILTER
# -------------------------------------------------------------------------

def compute_headphone_equalization(recording, excitation, beta, n_samp_out=1024,
                                   show=False, save_path=None):
    """
    Compute a regularized, minimum-phase inverse filter for headphone equalization.

    Parameters
    ----------
    recording : slab.Sound
        Binaural recording of the played chirp.
    excitation : slab.Sound
        The original test chirp.
    show : bool
        Plot intermediate results (TF, inverse, equalized response).

    Returns
    -------
    eq_filter : pyfar.Signal
        Two-channel equalization filter.
    """

    # ------------------------------------------------------------------
    # Convert to pyfar.Signal
    # ------------------------------------------------------------------
    hp_raw = pyfar.Signal(recording.data.T, recording.samplerate)
    sig = pyfar.Signal(excitation.data.T, excitation.samplerate)

    # ------------------------------------------------------------------
    # Compute headphone transfer function: HpTF = recording / signal
    # ------------------------------------------------------------------
    signal_inv = pyfar.dsp.regularized_spectrum_inversion(
        sig, frequency_range=(60, HIGH_CUTOFF)
    )
    hp_ir = hp_raw * signal_inv

    # Normalize peak to 1
    hp_ir.time *= 1 / numpy.max(hp_ir.time)

    # Align IR to 10 ms
    onset = pyfar.dsp.find_impulse_response_start(hp_ir, threshold=20)
    shift_s = 0.01 - onset / hp_ir.sampling_rate
    hp_ir = pyfar.dsp.time_shift(hp_ir, shift_s, unit="s")

    if show:
        plt.figure()
        plt.title("Raw HpTF / IR")
        pyfar.plot.time_freq(hp_ir)

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    reg = pyfar.dsp.filter.low_shelf(
        pyfar.signals.impulse(hp_ir.n_samples), 60, 20, 2
    )
    reg = pyfar.dsp.filter.high_shelf(reg, 6000, 20, 2) * 0.1

    hp_inv_reg = pyfar.dsp.regularized_spectrum_inversion(
        hp_ir, (0, 20e3), regu_final=reg.freq * beta
    )

    # if show:
    #     plt.figure()
    #     plt.title("Regularized inverse")
    #     pyfar.plot.freq(hp_inv_reg)

    # ------------------------------------------------------------------
    # Minimum-phase + time windowing
    # ------------------------------------------------------------------
    hp_inv_reg = pyfar.dsp.minimum_phase(hp_inv_reg, truncate=False)

    # window
    hp_windowed = pyfar.dsp.time_window(
        hp_inv_reg,
        [0, n_samp_out - 1],
        shape="right",
        window='boxcar',
        crop='window'
    )
    # ------------------------------------------------------------------
    # Final diagnostic plot
    # ------------------------------------------------------------------
    if show or save_path is not None:
        ears = ['left', 'right']
        if save_path is not None and not show:
            # isolated Agg canvas: save-only, never touches the interactive backend
            from hrtf_relearning.hrtf.processing.edge_shift import _agg_figure
            fig, axes = _agg_figure(2, (12, 4))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (ax, ear) in enumerate(zip(axes, ears)):
            ax.set_title(f'{ear} ear')
            pyfar.plot.freq(reg, linestyle="--", label="Regularization", ax=ax)
            pyfar.plot.freq(hp_ir[i], label="HpTF", ax=ax)
            pyfar.plot.freq(hp_windowed[i], label="Inverse (regularized)", ax=ax)
            pyfar.plot.freq(pyfar.dsp.convolve(hp_ir[i], hp_windowed[i]), label="Equalized HpTF", ax=ax)
            ax.set_ylim(-25, 20)
            ax.legend(fontsize=8)
        fig.suptitle('HP equalization — raw HpTF vs regularized inverse vs equalized')
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
            logging.info(f"Saved HP calibration figure to: {save_path}")
        if show:
            plt.show()

    return hp_windowed


# -------------------------------------------------------------------------
# SAVE / LOAD CALIBRATION
# -------------------------------------------------------------------------

def save_hp_filter(eq_filter: pyfar.Signal, path: Path):
    """Save a pyfar headphone equalization filter as .npz.

    Stores ``filter`` with shape (n_channels, n_samples) and ``samplerate``.

    Parameters
    ----------
    eq_filter : pyfar.Signal
        Two-channel equalization filter.
    path : Path
        Output file path (should end in .npz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(
        path,
        filter=eq_filter.time.astype("float32"),   # (n_channels, n_samples)
        samplerate=numpy.array(eq_filter.sampling_rate),
    )
    logging.info(f"Saved HP equalization filter to: {path}")


def load_hp_filter(path: Path, output='pyfar') -> Filter | Signal:
    """Load a headphone equalization filter from .npz (or .wav as fallback).

    Parameters
    ----------
    path : Path
        Path to a ``.npz`` file written by :func:`save_hp_filter`.
        If the file does not exist but a ``.wav`` with the same stem does,
        that is loaded instead for backward compatibility.

    Returns
    -------
    pyfar.Signal or slab.Filter
        Two-channel equalization filter.
    """
    path = Path(path)
    if path.exists():
        npz = numpy.load(path, allow_pickle=False)
        if output == 'pyfar':
            return pyfar.Signal(npz["filter"], int(npz["samplerate"]))
        elif output == 'slab':
            return slab.Filter(data=npz["filter"], samplerate=int(npz["samplerate"]), fir='FIR')
    # backward-compat: try .wav with the same stem
    wav_path = path.with_suffix(".wav")
    if wav_path.exists():
        logging.warning(f"No .npz found at {path} – loading legacy .wav: {wav_path}")
        # return pyfar.io.read_audio(wav_path)
        return slab.Filter(data=slab.Sound(wav_path).data, samplerate=48842, fir='FIR')
    raise FileNotFoundError(f"HP filter not found: tried {path} and {wav_path}")


def save_hp_repeats(recordings, subject_id, hp_id, samplerate=fs):
    """Persist the raw repeated HP recordings so the repeatability plot can be
    regenerated later, at REC_DIR/<id>/<hp>_hp_repeats.npz (stores ``repeats``
    with shape (n_rep, n_samples, n_channels) and ``samplerate``)."""
    data = numpy.stack([numpy.asarray(r.data) for r in recordings], axis=0)
    out = paths.REC_DIR / subject_id / f"{hp_id}_hp_repeats.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(out, repeats=data.astype("float32"), samplerate=numpy.array(samplerate))
    logging.info(f"Saved {data.shape[0]} raw HP repeats to: {out}")
    return out


def save_hp_repeats_deviation(recordings, subject_id, hp_id, plot_dir=None,
                              samplerate=fs, xlim=(2000, 18000), num_fractions=6):
    """Save a repeatability QC figure to judge whether HP calibration worked.

    Overlays the (1/``num_fractions``-octave smoothed) magnitude spectra of the
    individual repeated HP recordings with their mean +/- 1 SD, per ear
    (per-repeat level offset removed so only spectral-shape scatter shows).
    Fractional-octave smoothing strips the measurement fine-structure so the SD
    reflects genuine placement/seal drift rather than bin noise. A large SD --
    especially in the 2-16 kHz pinna-cue band -- means the earphone moved
    between repeats, so the averaged calibration is unreliable and should be
    redone. Rendered on an isolated Agg canvas (save-only). Returns Path."""
    from hrtf_relearning.hrtf.processing.edge_shift import _agg_figure
    data = numpy.stack([numpy.asarray(r.data) for r in recordings], axis=0)  # (n_rep, n_samp, n_ch)
    n_rep, n_samp, n_ch = data.shape

    # 1/n-octave smoothed magnitude per repeat, per channel (pyfar)
    smoothed = []
    for r in range(n_rep):
        sig = pyfar.Signal(data[r].T, samplerate)                 # (n_ch, n_samp)
        sm = pyfar.dsp.smooth_fractional_octave(sig, num_fractions)[0]
        smoothed.append(20.0 * numpy.log10(numpy.maximum(numpy.abs(sm.freq), 1e-9)))
    freqs = sm.frequencies
    mags = numpy.stack(smoothed, axis=0)                          # (n_rep, n_ch, n_bins)
    band = (freqs >= xlim[0]) & (freqs <= xlim[1])

    if plot_dir is None:
        plot_dir = paths.subject_acoustic_dir(subject_id)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = _agg_figure(n_ch, (7 * n_ch, 5))
    for ear, ax in enumerate(axes):
        m = mags[:, ear, :]
        m = m - m[:, band].mean(axis=1, keepdims=True)           # remove per-repeat level offset
        mean = m.mean(0)
        sd = m.std(0, ddof=1) if n_rep > 1 else numpy.zeros_like(mean)
        for r in range(n_rep):
            ax.semilogx(freqs[band], m[r][band], color="0.7", lw=0.8)
        ax.semilogx(freqs[band], mean[band], color="#1f77b4", lw=1.6, label="mean")
        ax.fill_between(freqs[band], (mean - sd)[band], (mean + sd)[band],
                        color="#1f77b4", alpha=0.25, label="±1 SD")
        ax.set_title(f"{'left' if ear == 0 else 'right'} ear  (max SD {float(sd[band].max()):.1f} dB)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("magnitude (dB, level removed)")
        ax.set_xlim(*xlim)
        ax.legend(fontsize=8, loc="lower center")
    fig.suptitle(f"{subject_id} — {hp_id} HP repeatability across {n_rep} recordings "
                 f"(1/{num_fractions}-oct)")
    fig.tight_layout()
    out = plot_dir / f"{hp_id}_hp_repeats_deviation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved HP repeatability figure to: {out}")
    return out


def pyfar2wav(eq_filter: pyfar.Signal, path: Path):
    """Save a pyfar filter as WAV (legacy, kept for backward compatibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, eq_filter.time.T.astype("float32"), eq_filter.sampling_rate, subtype="FLOAT")
    logging.info(f"Saved HP equalization to: {path}")

def ff_equalization(eq_filter, hp_id, save_freefield=True):
    """
    Save equalization filter to a pickle file.

    Parameters
    ----------
    eq_filter : pyfar.Signal
        Equalization filter.
    path : Path
        Output file path.
    """
    # convert to slab filter
    filter = slab.Filter(
        data=eq_filter.time,
        samplerate=fs,
        fir="IR",
    )
    speakers = freefield.pick_speakers([0, 1])
    equalization = dict()
    equalization.update({f"{speakers[0].index}": {"level": 0, "filter": filter.channel(0)}})
    equalization.update({f"{speakers[1].index}": {"level": 0, "filter": filter.channel(1)}})

    if save_freefield:
        with open(freefield.DIR / 'data' / f'calibration_{hp_id}.pkl', 'wb') as f:  # save the newly recorded calibration
            pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)
        print(f"Writing calibration to {freefield.DIR / 'data' / f'calibration_{hp_id}.pkl'}")
    return filter

# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------

def calibrate_headphones(subject_id=SUB_ID, hp_id=HP_ID, n_rec=N_REC, show=True,
                         save_freefield=False, overwrite=False):
    """Measure (or load) the headphone equalization filter for a subject.

    If a saved filter exists it is loaded from disk, unless ``overwrite=True``,
    in which case the filter is always re-measured and the existing file is
    replaced.
    """
    save_path = paths.REC_DIR / subject_id / f"{hp_id}_equalization.npz"

    if not overwrite:
        try:
            # load existing calibration from disk
            hp_filter = load_hp_filter(save_path, 'slab')
            print(f'Loading hp filter from disk: {hp_id}_equalization.npz')
            return hp_filter
        except FileNotFoundError:
            pass  # fall through to measurement

    # measure a fresh calibration (no file on disk, or overwrite requested)
    # Initialize freefield for headphone playback
    if not freefield.PROCESSORS.mode == 'bi_play_rec':
        freefield.initialize("headphones", default="bi_play_rec")

    # Generate chirp
    excitation = generate_chirp()

    # Load or measure HpIR
    recordings = []
    for i in range(n_rec):
        input('Press Enter to record...')
        recordings.append(measure_hp_raw(excitation, repeats=1))
    recording = slab.Sound(data=numpy.mean(recordings, axis=0))

    # QC (save-only, never blocks calibration): persist the raw repeats and save
    # the across-repeat deviation figure so HP calibration reliability is on record.
    try:
        save_hp_repeats(recordings, subject_id, hp_id)
        save_hp_repeats_deviation(recordings, subject_id, hp_id)
    except Exception as exc:
        logging.warning(f"HP repeatability QC skipped: {exc}")

    # Compute equalization. Do NOT save the pyfar filter/inverse diagnostic here;
    # the calibration RESULT figure (re-recorded before/after EQ) is saved below.
    eq_filter = compute_headphone_equalization(recording, excitation, beta=0.01,
                                               show=False, save_path=None)
    # adjust beta parameter if necessary

    # Save to npz
    save_hp_filter(eq_filter, save_path)

    # test and alternatively save to freefield
    hp_filter = ff_equalization(eq_filter, hp_id, save_freefield)

    # Calibration RESULT figure: re-record the chirp without and with the EQ
    # filter and plot the before/after spectra. Saved to plots/acoustic/ (on
    # record), and shown if show=True. Guarded so a playback hiccup never fails
    # the calibration itself.
    try:
        raw = freefield.play_and_record_headphones(speaker='both', sound=excitation, equalize=False)
        filtered = hp_filter.apply(excitation)
        equalized = freefield.play_and_record_headphones(speaker='both', sound=filtered, equalize=False)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        raw.spectrum(axis=axes[0]);       axes[0].set_title('Raw HpTF')
        equalized.spectrum(axis=axes[1]); axes[1].set_title('Equalized HpTF')
        fig.suptitle(f'{hp_id} equalization — before vs after')
        out_dir = paths.subject_acoustic_dir(subject_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{hp_id}_hp_calibration.png", dpi=150, bbox_inches='tight')
        logging.info(f"Saved HP calibration result figure to: {out_dir / f'{hp_id}_hp_calibration.png'}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception as exc:
        logging.warning(f"HP calibration result plot skipped: {exc}")

    return hp_filter

# if __name__ == "__main__":
#     main()
