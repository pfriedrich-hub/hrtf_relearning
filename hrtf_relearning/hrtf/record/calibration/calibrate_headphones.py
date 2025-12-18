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
matplotlib.use("tkagg")
from matplotlib import pyplot as plt
import numpy
import pyfar
import slab
import freefield
import warnings
from pathlib import Path
import pickle
import hrtf_relearning
ROOT = Path(hrtf_relearning.__file__).resolve().parent
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)
freefield.set_logger("info")

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
hp_id = 'MYSPHERE'
save_path = Path.cwd() / 'data' / 'sounds' / f'{hp_id}_equalization.wav'
fs = 48828
slab.set_default_samplerate(fs)

# sweep parameters
LEVEL = 85
N_REC = 5
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

def measure_hp_raw(signal, repeats=5):
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
    # Initialize freefield for headphone playback
    if not freefield.PROCESSORS.mode == 'bi_play_rec':
        freefield.initialize("headphones", default="bi_play_rec")
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

def compute_headphone_equalization(recording, excitation, beta, show=False):
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

    if show:
        plt.figure()
        plt.title("Regularized inverse")
        pyfar.plot.freq(hp_inv_reg)

    # ------------------------------------------------------------------
    # Minimum-phase + time windowing
    # ------------------------------------------------------------------
    hp_inv_reg = pyfar.dsp.minimum_phase(hp_inv_reg, truncate=False)

    # ------------------------------------------------------------------
    # Final diagnostic plot
    # ------------------------------------------------------------------
    if show:
        fig, ax = plt.subplots()
        pyfar.plot.freq(reg, linestyle="--", label="Regularization", ax=ax)
        pyfar.plot.freq(hp_ir[0], label="HpTF")
        pyfar.plot.freq(hp_inv_reg[0], label="Inverse (regularized)")
        pyfar.plot.freq(
            pyfar.dsp.convolve(hp_ir[0], hp_inv_reg[0]),
            label="Equalized HpTF"
        )
        ax.set_ylim(-25, 20)
        plt.legend()
        plt.show()

    return hp_inv_reg


# -------------------------------------------------------------------------
# SAVE CALIBRATION
# -------------------------------------------------------------------------
def pyfar2wav(eq_filter, path: Path):
    """
    Save a pyfar filter as WAV (for pyBinSim).

    Parameters
    ----------
    eq_filter : pyfar.Signal
        Filter to save.
    path : Path
        Output file path ending in .wav
    """
    path = str(ROOT / 'data' / 'sounds' / f'{hp_id}_equalization.wav')
    pyfar.io.write_audio(eq_filter, path, overwrite=True)  #todo test this
    print(f"Saved WAV filter to: {path}")

def save_equalization(eq_filter):
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
        data=eq_filter.time.T,
        samplerate=fs,
        fir="IR",
    )
    speakers = freefield.pick_speakers([0, 1])
    equalization = dict()
    equalization.update({f"{speakers[0].index}": {"level": 0, "filter": filter.channel(1)}})
    equalization.update({f"{speakers[1].index}": {"level": 0, "filter": filter.channel(0)}})

    freefield_path = freefield.DIR / 'data'
    equalization_path = freefield_path / f'calibration_{hp_id}.pkl'
    with open(equalization_path, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)
    print(f"Writing equalization to {equalization_path}")



# -------------------------------------------------------------------------
# MAIN PIPELINE
# # -------------------------------------------------------------------------

if __name__ == "__main__":

    # Generate chirp
    excitation = generate_chirp()

    # Load or measure HpIR
    recording = measure_hp_raw(excitation, repeats=N_REC)
    # recording = slab.Binaural.read(Path.cwd() / 'hrtf/record/calibration/hp_raw.wav')

    # Compute equalization
    eq_filter = compute_headphone_equalization(recording, excitation, beta=0.2, show=True)

    # Save filter
    pyfar2wav(eq_filter, save_path)
    save_equalization(eq_filter)

    # test
    freefield.load_equalization(freefield.DIR / 'data' / f'calibration_{hp_id}.pkl')
    equalized = freefield.play_and_record_headphones(speaker='both', sound=excitation, equalize=True)
    equalized.spectrum()

