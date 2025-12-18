import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt
import logging
import freefield
import slab
import numpy
import pyfar
import pickle
from copy import deepcopy
import hrtf_relearning
from pathlib import Path
ROOT = Path(hrtf_relearning.__file__).resolve().parent

# ------------------------ CONFIG ------------------------

SPEAKERS = 'center'          # 'full' or 'center'
FS = 48828
FILTER_LENGTH = 1024
LOW_FREQ = 20
HIGH_FREQ = FS/2
N_REPEATS = 3
BETA = 0.1
SHOW = False                 # plotting disabled for now

OUTPUT_FILE = freefield.DIR / 'data' / 'calibration_dome.pkl'

# ------------------------ SETUP ------------------------

def initialize_dome():
    freefield.initialize("dome", default="play_rec")
    freefield.set_logger("info")
    slab.set_default_samplerate(FS)
    return FS


def select_speakers():
    if SPEAKERS == 'full':
        speaker_idx = numpy.arange(0, 46).tolist()
    elif SPEAKERS == 'center':
        speaker_idx = [19, 20, 21, 22, 23, 24, 25, 26, 27]
    else:
        raise ValueError('SPEAKERS must be "full" or "center"')

    speakers = freefield.pick_speakers(speaker_idx)
    if not speakers:
        raise RuntimeError("No speakers selected.")
    return speakers


# ------------------------ EXCITATION ------------------------

def make_excitation():
    duration = 1.0
    ramp = duration / 50

    sig = slab.Sound.chirp(
        duration=duration,
        level=80,
        from_frequency=LOW_FREQ,
        to_frequency=FS / 2,
        kind="logarithmic",
    )

    return sig.ramp(duration=ramp)


# ------------------------ LEVEL MEASUREMENT (PASS 1) ------------------------

def measure_speaker_level(speaker, excitation):
    levels = []
    logging.info(f'Playing from {speaker}')
    for _ in range(N_REPEATS):
        rec = freefield.play_and_record(
            speaker=speaker,
            sound=excitation,
            compensate_delay=True,
            equalize=False,
            recording_samplerate=FS * 2,  # TDT workaround
        )
        levels.append(rec.level)

    return numpy.mean(levels)


def compute_level_equalization(levels):
    """
    Compute relative level corrections (old dome logic).
    """
    ref = numpy.mean(list(levels.values()))
    return {spk: ref - lvl for spk, lvl in levels.items()}


# ------------------------ IR MEASUREMENT (PASS 2) ------------------------

def measure_speaker_ir(speaker, excitation, level_correction):
    """
    Measure speaker IR with level equalization applied at playback.
    """
    attenuated = deepcopy(excitation)
    attenuated.level += level_correction

    recs = []

    for _ in range(N_REPEATS):
        rec = freefield.play_and_record(
            speaker=speaker,
            sound=attenuated,
            compensate_delay=True,
            equalize=False,
            recording_samplerate=FS * 2,  # TDT workaround
        )
        recs.append(rec.data)

    return slab.Sound(
        data=numpy.mean(recs, axis=0),
        samplerate=FS
    )


# ------------------------ FILTER CONSTRUCTION ------------------------

def build_inverse_filter(recording, excitation):
    speaker_raw = pyfar.Signal(recording.data.T, recording.samplerate)
    sig = pyfar.Signal(excitation.data.T, excitation.samplerate)

    # ------------------------------------------------------------------
    # Compute headphone transfer function: HpTF = recording / signal
    # ------------------------------------------------------------------
    signal_inv = pyfar.dsp.regularized_spectrum_inversion(
        sig, frequency_range=(LOW_FREQ, 20e3),
    )
    speaker_ir = speaker_raw * signal_inv

    # Normalize peak to 1
    speaker_ir.time *= 1 / numpy.max(speaker_ir.time)

    # Align IR to 10 ms
    onset = pyfar.dsp.find_impulse_response_start(speaker_ir, threshold=10)
    shift_s = 0.01 - onset / speaker_ir.sampling_rate
    speaker_ir = pyfar.dsp.time_shift(speaker_ir, shift_s, unit="s")

    # if show:
    #     plt.figure()
    #     plt.title("Raw Speaker TF / IR")
    #     pyfar.plot.time_freq(speaker_ir)

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    reg = pyfar.dsp.filter.low_shelf(
        pyfar.signals.impulse(speaker_ir.n_samples), 60, 20, 2
    )
    reg = pyfar.dsp.filter.high_shelf(reg, 6000, 20, 2) * 0.1

    speaker_inv_reg = pyfar.dsp.regularized_spectrum_inversion(
    speaker_ir, (LOW_FREQ, 20e3), regu_final=reg.freq * BETA)

    # if show:
    #     plt.figure()
    #     plt.title("Regularized inverse")
    #     pyfar.plot.freq(speaker_inv_reg)

    # ------------------------------------------------------------------
    # Minimum-phase + time windowing
    # ------------------------------------------------------------------
    eq_filter = pyfar.dsp.minimum_phase(speaker_inv_reg, truncate=False)

    eq_filter = pyfar.dsp.time_window(eq_filter, [0, 256 - 1],
                                      shape="right", window='boxcar', crop='window')

    # ------------------------------------------------------------------
    # Final diagnostic plot
    # ------------------------------------------------------------------
    # if show:
    #     fig, ax = plt.subplots()
    #     pyfar.plot.freq(reg, linestyle="--", label="Regularization", ax=ax)
    #     pyfar.plot.freq(speaker_ir, label="HpTF")
    #     pyfar.plot.freq(speaker_inv_reg
    #     , label="Inverse (regularized)")
    #     pyfar.plot.freq(
    #         pyfar.dsp.convolve(speaker_ir, eq_filter),
    #         label="Equalized Speaker TF"
    #     )
    #     ax.set_ylim(-25, 20)
    #     plt.legend()
    #     plt.show()

    filter = slab.Filter(
        data=eq_filter.time.T,
        samplerate=FS,
        fir="IR")

    return filter


# ------------------------ MAIN CALIBRATION ------------------------

def main():
    initialize_dome()
    speakers = select_speakers()
    excitation = make_excitation()

    # ---------- PASS 1: level measurement ----------
    print("\n--- Measuring speaker levels ---")
    levels = {
        spk.index: measure_speaker_level(spk, excitation)
        for spk in speakers
    }

    level_eq = compute_level_equalization(levels)

    # ---------- PASS 2: frequency equalization ----------
    print("\n--- Measuring IRs and building filters ---")
    calibration = {}

    for spk in speakers:
        print(f"Calibrating speaker {spk}")

        recording = measure_speaker_ir(
            spk,
            excitation,
            level_eq[spk.index]
        )

        filt = build_inverse_filter(recording, excitation)

        calibration.update({f'{spk.index}': {"level": level_eq[spk.index], "filter": filt}})


    # ---------- SAVE ----------
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(calibration, f)

    print(f"\nCalibration saved to {OUTPUT_FILE}")

    # test
    freefield.load_equalization(OUTPUT_FILE)
    recs = []
    for spk in speakers:
        rec = freefield.play_and_record(
            speaker=spk,
            sound=excitation,
            compensate_delay=True,
            equalize=True,
            recording_samplerate=FS * 2,  # TDT workaround
        )
        rec.spectrum()
        plt.title(f'Speaker {spk.index}')
        recs.append(rec.data)


# if __name__ == "__main__":
#     main()
