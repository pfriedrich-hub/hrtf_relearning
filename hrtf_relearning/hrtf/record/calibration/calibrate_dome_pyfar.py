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
ROOT = hrtf_relearning.PATH

# ------------------------ CONFIG ------------------------

SPEAKERS = 'center'
FS = 48828
FILTER_LENGTH = 1024
LOW_FREQ = 20
HIGH_FREQ = FS / 2
N_REPEATS = 3
BETA = 0.1
SHOW = False

OUTPUT_FILE = ROOT / 'hrtf' / 'record' / 'calibration' / 'calibration_dome.pkl'

# ------------------------ SETUP ------------------------

def initialize_dome():
    freefield.initialize("dome", default="play_rec")
    freefield.set_logger("info")
    slab.set_default_samplerate(FS)


def select_speakers():
    if SPEAKERS == 'full':
        speaker_idx = numpy.arange(0, 46).tolist()
    elif SPEAKERS == 'center':
        speaker_idx = [19, 20, 21, 22, 23, 24, 25, 26, 27]
    else:
        raise ValueError

    return freefield.pick_speakers(speaker_idx)


# ------------------------ EXCITATION ------------------------

def make_excitation():
    sig = slab.Sound.chirp(
        duration=1.0,
        level=80,
        from_frequency=LOW_FREQ,
        to_frequency=FS / 2,
        kind="logarithmic",
    )
    return sig.ramp(duration=0.02)


# ------------------------ RECORD ONCE ------------------------

def record_speakers(
    speakers,
    excitation,
    equalize=False,
    calibration=None,
):
    recordings = {}

    for spk in speakers:
        idx = spk.index
        logging.info(f"Recording speaker {idx}")

        recs = []

        for _ in range(N_REPEATS):
            sig = excitation

            if equalize:
                if calibration is None:
                    raise ValueError("Calibration required when equalize=True")

                sig = deepcopy(excitation)

                # apply level
                sig.level += calibration[idx]["level"]

                # apply FIR
                sig = calibration[idx]["filter"].apply(sig)

            rec = freefield.play_and_record(
                speaker=spk,
                sound=sig,
                compensate_delay=True,
                equalize=False,              # IMPORTANT
                recording_samplerate=FS  # TDT workaround
            )

            recs.append(rec.data)

        recordings[idx] = slab.Sound(
            data=numpy.mean(recs, axis=0),
            samplerate=FS
        )

    return recordings



# ------------------------ LEVEL EQUALIZATION ------------------------

def compute_level_equalization(recordings):
    """
    Compute relative level offsets from recordings.
    """
    levels = {
        idx: rec.level
        for idx, rec in recordings.items()
    }

    ref = numpy.mean(list(levels.values()))
    return {idx: ref - lvl for idx, lvl in levels.items()}

def update_levels_from_recordings(
    recordings,
    calibration,
    reference="mean",
):
    """
    Update level corrections in-place based on verified recordings.

    Parameters
    ----------
    recordings : dict
        {speaker_idx: slab.Sound}
        Recordings made with current calibration applied.
    calibration : dict
        {speaker_idx: {"level": float, "filter": slab.Filter}}
    reference : {"mean", "median"}

    Returns
    -------
    residuals : dict
        {speaker_idx: residual_level_correction}
    """
    levels = {idx: rec.level for idx, rec in recordings.items()}

    if reference == "mean":
        ref = numpy.mean(list(levels.values()))
    elif reference == "median":
        ref = numpy.median(list(levels.values()))
    else:
        raise ValueError("reference must be 'mean' or 'median'")

    residuals = {}
    for idx, lvl in levels.items():
        delta = ref - lvl
        calibration[idx]["level"] += delta
        residuals[idx] = delta

    return residuals

# ------------------------ FILTER CONSTRUCTION ------------------------

def build_inverse_filter(recording, excitation):
    speaker_raw = pyfar.Signal(recording.data.T, recording.samplerate)
    sig = pyfar.Signal(excitation.data.T, excitation.samplerate)

    # compute speaker TF
    sig_inv = pyfar.dsp.regularized_spectrum_inversion(
        sig, frequency_range=(LOW_FREQ, HIGH_FREQ))
    speaker_ir = speaker_raw * sig_inv  # convolution

    speaker_ir.time /= numpy.max(numpy.abs(speaker_ir.time))  # normalize

    # force causality (time shift)
    onset = pyfar.dsp.find_impulse_response_start(speaker_ir, threshold=10)
    shift_s = 0.01 - onset / speaker_ir.sampling_rate
    speaker_ir = pyfar.dsp.time_shift(speaker_ir, shift_s, unit="s")

    # regularize
    reg = pyfar.dsp.filter.low_shelf(
        pyfar.signals.impulse(speaker_ir.n_samples), 60, 20, 2)
    reg = pyfar.dsp.filter.high_shelf(reg, 6000, 20, 2) * 0.1
    inv = pyfar.dsp.regularized_spectrum_inversion(  #
        speaker_ir,
        frequency_range=(LOW_FREQ, HIGH_FREQ),
        regu_final=reg.freq * BETA)

    # ensure minimum phase
    eq = pyfar.dsp.minimum_phase(inv, truncate=False)
    # crop
    eq = pyfar.dsp.time_window(eq, [0, FILTER_LENGTH - 1],
                               shape="right", window="boxcar", crop="window")

    return slab.Filter(eq.time.T, samplerate=FS, fir="IR")


# ------------------------ MAIN ------------------------

def main():
    initialize_dome()
    speakers = select_speakers()
    excitation = make_excitation()

    # --------------------------------------------------
    # PASS A: RAW RECORDING (no EQ)
    # --------------------------------------------------
    raw_recordings = record_speakers(
        speakers,
        excitation,
        equalize=False,
    )

    # --------------------------------------------------
    # OFFLINE DESIGN
    # --------------------------------------------------
    level_eq = compute_level_equalization(raw_recordings)

    calibration = {}
    for spk in speakers:
        idx = spk.index
        filt = build_inverse_filter(raw_recordings[idx], excitation)
        calibration[idx] = {
            "level": level_eq[idx],
            "filter": filt,
        }

    # --------------------------------------------------
    # PASS B: VERIFICATION (explicit EQ)
    # --------------------------------------------------
    verified_recordings = record_speakers(
        speakers,
        excitation,
        equalize=True,
        calibration=calibration,
    )

    # --------------------------------------------------
    # UPDATE LEVELS (NEW, INTEGRATED)
    # --------------------------------------------------
    residuals = update_levels_from_recordings(
        verified_recordings,
        calibration,
        reference="mean",
    )

    # --------------------------------------------------
    # OPTIONAL: SECOND VERIFICATION PASS
    # --------------------------------------------------
    verified_recordings_2 = record_speakers(
        speakers,
        excitation,
        equalize=True,
        calibration=calibration,
    )

    if SHOW:  # inspect
        fig, axis = plt.subplots(1, 1)

        for idx, rec in verified_recordings_2.items():
            rec.spectrum(axis=axis)

            # catch the last plotted line
            line = axis.lines[-1]
            line.set_label(f"Speaker {idx}")

        axis.legend()
        plt.show()

    # --------------------------------------------------
    # SAVE FOR FREEFIELD (ONLY NOW)
    # --------------------------------------------------
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(calibration, f)
    with open(freefield.DIR / 'data' / 'calibration_dome.pkl', "wb") as f:
        pickle.dump(calibration, f)

    print(f"Calibration verified and saved to {OUTPUT_FILE}")


# 
# if __name__ == "__main__":
#     main()
