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
import hrtf_relearning
ROOT = Path(hrtf_relearning.__file__).resolve().parent

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
hp_id = 'MYSPHERE'  # MYSPHERE, DT990

fs = 48828
slab.set_default_samplerate(fs)

# sweep parameters
LEVEL = 85
N_REC = 5
LOW_CUTOFF = 20
HIGH_CUTOFF = fs/2
CHIRP_DURATION = 1.0
RAMP_DURATION = .001  # 1ms ramp

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
    signal.level = 75
    return signal

# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------

def main():

    # Generate chirp
    excitation = generate_chirp()
    kemar = slab.HRTF(ROOT / 'data' / 'hrtf' / 'sofa' / 'kemar_test.sofa')

    elevations = [37.5, 25, 12.5, 0, -12.5, -25, -37.5]
    az = 0

    # Record Headphones
    if not freefield.PROCESSORS.mode == 'bi_play_rec':
        freefield.initialize("headphones", default="bi_play_rec")
        freefield.load_equalization(freefield.DIR / 'data' / f'calibration_{hp_id}.pkl')
    spatialized = []
    for el in elevations:
        source_idx = kemar.get_source_idx(az, el)[0]
        spatial_signal = kemar.apply(source_idx, excitation)
        spatialized.append(freefield.play_and_record_headphones(speaker='both',
                                                                sound=spatial_signal, equalize=True))


    # Record Loudspeakers
    if not freefield.PROCESSORS.mode == 'play_birec':
        freefield.initialize("dome", default="play_birec")
        freefield.load_equalization(freefield.DIR / 'data' / f'calibration_dome.pkl')
    speaker_rec = []
    for el in elevations:
        speaker = freefield.pick_speakers((0, el))
        speaker_rec.append(freefield.play_and_record(speaker=speaker,
                                                     sound=excitation.channel(0), equalize=True))

    # plot
    # convert to  Signal for plotting
    for i in range(len(elevations)):
        spatialized[i] = pyfar.Signal(spatialized[i].data.T, fs)
        speaker_rec[i] = pyfar.Signal(speaker_rec[i].data.T, fs)

    fig, axes = plt.subplots(len(elevations), 2, sharex=True,
                             sharey=True, figsize=(12, 24), tight_layout=True)
    axes[0, 0].set_title(f'Left Ear')
    axes[0, 1].set_title(f'Right Ear')

    for i, el in enumerate(elevations):
        ax = axes[i]
        pyfar.plot.freq(spatialized[i][0], ax=ax[0])
        pyfar.plot.freq(speaker_rec[i][0], ax=ax[0])
        pyfar.plot.freq(spatialized[i][1], ax=ax[1])
        pyfar.plot.freq(speaker_rec[i][1], ax=ax[1])
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[0].annotate(
            f"Elevation {el:.1f}°",
            xy=(0.02, 1.1),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
            weight="bold"
        )
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_xlabel('Frequency (Hz)')

    plt.savefig(ROOT/'data' / 'img' / 'processing' / 'kemar_hrir_test.svg')

# if __name__ == "__main__":
#     main()
