"""
First-session pipeline
----------------------
1. Record (or load) HRIR
2. Calibrate headphones  (mics still in)
3. Acoustic sanity check (dome speaker vs HRIR rendering, spectrum comparison)
4. Dome localization     (real speakers, vertical midline)
5. Virtual localization  (pybinsim, same locations, independent randomisation)
6. Comparison plots      (dome vs virtual, side by side)
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import copy
import logging
import freefield
import slab
from hrtf_relearning.experiment.Localization.Localization_AR import Localization
from hrtf_relearning import PATH as ROOT
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.hrtf.record.record_hrir import record_hrir
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import calibrate_headphones, load_hp_filter
from hrtf_relearning.experiment.Localization.Localization_dome import LocalizationDome
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy, plot_localization, plot_elevation_response,
)

# --- session defaults (override via main() arguments) ---
subject_id   = 'NKa'
hp_id        = 'MYSPHERE'
reference_id = 'ref_03.04'
n_directions = 3  # directions for the hrir recording
n_recordings = 10  #
fs           = 48828
hp_freq      = 120
n_rec_hp     = 3
hrir_settings= None
show = True

slab.set_default_samplerate(fs)
freefield.set_logger('info')


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main(subject_id, reference_id, hp_id, hrir_settings,
         n_directions, n_recordings, n_rec_hp=3, show=True):
    """
    Full first-session pipeline.

    Parameters
    ----------
    subject_id : str
    reference_id : str
    hp_id : str
        Headphone model (e.g. 'MYSPHERE', 'DT990').
    hrir_settings : dict, optional
        Passed to hrtf2binsim. Defaults to binaural, no spectral modification.
    n_directions : int
        Head-tilt directions for HRIR recording.
    n_recordings : int
        Sweeps per speaker for HRIR recording.
    n_rec_hp : int
        HP calibration recordings (headphone re-placement repetitions).
    show : bool
        Show intermediate plots.
    """

    # ------------------------------------------------------------------
    # 1. Record / load HRIR
    # ------------------------------------------------------------------
    logging.info('--- Step 1: HRIR recording ---')
    hrir = record_hrir(
        subject_id   = subject_id,
        reference_id = reference_id,
        n_directions = n_directions,
        n_recordings = n_recordings,
        fs           = fs,
        hp_freq      = hp_freq,
        show         = show,
        overwrite = False ,
    )

    # ------------------------------------------------------------------
    # 2. HP calibration  (in-ear mics still in place)
    # ------------------------------------------------------------------
    logging.info('--- Step 2: HP calibration ---')
    # hp_id = 'MYSPHERE'
    logging.warning('--------- Check HP Jack and ID ---------')
    try:
        # alternatively load from disk
        hp_filter = load_hp_filter(ROOT / 'data' / 'hrtf' / 'rec' / subject_id / f'{hp_id}_equalization.npz','slab')
        print(f'Loading hp filter from disk: {hp_id}_equalization.npz')
    except FileNotFoundError:
        hp_filter = calibrate_headphones(
            subject_id    = subject_id,
            hp_id         = hp_id,
            n_rec         = n_rec_hp,
            show          = show,
            save_freefield = False,
        )

    # ------------------------------------------------------------------
    # 3. Acoustic sanity check
    # ------------------------------------------------------------------
    logging.info('--- Step 3: Acoustic test ---')
    acoustic_test(hrir, hp_filter, subject_id=subject_id, hp_id=hp_id, show=show)

# ---------------------------------------------------------------------
# Acoustic test
# ---------------------------------------------------------------------

def acoustic_test(hrir, hp_filter, subject_id, hp_id, show=True):
    """
    Compare real loudspeaker recordings against HRIR headphone renderings.

    Plays a log-chirp from every third vertical-midline speaker and records
    binaurally via the in-ear mics — once from the dome (remove headphones)
    and once via HP+HRIR (put headphones on). Overlays spectra per source.
    """
    fs = hrir.samplerate
    signal = slab.Sound.chirp(
        duration=1.0, level=70, samplerate=fs,
        kind='logarithmic', from_frequency=200, to_frequency=18000,
    )
    signal = signal.ramp(when='both', duration=0.01)

    src_idx = hrir.cone_sources(0)[::3]
    src_idx.sort()

    # --- headphones ---
    if freefield.PROCESSORS.mode != 'bi_play_rec':
        freefield.initialize('headphones', default='bi_play_rec')

    hp_signal = hp_filter.apply(signal)
    input('Put on headphones and press Enter to continue...')
    hp_recordings = {}
    for src in hrir.sources.vertical_polar[src_idx]:
        idx = hrir.get_source_idx(src[0], src[1])[0]
        filtered = hrir.apply(idx, hp_signal)
        filtered.level = 65
        hp_recordings[str(src)] = freefield.play_and_record_headphones(
            speaker='both', sound=filtered, compensate_delay=True, distance=0,
            compensate_attenuation=False, equalize=False, recording_samplerate=fs,
        )

    # --- dome speakers ---
    freefield.initialize('dome', default='play_birec')
    spk_signal = copy.deepcopy(signal)
    spk_signal.level = 85
    input('Remove headphones and press Enter to continue...')
    dome_recordings = {}
    for src in hrir.sources.vertical_polar[src_idx]:
        speaker = freefield.pick_speakers((src[0], src[1]))
        dome_recordings[str(src)] = freefield.play_and_record(
            speaker, spk_signal, compensate_delay=True,
            compensate_attenuation=False, equalize=True, recording_samplerate=fs,
        )

    if show:
        fmin, fmax = 2e3, 18.2e3
        ticks = 2 ** numpy.arange(numpy.log2(fmin), numpy.log2(fmax), 1)
        fig, axes = plt.subplots(
            nrows=len(src_idx), ncols=2, figsize=(12, 3 * len(src_idx)), layout='tight'
        )
        fig.suptitle(f'{subject_id} — acoustic test ({hp_id})')
        for row, (dome_item, hp_item) in enumerate(
            zip(dome_recordings.items(), hp_recordings.items())
        ):
            for col in range(2):
                ax = axes[row, col]
                dome_item[1].channel(col).spectrum(axis=ax)
                hp_item[1].channel(col).spectrum(axis=ax)
                ax.set_title(f'{dome_item[0]}° — {"L" if col == 0 else "R"}')
                ax.set_xlim(fmin, fmax)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{int(t/1000)}k" if t >= 1000 else str(int(t)) for t in ticks]
                )
                ax.legend(['Dome', 'HP+HRIR'])

        save_dir = ROOT / 'data' / 'results' / 'plot' / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'acoustic_test_{hp_id}.svg')
        plt.show()