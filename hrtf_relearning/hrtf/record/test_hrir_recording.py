import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt
from hrtf_relearning.hrtf.record.record_hrir import record_hrir
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import calibrate_headphones
import numpy
import freefield
freefield.set_logger('info')
import slab
fs=48828
slab.set_default_samplerate(fs)
from hrtf_relearning import PATH as root
import copy

subject_id='AS_test'
reference_id = 'ref_02.04'
hp_id = 'MYSPHERE'  # headphone model
n_rec=3  # how often to re-place the headphones for hp calibration

def main():
    # --- record or load HRIR
    hrir = record_hrir(subject_id=subject_id, reference_id=reference_id, n_directions=1, overwrite=False,
                       align_interaural=False, expand_az=False, show=True)

    # --- put on headphones and calibrate
    hp_filter = calibrate_headphones(subject_id=subject_id, hp_id=hp_id, n_rec=n_rec, show=True, save_freefield=False)

    # load hp filter from disk
    hp_filter_data = slab.Sound(
        "C:/projects/hrtf_relearning/hrtf_relearning/data/hrtf/rec/AS_test/MYSPHERE_equalization.wav").data
    hp_filter = slab.Filter(
        data=hp_filter_data,
        samplerate=fs,
        fir="IR",
    )

    # --- generate test signal
    signal = slab.Sound.chirp(duration=1.0, level=70, samplerate=fs, kind='logarithmic',
                              from_frequency=200, to_frequency=18000)
    signal = signal.ramp(when="both", duration=0.01)

    # ---- run acoustic test
    acoustic_test(hrir, hp_filter, signal)

    # --- run behavioral test
    behavioral_test(hrir, hp_filter, signal)


def acoustic_test(hrir, hp_filter, signal):
    src_idx = hrir.cone_sources(0)
    src_idx.sort()

    # --- adjust loudness and apply hp filter
    spk_signal, hp_signal = copy.deepcopy(signal), copy.deepcopy(signal)
    hp_signal = hp_filter.apply(signal)  # order should not matter for IR filters are linear
    spk_signal.level = 85

    # --- play and record from headphones
    if not freefield.PROCESSORS.mode == 'bi_play_rec':
        freefield.initialize('headphones', default='bi_play_rec')
    input('Put on headphones and press Enter to continue...')
    hp_recordings = dict()
    for src in hrir.sources.vertical_polar[src_idx]:
        idx = hrir.get_source_idx(src[0], src[1])[0]
        filtered_signal = hrir.apply(idx, hp_signal)
        filtered_signal.level = 70
        hp_recordings[str(src)] = freefield.play_and_record_headphones(speaker='both', sound=filtered_signal, compensate_delay=True, distance=0,
                                                  compensate_attenuation=False, equalize=False, recording_samplerate=48828) # equalize=True

    # --- play and record from speakers
    freefield.initialize('dome', default='play_birec')
    input('Remove headphones and press Enter to continue..')
    dome_recordings = dict()
    for src in hrir.sources.vertical_polar[src_idx]:
        speaker = freefield.pick_speakers((src[0], src[1]))
        dome_recordings[str(src)] = freefield.play_and_record(speaker, spk_signal, compensate_delay=True,
                                                              compensate_attenuation=False,
                                                              equalize=True, recording_samplerate=48828)

    # plot
    fig, axes = plt.subplots(nrows=len(src_idx), ncols=2, figsize=(12, 12), layout='tight')
    for idx, (dome_rec, hp_rec) in enumerate(zip(list(dome_recordings.items()), list(hp_recordings.items()))):
        for col in range(2): # left and right ear
            dome_rec[1].channel(col).spectrum(axis=axes[idx, col])
            hp_rec[1].channel(col).spectrum(axis=axes[idx, col])
            axes[idx, col].set_title(f'{dome_rec[0]}°')
            fmin, fmax = 2e3, 18.2e3
            axes[idx, col].set_xlim(fmin, fmax)
            # 1/3 octave spacing
            ticks = 2 ** numpy.arange(numpy.log2(fmin),numpy.log2(fmax), 1)
            axes[idx, col].set_xticks(ticks)
            def format_khz(x):
                if x >= 1000:
                    return f"{int(x / 1000)}k"
                return str(int(x))
            axes[idx, col].set_xticklabels([format_khz(t) for t in ticks])
    plt.savefig(root / 'data' / 'results' / 'plot' / subject_id / 'hrir_test.svg')

def behavioral_test(hrir, hp_filter):
    """
    # ----- PARTICIPANT TESTING ----- #
    Use open HP to test if participants can tell the difference between loudspeakers and headphones
    """

    # todo test signal
    signal = slab.Sound.pinknoise(duration=0.5, samplerate=hrir.samplerate)

    # generate random sequence
    sequence = numpy.random.randint(0,2, 50)
    elevations = hrir.sources.vertical_polar[hrir.cone_sources(0)][:, 1]
    ele_idx = numpy.random.randint(0,len(elevations), 50)

    # init freefield
    if not freefield.PROCESSORS.mode == 'play_birec':
        freefield.initialize('dome', default='play_birec')

    # adjust loudness and apply hp filter
    spk_signal, hp_signal = copy.deepcopy(signal), copy.deepcopy(signal)
    hp_signal = hp_filter.apply(hp_signal)
    spk_signal.level = 80

    input('Put on headphones and press Enter to continue...')
    responses = []
    for i, ele_idx in zip(sequence, ele_idx):
        elevation = elevations[ele_idx]

        if i == 0: # play speaker
            speaker = freefield.pick_speakers((0, elevation))
            freefield.set_signal_and_speaker(spk_signal, speaker)
            freefield.play()

        elif i == 1: # play hp
            src_idx = hrir.get_source_idx(0, elevation)[0]
            filtered = hrir.apply(src_idx, hp_signal)  # hrir filter
            filtered.level = 70  # todo adjust level
            filtered.play()

        print(f'playing from {elevation} at {i}')
        response = input("Enter response (0 for speaker, 1 for headphones): ")
        print(response)
        responses.append(response)

    return[sequence, responses]

def equalize_loudness(signal,hp_id):
    # --- play and record from speaker
    freefield.initialize('dome', default='play_birec')
    speaker = freefield.pick_speakers((0, 0))
    speaker_rec = freefield.play_and_record(speaker, signal, compensate_delay=True, compensate_attenuation=False,
                                                          equalize=True, recording_samplerate=48828)
    speaker_level = speaker_rec.level

    # --- play and record from headphones

    freefield.initialize('headphones', default='bi_play_rec')
    freefield.load_equalization(freefield.DIR / 'data' / f'calibration_{hp_id}.pkl')
    hp_rec = freefield.play_and_record_headphones(speaker='both', sound=signal,  compensate_delay=True, distance=0,
                                                                   compensate_attenuation=False, equalize=True,
                                                                   recording_samplerate=48828)
    hp_level = hp_rec.level
    return numpy.diff((speaker_level, hp_level), axis=0)  # diff is about 19 on both channels

