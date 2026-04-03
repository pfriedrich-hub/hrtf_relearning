from hrtf_relearning.hrtf.record.record_hrir import record_hrir
from hrtf_relearning.hrtf.record.calibration import calibrate_headphones
import freefield
freefield.set_logger('info')
import slab
import numpy
fs=48828
slab.set_default_samplerate(fs)

subject_id='kemar_test'
reference = 'kemar_reference'
hp_id = 'MYSPHERE'

# --- signal
signal = slab.Sound.chirp(duration=1.0, level=85, samplerate=fs, kind='logarithmic',
                          from_frequency=200, to_frequency=18000)
signal = signal.ramp(when="both", duration=0.01)  # matches the cos ramp in bi_play_buf.rcx # todo

# load headphone filter
#wav = slab.Sound(data="data/sounds/MYSPHERE_1024.wav")
#filt = slab.Filter(data=wav.data)

# --- play and record from speaker
freefield.initialize('dome', default='play_birec')
speaker = freefield.pick_speakers((0, 0))
dome_rec = freefield.play_and_record(speaker, signal, compensate_delay=True, compensate_attenuation=False, equalize=False,
                    recording_samplerate=48828)
dome_rec.spectrum()


# --- play and record from headphones
freefield.initialize('headphones', default='bi_play_rec')
freefield.load_equalization(freefield.DIR / 'data' / 'calibration_MYSPHERE.pkl')
src_idx = hrir.get_source_idx(0,0)
#filtered_signal = filt.apply(hrir.apply(src_idx[0], signal))
hp_rec = freefield.play_and_record_headphones(speaker='both', sound=signal, compensate_delay=True, distance=0,
                                              compensate_attenuation=False, equalize=False, recording_samplerate=48828) # equalize=True
hp_rec.spectrum()




# ----- PARTICIPANT TESTING ----- #
"""
Use open HP to test if participants can tell the difference between loudspeakers and headhones
"""
# --- record HRIR across the central dome
hrir = record_hrir(subject_id=subject_id, reference_id=reference,
                   n_directions=1, overwrite=False, show=True)

# --- calibrate HP
calibrate_headphones(sub_id=subject_id, hp_id=hp_id, n_rec=3, show=True)


sequence = numpy.random.randint(0,1, 50)
elevations = numpy.random.randint(0,6, 50)

# init freefield
freefield.initialize('dome', default='play_birec')

# load hp filter
# wav = slab.Sound(data="data/sounds/MYSPHERE_1024.wav")
# hp_filt = slab.Filter(data=wav.data)

# todo empirically equalize loudness with open headphones (use kemar to calibrate levels)
ff_level = 85
hp_level = 75

for i, ele in zip(sequence, elevations):

    if i == 0: # play speaker
        speaker = freefield.pick_speakers((0, ele))
        to_play = signal
        to_play.level = ff_level
        freefield.set_signal_and_speaker()
        freefield.play()
    elif i == 1: # play hp
        src_idx = hrir.get_source_idx(0, ele)
        hrir_filtered = hrir.apply(src_idx, signal)
        filtered = hp_filt.apply(hrir_filtered)
        filtered.level = hp_level
        filtered.play()  # todo test if results in the same

    response = input("Enter response (0 for speaker, 1 for headphones): ")
