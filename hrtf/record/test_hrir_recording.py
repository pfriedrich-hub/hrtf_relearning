"""
Record KEMAR HRIR, play back and record via Headphones, compare to initial recording
to validate HRIR recording and processing.
"""

#todo test this with kemar mics and pirates

from hrtf.record.record_hrir import *
import freefield
freefield.set_logger('info')
import slab
fs=48828
slab.set_default_samplerate(fs)

# --- record KEMAR HRIR across the central dome
hrir = record_hrir(subject_id='kemar_test', reference = 'kemar_reference', n_directions=1, n_recordings=5,
    fs=fs, overwrite=True, n_samples_out=256, show=True) #todo step by step

# --- signal
signal = slab.Sound.chirp(duration=1.0, level=85, samplerate=fs, kind='logarithmic',
                          from_frequency=200, to_frequency=18000)
signal = signal.ramp(when="both", duration=0.01)  # matches the cos ramp in bi_play_buf.rcx # todo

# --- play and record from speaker
freefield.initialize('dome', default='play_birec')
speaker = freefield.pick_speakers((0, 0))
dome_rec = freefield.play_and_record(speaker, signal, compensate_delay=True, compensate_attenuation=False, equalize=False,
                    recording_samplerate=48828)
dome_rec.spectrum()


# --- play and record from headphones
freefield.initialize('headphones', default='bi_play_rec')
src_idx = hrir.get_source_idx(0,0)
filtered_signal = hrir.apply(src_idx, signal)
hp_rec = freefield.play_and_record_headphones(speaker='both', sound=filtered_signal, compensate_delay=True, distance=0, compensate_attenuation=False,
                               equalize=True, recording_samplerate=48828)
hp_rec.spectrum()