"""
Record KEMAR HRIR, play back and record via Headphones, compare to initial recording
to validate HRIR recording and processing.
"""

#todo test this with kemar mics and pirates

from hrtf_relearning.hrtf.record.deprecated.record_hrir_old import *
import freefield
freefield.set_logger('info')
import slab
fs=48828
slab.set_default_samplerate(fs)
subject_id='kemar_test'
reference = 'kemar_reference'
n_directions=1
n_recordings=5
overwrite=False
n_samples_out=256
show=True

# --- record KEMAR HRIR across the central dome
hrir = record_hrir(subject_id, reference, n_directions, n_recordings, fs, overwrite, n_samples_out, show)

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