import freefield
import slab
import time
import numpy
from matplotlib import pyplot as plt
freefield.initialize('dome', default='play_rec')  # initialize setup
speaker_id = 23
[speaker] = freefield.pick_speakers(speaker_id)
signal = slab.Sound.chirp(duration=3.0, level=90)
rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
          compensate_attenuation=False, equalize=False)

# equalization

"""
equalize_speakers(speakers="all", reference_speaker=23, bandwidth=1 / 10, threshold=80,
                  low_cutoff=200, high_cutoff=16000, alpha=1.0, file_name=None):
                  
Equalize the loudspeaker array in two steps. First: equalize over all
level differences by a constant for each speaker. Second: remove spectral
difference by inverse filtering. For more details on how the
inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
"""

freefield.initialize('dome', default='play_rec')  # initialize setup

freefield.set_logger('warning')
speakers = [19,20,21,22,23,24,25,26,27]
low_cutoff=200
high_cutoff=16000
reference_speaker=23

threshold = 10
# change: set sound duration to 0.1 instead of 0.05 - lower frequency detection limited to 10 hz instead of 20 hz
sound = slab.Sound.chirp(duration=0.1, from_frequency=low_cutoff, to_frequency=high_cutoff)
if speakers == "all":  # use the whole speaker table
    speakers = freefield.read_speaker_table()
else:
    speakers = freefield.pick_speakers(picks=speakers)
reference_speaker = freefield.pick_speakers(reference_speaker)[0]

# step 1: level equalization
"""
equalization_levels = freefield._level_equalization(speakers, sound, reference_speaker, threshold)

Record the signal from each speaker in the list and return the level of each
speaker relative to the target speaker(target speaker must be in the list)
"""
# record from reference speaker
# proposed change I: average over 20 recordings, filter and ramp each recording
# filt = slab.Filter.band(frequency=(200, 16000), kind='bp')  # do this in the rcx file
temp_recs = []
for i in range(20):
    rec = freefield.play_and_record(reference_speaker, sound, equalize=False)
    # rec = filt.apply(rec)
    rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target_recording = slab.Sound(data=numpy.mean(temp_recs, axis=0))

# record from each speaker in the list
recordings = []
for speaker in speakers:
    temp_recs = []
    for i in range(20):
        rec = freefield.play_and_record(speaker, sound, equalize=False)
        # rec = filt.apply(rec)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)

# recordings.data[:, recordings.level < threshold] = target_recording.data  # old way of thresholding
# proposed change II:  thresholding: correct level on speakers which deviate more than <threshold> dB from reference
level_threshold = 0.5  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
recordings.data[:, numpy.logical_and(recordings.level > target_recording.level-level_threshold,
                                     recordings.level < target_recording.level+level_threshold)] = target_recording.data
equalization_levels = target_recording.level / recordings.level

# step 2: frequency equalization
"""

_frequency_equalization(speakers, sound, reference_speaker, calibration_levels, bandwidth,
                        low_cutoff, high_cutoff, alpha, threshold):
                        
play the level-equalized signal, record and compute and a bank of inverse filter
to equalize each speaker relative to the target one. Return filterbank and recordings
"""
from copy import deepcopy
bandwidth=1 / 10
alpha=1.0
# filt = slab.Filter.band(frequency=(200, 16000), kind='bp')

temp_recs = []
for i in range(20):
    rec = freefield.play_and_record(reference_speaker, sound, equalize=False)
    # rec = filt.apply(rec)
    rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target_recording = slab.Sound(data=numpy.mean(temp_recs, axis=0))
# reference = freefield.play_and_record(reference_speaker, sound, equalize=False)

recordings = []
for speaker, level in zip(speakers, equalization_levels):
    attenuated = deepcopy(sound)
    attenuated.level *= level
    temp_recs = []
    for i in range(20):
        rec = freefield.play_and_record(speaker, attenuated, equalize=False)
        # rec = filt.apply(rec)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
    # recordings.append(freefield.play_and_record(speaker, attenuated, equalize=False))
recordings = slab.Sound(recordings)


recordings.data[:, recordings.level < threshold] = target_recording.data
filter_bank = slab.Filter.equalizing_filterbank(target_recording, recordings, low_cutoff=low_cutoff,
                                                high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
# check for notches in the filter:
transfer_function = filter_bank.tf(show=False)[1][0:900, :]
if (transfer_function < -30).sum() > 0:
    print("Some of the equalization filters contain deep notches - try adjusting the parameters.")
return filter_bank, recordings





time.sleep(5)
freefield.equalize_speakers(speakers=speakers, reference_speaker=23, bandwidth=1 / 10, threshold=80,
                      low_cutoff=200, high_cutoff=16000, alpha=1.0, file_name=None)
