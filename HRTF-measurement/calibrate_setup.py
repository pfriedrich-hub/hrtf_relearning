import freefield
import slab
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
import time
import numpy
import matplotlib
from pathlib import Path
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy

"""
Equalize the loudspeaker array in two steps. First: equalize over all
level differences by a constant for each speaker. Second: remove spectral
difference by inverse filtering. For more details on how the
inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
"""

freefield.initialize('dome', default='play_rec')  # initialize setup
freefield.set_logger('warning')

# dictionary to hold equalization parameters
equalization = dict()

# dome parameters
reference_speaker = 23
azimuthal_angles = numpy.array([-52.5, -35, -17.5, 0, 17.5, 35, 52.5])
# speaker_idx = [19,20,21,22,23,24,25,26,27]  # central array

# signal parameters
low_cutoff = 200
high_cutoff = 20000
signal_length = 2.0  # how long should the chirp be?
rec_repeat = 20  # how often to repeat measurement for averaging

# filterbank parameters
freq_bins=1000 # can not be changes as of now
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
bandwidth=1 / 8
alpha=1.0

# sound for loudspeaker calibration
sound = slab.Sound.chirp(duration=signal_length, from_frequency=low_cutoff, to_frequency=high_cutoff)
sound = slab.Sound.ramp(sound, when='both', duration=0.01)

# record from reference speaker
reference_speaker = freefield.pick_speakers(reference_speaker)[0]
temp_recs = []
for i in range(rec_repeat):
    rec = freefield.play_and_record(reference_speaker, sound, equalize=False)
    rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target_recording = slab.Sound(data=numpy.mean(temp_recs, axis=0))

# get speaker id's for each column in the dome
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                         delimiter=",", dtype=float)
speaker_list = []
for az in azimuthal_angles:
    speaker_list.append((speaker_table[speaker_table[:, 1] == az][:, 0]).astype('int'))

# pick single column to calibrate
speakers = freefield.pick_speakers(speaker_list[0])

# hold on:
# place microphone 90° to source column at equal distance (recordings should be done in far field: > 1m)

# start calibration
# step 1: level equalization
"""
Record the signal from each speaker in the list and return the level of each
speaker relative to the target speaker(target speaker must be in the list)
"""
recordings = []
for speaker in speakers:
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, sound, equalize=False)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)

# thresholding
recordings.data[:, numpy.logical_and(recordings.level > target_recording.level-level_threshold,
                                     recordings.level < target_recording.level+level_threshold)] = target_recording.data
equalization_levels = target_recording.level - recordings.level

# set up plot
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
ax[3].set_xlabel('Frequency (Hz)')
for i in range(4):
    ax[i].set_ylabel('Power (dB/Hz)')
diff = freefield.spectral_range(recordings, plot=ax[0])
ax[0].set_title('raw')
fig.suptitle('Difference in power spectrum', fontsize=16)

# step 2: frequency equalization
"""
play the level-equalized signal, record and compute and a bank of inverse filter
to equalize each speaker relative to the target one. Return filterbank and recordings
"""
recordings = []
for speaker, level in zip(speakers, equalization_levels):
    attenuated = deepcopy(sound)
    attenuated.level += level
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, attenuated, equalize=False)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)
filter_bank = slab.Filter.equalizing_filterbank(target_recording, recordings, length=freq_bins, low_cutoff=low_cutoff,
                                                high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)

# plot
diff = freefield.spectral_range(recordings, plot=ax[1])
ax[1].set_title('level equalized')

# check for notches in the filter:
transfer_function = filter_bank.tf(show=False)[1][0:900, :]
if (transfer_function < -30).sum() > 0:
    print("Some of the equalization filters contain deep notches - try adjusting the parameters.")

# step 3: test filter bank
recordings = []
for idx, (speaker, level) in enumerate(zip(speakers, equalization_levels)):
    attenuated = deepcopy(sound)
    attenuated = filter_bank.channel(idx).apply(attenuated)
    attenuated.level += level  # which order? doesnt seem to matter much
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, attenuated, equalize=False)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)

# plot
diff = freefield.spectral_range(recordings, plot=ax[2])
ax[2].set_title('frequency equalized')

# step 4: adjust level after freq equalization: (?) -- definitely worth it!
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
recordings.data[:, numpy.logical_and(recordings.level > target_recording.level-level_threshold,
                                     recordings.level < target_recording.level+level_threshold)] = target_recording.data
equalization_levels += target_recording.level - recordings.level

# plot
diff = freefield.spectral_range(recordings, plot=ax[3])
ax[3].set_title('final level correction')


# write to pkl
array_equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
                for i in range(len(speakers))}
equalization.update(array_equalization)
