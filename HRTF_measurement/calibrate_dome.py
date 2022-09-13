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
import pickle
import copy

"""
Equalize the loudspeaker array in two steps. First: equalize over all
level differences by a constant for each speaker. Second: remove spectral
difference by inverse filtering. For more details on how the
inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
"""

freefield.initialize('dome', default='play_rec')  # initialize setup
freefield.set_logger('warning')

# dome parameters
reference_speaker = 23
azimuthal_angles = numpy.array([-52.5, -35, -17.5, 0, 17.5, 35, 52.5])
# speaker_idx = [19,20,21,22,23,24,25,26,27]  # central array

# signal parameters
low_cutoff = 200
high_cutoff = 18000
signal_length = 1.0  # how long should the chirp be?
rec_repeat = 20  # how often to repeat measurement for averaging
# signal for loudspeaker calibration
signal = slab.Sound.chirp(duration=signal_length, from_frequency=low_cutoff, to_frequency=high_cutoff, level=80, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=0.001)

# equalization parameters
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
freq_bins = 1000  # can not be changes as of now
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
bandwidth = 1 / 8
alpha = 1.0

# obtain target signal by recording from reference speaker
# reference_speaker = freefield.pick_speakers(reference_speaker)[0]
temp_recs = []
for i in range(rec_repeat):
    rec = freefield.play_and_record(reference_speaker, signal, equalize=False)
    # rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target = slab.Sound(data=numpy.mean(temp_recs, axis=0))

# use original signal as reference
target = deepcopy(signal)
target.level = 20

# get speaker id's for each column in the dome
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
speaker_list = []
for az in azimuthal_angles:
    speaker_list.append((speaker_table[speaker_table[:, 1] == az][:, 0]).astype('int'))

dome_rec = []  # store all recordings from the dome for final spectral difference
equalization = dict()  # dictionary to hold equalization parameters


#------------------- hold on --------------------#
# pick single column to calibrate speaker_list[0] to speaker_list[6]
speakers = freefield.pick_speakers(speaker_list[6])
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
        rec = freefield.play_and_record(speaker, signal, equalize=False)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)

# thresholding
recordings.data[:, numpy.logical_and(recordings.level > target.level-level_threshold,
                                     recordings.level < target.level+level_threshold)] = target.data
equalization_levels = target.level - recordings.level

# set up plot
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(25, 10))
ax[3].set_xlabel('Frequency (Hz)')
for i in range(4):
    ax[i].set_ylabel('Power (dB/Hz)')
diff = freefield.spectral_range(recordings, plot=ax[0])
ax[0].set_title('raw')

# step 2: frequency equalization
"""
play the level-equalized signal, record and compute and a bank of inverse filter
to equalize each speaker relative to the target one. Return filterbank and recordings
"""
recordings = []
for speaker, level in zip(speakers, equalization_levels):
    attenuated = deepcopy(signal)
    attenuated.level += level
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, attenuated, equalize=False)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)
filter_bank = slab.Filter.equalizing_filterbank(target, recordings, length=freq_bins, low_cutoff=low_cutoff,
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
    attenuated = deepcopy(signal)
    attenuated = filter_bank.channel(idx).apply(attenuated)
    attenuated.level += level  # which order? doesnt seem to matter much
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, attenuated, equalize=False)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
dome_rec.extend(recordings)  # collect equalized recordings from the whole dome for final evaluation
recordings = slab.Sound(recordings)

# plot
diff = freefield.spectral_range(recordings, plot=ax[2])
ax[2].set_title('frequency equalized')


#------ OPTIONAL -----#
# # step 4: adjust level after freq equalization: (?) -- sometimes worth doing!
# level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
# recordings.data[:, numpy.logical_and(recordings.level > target.level-level_threshold,
#                                      recordings.level < target.level+level_threshold)] = target.data
# final_equalization_levels = equalization_levels + (target.level - recordings.level)
# recordings = []
# for idx, (speaker, level) in enumerate(zip(speakers, final_equalization_levels)):
#     attenuated = deepcopy(signal)
#     attenuated = filter_bank.channel(idx).apply(attenuated)
#     attenuated.level += level  # which order? doesnt seem to matter much
#     temp_recs = []
#     for i in range(rec_repeat):
#         rec = freefield.play_and_record(speaker, attenuated, equalize=False)
#         rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
#         temp_recs.append(rec.data)
#     recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
# recordings = slab.Sound(recordings)
# # plot
# diff = freefield.spectral_range(recordings, plot=ax[3])
# ax[3].set_title('final level correction')

az = speakers[0].azimuth
fig.suptitle('Calibration for dome speaker column at %.1f° azimuth. \n Difference in power spectrum' % az, fontsize=16)

# save equalization
array_equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
                for i in range(len(speakers))}
equalization.update(array_equalization)


# ----------  repeat for next speaker column ----------- #

# write final equalization to pkl file
file_name = freefield.DIR / 'data' / f'calibration_dome.pkl'
with open(file_name, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)

# check spectral difference across dome
dome_recs = slab.Sound(dome_rec)
diff = freefield.spectral_range(dome_recs)

# test calibration
import freefield
import slab
import numpy
import time
freefield.initialize('dome', default='play_rec')  # initialize setup

sound = slab.Sound.chirp(duration=1.0, from_frequency=20, to_frequency=20000, level=80)
sound = slab.Sound.pinknoise(duration=0.1, level=80)
sound = slab.Sound.ramp(sound, when='both', duration=0.01)
speaker_ids = list(numpy.arange(46))

# - on human listener
time.sleep(10)
for speaker_id in speaker_ids:
    freefield.set_signal_and_speaker(sound, speaker_id, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()

# - spectral range
recordings = []
for speaker_id in speaker_ids:
    temp_recs = []
    for i in range(20):
        rec = freefield.play_and_record(speaker_id, sound, equalize=False)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)
diff2 = freefield.spectral_range(recordings)


# build in test function
raw, level, spectrum = freefield.test_equalization()
diff_raw = freefield.spectral_range(raw)
diff_level = freefield.spectral_range(level)
diff_spectrum = freefield.spectral_range(spectrum)

"""
### extra: arrange dome ####
import numpy as numpy
radius = 1.4 # meter
az_angles = numpy.radians((0, 17.5, 35, 52.5))
ele_angles = numpy.radians((12.5, 25, 37.5, 50))
# horizontal_dist = numpy.cos((numpy.pi / 2) - az_angles) * radius
horizontal_dist = numpy.sin(az_angles) * radius

# this would be the correct vertical distances for interaural speaker locations;
radii = numpy.sin(numpy.pi / 2 - az_angles) * radius
vertical_dist = []
for elevation in ele_angles:
    vertical_dist.append(numpy.sin(elevation) * radii)
vertical_dist = numpy.asarray(vertical_dist)

# but we are using these for simplicity (just think of head tracking based training):
vertical_dist = numpy.sin(ele_angles) * radius

vert_abs = []
for i in range(len(vertical_dist)):
    vert_abs.append(0.22 + vertical_dist[i])
    
    

"""