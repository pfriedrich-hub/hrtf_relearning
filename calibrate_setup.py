import freefield
import slab
import time
import numpy
import matplotlib
from pathlib import Path
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy

# freefield.initialize('dome', default='play_rec')  # initialize setup
# speaker_id = 23
# [speaker] = freefield.pick_speakers(speaker_id)
# signal = slab.Sound.chirp(duration=3.0, level=90)
# rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
#           compensate_attenuation=False, equalize=False)

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

# dome parameters
reference_speaker = 23
azimuthal_angles = numpy.array([-52.5, -35, -17.5, 0, 17.5, 35, 52.5])

# signal parameters
low_cutoff=200
high_cutoff=16000
rec_time=0.1 # how long should the chirp be?
rec_repeat=5 # how often to repeat measurement for averaging

# filterbank parameters
freq_bins=1000 # can not be changes as of now
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
bandwidth=1 / 8
alpha=1.0

# change: set sound duration to 0.1 instead of 0.05 - lower frequency detection limited to 10 hz instead of 20 hz
sound = slab.Sound.chirp(duration=rec_time, from_frequency=low_cutoff, to_frequency=high_cutoff)

table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                         delimiter=",", dtype=float)

reference_speaker = freefield.pick_speakers(reference_speaker)[0]
# record from reference speaker
# proposed change I: average over 20 recordings, filter and ramp each recording
temp_recs = []
for i in range(rec_repeat):
    rec = freefield.play_and_record(reference_speaker, sound, equalize=False)
    rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target_recording = slab.Sound(data=numpy.mean(temp_recs, axis=0))

# ready to start:
# turn microphone
# press button to record and equalize

for az in azimuthal_angles:

    print('Place Mic orthogonal to %f Loudspeaker array and press Button to start calibration.')
    freefield.wait_for_button()
    print('recording...')

    speaker_idx = speaker_table[speaker_table[:, 1] == az][:, 0]
    speakers = freefield.pick_speakers(picks=list(speaker_idx.astype('int')))
    # print(speaker_idx)

    # step 1: level equalization
    """
    equalization_levels = freefield._level_equalization(speakers, sound, reference_speaker, threshold)
    
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker(target speaker must be in the list)
    """

    # record from each speaker in the list
    recordings = []
    for speaker in speakers:
        temp_recs = []
        for i in range(rec_repeat):
            rec = freefield.play_and_record(speaker, sound, equalize=False)
            rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
            temp_recs.append(rec.data)
        recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
    recordings = slab.Sound(recordings)

    # proposed change II:  thresholding: correct level on speakers which deviate more than <threshold> dB from reference
    recordings.data[:, numpy.logical_and(recordings.level > target_recording.level-level_threshold,
                                         recordings.level < target_recording.level+level_threshold)] = target_recording.data
    # proposed change
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
    
    _frequency_equalization(speakers, sound, reference_speaker, calibration_levels, bandwidth,
                            low_cutoff, high_cutoff, alpha, threshold):
                            
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
