import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path
from stats.localization_accuracy import localization_accuracy
import os
default_dir = os.getcwd()
os.chdir(default_dir + '/data/localization_data/')
# import head_tracking.cam_tracking.aruco_pose as aruco
import head_tracking.meta_motion.mm_pose as motion_sensor
# import head_tracking.sensor_tracking.sensor_pose as sensor

fs = 48828
slab.set_default_samplerate(fs)
# data_dir = Path.cwd() / 'data' / 'localization_data'
tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)

subject_id = 'jakab_mold_1.0'

def localization_test():
    global speakers, stim, sensor
    sensor = motion_sensor.start_sensor()
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_birec')
    freefield.set_logger('warning')

    # generate stimulus
    noise = slab.Sound.pinknoise(duration=0.025, level=90)
    noise = noise.ramp(when='both', duration=0.01)
    silence = slab.Sound.silence(duration=0.025)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                               silence, noise, silence, noise)
    stim = stim.ramp(when='both', duration=0.01)
    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                     delimiter=",", dtype=float)

    # speakers = numpy.delete(speakers, 23, 0)
    # create sequence of speakers to play from, without direct repetition of azimuth or elevation
    n_conditions = len(speakers)
    sequence = numpy.random.permutation(numpy.tile(list(range(n_conditions)), 2))
    sequence = numpy.delete(sequence, numpy.where(sequence == 27))  # remove 0, -50 target

    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(trials=speakers[sequence, 0].astype('int'))
    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        trial_sequence.add_response(play_trial(speaker_id))  # play n trials
    trial_sequence.save_pickle(str(subject_id + date.strftime('_%d_%b')))
    freefield.halt()
    motion_sensor.disconnect(sensor)
    print('localization test completed!')
    return

def play_trial(speaker_id):
    time.sleep(.5)
    offset = motion_sensor.calibrate_pose(sensor)
    target = speakers[speaker_id, 1:]
    print('TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
    time.sleep(.5)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=False)
    freefield.play()
    freefield.wait_to_finish_playing()
    response = 0
    while not response:
        pose = motion_sensor.get_pose(sensor, 30)  # set initial isi based on pose-target difference
        if all(pose):
            pose = pose - offset
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        response = freefield.read('response', processor='RP2')
    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    freefield.set_signal_and_speaker(signal=tone, speaker=23)
    freefield.play()
    return numpy.array((pose, target))

if __name__ == "__main__":
    trialsequence = localization_test()
    localization_accuracy(str(subject_id + date.strftime('_%d_%b')), show=True)
