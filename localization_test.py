import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path
from stats.localization_stats import localization_accuracy
import head_tracking.meta_motion.mm_pose as motion_sensor

subject_id = 'toni_mold'

fs = 48828
slab.set_default_samplerate(fs)
tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)
data_dir = Path.cwd() / 'data' / 'localization_data' / 'pilot'
filename = str(subject_id + date.strftime('_%d.%m'))
filepath = str(data_dir / filename)
n = 3  # number of repetitions per speaker

def localization_test():
    global speakers, stim, sensor
    sensor = motion_sensor.start_sensor()
    freefield.set_logger('warning')
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_birec')

    # generate stimulus
    noise = slab.Sound.pinknoise(duration=0.025, level=90)
    noise = noise.ramp(when='both', duration=0.01)
    silence = slab.Sound.silence(duration=0.025)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                               silence, noise, silence, noise)
    stim = stim.ramp(when='both', duration=0.01)
    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), n))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while any([az_dist[i] == 0 and ele_dist[i] == 0 for i in range(len(az_dist))]):
        sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), n))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    sequence = numpy.delete(sequence, [numpy.where(sequence == 19), numpy.where(sequence == 27)])
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(trials=range(len(sequence)))

    # loop over trials
    for index in trial_sequence:
        progress = int(trial_sequence.this_n / trial_sequence.n_trials * 100)
        trial_sequence.add_response(play_trial(sequence[index], progress))
    trial_sequence.save_pickle(filepath, clobber=True)
    freefield.halt()
    motion_sensor.disconnect(sensor)
    print('localization test completed!')
    return

def play_trial(speaker_id, progress):
    time.sleep(.5)
    offset = motion_sensor.calibrate_pose(sensor)
    target = speakers[speaker_id, 1:]
    print('%i%%: TARGET| azimuth: %.1f, elevation %.1f' % (progress, target[0], target[1]))
    time.sleep(.5)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
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
    elevation_gain, rmse, sd = localization_accuracy(filename, show=True, plot_dim=1)
    elevation_gain, rmse, sd = localization_accuracy(filename, show=True, plot_dim=2, binned=False)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))
