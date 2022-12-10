import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
import head_tracking.meta_motion.mm_pose as motion_sensor
fs = 48828
slab.set_default_samplerate(fs)

subject_id = 'ma'
condition = 'earmolds'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition

repetitions = 3  # number of repetitions per speaker

def localization_test(subject_id, data_dir, condition, repetitions):
    global speakers, stim, sensor, tone
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
    bell = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'bell.wav')
    bell.level = 75
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)

    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), repetitions))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while any([az_dist[i] == 0 and ele_dist[i] == 0 for i in range(len(az_dist))]):
        sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), repetitions))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    sequence = numpy.delete(sequence, [numpy.where(sequence == 19), numpy.where(sequence == 27)])
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(trials=range(len(sequence)))
    # loop over trials
    for index in trial_sequence:
        progress = int(trial_sequence.this_n / trial_sequence.n_trials * 100)
        if progress == 50:
            freefield.set_signal_and_speaker(signal=bell, speaker=23)
            freefield.play()
            freefield.wait_to_finish_playing()
        trial_sequence.add_response(play_trial(sequence[index], progress))
    data_dir.mkdir(parents=True, exist_ok=True)  # create subject data directory if it doesnt exist
    file_name = 'localization_' + subject_id + '_' + condition + date.strftime('_%d.%m')
    counter = 1
    while Path.exists(data_dir / file_name):
        file_name = file_name + '_' + str(counter)
        counter += 1
    trial_sequence.save_pickle(data_dir / file_name, clobber=True)
    freefield.halt()
    # motion_sensor.disconnect(sensor)
    print('localization test completed!')
    return trial_sequence

def play_trial(speaker_id, progress):
    time.sleep(.5)
    offset = motion_sensor.calibrate_pose(sensor)
        # if any(offset > 140 * numpy.tan(numpy.deg2rad(1.5))):
        #     freefield.play_warning_sound(0.25, 23)
        # else:  # check if head position is within tolerance margin of 1.5 cm
        #     break  # todo test this - doesnt work with drifting sensor..
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
    freefield.wait_to_finish_playing()
    return numpy.array((pose, target))

if __name__ == "__main__":
    sequence = localization_test(subject_id, data_dir, condition, repetitions)
    elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
    elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=1)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))

