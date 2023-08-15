import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
fs = 48828
slab.set_default_samplerate(fs)

subject_id = 'mh'
condition = 'Earmolds Week 2'
data_dir = Path.cwd() / 'final_data' / 'experiment' / 'bracket_4' / subject_id / condition

repetitions = 3  # number of repetitions per speaker

def localization_test(subject_id, data_dir, condition, repetitions):
    global speakers, sensor, tone, uso_list
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_rec', sensor_tracking=True)
    freefield.load_equalization(Path.cwd() / 'final_data' / 'calibration' / 'calibration_dome_23.05')
    # load sounds
    bell = slab.Sound.read(Path.cwd() / 'final_data' / 'sounds' / 'bell.wav')
    bell.level = 75
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)
    uso_dir = Path.cwd() / 'final_data' / 'sounds' / 'uso'
    uso_list = []
    for file_name in list(uso_dir.iterdir()):
        if file_name.is_file() and file_name.suffix == '.wav':
            uso_list.append(slab.Sound.read(file_name))
    # read list of speaker locations
    table_file = freefield.DIR / 'final_data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    c_speakers = numpy.delete(speakers, [19, 23, 27], axis=0)  # remove disconnected speaker from speaker_list
    speaker_sequence = numpy.zeros(repetitions * len(c_speakers)).astype('int')
    print('Setting speaker sequence...')
    while True:  # create n_repetitions sequences with more than 35° angular distance between successive targets
        for s in range(repetitions):
            seq = numpy.random.choice(c_speakers[:, 0], replace=False, size=(len(c_speakers))).astype('int')
            diff = numpy.diff(speakers[seq, 1:], axis=0)
            euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            while any(euclidean_dist < 35):  # or any(diff[:, 1] == 0):  # avoid similar targets in successive trials
                seq = numpy.random.choice(c_speakers[:, 0], replace=False, size=(len(c_speakers))).astype('int')
                diff = numpy.diff(speakers[seq, 1:], axis=0)
                euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            speaker_sequence[s*len(seq):s*len(seq)+len(seq)] = seq
            dist = numpy.zeros(repetitions * len(c_speakers) - 1)
        for i in range(len(speaker_sequence)-1):  #
            [diff] = numpy.diff((speakers[int(speaker_sequence[i]), 1:], speakers[int(speaker_sequence[i+1]), 1:]), axis=0)
            dist[i] = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
        if all(dist >= 35):  # check if distance is never smaller than 35°
            break
    trial_sequence = slab.Trialsequence(conditions=numpy.arange(0, len(uso_list)).tolist())
    trial_sequence.n_trials = len(speaker_sequence)
    trial_sequence.trials = trial_sequence.trials[:len(speaker_sequence)]
    trial_sequence.data = trial_sequence.data[:len(speaker_sequence)]
    trial_sequence.n_remaining = len(speaker_sequence)
    # loop over trials
    data_dir.mkdir(parents=True, exist_ok=True)  # create subject final_data directory if it doesnt exist
    file_name = 'uso_localization_' + subject_id + '_' + condition + date.strftime('_%d.%m')
    counter = 1
    while Path.exists(data_dir / file_name):
        file_name = 'uso_localization_' + subject_id + '_' + condition + date.strftime('_%d.%m') + '_' + str(counter)
        counter += 1
    played_bell = False
    print('Starting...')
    for index in trial_sequence:
        progress = int(trial_sequence.this_n / trial_sequence.n_trials * 100)
        if progress == 50 and played_bell is False:
            freefield.set_signal_and_speaker(signal=bell, speaker=23)
            freefield.play()
            freefield.wait_to_finish_playing()
            played_bell = True
        trial_sequence.add_response(play_trial(speaker_sequence[trial_sequence.this_n], uso_list[index], progress))
        trial_sequence.save_pickle(data_dir / file_name, clobber=True)
    freefield.halt()
    print('localization test completed!')
    return trial_sequence

def play_trial(speaker_id, uso, progress):
    time.sleep(.5)
    freefield.calibrate_sensor()
    target = speakers[speaker_id, 1:]
    print('%i%%: TARGET| azimuth: %.1f, elevation %.1f' % (progress, target[0], target[1]))
    time.sleep(.5)
    uso.level = 88
    freefield.set_signal_and_speaker(signal=uso, speaker=speaker_id, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()
    response = 0
    while not response:
        pose = freefield.get_head_pose(method='sensor')
        if all(pose):
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
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=1)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, ele_rmse, ele_var))


"""
import slab
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
subject_id = 'lk'
condition = 'Earmolds Week 1'
data_dir = Path.cwd() / 'final_data' / 'experiment' / 'bracket_4' / subject_id / condition
file_name = 'uso_localization_mh_Earmolds Week 1_06.08'
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_name=data_dir / file_name)

# plot
elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
"""