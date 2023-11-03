import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from matplotlib import pyplot as plt
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
fs = 48828
slab.set_default_samplerate(fs)

subject_id = 'mh'
condition = 'Ears Free'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_4' / subject_id / condition

repetitions = 3  # number of repetitions per speaker

def localization_test(subject_id, data_dir, condition, repetitions):
    global speakers, stim, sensor, tone, file_name
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_rec', sensor_tracking=True)
    freefield.load_equalization(Path.cwd() / 'data' / 'calibration' / 'calibration_dome_23.05')

    # generate stimulus
    bell = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'bell.wav')
    bell.level = 75
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)

    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    c_speakers = numpy.delete(speakers, [19, 23, 27], axis=0)  # remove disconnected speaker from speaker_list
    sequence = numpy.zeros(repetitions * len(c_speakers)).astype('int')
    print('Setting target sequence...')
    while True:  # create n_repetitions sequences with more than 35° angular distance between successive targets
        for s in range(repetitions):
            seq = numpy.random.choice(c_speakers[:, 0], replace=False, size=(len(c_speakers))).astype('int')
            diff = numpy.diff(speakers[seq, 1:], axis=0)
            euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            while any(euclidean_dist < 35):  # or any(diff[:, 1] == 0):  # avoid similar targets in successive trials
                seq = numpy.random.choice(c_speakers[:, 0], replace=False, size=(len(c_speakers))).astype('int')
                diff = numpy.diff(speakers[seq, 1:], axis=0)
                euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            sequence[s*len(seq):s*len(seq)+len(seq)] = seq
            dist = numpy.zeros(repetitions * len(c_speakers) - 1)
        for i in range(len(sequence)-1):  #
            [diff] = numpy.diff((speakers[int(sequence[i]), 1:], speakers[int(sequence[i+1]), 1:]), axis=0)
            dist[i] = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
        if all(dist >= 35):  # check if distance is never smaller than 35°
            break
    trial_sequence = slab.Trialsequence(trials=range(len(sequence)))
    # loop over trials
    data_dir.mkdir(parents=True, exist_ok=True)  # create subject data directory if it doesnt exist
    file_name = 'localization_' + subject_id + '_' + condition + date.strftime('_%d.%m')
    counter = 1
    while Path.exists(data_dir / file_name):
        file_name = 'localization_' + subject_id + '_' + condition + date.strftime('_%d.%m') + '_' + str(counter)
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
        trial_sequence.add_response(play_trial(sequence[index], progress))
        trial_sequence.save_pickle(data_dir / file_name, clobber=True)
    # freefield.halt()
    print('localization ole_test completed!')
    return trial_sequence

def play_trial(speaker_id, progress):
    freefield.calibrate_sensor()
    target = speakers[speaker_id, 1:]
    print('%i%%: TARGET| azimuth: %.1f, elevation %.1f' % (progress, target[0], target[1]))
    noise = slab.Sound.pinknoise(duration=0.025, level=90)
    noise = noise.ramp(when='both', duration=0.01)
    silence = slab.Sound.silence(duration=0.025)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                               silence, noise, silence, noise)
    stim = stim.ramp(when='both', duration=0.01)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
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
    fig, axis = plt.subplots()
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, axis=axis,
                                                                            show=True, plot_dim=2, binned=True)
    axis.set_title(file_name)
    (data_dir / 'images').mkdir(parents=True, exist_ok=True)  # create subject image directory
    fig.savefig(data_dir / 'images' / str(file_name + '.png'), format='png')  # save image
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=1)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, ele_rmse, ele_var))



"""
import slab
from pathlib import Path
from analysis.localization_analysis import localization_accuracy

file_name = 'localization_lw_ears_free_10.12'

for path in Path.cwd().glob("**/"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_path)

# plot
from matplotlib import pyplot as plt
fig, axis = plt.subplots(1, 1)
elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2,
 binned=True, axis=axis)
axis.set_xlabel('Response Azimuth (degrees)')
axis.set_ylabel('Response Elevation (degrees)')
fig.suptitle(file_name)
"""


"""


#--------- stitch incomplete sequences ------------------#

filename_1 = 'localization_sm_Earmolds Week 1_6_29.01'
filename_2 = 'localization_sm_Earmolds Week 1_29.01_1'
sequence_1 = slab.Trialsequence(conditions=45, n_reps=1)
sequence_2 = deepcopy(sequence_1)
sequence_1.load_pickle(file_name=data_dir / filename_1)
sequence_2.load_pickle(file_name=data_dir / filename_2)
data_1 = sequence_1.data[:-sequence_1.n_remaining]
data_2 = sequence_2.data[:-sequence_2.n_remaining]
data = data_1 + data_2
sequence = sequence_1
file_name = filename_1
sequence.data = data

#  save
sequence.save_pickle(data_dir / file_name, clobber=True)

# ----------- correct azimuth for >300° ---------- #

file_name = 'localization_lm_Ears Free_05.06_1'
for path in Path.cwd().glob("**/*"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_path)

for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] > 180] -= 360
    
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] < -180] += 360
    
# -------------- save ------------------#

sequence.save_pickle(file_path, clobber=True)

"""