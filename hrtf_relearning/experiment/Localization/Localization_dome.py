import matplotlib
matplotlib.use('TkAgg')
import freefield
import slab
import numpy
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from analysis.plotting.localization_plot import localization_accuracy

# --- Config ---
fs = 48828
slab.set_default_samplerate(fs)

subject_id = 'PC'
condition = 'Free Ears'
data_dir = Path.cwd() / 'data' / 'control' / subject_id / condition
repetitions = 3


def get_central_speakers():
    """Load speaker table and return only central column speakers (az=0, -37.5<=el<=37.5)."""
    table_file = freefield.DIR / 'data' / 'tables' / 'speakertable_dome.txt'
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    mask = (speakers[:, 1] == 0) & (speakers[:, 2] >= -37.5) & (speakers[:, 2] <= 37.5)
    return speakers, speakers[mask]


def build_sequence(speakers, c_speakers, repetitions, min_dist=35):
    """Build a trial sequence ensuring >min_dist° between successive targets."""
    n = len(c_speakers)
    sequence = numpy.zeros(repetitions * n, dtype=int)

    print('Setting target sequence...')
    for s in range(repetitions):
        while True:
            seq = numpy.random.choice(c_speakers[:, 0], replace=False, size=n).astype(int)
            diffs = numpy.diff(speakers[seq, 1:], axis=0)
            if numpy.all(numpy.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) >= min_dist):
                break
        sequence[s*n:(s+1)*n] = seq

    # Validate across repetition boundaries
    while True:
        diffs = numpy.diff([speakers[int(s), 1:] for s in sequence], axis=0)
        if numpy.all(numpy.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) >= min_dist):
            break
        # Re-shuffle only the boundary-violating repetition joins
        for s in range(repetitions):
            seq = sequence[s*n:(s+1)*n]
            diffs = numpy.diff(speakers[seq, 1:], axis=0)
            if numpy.any(numpy.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) < min_dist):
                numpy.random.shuffle(seq)
                sequence[s*n:(s+1)*n] = seq

    return slab.Trialsequence(sequence)


def play_trial(speaker_id, speakers, tone, progress):
    freefield.calibrate_sensor()
    target = speakers[speaker_id, 1:]
    print('%i%%: TARGET| azimuth: %.1f, elevation %.1f' % (progress, target[0], target[1]))

    noise = slab.Sound.pinknoise(duration=0.025, level=90).ramp(when='both', duration=0.01)
    silence = slab.Sound.silence(duration=0.025)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise, silence, noise, silence, noise)
    stim = stim.ramp(when='both', duration=0.01)

    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=False)
    freefield.play()
    freefield.wait_to_finish_playing()

    pose = numpy.array([0.0, 0.0])
    response = 0
    while not response:
        pose = freefield.get_head_pose(method='sensor')
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end='\r', flush=True)
        else:
            print('no head pose detected', end='\r', flush=True)
        response = freefield.read('response', processor='RP2')

    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))

    freefield.set_signal_and_speaker(signal=tone, speaker=23, equalize=False)
    freefield.play()
    freefield.wait_to_finish_playing()

    return numpy.array((pose, target))


def localization_test(subject_id, data_dir, condition, repetitions):
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_rec', sensor_tracking=True)
    freefield.set_logger('warning')

    bell = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'bell.wav')
    bell.level = 75
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)

    speakers, c_speakers = get_central_speakers()
    trial_sequence = build_sequence(speakers, c_speakers, repetitions)

    data_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.now().strftime('_%d.%m')
    file_name = f'localization_{subject_id}_{condition}{date_str}'
    counter = 1
    while (data_dir / file_name).exists():
        file_name = f'localization_{subject_id}_{condition}{date_str}_{counter}'
        counter += 1

    played_bell = False
    print('Starting...')
    for speaker_id in trial_sequence:
        progress = int(trial_sequence.this_n / trial_sequence.n_trials * 100)
        if progress >= 50 and not played_bell:
            freefield.set_signal_and_speaker(signal=bell, speaker=23, equalize=False)
            freefield.play()
            freefield.wait_to_finish_playing()
            played_bell = True
        trial_sequence.add_response(play_trial(speaker_id, speakers, tone, progress))
        trial_sequence.save_pickle(data_dir / file_name, clobber=True)

    freefield.halt()
    print('Localization test completed!')
    return trial_sequence, file_name


if __name__ == '__main__':
    sequence, file_name = localization_test(subject_id, data_dir, condition, repetitions)

    fig, axis = plt.subplots()
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(
        sequence, axis=axis, show=True, plot_dim=2, binned=True
    )
    axis.set_title(f'{file_name} EG: {elevation_gain}')

    (data_dir / 'images').mkdir(parents=True, exist_ok=True)
    fig.savefig(data_dir / 'images' / f'{file_name}.png', format='png')
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, ele_rmse, ele_var))
