import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path
from aruco_pose import cams, az_dict, ele_dict
from aruco_pose import get_pose, calibrate_aruco, init_cams, deinit_cams
fs = 48828
slab.set_default_samplerate(fs)
data_dir = Path.cwd() / 'data'
tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)
subj_id = '001'


def localization_test():
    global speakers, stim
    # # initialize processors and cameras
    freefield.initialize('dome', default="loctest_freefield")
    freefield.set_logger('warning')
    init_cams(cams)
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
    # create sequence of speakers to play from, without direct repetition of azimuth or elevation
    n_conditions = len(speakers)
    sequence = numpy.random.permutation(numpy.tile(list(range(n_conditions)), 1))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while numpy.min(numpy.abs(az_dist)) == 0.0 or numpy.min(numpy.abs(ele_dist)) == 0.0:
        sequence = numpy.random.permutation(numpy.tile(list(range(n_conditions)), 1))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(trials=speakers[sequence, 0].astype('int'))
    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        trial_sequence.add_response(play_trial(speaker_id))  # play n trials
    trial_sequence.save_pickle(data_dir / 'localization_data' / str(subj_id + date.strftime('_%d_%b')))
    freefield.halt()
    deinit_cams(cams)
    print('localization test completed!')
    return

def play_trial(speaker_id):
    time.sleep(.5)
    offset = calibrate_aruco(cams, limit=0.5, report=False)  # get orientation offset
    target = speakers[speaker_id, 1:]
    print('STARTING..\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
    time.sleep(.5)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=False)
    freefield.play()
    freefield.wait_to_finish_playing()
    azimuth, elevation = None, None
    response = 0
    while not response:
        azimuth = get_pose(cams[1], aruco_dict=az_dict)
        elevation = get_pose(cams[0], aruco_dict=az_dict)
        if azimuth != None and elevation != None:
            pose = numpy.array((azimuth, elevation)) - offset
            print(pose)
            response = freefield.read('response', processor='RP2')
        else:
            print('no marker detected', end="\r", flush=True)
    freefield.set_signal_and_speaker(signal=tone, speaker=23)
    freefield.play()
    return numpy.array((pose, target))

if __name__ == "__main__":
    trialsequence = localization_test()
