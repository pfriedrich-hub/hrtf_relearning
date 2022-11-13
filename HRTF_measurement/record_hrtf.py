# record form in-ear microphones
import numpy
from pathlib import Path
import slab
import freefield
import datetime
import stats.plots as plots
date = datetime.datetime.now()
from copy import deepcopy
from matplotlib import pyplot as plt

subject_id = 'kemar'
kemar = True
speakers = numpy.arange(20, 27).tolist()  # central cone, with top and bottom speaker removed
# speakers = numpy.arange(28, 35).tolist()  # 17.5 cone
# speakers = numpy.arange(12, 19).tolist()  # -17.5 cone

# speakers = 'all'
safe = 'sofa'

data_dir = Path.cwd() / 'data' / 'hrtfs' / 'pilot'
filename = str(subject_id + date.strftime('_%d.%m'))
filepath = str(data_dir / filename)
fs = 48828  # sampling rate
duration = 0.1  # short chirps <0.05s introduce variability in low freq (4-5 kHz). no improvement above 0.1s
low_freq = 1000
high_freq = 17000  # window of interes is 4-16
repetitions = 10  # works on kemar
n_directions = 1  # only from the front (1) or front-back recordings (2)
n_bins = 2400
plot_ear = 'left'
ramp_duration = duration/20
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=duration, level=85, from_frequency=low_freq, to_frequency=high_freq, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)

def record_hrtfs(subject_id, repetitions, signal, n_directions, safe=safe, speakers=speakers):
    global filt
    # filt = slab.Filter.band('bp', (low_freq, high_freq))
    filt = slab.Filter.band('hp', (200)) # makes no diff
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_birec')
    freefield.set_logger('warning')
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')  # get speaker coordinates
    if isinstance(speakers, str) and speakers == 'all':
        source_locations = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                         delimiter=",", dtype=float)
        source_locations = numpy.delete(source_locations, [numpy.where(source_locations[:, 0] == 19),
                           numpy.where(source_locations[:, 0] == 27)], axis=0)  # remove 0, 50 and -50 speakers
    elif isinstance(speakers, list):
        source_locations = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                         delimiter=",", dtype=float)[speakers]
    else:
        raise ValueError('Speakers must be >>all<< or list of indices.')
    speaker_ids = source_locations[:, 0].astype('int')
    sources = deepcopy(source_locations)
    if not kemar:
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
        freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                    processors=led_speaker.digital_proc)  # illuminate LED
        print('Face fixpoint and press button to start recording.')
        freefield.wait_for_button()
    recordings = []
    for i in range(n_directions):  # record for n listener orientations, 2 = front + back
        recordings = recordings + (dome_rec(signal, speaker_ids, sources, repetitions))
        if i < n_directions-1:
            sources[:, 1] += 360/n_directions
            print('Rotate chair %i degrees clockwise and look at fixpoint. \nPress button to start recording.' % 360/n_directions)
            freefield.wait_for_button()
    for i in range(len(recordings)):  # bandpass filter recordings 200 - 18000 hz
        filt.apply(recordings[i][2])
    freefield.set_logger('INFO')
    if not kemar:
        freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED

    # save .sofa / recordings.wav and sources.txt
    sources = create_src_txt(recordings)  # create source coordinate array
    if safe == 'sofa' or safe == 'both':  # compute HRTFs and write to sofa file
        print('Creating sofa file...')
        recorded_hrtf = slab.HRTF.estimate_hrtf([rec[2] for rec in recordings], signal, sources)
        recorded_hrtf.write_sofa(filepath + '.sofa')

    if safe == 'wav' or safe == 'both':  # write recordings to wav files
        print('Creating wav files...')
        for idx, bi_rec in enumerate(recordings):    # save recordings as .wav
            filename = '%s_src_id%02d_az%i_el%i.wav' % (subject_id, idx, bi_rec[0], bi_rec[1])
            bi_rec[2].write('in-ear_recordings' / filename)

        numpy.savetxt(str('in-ear_recordings') + '/sources_%s.txt' % subject_id,
                   sources, fmt='%1.1f')   # save source coordinates to a text file
    return recordings, sources, recorded_hrtf

def dome_rec(signal, speaker_ids, sources, repetitions):
    print('Recording...')
    recordings = []  # list to store binaural recordings and source coordinates
    for idx, speaker_id in enumerate(speaker_ids):
        [speaker] = freefield.pick_speakers(speaker_id)
        # get avg of n recordings from each sound source location
        recs = []
        for r in range(repetitions):
            recs.append(freefield.play_and_record(speaker, signal, equalize=True))
        rec = slab.Binaural(numpy.mean(numpy.asarray(recs), axis=0))
        rec = slab.Binaural.ramp(rec, when='both', duration=ramp_duration)
        azimuth = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][1]
        elevation = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][2]
        recordings.append([azimuth, elevation, rec])
        print('progress: %i %%' % (int((idx+1)/len(speaker_ids)*100)), end="\r", flush=True)
    return recordings

def create_src_txt(recordings):
    # interaural polar to cartesian
    interaural_polar = numpy.asarray(recordings)[:, :2].astype('float')
    cartesian = numpy.zeros((len(interaural_polar), 3))
    vertical_polar = numpy.zeros((len(interaural_polar), 3))
    azimuths = numpy.deg2rad(interaural_polar[:, 0])
    elevations = numpy.deg2rad(90 - interaural_polar[:, 1])
    r = 1.4  # get radii of sound sources
    cartesian[:, 0] = r * numpy.cos(azimuths) * numpy.sin(elevations)
    cartesian[:, 1] = r * numpy.sin(azimuths)
    cartesian[:, 2] = r * numpy.cos(elevations) * numpy.cos(azimuths)
    # cartesian to vertical polar
    xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
    vertical_polar[:, 0] = numpy.rad2deg(numpy.arctan2(cartesian[:, 1], cartesian[:, 0]))
    vertical_polar[vertical_polar[:, 0] < 0, 0] += 360
    vertical_polar[:, 1] = 90 - numpy.rad2deg(numpy.arctan2(numpy.sqrt(xy), cartesian[:, 2]))
    vertical_polar[:, 2] = numpy.sqrt(xy + cartesian[:, 2] ** 2)
    return vertical_polar.astype('float16')

if __name__ == "__main__":
    recordings, sources, hrtf = record_hrtfs(subject_id, repetitions, signal, n_directions, safe=safe, speakers=speakers)
    sources = list(range(hrtf.n_sources-1, -1, -1))
    fig, axis = plt.subplots(2, 1)
    hrtf.plot_tf(sources, n_bins=n_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000))
    plots.plot_vsi(hrtf, sources, n_bins=n_bins, axis=axis[1])
    axis[0].set_title(subject_id)
    hrtf.plot_tf(sources, xlim=(low_freq, high_freq), ear=plot_ear)

# example - from terminal/shell:
# python record_hrtf.py --id paul_hrtf

# todo: implement this into freefield?
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--id", type=str,
# 	default="paul_hrtf",
# 	help="enter subject id")
# args = vars(ap.parse_args())
# id = args["id"]
# print('record from %s speakers, subj_id: %i'%(id, 9))

"""
### extra: arrange dome ####
import numpy as numpy
radius = 1.4 # meter
az_angles = numpy.radians((17.5, 35, 52.5))
ele_angles = numpy.radians((12.5, 25, 37.5, 50))
horizontal_dist = numpy.sin(az_angles) * radius
vertical_dist = numpy.sin(ele_angles) * radius
vert_abs = []
for i in range(len(vertical_dist)):
    vert_abs.append(0.22 + vertical_dist[i])
    
    
# test molds on kemar
from matplotlib import pyplot as plt

fname='varvara_ears_free_23.09.sofa'
hrtf=slab.HRTF(Path.cwd() / 'data' / 'hrtfs' / fname)
src=hrtf.cone_sources(0)
hrtf.plot_tf(src, n_bins=300)
plt.title(fname)
"""

