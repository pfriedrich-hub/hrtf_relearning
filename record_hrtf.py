# record form in-ear microphones
import numpy
from pathlib import Path
import slab
import freefield
import datetime
import analysis.hrtf_analysis as hrtf_analysis
date = datetime.datetime.now()
from copy import deepcopy
fs = 97656  # 97656.25, 195312.5
from matplotlib import pyplot as plt
slab.set_default_samplerate(fs)

# file settings
subject_id = 'lw'
condition = 'earmolds_1'  # can be 'ears_free' or 'earmolds' - important for file naming!
kemar = False  # requires no button press if true
safe = 'both'  # decide if additionally save in-ear-recordings
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition

# HRTF recording settings
speakers = numpy.arange(20, 27).tolist()  #sources to record HRTF from # central cone, with top and bottom speaker removed
# speakers = numpy.arange(28, 35).tolist()  # 17.5 cone  # still to be calibrated
# speakers = numpy.arange(12, 19).tolist()  # -17.5 cone
level = 80  # minimize to reduce reverb ripple effect, apparently kemar recordings are not affected?
duration = 0.1  # short chirps <0.05s introduce variability in low freq (4-5 kHz). improvement at 0.1s for kemar vsi
low_freq = 1000
high_freq = 17000  # window of interes is 4-16
repetitions = 30  # 10 work for kemar, 30-50 for in ear mics
n_directions = 1  # only from the front (1) or front-back recordings (2)
ramp_duration = duration/20
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
# todo replace signal with mean central arc recording?
# signal = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'mean_central_arc_rec.wav')

# plotting options
dfe = False  # whether to use diffuse field equalization to plot hrtf and compute vsi
plot_bins = 2400  # number of bins also used to calculate vsi across bands (use 80 to minimize´frequency-resolution dependend vsi change)
plot_ear = 'left'  # ear for which to plot HRTFs


def record_hrtf(subject_id, data_dir, condition, signal, repetitions, n_directions, safe, speakers, kemar=False):
    global filt
    # filt = slab.Filter.band('bp', (low_freq, high_freq))
    filt = slab.Filter.band('hp', (200))  # makes no diff
    if not freefield.PROCESSORS.mode:
        proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_rec_buf.rcx'],
                     ['RX81', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx'],
                     ['RX82', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx']]
        freefield.initialize('dome', device=proc_list)
        freefield.PROCESSORS.mode = 'play_birec'
    freefield.load_equalization(file=Path.cwd() / 'data' / 'calibration' / 'central_arc_calibration_100k')
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
            print('Rotate chair %i degrees clockwise and look at fixpoint. \nPress button to start recording.'
                  % 360/n_directions)
            freefield.wait_for_button()
    for i in range(len(recordings)):  # highpass filter recordings 200
        recordings[i][2] = filt.apply(recordings[i][2])
    freefield.set_logger('INFO')
    if not kemar:
        freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    sources = create_src_txt(recordings)  # create source coordinate array

    # save files
    data_dir.mkdir(parents=True, exist_ok=True)  # create condition directory if it doesnt exist
    if safe == 'sofa' or safe == 'both':  # compute HRTFs and write to sofa file
        print('Creating sofa file...')
        recorded_hrtf = slab.HRTF.estimate_hrtf([rec[2] for rec in recordings], signal, sources)

        # file_path = str(data_dir / (subject_id + '_' + condition + date.strftime('_%d.%m')))
        # counter = 1
        # while Path.exists(file_path + '.sofa'):
        #     file_path = file_path + ('_%i' % counter)
        #     counter += 1
        # recorded_hrtf.write_sofa(file_path + '.sofa')

        recorded_hrtf.write_sofa(str(data_dir / (subject_id + '_' + condition + date.strftime('_%d.%m'))) + '.sofa')
    if safe == 'wav' or safe == 'both':  # write recordings to wav files
        wav_dir = data_dir / str('in_ear_recordings' + date.strftime('_%d.%m'))
        wav_dir.mkdir(parents=True, exist_ok=True)
        print('Creating wav files...')
        for idx, bi_rec in enumerate(recordings):    # save recordings as .wav
            filename = '%s_%s_%02d_az%i_el%i.wav' % (subject_id, condition, idx, bi_rec[0], bi_rec[1])
            bi_rec[2].write(wav_dir / filename)
        numpy.savetxt(wav_dir / ('sources_%s_%s.txt' % (subject_id, condition)), sources, fmt='%1.1f')
    return recordings, sources, recorded_hrtf

def dome_rec(signal, speaker_ids, sources, repetitions):
    print('Recording...')
    recordings = []  # list to store binaural recordings and source coordinates
    for idx, speaker_id in enumerate(speaker_ids):
        [speaker] = freefield.pick_speakers(speaker_id)
        to_play = freefield.apply_equalization(signal, speaker)
        # get avg of n recordings from each sound source location
        recs = []
        for r in range(repetitions):
            recs.append(freefield.play_and_record(speaker, to_play, equalize=False))
        rec = slab.Binaural(numpy.mean(numpy.asarray(recs), axis=0))
        rec = slab.Binaural.ramp(rec, when='both', duration=ramp_duration)
        azimuth = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][1]
        elevation = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][2]
        recordings.append([azimuth, elevation, rec])
        print('progress: %i %%' % (int((idx+1)/len(speaker_ids)*100)), end="\r", flush=True)
    return recordings

def create_src_txt(recordings):
    # convert interaural_polar to vertical_polar coordinates for sofa file
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
    recordings, sources, hrtf = record_hrtf(subject_id, data_dir, condition, signal, repetitions, n_directions, safe, speakers, kemar)
    sources = list(range(hrtf.n_sources-1, -1, -1))  # works for 0°/+/-17.5° cone
    fig, axis = plt.subplots(2, 1)
    hrtf_analysis.plot_tf(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000), dfe=dfe)
    hrtf_analysis.vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
    axis[0].set_title(subject_id)
    # hrtf.plot_tf(sources, xlim=(low_freq, high_freq), ear=plot_ear)
    hrtf.plot_tf(sources, xlim=(4000, 16000), ear=plot_ear)


"""
# example - from terminal/shell:
python record_hrtf.py --id paul_hrtf

todo: implement this into freefield?
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--id", type=str,
	default="paul_hrtf",
	help="enter subject id")
args = vars(ap.parse_args())
id = args["id"]
print('record from %s speakers, subj_id: %i'%(id, 9))
"""

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
