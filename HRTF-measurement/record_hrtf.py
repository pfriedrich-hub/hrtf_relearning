# record form in-ear microphones
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import slab
import freefield
import argparse
from copy import deepcopy
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=0.25, level=90, from_frequency=0, to_frequency=18000)
repetitions = 10
subject_id = 'paul'
n_directions = 1

def record_hrtfs(subject_id, repetitions, signal, n_directions=1, safe='sofa'):
    freefield.initialize('dome', default='play_birec')  # initialize setup
    #freefield.load_equalization(data_dir / 'dome_equalization_65')
    freefield.set_logger('WARNING')
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')  # get speaker coordinates
    source_locations = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    speaker_ids = source_locations[:, 0].astype('int')
    recordings = []
    sources = deepcopy(source_locations)
    print('Face fixpoint and press button to start recording.')
    # record from various locations at given listener orientations
    for i in range(n_directions):  # record for n listener orientations, 2 = front + back
        freefield.wait_for_button()
        recordings = recordings + (dome_rec(signal, speaker_ids, sources, repetitions))
        if i < n_directions-1:
            sources[:, 1] += 360/n_directions
            print('Rotate chair 180 degrees clockwise \nLook at fixpoint. Press button to start recording.')
    sources = create_src_txt(recordings)
    if safe == 'wav':
        print('Creating wav files...')
        for idx, bi_rec in enumerate(recordings):    # save recordings as .wav
            filename = 'in-ear_recordings\in-ear_%s_src_id%02d_az%i_el%i.wav' % (subject_id,
                idx, bi_rec[0], bi_rec[1])
            bi_rec[2].write(data_dir / filename)
        np.savetxt(str(data_dir / 'in-ear_recordings') + '/sources_%s.txt' % subject_id,
                   sources, fmt='%1.1f')   # save source coordinates to a text file
    if safe == 'sofa':
        print('Creating sofa file...')
        recorded_hrtf = slab.HRTF.estimate_hrtf([el[2] for el in recordings], signal, sources)
        recorded_hrtf.write_sofa(data_dir / 'hrtfs' / str('%s.sofa' % subject_id))
    freefield.set_logger('INFO')
    return recordings, sources

def dome_rec(signal, speaker_ids, sources, repetitions):
    print('Recording from various sound source locations..')
    recordings = []  # list to store binaural recordings and source coordinates
    for speaker_id in speaker_ids:
        [speaker] = freefield.pick_speakers(speaker_id)
        # get avg of n recordings from each sound source location
        recs = []
        for r in range(repetitions):
            rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
                  compensate_attenuation=False, equalize=False)
            recs.append(rec.data)
        recs = np.mean(np.asarray(recs), axis=0)
        azimuth = sources[speaker.index, 1]
        elevation = sources[speaker.index, 2]
        rec = [azimuth, elevation, slab.Binaural(data=recs, samplerate=rec.samplerate)]
        recordings.append(rec)
        print('Progress: %i %%' % (speaker_id*2))
    return recordings

def create_src_txt(recordings):
    sources = np.asarray(recordings)[:, :2]
    sources = np.c_[sources, np.ones(len(sources))*1.4].astype('float16')
    return sources

# def remove_mic_tf(recordings):

# generate probe signal
if __name__ == "__main__":
    data = record_hrtfs(subject_id=subject_id, repetitions=repetitions, signal=signal, safe='sofa')

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

# # equalize speaker level and transfer functions
# # todo reduce spectral range
# freefield.initialize('dome', "play_rec")
# freefield.equalize_speakers(speakers='all', threshold=65, file_name=data_dir / 'dome_equalization_65')
# rec_raw, rec_lvl, rec_full = freefield.test_equalization(speakers='all')
# freefield.spectral_range(rec_full)

"""
### extra: arrange dome ####
import numpy as np
radius = 1.4 # meter
az_angles = np.radians((17.5, 35, 52.5))
ele_angles = np.radians((12.5, 25, 37.5, 50))
horizontal_dist = np.sin(az_angles) * radius
vertical_dist = np.sin(ele_angles) * radius
vert_abs = []
for i in range(len(vertical_dist)):
    vert_abs.append(0.22 + vertical_dist[i])
"""

