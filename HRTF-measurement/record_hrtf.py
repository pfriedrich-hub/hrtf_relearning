# record form in-ear microphones
# example - from terminal/shell:
# python record_hrtf.py --id paul_hrtf

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import slab
import freefield
import argparse
from copy import deepcopy
data_dir = Path.cwd() / 'data'
fs = 48828  # sampling rate

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



def record_hrtfs(subject_id, repetitions, signal):
    # initialize setup
    freefield.initialize('dome', default='play_birec')
    freefield.load_equalization(data_dir / 'dome_equalization_65')
    # generate probe signal
    slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
    signal = slab.Sound.chirp(duration=0.1, level=90, from_frequency=200, to_frequency=16000)
    # get speaker coordinates
    freefield.set_logger('WARNING')
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    source_locations = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    speaker_ids = source_locations[:, 0].astype('int')
    # record
    n = 2  # record for n listener positions, 2 = front + back
    recordings = []
    sources = deepcopy(source_locations)
    print('Face fixpoint and press button to start recording.')
    for i in range(n):
        freefield.wait_for_button()
        recordings = recordings + (dome_rec(signal, speaker_ids, sources, repetitions))
        if i < n:
            sources = rotate(sources, angle=int(360/n))  # rotate listener 180°
    for idx, bi_rec in enumerate(recordings):
        filename = 'in-ear_recordings\in-ear_%s_src_idx%i_az%i_el%i.wav'%(subject_id,
                    idx, bi_rec[0], bi_rec[1])
        bi_rec[2].write(data_dir / filename)
    np.savetxt(str(data_dir) + '\in-ear_recordings\sources_%s.txt'%(subject_id),
               create_src_txt(recordings), fmt='%1.1f')
    freefield.set_logger('INFO')
    return recordings

def dome_rec(signal, speaker_ids, source_locations, repetitions):
    print('Recording from various sound source locations..')
    recordings = []  # list to store binaural recordings and source coordinates
    for speaker_id in speaker_ids:
        [speaker] = freefield.pick_speakers(speaker_id)
        # get avg of n recordings from each sound source location
        recs = []
        for r in range(repetitions):
            rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
                                            compensate_attenuation=False, equalize=True)
            recs.append(rec.data)
        recs = np.asarray(recs)
        azimuth = source_locations[speaker.index, 1]
        elevation = source_locations[speaker.index, 2]
        rec = [azimuth, elevation, slab.Binaural(data=recs.mean(axis=0))]
        recordings.append(rec)
        print('Progress: %i %%'%(speaker_id*2))
    return recordings

def rotate(source_locations, angle):
    # rotate speaker coordinates by 90 degrees
    source_locations[:, 1] += angle
    print('Rotate chair %i degrees clockwise \nLook at fixpoint. Press button to start recording.'%angle)
    return source_locations

def create_src_txt(recordings):
    sources = np.asarray(recordings)[:,:2]
    sources = np.c_[np.arange(len(sources)), sources, np.ones(len(sources))*1.4].astype('float16')
    return sources

def read_wav(path):
    from natsort import natsorted
    recordings = []  # list to hold slab.Binaural objects
    path_list = []
    for file_path in path.rglob('*.wav'):
        path_list.append(str(file_path))
    path_list = natsorted(path_list)
    for file_path in path_list:
        recordings.append(slab.Sound.read(file_path).data)
    return slab.Sound(data=recordings)

# def remove_mic_tf(recordings):
#     mic_tf = ?
#     recordings = recordings * mic_tf
#
# if __name__ == "__main__":
#     recordings = record_hrtfs(subject_id='kemar_test', repetitions=5, signal=signal)

# write sofa
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
signal = slab.Sound.chirp(duration=0.1, level=90, from_frequency=200, to_frequency=16000)
recs = read_wav(path = data_dir / 'in-ear_recordings' / 'KEMAR')
sources = np.loadtxt(data_dir / 'in-ear_recordings' / 'KEMAR' /'sources_KEMAR.txt')
kemar_hrtf = slab.HRTF.estimate_hrtf(recs, signal, sources)
kemar_hrtf.write_sofa(filename=data_dir / 'hrtfs' / 'KEMAR.sofa')


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

