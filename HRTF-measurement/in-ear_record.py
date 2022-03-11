# record form in-ear microphones
# example - from terminal/shell:
# python in-ear_record.py --id paul_hrtf

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import slab
import freefield
import argparse

# todo: implement this into freefield?

# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--id", type=str,
# 	default="paul_hrtf",
# 	help="enter subject id")
# args = vars(ap.parse_args())
# id = args["id"]
# print('record from %s speakers, subj_id: %i'%(id, 9))

# get speakers and locations(az,ele) to play from
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
table = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
# todo record from whole dome
sources = table[20:27]  # for now only use positive az (half dome)

fs = 48828  # sampling rate
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
probe_len = 0.5  # length of the sound probe in seconds

#tone = slab.Sound.whitenoise(duration=probe_len) # chirp?
chirp = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz

def dome_rec(sources, subject=id, n_reps=50):
    # initialize setup
    freefield.initialize('dome', default='play_birec')
    freefield.load_equalization()

    # equalize speaker transfer functions
    # freefield.equalize_speakers(speakers=speaker_coordinates)
    # rec_raw, rec_lvl, rec_full = freefield.test_equalization(speakers=speaker_speaker_coordinates)
    # play , record  and average n_reps times, from all speakers in the list + save as .wav
    freefield.set_logger('WARNING')
    recordings = np.zeros([len(sources), int(probe_len*fs), 2])  # array to store recordings as data arrays
    avg_rec_list = []  # list to hold recordings as slab binaural objects
    #todo sort out left and right (ff setup uses [0] as right, while slab uses [0] as left -> change in ff lab
    for i, source_location in enumerate(sources):
        print(source_location)
        speaker = freefield.pick_speakers(tuple((source_location[1], source_location[2])))
        # get avg of 20 recordings from each sound source location
        recs = []
        for r in range(n_reps):
            rec = freefield.play_and_record(speaker, chirp, compensate_delay=True,
                                            compensate_attenuation=False, equalize=True)
            recs.append(rec.data)
        recs = np.asarray(recs)
        avgrec = slab.Binaural(data=recs.mean(axis=0))
        avgrec.write(os.getcwd() + '/data/in-ear_recordings/in-ear_%s_%s_%s.wav'
                     % (subject, str(source_location[1]), str(source_location[2])))
        recordings[i] = avgrec.data # as np.array
        avg_rec_list.append(avgrec)
    freefield.set_logger('INFO')
    return avg_rec_list, recordings

def remove_mic_tf(recordings):
    mic_tf = ?
    recordings = recordings * mic_tf


if __name__ == "__main__":
    dome_rec(speakers, )

"""
### extra: arrange dome ####
radius = 1.5 # meter
az_angles = np.radians((17.5, 35, 52.5))
ele_angles = np.radians((12.5, 25, 37.5, 50))
horizontal_dist = np.sin(az_angles) * radius
vertical_dist = np.sin(ele_angles) * radius
"""

