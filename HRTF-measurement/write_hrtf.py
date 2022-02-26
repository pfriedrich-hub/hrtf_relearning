# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   write_hrtf.py
#   read in-the-ear recordings (.wav), compute HRTF via numpy and store in file (.sofa)
#   after Andrés Pérez-López - Eurecat / UPF
#   24/08/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from netCDF4 import Dataset
import time
import numpy as np
import os
import slab
import freefield
from pathlib import Path
import argparse

subject = 'paul_hrtf'
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--id", type=str,
# 	default="paul_hrtf",
# 	help="enter subject id")
# args = vars(ap.parse_args())
# subject = args["id"]
# print('record from %s speakers, subj_id: %i' %(id, 9))

#---------Load Recordings-----------#

# get speakers and locations(az,ele)
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
table = np.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
speakers = table[20:27]  # for now only use 7 central speakers
probe_len = 0.5  # length of the sound probe in seconds
fs = 48828  # sampling rate

# read recorded .wav files and return slab object list
def read_wav(speakers, subject):
    recordings = np.zeros([len(speakers), int(probe_len*fs), 2])  # array to store recordings
    for i, source_location in enumerate(speakers):
        recording = slab.Sound.read(Path.cwd() / 'data' / 'in-ear_recordings' / ('in-ear_%s_%s_%s.wav'\
                    %(subject, str(source_location[1]), str(source_location[2]))))
        recordings[i] = recording.data
    return recordings.reshape(recordings.shape[0], 2, recordings.shape[1])
recs = read_wav(speakers, subject)

#---------compute HRTFs-----------#

# generate probe signal
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.
chirp = slab.Sound.chirp(duration=probe_len, level=90)  # create chirp from 100 to fs/2 Hz
# compute HRTFs and safe them in MRN array - Measurements, Receivers, N_frequencies
def HRTF_estimate(signal, recordings):
    x = signal.data[:, 0]
    N = int(len(x)/2)+1
    hrtfs = np.zeros([len(recordings), 2, N], dtype=complex)
    for i, recfile in enumerate(recordings):
        yr = recfile[1, :]
        yl = recfile[0, :]
        # input
        xfft = np.fft.rfft(x, axis=0)  # compute discrete fourier transform
        # output
        yr_fft = np.fft.rfft(yr, axis=0)  # compute discrete fourier transform
        yl_fft = np.fft.rfft(yl, axis=0)  # compute discrete fourier transform
        # transfer function: h = y / x
        tf_r = yr_fft / xfft
        tf_l = yl_fft / xfft
        hrtfs[i, 0] = tf_r
        hrtfs[i, 1] = tf_l
    return hrtfs

hrtfs = HRTF_estimate(signal=chirp, recordings=recs)

#----------Create sound source array---------#
radius = 1
# sound source array should be of shape [azimuths, elevation, radius]
sources = np.c_[speakers[:, 1:], np.ones(speakers.shape[0])*radius]

#----------Create SOFA file----------#

filePath = Path.cwd() / 'data' / 'hrtfs' / (subject + '.sofa')
# Need to delete it first if file already exists
if os.path.exists(filePath):
    os.remove(filePath)
rootgrp = Dataset(filePath, 'w', format='NETCDF4')

#----------Required Attributes----------#

rootgrp.Conventions = 'SOFA'
rootgrp.Version = '2.0'
rootgrp.SOFAConventions = 'SimpleFreeFieldHRTF'
rootgrp.SOFAConventionsVersion = '2.0'
rootgrp.APIName = 'pysofaconventions'
rootgrp.APIVersion = '0.1'
rootgrp.APIVersion = '0.1'
rootgrp.AuthorContact = 'andres.perez@eurecat.org'
rootgrp.DataType = 'TF'
rootgrp.License = 'PublicLicence'
rootgrp.ListenerShortName = 'PF01'
rootgrp.Organization = 'Eurecat - UPF'
rootgrp.RoomType = 'free field'
rootgrp.DateCreated = time.ctime(time.time())
rootgrp.DateModified = time.ctime(time.time())
rootgrp.Title = 'testpysofaconventions'
rootgrp.DatabaseName = 'UniLeipzig Freefield'

#----------Required Dimensions----------#

m = len(speakers)  # number of measurements
n = hrtfs.shape[2]  # number of datapoints per measurement
r = 2  # number of receivers (HRTFs measured for 2 ears)
e = 1  # number of emitters (1 speaker per measurement)
i = 1  # always 1
c = 3  # number of dimensions in space (elevation, azimuth, radius)

rootgrp.createDimension('M', m)
rootgrp.createDimension('N', n)
rootgrp.createDimension('E', e)
rootgrp.createDimension('R', r)
rootgrp.createDimension('I', i)
rootgrp.createDimension('C', c)

#----------Required Variables----------#

    # listener position
listenerPositionVar = rootgrp.createVariable('ListenerPosition', 'f8', ('I', 'C'))
listenerPositionVar.Units   = 'metre'
listenerPositionVar.Type    = 'cartesian'
listenerPositionVar[:] = np.zeros(c)

    # receiver position
receiverPositionVar = rootgrp.createVariable('ReceiverPosition', 'f8', ('R', 'C', 'I'))
receiverPositionVar.Units   = 'metre'
receiverPositionVar.Type    = 'cartesian'
receiverPositionVar[:]      = np.zeros((r, c, i))

    # source position
sourcePositionVar = rootgrp.createVariable('SourcePosition', 'f8', ('M', 'C'))
sourcePositionVar.Units   = 'degree, degree, metre'
sourcePositionVar.Type    = 'spherical'
sourcePositionVar[:]      = sources  # array of speaker positions

    # emitter position
emitterPositionVar  = rootgrp.createVariable('EmitterPosition', 'f8', ('E', 'C', 'I'))
emitterPositionVar.Units   = 'metre'
emitterPositionVar.Type    = 'cartesian'
# Equidistributed speakers in circle
emitterPositionVar[:] = np.zeros((e, c, i))

    # ListenerUp / ListenerView
listenerUpVar       = rootgrp.createVariable('ListenerUp', 'f8', ('I', 'C'))
listenerUpVar.Units         = 'metre'
listenerUpVar.Type          = 'cartesian'
listenerUpVar[:]    = np.asarray([0, 0, 1])

    # Listener looking to the left (+Y axis)
listenerViewVar     = rootgrp.createVariable('ListenerView', 'f8', ('I', 'C'))
listenerViewVar.Units       = 'metre'
listenerViewVar.Type        = 'cartesian'
listenerViewVar[:]  = np.asarray([0, 1, 0])

    # data
dataRealVar = rootgrp.createVariable('Data.Real', 'f8', ('M', 'R', 'N'))
dataRealVar[:] = np.real(hrtfs)

dataImagVar = rootgrp.createVariable('Data.Imag', 'f8', ('M', 'R', 'N'))
dataImagVar[:] = np.imag(hrtfs)

NVar = rootgrp.createVariable('N', 'f8', ('N'))
NVar.LongName        = 'frequency'
NVar.Units       = 'hertz'
NVar[:] = n

samplingRateVar =   rootgrp.createVariable('Data.SamplingRate', 'f8',   ('I'))
samplingRateVar.Units = 'hertz'
samplingRateVar[:] = fs

#----------Close it----------#

rootgrp.close()

