import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import numpy
import slab
import time
import logging
from hrtf.processing.tf2ir import *
from hrtf.processing.add_interaural import *

wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
sound_path = Path.cwd() / 'data' / 'sounds'

# wrapper for hrtf2wav
def make_wav(filename, overwrite=False, show=False):
    if not (wav_path / filename).exists() or overwrite:
        hrtf2wav(f'{filename}.sofa', show)

def hrtf2wav(filename, show):
    """
    Convert HRIR filters from a sofa file to wav files for use with pybinsim.
    """
    # create folder structure for HRTF wav files
    global dir_name, filter_list_fname, hrir, blocksize
    dir_name = Path(filename).stem
    if not (wav_path / dir_name).exists():
        (wav_path / dir_name).mkdir(exist_ok=True)
        (wav_path / dir_name / 'IR_data').mkdir(exist_ok=True)
        (wav_path / dir_name / 'sounds').mkdir(exist_ok=True)
    filter_list_fname = wav_path / dir_name / f"filter_list_{dir_name}.txt"

    # load HRTF, convert to IR and interpolate if necessary
    hrtf = slab.HRTF(sofa_path / filename)
    slab.set_default_samplerate(hrtf.samplerate)
    if hrtf.datatype == 'TF':
        hrir = hrtf2hrir(hrtf)
    elif hrtf.datatype == 'FIR':
        hrir = hrtf
    else: raise ValueError('Unknown datatype.')
    hrir.name = filename
    blocksize = int(hrir[0].n_taps / 2)

    logging.info(f'Resampling sounds from sounds directory')
    for file in sound_path.glob('*.wav'):
        sound = slab.Sound.read(file)
        sound.resample(hrir.samplerate).write(wav_path / dir_name / 'sounds' / file.name)

    # write direct sound filters from hrtf
    write_ds_filter(hrir)
    # write reverb
    write_lr_filter(drr=20, show=show)
    # write pybinsim settings
    write_settings()

def write_ds_filter(hrir):
    # zero pad and write IR to wav and coordinates to filter_list.txt
    logging.info(f'Writing {hrir.name} to wav files and filter_list.txt')
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fname = wav_path / dir_name / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        fir_coefs = hrir[source_idx].data
        # n_bins = hrir[0].n_taps
        # if not n_bins == hrir[source_idx].n_taps:  # interpolate bins if necessary
        #     logging.info(f'Interpolating IR to {n_bins} bins.')
        #     t = numpy.linspace(0, hrir[source_idx].duration, hrir[source_idx].n_taps)
        #     t_interp = numpy.linspace(0, t[-1], n_bins)
        #     fir_coefs = numpy.zeros((n_bins, 2))
        #     for idx in range(2):
        #         fir_coefs[:, idx] = numpy.interp(t_interp, t, hrir[source_idx].data[:, idx])
        # else:
        #     fir_coefs = hrir[source_idx].data
        directional_ir = (slab.Sound(data=fir_coefs))
        directional_ir.write(filename=fname, normalise=False)
        # write to filter_list.txt
        with open(filter_list_fname, 'a') as file:
            file.write(f'DS'
                       f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                       f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                       f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                       f' {coordinates[0]} {coordinates[1]} 0'  # Value 10 - 12: source position[x, y, z]
                       f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                       f' {fname}\n')

def write_lr_filter(drr=20, show=False):
    global reverb_n_samples
    logging.info(f'Writing reverb (DRR = {drr} dB) to wav file and add late reverb to filter_list.txt')
    fname = wav_path / dir_name / 'sounds' / 'reverb_IR.wav'
    reverb = slab.Sound(wav_path / dir_name / 'sounds' / 'reverb.wav').data
    # crop to 100 ms and multiple of block size (hrir taps / 2)
    cropped_len = int((int(hrir.samplerate * 0.1) // blocksize) * blocksize)
    reverb = reverb[:cropped_len]
    reverb_n_samples = len(reverb)
    # ramp reverb tail at the max impulse response
    reverb = slab.Sound(reverb).ramp(duration=0.005, when='onset').data  # ramp reverb onset
    mean_ir_onset = int(numpy.mean(  # average onset time of the direct IR
                [numpy.where(hrir[idx].data == (hrir[idx].data).max())[0][0] for idx in range(hrir.n_sources)]))
    reverb = numpy.concatenate((numpy.zeros((mean_ir_onset, 2)), reverb[:-mean_ir_onset]), axis=0)
    # adjust reverb level to DRR
    ir_level = numpy.mean(
                [20.0 * numpy.log10(numpy.sqrt(numpy.mean(numpy.square(hrir[idx].data))) / 2e-5)
                for idx in range(hrir.n_sources)]) # get mean ir level to apply DRR
    reverb = slab.Sound(data=reverb)
    reverb.level = numpy.mean(ir_level) - drr
    #  write reverb IR and filter list entry
    reverb.write(fname, normalise=False)
    with open(filter_list_fname, 'a') as file:
        file.write(f'LR'
                   f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                   f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 10 - 12: source position[x, y, z]
                   f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                   f' {fname}\n')
    if show:  # plot example IR and reverb envelope
        fig, axis = plt.subplots(nrows=1, ncols=1)
        idx = hrir.get_source_idx((85,95),(-1,1))[0]
        ds = numpy.concatenate((hrir[idx].data, numpy.zeros((len(reverb) - hrir[idx].n_taps, 2))), axis=0)
        ds_lr = ds + reverb.data
        ds_lr = 20.0 * numpy.log10(numpy.abs(ds_lr) / 2e-5)  # convert to dB
        times = numpy.linspace(0, len(ds_lr) / hrir.samplerate, len(ds_lr))
        axis.plot(times, ds_lr)
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Amplitude (dB)')
        fig.show()

def write_hp_filter(mute_ear='left'):
    return


def write_settings():
    # write settings.txt for training and testing:
    logging.info(f'Writing {dir_name}_settings.txt')
    filename = f'{dir_name}_training_settings.txt'
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / dir_name / "sounds" / "noise_pulse.wav")}\n'
            f'blockSize {blocksize}\n'  # low values reduce delay but increase cpu load.
            f'ds_filterSize {hrir[0].n_samples}\n'
            f'early_filterSize {hrir[0].n_samples}\n'
            f'late_filterSize {reverb_n_samples}\n'  # reverb filter
            f'headphone_filterSize {hrir[0].n_samples}\n'  # headphone equalizer
            f'filterSource[mat/wav] wav\n'
            f'filterList {filter_list_fname}\n'
            f'maxChannels 1\n'
            f'samplingRate {int(hrir.samplerate)}\n'
            f'enableCrossfading True\n'
            f'loudnessFactor 0\n'
            f'loopSound False\n'
            # convolver settings 
            f'torchConvolution[cpu/cuda] cpu\n'
            f'torchStorage[cpu/cuda] cpu\n'
            f'pauseConvolution False\n'
            f'pauseAudioPlayback False\n'
            f'useHeadphoneFilter False\n'
            f'ds_convolverActive True\n'
            f'early_convolverActive False\n'
            f'late_convolverActive True\n'
            # osc receiver settings
            f'recv_type osc\n'
            f'recv_protocol udp\n'
            f'recv_ip 127.0.0.1\n'
            f'recv_port 10000\n'
            )

    filename = f'{dir_name}_test_settings.txt'
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
        f'soundfile {str(wav_path / dir_name / "sounds" / "localization.wav")}\n'
        f'blockSize {blocksize}\n'  # low values reduce delay but increase cpu load.
        f'ds_filterSize {hrir[0].n_samples}\n'
        f'early_filterSize {hrir[0].n_samples}\n'
        f'late_filterSize {reverb_n_samples}\n'  # reverb filter
        f'headphone_filterSize {hrir[0].n_samples}\n'  # headphone equalizer
        f'filterSource[mat/wav] wav\n'
        f'filterList {filter_list_fname}\n'
        f'maxChannels 1\n'
        f'samplingRate {int(hrir.samplerate)}\n'
        f'enableCrossfading True\n'
        f'loudnessFactor 0\n'
        f'loopSound False\n'
        # convolver settings 
        f'torchConvolution[cpu/cuda] cpu\n'
        f'torchStorage[cpu/cuda] cpu\n'
        f'pauseConvolution False\n'
        f'pauseAudioPlayback False\n'
        f'useHeadphoneFilter False\n'
        f'ds_convolverActive True\n'
        f'early_convolverActive False\n'
        f'late_convolverActive True\n'
        # osc receiver settings
        f'recv_type osc\n'
        f'recv_protocol udp\n'
        f'recv_ip 127.0.0.1\n'
        f'recv_port 10000\n'
        )