import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import numpy
import slab
import time
import logging
from hrtf.processing.tf2ir import *
from hrtf.processing.flatten_hrir import *


wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
sound_path = Path.cwd() / 'data' / 'sounds'

# wrapper for hrtf2wav
def make_wav(filename, overwrite=False, ear=None, show=False):
    if not (wav_path / filename).exists() or overwrite:
        hrtf2wav(filename, ear, show)

def hrtf2wav(filename, ear, show):
    """
    Convert HRIR filters from a sofa file to wav files for use with pybinsim.
    Args:
        filename (str): HRIR sofa file
        ear (str, optional): flatten DTFs at specified Ear (and keep ITD / ILD)
        show (bool, optional): whether to show a plot of a final IR with reverb tail
    """
    global hrir, blocksize

    # load and process HRTF (convert to HRIR, interpolate, flatten ...)
    hrtf = slab.HRTF(sofa_path / f'{filename}.sofa')
    slab.set_default_samplerate(hrtf.samplerate)
    if hrtf.datatype == 'TF':
        hrir = hrtf2hrir(hrtf)
    elif hrtf.datatype == 'FIR':  # convert to IR
        hrir = hrtf
    else: raise ValueError('Unknown datatype.')
    if ear:  # flatten DTFs of specified channel
        logging.info(f'Flatten DTFs at the {ear} ear')
        hrir = flatten_dtf(hrir, ear=ear)
        filename += f'_{ear[0]}_flat'
    hrir.name = filename
    blocksize = int(hrir[0].n_taps / 2)

    # create folder structure for HRTF wav files
    if not (wav_path / filename).exists():
        (wav_path / filename / 'IR_data').mkdir(parents=True, exist_ok=True)
        (wav_path / filename / 'sounds').mkdir(exist_ok=True)

    # write files
    write_ds_filter(hrir)  # write direct sound filters from hrtf
    for file in sound_path.glob('*.wav'): # resample sound files
        sound = slab.Sound.read(file)
        sound.resample(hrir.samplerate).write(wav_path / filename / 'sounds' / file.name)
    write_lr_filter(drr=20, show=show)  # write reverb
    # write_hp_filter(mute_ear='left')
    write_settings()  # write pybinsim settings

def write_ds_filter(hrir):
    # zero pad and write IR to wav and coordinates to filter_list.txt
    logging.info(f'Writing {hrir.name} to wav files and filter_list.txt')
    for source_idx in range(hrir.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        fname = wav_path / hrir.name / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        fir_coefs = hrir[source_idx].data
        directional_ir = (slab.Sound(data=fir_coefs))
        directional_ir.write(filename=fname, normalise=False)  # write IR to wav
        with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:  # write to filter_list.txt
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
    fname = wav_path / hrir.name / 'sounds' / 'reverb_IR.wav'
    reverb = slab.Sound(wav_path / hrir.name / 'sounds' / 'reverb.wav').data  # load reverb ir
    # crop to 100 ms and multiple of block size (hrir taps / 2)
    cropped_len = int((int(hrir.samplerate * 0.1) // blocksize) * blocksize)
    reverb = reverb[:cropped_len]
    reverb_n_samples = len(reverb)
    # ramp up reverb tail starting at the max impulse response
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
    with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:
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

def write_hp_filter(mute_ear=None):
    """
    Apply headphone filter to mute a channel
    Arguments:
        mute_ear (String): Whether to mute the left or the right ear. If None, don't apply the HP filter.
    """
    logging.info(f'Writing headphone filters')
    fname = wav_path / hrir.name / 'sounds' / 'hp_filter.wav'
    hp_filter = numpy.zeros((2, hrir[0].n_samples))
    hp_filter[:,0] = 1
    if mute_ear == 'left':
        hp_filter[0] *= 0
    elif mute_ear == 'right':
        hp_filter[1] *= 0
    hp_filter = slab.Sound(data=hp_filter)
    hp_filter.write(fname, normalise=False)  # write hp filter to wav
    with open(wav_path / hrir.name / f"filter_list_{hrir.name}.txt", 'a') as file:  # write filename to filter list
        file.write(f'HP {fname}\n')

def write_settings():
    # write settings.txt for training and testing:
    logging.info(f'Writing {hrir.name}_settings.txt')
    filename = f'{hrir.name}_training_settings.txt'
    with open(wav_path / hrir.name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / hrir.name / "sounds" / "noise_pulse.wav")}\n'
            f'blockSize {blocksize}\n'  # low values reduce delay but increase cpu load.
            f'ds_filterSize {hrir[0].n_samples}\n'
            f'early_filterSize {hrir[0].n_samples}\n'
            f'late_filterSize {reverb_n_samples}\n'  # reverb filter
            f'headphone_filterSize {hrir[0].n_samples}\n'  # headphone equalizer
            f'filterSource[mat/wav] wav\n'
            f'filterList {wav_path / hrir.name / f"filter_list_{hrir.name}.txt"}\n'
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
            f'useHeadphoneFilter True\n'
            f'ds_convolverActive True\n'
            f'early_convolverActive False\n'
            f'late_convolverActive True\n'
            # osc receiver settings
            f'recv_type osc\n'
            f'recv_protocol udp\n'
            f'recv_ip 127.0.0.1\n'
            f'recv_port 10000\n'
            )

    filename = f'{hrir.name}_test_settings.txt'
    with open(wav_path / hrir.name / filename, 'w') as file:
        file.write(
        f'soundfile {str(wav_path / hrir.name / "sounds" / "localization.wav")}\n'
        f'blockSize {blocksize}\n'  # low values reduce delay but increase cpu load.
        f'ds_filterSize {hrir[0].n_samples}\n'
        f'early_filterSize {hrir[0].n_samples}\n'
        f'late_filterSize {reverb_n_samples}\n'  # reverb filter
        f'headphone_filterSize {hrir[0].n_samples}\n'  # headphone equalizer
        f'filterSource[mat/wav] wav\n'
        f'filterList {wav_path / hrir.name / f"filter_list_{hrir.name}.txt"}\n'
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
        f'useHeadphoneFilter True\n'
        f'ds_convolverActive True\n'
        f'early_convolverActive False\n'
        f'late_convolverActive True\n'
        # osc receiver settings
        f'recv_type osc\n'
        f'recv_protocol udp\n'
        f'recv_ip 127.0.0.1\n'
        f'recv_port 10000\n'
        )