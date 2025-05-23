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
def make_wav(filename, overwrite=False):
    if overwrite:
        hrtf2wav(f'{filename}.sofa')
    else:
        if not (wav_path / filename).exists():
            hrtf2wav(f'{filename}.sofa')

def hrtf2wav(filename, n_bins=None):
    """
    Convert HRIR filters from a sofa file to wav files for use with pybinsim.
    """
    ir_level = 10  # todo has no effect as of now - maybe the wav file writing overrides levels?
    reverb_level = 10
    # create folder structure for HRTF
    dir_name = Path(filename).stem
    if not (wav_path / dir_name).exists():
        (wav_path / dir_name).mkdir(exist_ok=True)
        (wav_path / dir_name / 'IR_data').mkdir(exist_ok=True)
        (wav_path / dir_name / 'sounds').mkdir(exist_ok=True)
    hrtf = slab.HRTF(sofa_path / filename)
    slab.set_default_samplerate(hrtf.samplerate)

    # convert to TF to IR
    if hrtf.datatype == 'TF':
        hrir = tf2ir(hrtf)
    elif hrtf.datatype == 'FIR':
        hrir = hrtf
    else: raise ValueError('Unknown datatype.')

    if n_bins is None:
        n_bins = hrir[0].n_taps
    else:
        print(f'interpolating IR to {n_bins} bins.')
        # todo

    logging.info(f'Resampling sounds from sounds directory ...')
    for file in sound_path.glob('*.wav'):
        sound = slab.Sound.read(file)
        sound.resample(hrir.samplerate).write(wav_path / dir_name / 'sounds' / file.name)

    # write IR to wav and coordinates to filter_list.txt
    logging.info(f'Writing {filename} to wav files and filter_list.txt ...')
    for source_idx in range(hrtf.n_sources):
        coordinates = hrir.sources.vertical_polar[source_idx]
        if not n_bins == hrir[source_idx].n_taps:  # interpolate bins if necessary
            t = numpy.linspace(0, hrir[source_idx].duration, hrir[source_idx].n_taps)
            t_interp = numpy.linspace(0, t[-1], n_bins)
            fir_coefs = numpy.zeros((n_bins, 2))
            for idx in range(2):
                fir_coefs[:, idx] = numpy.interp(t_interp, t, hrir[source_idx].data[:, idx])
        else:
            fir_coefs = hrir[source_idx].data
        fname = wav_path / dir_name / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        directional_ir = (slab.Sound(data=fir_coefs))
        directional_ir.level = ir_level
        directional_ir.write(filename=fname)
        # write to filter_list.txt
        filter_list_fname = wav_path / dir_name / f"filter_list_{dir_name}.txt"
        with open(filter_list_fname, 'a') as file:
            file.write(f'DS'
                       f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                       f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                       f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                       f' {coordinates[0]} {coordinates[1]} 0'  # Value 10 - 12: source position[x, y, z]
                       f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                       f' {fname}\n')

    # reverb tail:
    # crop duration, interpolate to n_bins and rescale level
    reverb = slab.Sound(wav_path / dir_name / 'sounds' / 'reverb.wav').data
    # duration = 0.1
    fname = wav_path / dir_name / 'sounds' / 'reverb_IR.wav'
    # reverb = reverb[:int(hrtf.samplerate * duration)]  # crop to 100 ms
    if not n_bins == reverb.shape[0]:  # interpolate bins if necessary
        t = numpy.linspace(0, 1, reverb.shape[0])
        t_interp = numpy.linspace(0, 1, n_bins)
        reverb_interp = numpy.zeros((n_bins, 2))
        for idx in range(2):
            reverb_interp[:, idx] = numpy.interp(t_interp, t, reverb[:, idx])
    # reverb = reverb_interp * level / numpy.max(numpy.abs(reverb_interp))  # rescale
    reverb = slab.Sound(data=reverb_interp)
    reverb.level = reverb_level
    reverb.write(fname)
    #  write IR and filter list entry
    with open(filter_list_fname, 'a') as file:
        file.write(f'LR'
                   f' 0 0 0'  # Value 1 - 3: listener orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 4 - 6: listener position[x, y, z]
                   f' 0 0 0'  # Value 7 - 9: source orientation[yaw, pitch, roll]
                   f' 0 0 0'  # Value 10 - 12: source position[x, y, z]
                   f' 0 0 0'  # Value 13 - 15: custom values[a, b, c]
                   f' {fname}\n')

    # write settings.txt for training and testing:
    logging.info(f'Writing {dir_name}_settings.txt ...')
    filename = f'{dir_name}_test_settings.txt'
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
        f'soundfile {str(wav_path / dir_name / "sounds" / "localization.wav")}\n'
        f'blockSize {int(hrir[0].n_samples / 2)}\n' # low values reduce delay but increase cpu load.
        f'ds_filterSize {hrir[0].n_samples}\n'
        f'early_filterSize {hrir[0].n_samples}\n'
        f'late_filterSize {hrir[0].n_samples}\n'  # reverb filter
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
        f'late_convolverActive False\n'
        # osc receiver settings
        f'recv_type osc\n'
        f'recv_protocol udp\n'
        f'recv_ip 127.0.0.1\n'
        f'recv_port 10000\n'
        )

    filename = f'{dir_name}_training_settings.txt'
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / dir_name / "sounds" / "noise_pulse.wav")}\n'
            f'blockSize {int(hrir[0].n_samples / 2)}\n'  # low values reduce delay but increase cpu load.
            f'ds_filterSize {hrir[0].n_samples}\n'
            f'early_filterSize {hrir[0].n_samples}\n'
            f'late_filterSize {hrir[0].n_samples}\n'  # reverb filter
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

