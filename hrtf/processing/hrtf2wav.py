from pathlib import Path
import numpy
import slab
import time
from hrtf.processing.tf2ir import tf2ir
wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
sound_path = Path.cwd() / 'data' / 'sounds'

def make_wav(filename):
    if not (wav_path /filename).exists():
        hrtf2wav(f'{filename}.sofa')

def hrtf2wav(filename, n_bins=None, add_itd=True):
    """
    Convert HRIR filters from a sofa file to wav files for use with pybinsim.
    """
    # create folder structure for HRTF
    dir_name = Path(filename).stem
    if not (wav_path / dir_name).exists():
        (wav_path / dir_name).mkdir(exist_ok=True)
        (wav_path / dir_name / 'IR_data').mkdir(exist_ok=True)
        (wav_path / dir_name / 'sounds').mkdir(exist_ok=True)
    hrtf = slab.HRTF(sofa_path / filename)
    sources = hrtf.sources.vertical_polar
    sources[sources[:, 0] < 0, 0] = sources[sources[:, 0] < 0, 0] + 360  # convert sources to sofa convention (0, 360)°
    slab.set_default_samplerate(hrtf.samplerate)

    # convert to TF to IR
    if hrtf.datatype not in ['TF', 'FIR']:
        raise ValueError('Unknown datatype.')
    if hrtf.datatype == 'TF':
        hrtf = tf2ir(hrtf)
    if n_bins is None:
        n_bins = hrtf[0].n_taps
    else:
        print(f'interpolating IR to {n_bins} bins.')

    # write IR to wav and coordinates to filter_list.txt
    print(f'Writing wav files from {filename} and filter list "{dir_name}.text"')
    for source_idx in range(hrtf.n_sources):
        coordinates = sources[source_idx]
        if not n_bins == hrtf[source_idx].n_taps:  # interpolate bins if necessary
            t = numpy.linspace(0, hrtf[source_idx].duration, hrtf[source_idx].n_taps)
            t_interp = numpy.linspace(0, t[-1], n_bins)
            fir_coefs = numpy.zeros((n_bins, 2))
            for idx in range(2):
                fir_coefs[:, idx] = numpy.interp(t_interp, t, hrtf[source_idx].data[:, idx])
        else:
            fir_coefs = hrtf[source_idx].data
        fname = wav_path / dir_name / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        slab.Sound(data=fir_coefs).write(filename=fname)
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

    # resample sounds from sound folder
    for file in sound_path.glob('*.wav'):
        sound = slab.Sound.read(file)
        sound.resample(hrtf.samplerate).write(wav_path / dir_name / 'sounds' / file.name)

    # write settings.txt:
    filename = f'{dir_name}_test_settings.txt'
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
        f'soundfile {str(wav_path / dir_name / "sounds" / "localization.wav")}\n'
        f'blockSize {int(hrtf[0].n_samples / 2)}\n' # low values reduce delay but increase cpu load.
        f'ds_filterSize {hrtf[0].n_samples}\n'
        f'early_filterSize {hrtf[0].n_samples}\n'
        f'late_filterSize {hrtf[0].n_samples}\n'  # reverb filter
        f'headphone_filterSize {hrtf[0].n_samples}\n'  # headphone equalizer
        f'filterSource[mat/wav] wav\n'
        f'filterList {filter_list_fname}\n'
        f'maxChannels 1\n'
        f'samplingRate {int(hrtf.samplerate)}\n'
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

    # write settings.txt:
    filename = f'{dir_name}_training_settings.txt'
    print(f'Writing {dir_name}_settings.txt ...')
    with open(wav_path / dir_name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / dir_name / "sounds" / "pinknoise.wav")}\n'
            f'blockSize {int(hrtf[0].n_samples / 2)}\n'  # low values reduce delay but increase cpu load.
            f'ds_filterSize {hrtf[0].n_samples}\n'
            f'early_filterSize {hrtf[0].n_samples}\n'
            f'late_filterSize {hrtf[0].n_samples}\n'  # reverb filter
            f'headphone_filterSize {hrtf[0].n_samples}\n'  # headphone equalizer
            f'filterSource[mat/wav] wav\n'
            f'filterList {filter_list_fname}\n'
            f'maxChannels 1\n'
            f'samplingRate {int(hrtf.samplerate)}\n'
            f'enableCrossfading True\n'
            f'loudnessFactor 0\n'
            f'loopSound True\n'
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

        # if add_itd:
        #     itd = slab.Binaural.azimuth_to_itd(azimuth=coordinates[0], head_radius=11)  # head radius in cm
        #     if itd >= 0:  # add left delay
        #         delay = numpy.array((int(itd / 2 * hrtf.samplerate), 0))
        #     elif itd < 0:  # add right delay
        #         delay = numpy.array((0, int(- itd / 2 * hrtf.samplerate)))
        #     fir_coefs = numpy.vstack((fir_coefs, delay))  # add group delays
        # else:
        #     fir_coefs = numpy.vstack((fir_coefs, numpy.zeros(2)))  # add zero group delays

