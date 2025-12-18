from hrtf_relearning.hrtf.binsim.tf2ir import hrtf2hrir
from hrtf_relearning.hrtf.binsim.flatten import flatten_dtf
from hrtf_relearning.hrtf.binsim.hrir2wav import *
from pathlib import Path
import slab
import logging
data_dir = Path(hrtf_relearning.__file__).resolve().parent / 'data' / 'hrtf'

def hrtf2binsim(sofa_name, ear=None, reverb=True, hp_filter=True, convolution='cpu', storage='cpu', overwrite=False):
    hrir = slab.HRTF(data_dir / 'sofa' / f'{sofa_name}.sofa')  # read original sofa file
    hrir.name = sofa_name
    slab.set_default_samplerate(hrir.samplerate)
    # convert to IR if necessary
    if hrir.datatype != 'FIR':
        hrir = hrtf2hrir(hrir)
    if ear:        # flatten DTF at specified ear
        hrir = flatten_dtf(hrir, ear)
        hrir.name += f'_{ear}'
    if not (data_dir / 'binsim' / hrir.name).exists() or overwrite: # create wav files
        # create folder structure for HRTF wav files
        (data_dir / 'binsim' / hrir.name / 'IR_data').mkdir(parents=True, exist_ok=True)
        (data_dir / 'binsim' / hrir.name / 'sounds').mkdir(exist_ok=True)
        (data_dir / 'binsim' / hrir.name / 'plot').mkdir(exist_ok=True)
        # ---- write to wav files for pybinsim
        # resample sound files and adjust SPL
        resample_sounds(target_samplerate=hrir.samplerate, target_directory=wav_path / hrir.name / 'sounds')
        # write filters to wav files and filter_list.txt for use with pybinsim
        write_ds_filter(hrir)
        write_lr_filter(hrir, drr=5)  # larger drr results in weaker reverb
        write_hp_filter(hrir=hrir, fname='MYSPHERE_equalization.wav')
        # plot(hrir, title=f'{hrir.name} raw')  # plot raw example IR at 90°
        # write pybinsim settings (should be done after writing wav files)
    write_settings(hrir, reverb, hp_filter, convolution, storage)
    return hrir

def write_settings(hrir, reverb, hp_filter, convolution, storage):
    # write settings.txt for training and testing:
    logging.info(f'Writing {hrir.name}_settings.txt: Reverb: {reverb}, HP filter: {hp_filter} '
                 f'Convolution: {convolution}, Storage: {storage}')
    filename = f'{hrir.name}_training_settings.txt'
    wav_path = data_dir / 'binsim'
    reverb_n_samples = int((int(hrir.samplerate * 0.1) // int(hrir[0].n_taps / 2)) * int(hrir[0].n_taps / 2))
    hp_filtersize = slab.Sound(str(wav_path / hrir.name / "sounds" / "HP_filter.wav")).n_samples
    with open(wav_path / hrir.name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / hrir.name / "sounds" / "noise_pulse.wav")}\n'
            f'blockSize {int(hrir[0].n_taps / 2)}\n'  # low values reduce delay but increase cpu load.
            f'ds_filterSize {hrir[0].n_samples}\n'
            f'early_filterSize {hrir[0].n_samples}\n'
            f'late_filterSize {reverb_n_samples}\n'  # reverb filter
            f'headphone_filterSize {hp_filtersize}\n'  # headphone equalizer
            f'filterSource[mat/wav] wav\n'
            f'filterList {wav_path / hrir.name / f"filter_list_{hrir.name}.txt"}\n'
            f'maxChannels 1\n'
            f'samplingRate {int(hrir.samplerate)}\n'
            f'enableCrossfading True\n'
            f'loudnessFactor 0\n'
            f'loopSound False\n'
            # convolver settings 
            f'torchConvolution[cpu/cuda] {convolution}\n'
            f'torchStorage[cpu/cuda] {storage}\n'
            f'pauseConvolution False\n'
            f'pauseAudioPlayback False\n'
            f'useHeadphoneFilter {hp_filter}\n'
            f'ds_convolverActive True\n'
            f'early_convolverActive False\n'
            f'late_convolverActive {reverb}\n'
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
        f'blockSize {int(hrir[0].n_taps / 2)}\n'  # low values reduce delay but increase cpu load.
        f'ds_filterSize {hrir[0].n_samples}\n'
        f'early_filterSize {hrir[0].n_samples}\n'
        f'late_filterSize {reverb_n_samples}\n'  # reverb filter
        f'headphone_filterSize {hp_filtersize}\n'  # headphone equalizer
        f'filterSource[mat/wav] wav\n'
        f'filterList {wav_path / hrir.name / f"filter_list_{hrir.name}.txt"}\n'
        f'maxChannels 1\n'
        f'samplingRate {int(hrir.samplerate)}\n'
        f'enableCrossfading True\n'
        f'loudnessFactor 0\n'
        f'loopSound False\n'
        # convolver settings 
        f'torchConvolution[cpu/cuda] {convolution}\n'
        f'torchStorage[cpu/cuda] {storage}\n'
        f'pauseConvolution False\n'
        f'pauseAudioPlayback False\n'
        f'useHeadphoneFilter {hp_filter}\n'
        f'ds_convolverActive True\n'
        f'early_convolverActive False\n'
        f'late_convolverActive {reverb}\n'
        # osc receiver settings
        f'recv_type osc\n'
        f'recv_protocol udp\n'
        f'recv_ip 127.0.0.1\n'
        f'recv_port 10000\n'
        )

