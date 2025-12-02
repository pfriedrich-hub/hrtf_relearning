from hrtf.analysis.plot_ir import plot
from hrtf.processing.binsim.tf2ir import hrtf2hrir
from hrtf.processing.binsim.flatten import flatten_dtf
from hrtf.processing.binsim.hrir2wav import hrir2wav
from pathlib import Path
import slab
import logging
data_dir = Path.cwd() / 'data' / 'hrtf'

def hrtf2binsim(sofa_name, ear=None, reverb=True, overwrite=False):
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
        # write to wav files for pybinsim
        hrir = hrir2wav(hrir)
        plot(hrir, title=f'{hrir.name} raw')  # plot raw example IR at 90°
    write_settings(hrir, reverb)  # write pybinsim settings
    return hrir

def write_settings(hrir, reverb):
    # write settings.txt for training and testing:
    logging.info(f'Writing {hrir.name}_settings.txt')
    filename = f'{hrir.name}_training_settings.txt'
    wav_path = data_dir / 'binsim'
    reverb_n_samples = int((int(hrir.samplerate * 0.1) // int(hrir[0].n_taps / 2)) * int(hrir[0].n_taps / 2))
    with open(wav_path / hrir.name / filename, 'w') as file:
        file.write(
            f'soundfile {str(wav_path / hrir.name / "sounds" / "noise_pulse.wav")}\n'
            f'blockSize {int(hrir[0].n_taps / 2)}\n'  # low values reduce delay but increase cpu load.
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
            f'useHeadphoneFilter False\n'
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
        f'useHeadphoneFilter False\n'
        f'ds_convolverActive True\n'
        f'early_convolverActive False\n'
        f'late_convolverActive {reverb}\n'
        # osc receiver settings
        f'recv_type osc\n'
        f'recv_protocol udp\n'
        f'recv_ip 127.0.0.1\n'
        f'recv_port 10000\n'
        )

