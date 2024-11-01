from pathlib import Path
import numpy
import slab
from array import array
from dev.hrtf.tf2ir import tf2ir
fs = 48828
wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

def hrtf2wav(hrtf, filename, n_bins=None, add_itd=True):
    """
    Convert HRIR sofa to binary file to be used with HRTFCoef in RPvdsEx.
    """
    if hrtf.datatype not in ['TF', 'FIR']:
        raise ValueError('Unknown datatype.')
    if hrtf.datatype == 'TF':
        hrtf = tf2ir(hrtf)
    if n_bins is None:
        n_bins = hrtf[0].n_taps
    else:
        print(f'interpolating IR to {n_bins} bins.')

    # create subfolder
    if not (wav_path / filename).exists():
        (wav_path / filename).mkdir(exist_ok=False)
        (wav_path / filename / 'IR_data').mkdir(exist_ok=False)
    # write to pybinsim settings.txt:
    filter_list_fname = wav_path / filename / f"filter_list_{filename}.txt"
    with open(wav_path / filename / f'{filename}_settings.txt', 'w') as file:
        file.write(
        f'soundfile C:/Users/admpf18fixi/Projects/hrtf_relearning/data/sounds/pinknoise_44100.wav\n'
        f'blockSize {int(hrtf[0].n_samples / 2)}\n'
        f'filterSize {hrtf[0].n_samples}\n'
        f'filterList {filter_list_fname}\n'
        'maxChannels 1\n'
        f'samplingRate {int(hrtf.samplerate)}\n'
        'enableCrossfading True\n'
        'useHeadphoneFilter False\n'
        'loudnessFactor 0.5\n'
        'loopSound True\n'
        )

    # write IR Filters to wav files
    for source_idx in range(hrtf.n_sources):
        coordinates = hrtf.sources.vertical_polar[source_idx].astype('int')
        if not n_bins == hrtf[source_idx].n_taps:  # interpolate bins if necessary
            t = numpy.linspace(0, hrtf[source_idx].duration, hrtf[source_idx].n_taps)
            t_interp = numpy.linspace(0, t[-1], n_bins)
            fir_coefs = numpy.zeros((n_bins, 2))
            for idx in range(2):
                fir_coefs[:, idx] = numpy.interp(t_interp, t, hrtf[source_idx].data[:, idx])
        else:
            fir_coefs = hrtf[source_idx].data

        # if add_itd:
        #     itd = slab.Binaural.azimuth_to_itd(azimuth=coordinates[0], head_radius=11)  # head radius in cm
        #     if itd >= 0:  # add left delay
        #         delay = numpy.array((int(itd / 2 * hrtf.samplerate), 0))
        #     elif itd < 0:  # add right delay
        #         delay = numpy.array((0, int(- itd / 2 * hrtf.samplerate)))
        #     fir_coefs = numpy.vstack((fir_coefs, delay))  # add group delays
        # else:
        #     fir_coefs = numpy.vstack((fir_coefs, numpy.zeros(2)))  # add zero group delays

        fname = wav_path / filename / 'IR_data' / f'{coordinates[0]}_{coordinates[1]}.wav'
        # write to wav
        slab.Sound(data=fir_coefs, samplerate=int(hrtf.samplerate)).write(filename=fname)

        # write to filter_list
        with open(filter_list_fname, 'a') as file:
            file.write(f'{source_idx} 0 0 0 0 0 {fname}\n')

