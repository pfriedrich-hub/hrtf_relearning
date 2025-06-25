import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy
from pathlib import Path
wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav'

def plot(hrir, title):
    src_idx = hrir.get_source_idx((85, 95), (-5, 5))[0]  # pick a source 90° to the left
    fig, ax = plt.subplots()
    ax.plot(hrir[src_idx], label=['left', 'right'])
    ax.legend()
    ax.set_title(f'{title} at {hrir.sources.vertical_polar[src_idx][:2]}')
    plt.savefig(wav_path / hrir.name / 'plot' / f'{hrir.name}_{title}.png')
    return ax

def plot_reverb(hrir, reverb):
    fig, axis = plt.subplots(nrows=1, ncols=1)
    idx = hrir.get_source_idx((85, 95), (-5, 5))[0]
    ds = numpy.concatenate((hrir[idx].data, numpy.zeros((len(reverb) - hrir[idx].n_taps, 2))), axis=0)
    ds_lr = ds + reverb.data
    ds_lr = 20.0 * numpy.log10(numpy.abs(ds_lr) / 2e-5)  # convert to dB
    times = numpy.linspace(0, len(ds_lr) / hrir.samplerate, len(ds_lr))
    axis.plot(times, ds_lr)
    axis.set_xlabel('Time (s)')
    axis.set_ylabel('Amplitude (dB)')
    axis.set_title(f'{hrir.name} final IR + reverb')
    fig.show()
    plt.savefig(wav_path / hrir.name / 'plot' / f'{hrir.name} final with reverb.png')