from matplotlib import pyplot as plt
import numpy
from pathlib import Path
import pyfar
wav_path = Path.cwd() / 'data' / 'hrtf' / 'binsim'

def plot(hrir, title):
    try:
        az = 90
        ele = 0
        src_idx = hrir.get_source_idx(az, ele)[0]
    except IndexError:
        src_idx = -1
    fig, ax = plt.subplots()
    times = numpy.linspace(0, hrir[src_idx].n_samples/hrir.samplerate,  hrir[src_idx].n_samples)
    ax.plot(times, (hrir[src_idx]), label=['left', 'right'])  # convert to dB
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [dB]')
    ax.legend()
    ax.set_title(f'{title} at ({az:.2f}, {ele:.1f})')
    plt.savefig(wav_path / hrir.name / 'plot' / f'{hrir.name}_{title}.png')
    return ax

def plot_reverb(hrir, reverb):
    # fig, axis = plt.subplots(nrows=1, ncols=1)
    # try:
    #     src_idx = hrir.get_source_idx((85, 95), (-5, 5))[0]
    #     src = hrir.sources.vertical_polar[src_idx]  # pick a source 90° to the right
    # except IndexError:
    #     src_idx = -1
    #     src = hrir.sources.vertical_polar[src_idx]  # pick a random source instead
    # az = src[0]
    # ele = src[1]
    # ds = numpy.concatenate((hrir[src_idx].data, numpy.zeros((len(reverb) - hrir[src_idx].n_taps, 2))), axis=0)
    # ds_lr = ds + reverb.data
    # ds_lr =  20.0 * numpy.log10(numpy.maximum(numpy.abs(ds_lr), 1e-12))  # convert to dB
    # times = numpy.linspace(0, len(ds_lr) / hrir.samplerate, len(ds_lr))
    # axis.plot(times, ds_lr)
    # axis.set_xlabel('Time (s)')
    # axis.set_ylabel('Amplitude (dB)')
    # axis.set_title(f'{hrir.name} final IR with reverb at ({az:.2f}, {ele:.1f})')
    # plt.savefig(wav_path / hrir.name / 'plot' / f'{hrir.name}_final_with_reverb.png')
    #
    fig, axis = plt.subplots(nrows=1, ncols=1)

    try:
        src_idx = hrir.get_source_idx((85, 95), (-5, 5))[0]
        src = hrir.sources.vertical_polar[src_idx]
    except IndexError:
        src_idx = -1
        src = hrir.sources.vertical_polar[src_idx]

    az = src[0]
    ele = src[1]

    ds = numpy.concatenate(
        (
            hrir[src_idx].data,
            numpy.zeros((len(reverb) - hrir[src_idx].n_taps, 2))
        ),
        axis=0
    )

    ds_lr = ds + reverb  # not reverb.data

    def moving_rms(x, win=32):
        kernel = numpy.ones(win) / win
        return numpy.sqrt(numpy.convolve(x**2, kernel, mode="same"))

    env_l = moving_rms(ds_lr[:, 0], win=32)
    env_r = moving_rms(ds_lr[:, 1], win=32)

    env_l_db = 20.0 * numpy.log10(numpy.maximum(env_l, 1e-12))
    env_r_db = 20.0 * numpy.log10(numpy.maximum(env_r, 1e-12))

    times = numpy.arange(len(ds_lr)) / hrir.samplerate

    axis.plot(times, env_l_db, label="left")
    axis.plot(times, env_r_db, label="right")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("RMS envelope (dB)")
    axis.set_title(f"{hrir.name} final IR with reverb at ({az:.2f}, {ele:.1f})")
    axis.legend()

    plt.savefig(wav_path / hrir.name / "plot" / f"{hrir.name}_final_with_reverb.png")

def plot_reverb(hrir, reverb):

    fig, axis = plt.subplots(1, 1)

    try:
        src_idx = hrir.get_source_idx((85, 95), (-5, 5))[0]
        src = hrir.sources.vertical_polar[src_idx]
    except IndexError:
        src_idx = -1
        src = hrir.sources.vertical_polar[src_idx]

    az = src[0]
    ele = src[1]

    ds_dB = numpy.log10(20*numpy.abs(hrir[src_idx].data))
    final_ir = numpy.concatenate(
        (
            ds_dB,
            numpy.zeros((len(reverb) - hrir[src_idx].n_taps, 2))
        ),
        axis=0
    )

    final_ir += reverb

    final_ir_pf = pyfar.Signal(
        final_ir.T,
        sampling_rate=hrir.samplerate
    )

    pyfar.plot.time_freq(final_ir_pf)
    plt.suptitle(f"{hrir.name} final IR with reverb at ({az:.2f}, {ele:.1f})")
    plt.savefig(
        wav_path / hrir.name / "plot" / f"{hrir.name}_final_with_reverb_timefreq.svg"
    )
