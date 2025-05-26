import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from win32com.client import Dispatch
from dev.hrtf.processing.hrtf2rcx import hrtf2binary
from pathlib import Path
import numpy
import slab
import random
data_path = Path.cwd() / 'data'

# select sofa file to test
filename = 'MRT01'
sofa_path = data_path / 'hrtf' / 'sofa' / f'{filename}.sofa'
binary_path = data_path / 'hrtf' / 'binary' / f'{filename}.f32'
n_bins = None  # number of bins used for binary file
hrtf = slab.HRTF(data_path / 'hrtf' / 'sofa' / f'{filename}.sofa')

# connect and write fm sweep to processor
init = numpy.zeros((6))
RM1 = Dispatch('RPco.X')
init[0] = RM1.ConnectRM1('USB', 1)
init[1] = RM1.ClearCOF()
init[2] = RM1.LoadCOF(data_path / 'rcx' / 'test_fir_coef.rcx')
fs = RM1.GetSFreq()
slab.set_default_samplerate(fs)
signal_in = slab.Sound.chirp(duration=1.0, from_frequency=50, to_frequency=20000)
init[3] = RM1.SetTagVal('signal_n', signal_in.n_samples)
init[4] = RM1._oleobj_.InvokeTypes(15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
                    'signal', 0,  signal_in.data.flatten())
init[5] = RM1.Run()
if not all(init):
    print('processor could not be initialized')


def test_hrtf2rcx(hrtf):
    # create binary from sofa
    hrtf2binary(hrtf, filename, n_bins=n_bins)
    # select random spatial filter location to test
    sources = hrtf.sources.vertical_polar
    az = random.choice(numpy.unique(sources[:, 0]))
    ele = random.choice(numpy.unique(sources[:, 1]))
    source_idx, = numpy.where((sources[:, 1]==ele) & (sources[:, 0]==az))[0]
    # ---- filter, play and record signal on processor --- #
    RM1.SetTagVal('az', az)
    RM1.SetTagVal('ele', ele)
    RM1.SoftTrg(1)
    while True:
        if RM1.GetTagVal('running') == 0:
            last_loop = True
        else:
            last_loop = False
        if last_loop:
            break
    signal_out = numpy.array((RM1.ReadTagV('filtered_left', 0, signal_in.n_samples),
                       RM1.ReadTagV('filtered_right', 0, signal_in.n_samples)))

    # ---- plot slab filtered vs dsp filtered signal for comparison ---- #
    filtered = hrtf.apply(source_idx, signal_in)  # signal filtered by slab HRTF.apply method
    recording = slab.Binaural(data=signal_out, samplerate=fs)  # signal filtered by RPvdsEx HrtfFir component
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    filtered.spectrum(axis=axes[0])
    recording.spectrum(axis=axes[1])
    fig.suptitle('directionally filtered fm sweep spectra\nfor {:.2f}° azimuth and {:.2f}° elevation'.format(az, ele))
    axes[0].set_title('slab filtered')
    axes[1].set_title('DSP filtered')
    fig.tight_layout()




"""
    # RM1.SetTagVal('component_nr', 17) change nr of hrtfcoef component to use to dynamically switch between hrtfs

   # ------ retrieve FIR at selected sound location and convert to DTF ----- #
    source_idx, = numpy.where((sources[:, 1]==ele) & (sources[:, 0]==az))[0]
    fir_in = hrtf[source_idx].data
    # dtfs = hrtf.tfs_from_sources([source_idx], ear='both', n_bins=None)[0]
    freqs_in, dtfs_in = hrtf[source_idx].tf(n_bins=n_bins, show=False)

    # compute DTF and impulse response
    sig_in_fft = numpy.fft.rfft(signal_in.data.flatten())
    dtfs_out = numpy.fft.rfft(signal_out) / sig_in_fft  # compute DTF from signal / recording
    fir_out = numpy.fft.irfft(dtfs_out)  # compute FIR from DTF
    dtfs_out = 20 * numpy.log10(numpy.abs(dtfs_out))  # convert TF to power
    freqs_out = numpy.fft.rfftfreq(signal_out.shape[1], d=1 / fs)
    
    
# # DTFs
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
# axes[0].plot(freqs_out, dtfs_out[0])
# axes[1].plot(freqs_out, dtfs_out[1])
# fig.suptitle('measured DTFs for {}° azimuth and {}° elevation'.format(az, ele))
# axes[0].set_title('left')
# axes[1].set_title('right')
#
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
# axes[0].plot(freqs_in, dtfs_in[:, 0])
# axes[1].plot(freqs_in, dtfs_in[:, 1])
# fig.suptitle('DTFs for {}° azimuth and {}° elevation'.format(az, ele))
# axes[0].set_title('left')
# axes[1].set_title('right')

# IR
# plt.figure()
# plt.plot(fir_out[0])
#
# plt.figure()
# plt.plot(fir_coefs[:, 0])
"""