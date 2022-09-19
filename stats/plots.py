import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import stats.HRTF_stats as hrtf_corr
import numpy

hrtf_dir = Path.cwd() / 'data' / 'hrtfs' / 'pilot'
loc_dir = Path.cwd() / 'data' / 'localization_accuracy' / 'pilot'

sofa1 = 'jakab_ears_free_12_Sep.sofa'
sofa2 = 'jakab_mold_1_12_Sep.sofa'
kemar_sofa = 'kemar_free.sofa'
hrtf_free = slab.HRTF(hrtf_dir / sofa1)
hrtf_mold = slab.HRTF(hrtf_dir / sofa2)
kemar_hrtf = slab.HRTF(hrtf_dir / kemar_sofa)
sources = hrtf_free.cone_sources(cone=0, full_cone=False)

# plot waterfall
hrtf_free.plot_tf(sources, n_bins=200, kind='waterfall', ear='left', xlim=(4000, 12000))

# compare heatmap
fig, axis = plt.subplots(1, 2, sharey=True)
hrtf_free.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 11000), axis=axis[0])
hrtf_mold.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 11000), axis=axis[1])
fig.text(0.3, 0.9, 'Ears Free', ha='center')
fig.text(0.7, 0.9, 'With Mold', ha='center')

# plot hrtf correlation
corr_mtx = hrtf_corr.dtf_correlation(hrtf_free, hrtf_free, show=True, bandwidth=None, n_bins=96)

# plot vsi across 1/2 octave frequency bands
hrtf = kemar_hrtf
n_bins = 96
dtfs = hrtf.tfs_from_sources(sources, n_bins)
frequencies = numpy.linspace(0, hrtf[0].frequencies[-1], n_bins)
bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
vsi = numpy.zeros(len(bandwidths))
# extract vsi for each band
for idx, bw in enumerate(bandwidths):
    dtf_band = dtfs[numpy.logical_and(frequencies >= bw[0], frequencies <= bw[1])]
    sum_corr = 0
    n = 0
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sum_corr += numpy.corrcoef(dtf_band[:, i], dtf_band[:, j])[1, 0]
            n += 1
    vsi[idx] = 1 - sum_corr / n
# plot
fig, axis = plt.subplots()
axis.plot(vsi)
axis.set_xticks([0, 1, 2, 3, 4])
labels = [item.get_text() for item in axis.get_xticklabels()]
for idx, band in enumerate(bandwidths / 1000):
    labels[idx] = '%.1f - %.1f' % (band[0], band[1])
axis.set_xticklabels(labels)
axis.set_xlabel('Frequency bands (kHz)')
axis.set_ylabel('VSI')

