import os
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import stats.HRTF_stats as hrtf_corr
import stats.localization_stats as loc_acc
import numpy

### ---- HRTF plots ----- ###
hrtf_dir = Path.cwd() / 'data' / 'hrtfs' / 'pilot'

sofa1 = 'meike_mold_1_30.09.sofa'
sofa2 = 'varvara_mold_1_23.09.sofa'
kemar_sofa = 'kemar_free.sofa'
hrtf_free = slab.HRTF(hrtf_dir / sofa1)
hrtf_mold = slab.HRTF(hrtf_dir / sofa2)
kemar_hrtf = slab.HRTF(hrtf_dir / kemar_sofa)
sources = hrtf_free.cone_sources(cone=0, full_cone=False)
hrtf_free, hrtf_mold = hrtf_free.diffuse_field_equalization(), hrtf_mold.diffuse_field_equalization()

# plot waterfall
hrtf_free.plot_tf(sources, n_bins=200, kind='image', ear='left', xlim=(4000, 16000))

# compare heatmap of hrtf free and with mold
fig, axis = plt.subplots(2, 2, sharey=True)
hrtf_free.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 0])
hrtf_mold.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 1])
fig.text(0.3, 0.9, 'Ear Free', ha='center')
fig.text(0.7, 0.9, 'With Mold', ha='center')
# plot hrtf autocorrelation free
corr_mtx, cbar_1 = hrtf_corr.dtf_correlation(hrtf_free, hrtf_free, show=True, bandwidth=None,
                                     n_bins=96, axis=axis[1, 0])
# plot hrtf correlation free vs mold
cross_corr_mtx, cbar_2 = hrtf_corr.dtf_correlation(hrtf_free, hrtf_mold, show=True, bandwidth=None,
                                           n_bins=96, axis=axis[1, 1])
fig.text(0.3, 0.5, 'Autocorrelation Ear Free', ha='center')
fig.text(0.7, 0.5, 'Correlation Free vs. Mold', ha='center')
cbar_1.remove()

# axis[1, 1]


# plot vsi across 1/2 octave frequency bands
hrtf = hrtf_free
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
axis.plot(vsi, c='k')
# axis.errorbar(numpy.mean(vsi, axis=1), capsize=3, yerr=numpy.abs(numpy.diff(vsi, axis=0)),
#                fmt="o", c='0.6', elinewidth=0.5, markersize=3)
axis.set_xticks([0, 1, 2, 3, 4])
labels = [item.get_text() for item in axis.get_xticklabels()]
for idx, band in enumerate(bandwidths / 1000):
    labels[idx] = '%.1f - %.1f' % (band[0], band[1])
axis.set_xticklabels(labels)
axis.set_xlabel('Frequency bands (kHz)')
axis.set_ylabel('VSI')



# plot Elevation Gain, RMSE and response variability across experiment
# get files
loc_dir = Path.cwd() / 'data' / 'localization_data' / 'pilot'
subject_list = next(os.walk(loc_dir))[1]
files = []
ele_gain = numpy.zeros((len(subject_list), 6))  # for now, 6 measurements per participant
rmse = numpy.zeros((len(subject_list), 6))
sd = numpy.zeros((len(subject_list), 6))
for subj_idx, subject_folder in enumerate(subject_list):
    subj_files = os.listdir(str(loc_dir / subject_folder))
    if '.DS_Store' in subj_files: subj_files.remove('.DS_Store')
    subj_files.sort()
    files.append(subj_files)
    for file_idx, subj_file in enumerate(subj_files):
        ele_gain[subj_idx, file_idx], rmse[subj_idx, file_idx], sd[subj_idx, file_idx]\
        = loc_acc.localization_accuracy(loc_dir / subject_folder / subj_file, show=False)
# plot participants EG
fig, ax = plt.subplots(1, 3)
days = numpy.arange(0, 6)
days[0] = 1
for i in range(ele_gain.shape[0]):
    ax[0].scatter(days, ele_gain[i, :], label='participant %i'% i)
    ax[0].plot(days, ele_gain[i, :])
ax[0].set_xticks(numpy.arange(1,7,1))
ax[0].set_ylabel('Elevation Gain')
ax[0].set_xlabel('Days')
ax[0].set_yticks(numpy.arange(0,1.2,0.1))
ax[0].legend()
# EG
ax[1].plot(days, numpy.mean(ele_gain, axis=0), c='k', linewidth=0.5)
ax[1].errorbar(days, numpy.mean(ele_gain, axis=0), capsize=3, yerr=numpy.abs(numpy.diff(ele_gain, axis=0)),
               fmt="o", c='k', elinewidth=0.5, markersize=3)
ax[1].set_yticks(numpy.arange(0,1.2,0.1))
ax[1].set_xlabel('Days')
ax[1].set_ylabel('Elevation Gain')
ax[1].set_title('Elevation Gain')
# RMSE
ax[2].plot(days, numpy.mean(rmse, axis=0), c='k', linewidth=0.5, label='RMSE')
ax[2].errorbar(days, numpy.mean(rmse, axis=0), capsize=3, yerr=numpy.abs(numpy.diff(rmse, axis=0)),
               fmt="o", c='k', elinewidth=0.5, markersize=3)
# SD
ax[2].plot(days, numpy.mean(sd, axis=0), c='0.6', linewidth=0.5, label='SD')
ax[2].errorbar(days, numpy.mean(sd, axis=0), capsize=3, yerr=numpy.abs(numpy.diff(sd, axis=0)),
               fmt="o", c='0.6', elinewidth=0.5, markersize=3)
ax[2].set_yticks(numpy.arange(0,23,2))
ax[2].set_xlabel('Days')
ax[2].set_title('RMSE')
ax[2].legend()
ax[2].set_ylabel('Elevation in degrees')


