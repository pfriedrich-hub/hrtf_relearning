import analysis.hrtf_analysis as hrtf_analysis
import analysis.statistics.stats_df as stats_df
import analysis.plot.spectral_behavior_collection as stats_plot
import analysis.build_dataframe as build_df
import misc.octave_spacing
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
import analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
# from methods: filter='erb', bandwidth=(500, 16000), baseline=False, dfe=True
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)

""" free ears VSI across non overlapping 1/2 octave bands """
bands = misc.octave_spacing.overlapping_bands()[0]
condition = 'Ears Free'
vsis = []
for hrtf in list(hrtf_df[hrtf_df['condition'] == condition]['hrtf']):
    vsis.append(hrtf_analysis.vsi_across_bands(hrtf, bands, show=False, ear_idx=[0]))  # left and right separately
    vsis.append(hrtf_analysis.vsi_across_bands(hrtf, bands, show=False, ear_idx=[1]))
vsis = numpy.asarray(vsis)
statistic, pval = scipy.stats.kruskal(vsis[:, 0], vsis[:, 1], vsis[:, 2], vsis[:, 3], vsis[:, 4])

""" spectral strength across non overlapping 1/2 octave bands """
bands = misc.octave_spacing.overlapping_bands()[0]
condition = 'Ears Free'
sp_st = []
for hrtf in list(hrtf_df[hrtf_df['condition'] == condition]['hrtf']):
    sp_st.append(hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=False, ear='left'))  # left and right separately
    sp_st.append(hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=False, ear='right'))  # left and right separately
sp_st = numpy.asarray(sp_st)
statistic, pval = scipy.stats.kruskal(sp_st[:, 0], sp_st[:, 1], sp_st[:, 2], sp_st[:, 3], sp_st[:, 4])


""" relation free ears VSI and vertical localization performance in the 5.7 - 11.3 kHz band """
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300))
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 13700))
x = numpy.array([item[0] for item in main_df['EFD0']])  # EG
x = numpy.array([item[1] for item in main_df['EFD0']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['EFD0']])  # SD ele - only correlation
y = main_df['EF VSI'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')
y1 = main_df['EF spectral strength'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')

""" Fig 5 A free ears VSI and vertical localization performance in the 3.7 - 12.9 kHz band """
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
stats_plot.ef_vsi(main_df, 'RMSE ele', axis=None)
stats_plot.ef_spstr(main_df, 'RMSE ele', axis=None)

""" relation free ears VSI and vertical localization performance in the 3.7 - 12.9 kHz band """
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
x = numpy.array([item[0] for item in main_df['EFD0']])  # EG
x = numpy.array([item[1] for item in main_df['EFD0']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['EFD0']])
y = main_df['EF VSI'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI
y1 = main_df['EF spectral strength'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')  # spectral strength


""" difference in vsi dissimilarity  free ears / M1 / M2"""
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
x = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
p = scipy.stats.wilcoxon(x, y, alternative='less', nan_policy='omit')[1]
efm1_mean = numpy.round(numpy.nanmean(x), 2)
efm1_sem = numpy.round(scipy.stats.sem(x, nan_policy='omit', axis=0), 2)
efm2_mean = numpy.round(numpy.nanmean(y), 2)
efm2_sem = numpy.round(scipy.stats.sem(y, nan_policy='omit', axis=0), 2)
plt.boxplot()


""" relation between behavioral and acoustic effect of earmolds """
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
# EF vs Mold 1 day 0
x = numpy.array([item[0] for item in main_df['M1 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M1 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M1 drop']])  # SD ele
y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity

# EF vs Mold 2 day 5
x = numpy.array([item[0] for item in main_df['M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M2 drop']])  # SD ele
y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity

# Mold 1 vs Mold 2 day 5
x = numpy.array([item[0] for item in main_df['M1M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M1M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M1M2 drop']])  # SD ele
y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity


# day 5

x = numpy.array([item[0] for item in main_df['M1 gain']])  # EG
x = numpy.array([item[1] for item in main_df['M1 gain']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M1 gain']])  # SD ele
y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity
y1 = main_df['EF M1 spectral difference'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')  # spectral difference


""" Fig 5 BCDEF relation between VSI dissimilarity and behavioral effect of the first set of molds"""
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
fig, axes = plt.subplots(1, 4, figsize=(20, 3))
stats_plot.d0dr_vsi_dis(main_df, 'RMSE ele', axis=axes[0])
stats_plot.d0dr_vsi_dis(main_df, 'EG', axis=axes[1])
stats_plot.d5ga_vsi_dis(main_df, 'RMSE ele', axis=axes[2])
stats_plot.d5ga_vsi_dis(main_df, 'EG', axis=axes[3])
axes[0].set_xlabel('RMSE')
axes[1].set_xlabel('Elevation Gain')
axes[2].set_xlabel('RMSE')
axes[3].set_xlabel('Elevation Gain')
axes[0].set_ylabel('VSI dissimilarity')
fig.text(.22, .92, 'Ears Free / M1 difference day 0', fontsize=12)
fig.text(.61, .92, 'Ears Free / M1 difference day 5', fontsize=12)

""" relation between behavioral and acoustic effect of the second molds """
# compared to VSI dissimilarity EF M2
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
x = numpy.array([item[0] for item in main_df['M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M2 drop']])  # SD ele
y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity
y1 = main_df['EF M2 spectral difference'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')  # spectral difference
# compared to VSI dissimilarity M1 M2
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
x = numpy.array([item[0] for item in main_df['M1M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M1M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M1M2 drop']])  # SD ele
y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity
y1 = main_df['M1 M2 spectral difference'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')  # spectral difference

