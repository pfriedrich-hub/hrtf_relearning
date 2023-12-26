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
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300))# bandwidth of analyses

""" free ears VSI and spectral strength across non overlapping 1/2 octave bands """
bands = misc.octave_spacing.overlapping_bands()[0]
condition = 'Ears Free'
vsis = []
sp_st = []
for hrtf in list(hrtf_df[hrtf_df['condition'] == condition]['hrtf']):
    vsis.append(hrtf_analysis.vsi_across_bands(hrtf, bands, show=False, ear_idx=[0]))  # left and right separately
    vsis.append(hrtf_analysis.vsi_across_bands(hrtf, bands, show=False, ear_idx=[1]))
    sp_st.append(
        hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=False, ear='left'))  # left and right separately
    sp_st.append(
        hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=False, ear='right'))  # left and right separately
sp_st = numpy.asarray(sp_st)
vsis = numpy.asarray(vsis)
statistic, pval = scipy.stats.kruskal(vsis[:, 0], vsis[:, 1], vsis[:, 2], vsis[:, 3], vsis[:, 4])
statistic, pval = scipy.stats.kruskal(sp_st[:, 0], sp_st[:, 1], sp_st[:, 2], sp_st[:, 3], sp_st[:, 4])

"""spectral information of left and right ears in the 5300, 11700 kHz band """
vsi_l, vsi_r = stats_plot.vsi_l_r(main_df, show=False)
r, p_val = numpy.round(scipy.stats.spearmanr(vsi_l[0], vsi_r[0], nan_policy='omit'), 5)
m1_r, m1_pval = numpy.round(scipy.stats.spearmanr(vsi_l[1], vsi_r[1], nan_policy='omit'), 5)
m2_r, m2_pval = numpy.round(scipy.stats.spearmanr(vsi_l[2], vsi_r[2], nan_policy='omit'), 5)

""" relation free ears VSI and vertical localization performance in the 5.7 - 13.5 kHz band """
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 13500))
x = numpy.array([item[0] for item in main_df['EFD0']])  # EG
x = numpy.array([item[1] for item in main_df['EFD0']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['EFD0']])  # SD ele
y = main_df['EF VSI'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')
y1 = main_df['EF spectral strength'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y1, nan_policy='omit')

