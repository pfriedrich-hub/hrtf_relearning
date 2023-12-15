import analysis.hrtf_analysis as hrtf_analysis
import analysis.statistics.stats_df as stats_df
import analysis.plot.plot_spectral_behavior_stats as stats_plot
import analysis.plot.hrtf_plot as hrtf_plot
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
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
# from methods: filter='erb', bandwidth=(500, 16000), baseline=False, dfe=True
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)


""" II Acoustic effect of the molds """
bandwidth = (4000, 16000)
# use this bandwidth for spectral change p
vsi_bandwidth = (5700, 11300)
# use this bandwidth for vsi and vsi dissimilarity



""" figure 2 - acoustic effect of the earmolds """
# hrtf_plot.hrtf_overwiev(hrtf_df, to_plot='average', dfe=False, n_bins=None)
fig, axes = hrtf_plot.hrtf_compare(hrtf_df, axis=None, average_ears=True, hrtf_diff=False, zlim=(-12,8),
                                   width=14, height=5.2)
plt.savefig(plot_path / 'hrtf_compare/hrtf_compare.svg', format='svg', bbox_inches='tight')


""" Fig 3 - spectral change probability """
fig, axes = hrtf_plot.compare_spectral_change_p(main_df, axis=None, bandwidth=(4000, 16000),  width=21, height=8)
plt.savefig(plot_path / 'spectral_change_p/spectral_change_p.svg', format='svg', bbox_inches='tight')



# todo maybe only plot 3 conditons here w/out difference spectra

""" mean and SE of spectral change thresholds """
_, thresholds = stats_df.spectral_change_p(main_df, threshold=None, bandwidth=bandwidth)
mean = numpy.round(numpy.mean(thresholds), 2)
se = numpy.round(scipy.stats.sem(thresholds), 2)



""" VSI across conditions """
vsis = numpy.zeros((3, 15, 2))
for subject_id, row in main_df.iterrows():
    for c_idx, condition in enumerate(['EF hrtf', 'M1 hrtf', 'M2 hrtf']):
        for ear in [0, 1]:
            try:
                vsis[c_idx, subject_id, ear] = hrtf_analysis.vsi(main_df.iloc[subject_id][condition], vsi_bandwidth, ear_idx=[ear])
            except TypeError:
                vsis[c_idx, subject_id, ear] = numpy.nan
                print(f'{condition} not found for {row["subject"]}')
efvsi = numpy.append(vsis[0, :, 0], vsis[0, :, 1])
m1vsi = numpy.append(vsis[1, :, 0], vsis[1, :, 1])
m2vsi = numpy.append(vsis[2, :, 0], vsis[2, :, 1])
ef_m1_diff = scipy.stats.wilcoxon(efvsi, m1vsi, alternative='greater', nan_policy='omit')
ef_m2_diff = scipy.stats.wilcoxon(efvsi, m2vsi, alternative='greater', nan_policy='omit')
efvsi_mean = numpy.round(numpy.nanmean(efvsi), 2)
efvsi_se = numpy.round(scipy.stats.sem(efvsi), 2)

""" correlation left and right ear vsi across conditions """
main_df = stats_df.add_l_r_comparison(main_df, vsi_bandwidth)
vsi_l, vsi_r = stats_plot.vsi_l_r(main_df, show=False)
r, p_val = numpy.round(scipy.stats.spearmanr(vsi_l[0], vsi_r[0], nan_policy='omit'), 5)
m1_r, m1_pval = numpy.round(scipy.stats.spearmanr(vsi_l[1], vsi_r[1], nan_policy='omit'), 5)
m2_r, m2_pval = numpy.round(scipy.stats.spearmanr(vsi_l[2], vsi_r[2], nan_policy='omit'), 5)

""" Fig 4 A - correlation left and right ear vsi of free ears"""
stats_plot.vsi_l_r(main_df, show=True)

""" Fig 4 B - comparison vsi dissimilarity left vs right and ears vs free earmolds 1 / 2 """
main_df = stats_df.add_l_r_comparison(main_df, vsi_bandwidth)
stats_plot.scatter_perm_vsi_dis(main_df, vsi_bandwidth)

""" Fig VSI across bands """
