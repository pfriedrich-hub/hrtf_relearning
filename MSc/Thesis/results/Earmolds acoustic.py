import MSc.analysis.statistics.stats_df as stats_df
import MSc.analysis.plot.spectral_behavior_collection as stats_plot
import MSc.analysis.plot.hrtf_plot as hrtf_plot
import MSc.analysis.build_dataframe as build_df
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
import MSc.analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
# from methods: filter='erb', bandwidth=(500, 16000), baseline=False, dfe=True
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))  # bandwidth of analyses

""" II Acoustic effect of the molds """

# use this bandwidth for vsi and vsi dissimilarity
""" differences between VSI of free and modified ears in the 5.7–11.3 kHz band """
vsis = numpy.zeros((3, 15, 2))
for subject_id, row in main_df.iterrows():
    for c_idx, condition in enumerate(['EF VSI', 'M1 VSI', 'M2 VSI']):
        vsis[c_idx, subject_id, 0] = row[f'{condition} l']
        vsis[c_idx, subject_id, 1] = row[f'{condition} r']
efvsi = numpy.append(vsis[0, :, 0], vsis[0, :, 1])
m1vsi = numpy.append(vsis[1, :, 0], vsis[1, :, 1])
m2vsi = numpy.append(vsis[2, :, 0], vsis[2, :, 1])
ef_m1 = scipy.stats.wilcoxon(efvsi, m1vsi, alternative='greater', nan_policy='omit')
ef_m2 = scipy.stats.wilcoxon(efvsi, m2vsi, alternative='greater', nan_policy='omit')
efvsi_mean = numpy.round(numpy.nanmean(efvsi), 2)
efvsi_se = numpy.round(scipy.stats.sem(efvsi), 2)
m1vsi_mean = numpy.round(numpy.nanmean(m1vsi), 2)
m1vsi_se = numpy.round(scipy.stats.sem(m1vsi, nan_policy='omit'), 2)
m2vsi_mean = numpy.round(numpy.nanmean(m2vsi), 2)
m2vsi_se = numpy.round(scipy.stats.sem(m2vsi, nan_policy='omit'), 2)

""" free ears VSI reduction caused by Mold 1 vs Mold 2 """
efm1diff = efvsi - m1vsi
efm2diff = efvsi - m2vsi
efm1diff_mean = numpy.round(numpy.nanmean(efm1diff), 2)
efm1diff_se = numpy.round(scipy.stats.sem(efm1diff, nan_policy='omit'), 2)
efm2diff_mean = numpy.round(numpy.nanmean(efm2diff), 2)
efm2diff_se = numpy.round(scipy.stats.sem(efm2diff, nan_policy='omit'), 2)
m1_m2 = scipy.stats.wilcoxon(efm1diff, efm2diff, alternative='two-sided', nan_policy='omit')

""" correlation between VSIs of free left and right ears persisted after mold insertion  """
vsi_l, vsi_r = stats_plot.vsi_l_r(main_df, show=False)
m1_r, m1_pval = numpy.round(scipy.stats.spearmanr(vsi_l[1], vsi_r[1], nan_policy='omit'), 5)
m2_r, m2_pval = numpy.round(scipy.stats.spearmanr(vsi_l[2], vsi_r[2], nan_policy='omit'), 5)

"""  VSI dissimilarity in the 5.7–13.5 kHz band across conditions """
vsi_dis_efm1 = numpy.zeros((30))
vsi_dis_efm2 = numpy.zeros((30))
vsi_dis_m1m2 = numpy.zeros((30))
vsi_dis_efm1[:15] = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16')
vsi_dis_efm1[15:] = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16')
vsi_dis_efm2[:15] = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16')
vsi_dis_efm2[15:] = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16')
vsi_dis_m1m2[:15] = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16')
vsi_dis_m1m2[15:] = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16')
p = scipy.stats.wilcoxon(vsi_dis_efm1, vsi_dis_efm2, alternative='less', nan_policy='omit')[1]
efm1_mean = numpy.round(numpy.nanmean(vsi_dis_efm1), 2)
efm1_sem = numpy.round(scipy.stats.sem(vsi_dis_efm1, nan_policy='omit', axis=0), 2)
efm2_mean = numpy.round(numpy.nanmean(vsi_dis_efm2), 2)
efm2_sem = numpy.round(scipy.stats.sem(vsi_dis_efm2, nan_policy='omit', axis=0), 2)
m1m2_m1ef_p = scipy.stats.wilcoxon(vsi_dis_efm1, vsi_dis_m1m2, alternative='greater', nan_policy='omit')[1]
m1m2_m2ef_p = scipy.stats.wilcoxon(vsi_dis_efm2, vsi_dis_m1m2, alternative='greater', nan_policy='omit')[1]
m1m2_mean = numpy.round(numpy.nanmean(vsi_dis_m1m2), 2)
m1m2_sem = numpy.round(scipy.stats.sem(vsi_dis_m1m2, nan_policy='omit', axis=0), 2)


""" MISC """
""" spectral change probability """
fig, axes = hrtf_plot.compare_spectral_change_p(main_df, axis=None, bandwidth=(4000, 16000), width=14, height=5.2)
plt.savefig(plot_path / 'spectral_change_p/spectral_change_p.svg', format='svg', bbox_inches='tight')
""" mean and SE of spectral change thresholds """
_, thresholds = stats_df.spectral_change_p(main_df, threshold=None, bandwidth=bandwidth)
mean = numpy.round(numpy.mean(thresholds), 2)
se = numpy.round(scipy.stats.sem(thresholds), 2)
