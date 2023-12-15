import analysis.build_dataframe as build_df
import analysis.plot.localization_plot as loc_plot
import analysis.statistics.stats_df as stats_df
import analysis.plot.learning as learn
import analysis.plot.elevation_learning as ele_learn
import analysis.plot.azimuth_learning as az_learn
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

# from methods: filter='erb', bandwidth=(500, 16000), baseline=False, dfe=True
loc_df = get_df.get_localization_dataframe()
main_df = get_df.main_dataframe(path, processed_hrtf=True)


""" fig plot evolution of response pattern across the experiment """
loc_plot.response_evolution(loc_df, to_plot='average', axis=None, width=14, height=8)
plt.savefig(plot_path / 'response_pattern/response_pattern.svg', format='svg', bbox_inches='tight')


""" adaptation comparing first vs last test with molds - one tailed wilcoxon signed rank test"""
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
efd0 = numpy.stack((main_df['EFD0']).to_numpy())  # ears free day 0
efd5 = numpy.stack((main_df['EFD5']).to_numpy())  # ears free day 5
m1d0 = numpy.stack((main_df['M1D0']).to_numpy())  # molds 1 day 0
m1d5 = numpy.stack((main_df['M1D5']).to_numpy())  # molds 1 day 0
m2d0 = numpy.stack((main_df['M2D0']).to_numpy())  # molds 2 day 0
m2d5 = numpy.stack((main_df['M2D5']).to_numpy())  # molds 2 day 5
m1_results = {}
for m_idx, measure in enumerate(measures):
    if m_idx == 1 or m_idx == 3:
        alternative = 'greater'
    else: alternative = 'less'
    m1_results[measure] = dict()
    m1_results[measure]['D0 mean'] = numpy.round(numpy.nanmean(m1d0[:, m_idx], axis=0), 2)
    m1_results[measure]['D0 SE'] = numpy.round(scipy.stats.sem(m1d0[:, m_idx], nan_policy='omit', axis=0), 2)
    m1_results[measure]['D5 mean'] = numpy.round(numpy.nanmean(m1d5[:, m_idx], axis=0), 2)
    m1_results[measure]['D5 SE'] = numpy.round(scipy.stats.sem(m1d5[:, m_idx], nan_policy='omit', axis=0), 2)
    m1_results[measure]['p'] = scipy.stats.wilcoxon(m1d0[:, m_idx], m1d5[:, m_idx],
                                                    alternative=alternative, nan_policy='omit')[1]
m2_results = {}
for m_idx, measure in enumerate(measures):
    if m_idx == 1 or m_idx == 3:
        alternative = 'greater'
    else: alternative = 'less'
    m2_results[measure] = dict()
    m2_results[measure]['D0 mean'] = numpy.round(numpy.nanmean(m2d0[:, m_idx], axis=0), 2)
    m2_results[measure]['D0 SE'] = numpy.round(scipy.stats.sem(m2d0[:, m_idx], nan_policy='omit', axis=0), 2)
    m2_results[measure]['D5 mean'] = numpy.round(numpy.nanmean(m2d5[:, m_idx], axis=0), 2)
    m2_results[measure]['D5 SE'] = numpy.round(scipy.stats.sem(m2d5[:, m_idx], nan_policy='omit', axis=0), 2)
    m2_results[measure]['p'] = scipy.stats.wilcoxon(m2d0[:, m_idx], m2d5[:, m_idx],
                                                    alternative=alternative, nan_policy='omit')[1]

""" compare individual rate of adaptation: d5 gain m1 vs m2 normalized by initial disruption (d0 drop) """

m1drop = numpy.stack((main_df['M1 drop']).to_numpy())
m1gain = numpy.stack((main_df['M1 gain']).to_numpy())
m2drop = numpy.stack((main_df['M2 drop']).to_numpy())
m2gain = numpy.stack((main_df['M2 gain']).to_numpy())
# check: test relation between initial drop and gain  - smaller disruption -> smaller gain
x = m1drop[:, 1]
y = m1gain[:, 1]
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')
plt.scatter(x, y)
# normalize gain in elevation performance
m1gain_normalized = m1gain[:, 1] / m1drop[:, 1]
m2gain_normalized = m2gain[:, 1] / m2drop[:, 1]

# distribution of individual adaptation rates
m1gain_mean = numpy.round(numpy.mean(m1gain_normalized, axis=0), 2)
m1gain_se = numpy.round(scipy.stats.sem(m1gain_normalized, nan_policy='omit', axis=0), 2)
m2gain_mean = numpy.round(numpy.nanmean(m2gain_normalized, axis=0), 2)
m2gain_se = numpy.round(scipy.stats.sem(m2gain_normalized, nan_policy='omit', axis=0), 2)
# differences in individual adaptation rates m1 vs m2
x = m1gain_normalized
y = m2gain_normalized
statistic, p_val = scipy.stats.wilcoxon(x, y, alternative='two-sided', nan_policy='omit')
# relation individual adaptation rates m1 m2
x = m1gain_normalized
y = m2gain_normalized
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')
plt.scatter(x, y)

""" VSI and learning rate """
# ---- M1 ---- #
main_df = get_df.main_dataframe(path, processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
m1drop = numpy.stack((main_df['M1 drop']).to_numpy())
m1gain = numpy.stack((main_df['M1 gain']).to_numpy())
m1_vsi = numpy.stack((main_df['M1 VSI']).to_numpy())
efm1_vsi_dissimilarity = numpy.stack((main_df['EF M1 VSI dissimilarity']).to_numpy())
m1_spectral_strength = numpy.stack((main_df['M1 spectral strength']).to_numpy())
m1gain_normalized = m1gain[:, 1] / m1drop[:, 1]
plt.scatter(m1gain[:, 1], m1_vsi)
R, p_val = scipy.stats.spearmanr(m1gain[:, 1], m1_vsi, nan_policy='omit')
plt.scatter()


""" difference USO vs pink noise test: difference between conditions? """
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
main_df = get_df.main_dataframe(path, processed_hrtf=True)
ef = numpy.stack((main_df['EFD10']).to_numpy())  # ears free day 10
efuso = numpy.stack((main_df['EF USO']).to_numpy())
m1 = numpy.stack((main_df['M1D5']).to_numpy())
m1uso = numpy.stack((main_df['M1 USO']).to_numpy())
m2 = numpy.stack((main_df['M2D5']).to_numpy())
m2uso = numpy.stack((main_df['M2 USO']).to_numpy())
nan_mask = numpy.where(~numpy.isnan(m2uso[:,0]))
efdiff = (efuso - ef)[nan_mask]
m1diff = (m1uso - m1)[nan_mask]
m2diff = (m2uso - m2)[nan_mask]
results = dict()
for m_idx, measure in enumerate(measures):
    results[measure] = dict()
    results[measure]['ef mean'] = numpy.round(numpy.nanmean(efdiff[:, m_idx]), 2)
    results[measure]['ef se'] = numpy.round(scipy.stats.sem(efdiff[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['m1 mean'] = numpy.round(numpy.nanmean(m1diff[:, m_idx]), 2)
    results[measure]['m1 se'] = numpy.round(scipy.stats.sem(m1diff[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['m2 mean'] = numpy.round(numpy.nanmean(m2diff[:, m_idx]), 2)
    results[measure]['m2 se'] = numpy.round(scipy.stats.sem(m2diff[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['p'] = scipy.stats.friedmanchisquare(efdiff[:, m_idx], m1diff[:, m_idx], m2diff[:, m_idx])
    # dependent (metric) response variables, 3 level factorial predictor

""" Aftereffect """
main_df = get_df.main_dataframe(path, processed_hrtf=True)
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
ef0 = numpy.stack((main_df['EFD0']).to_numpy())
ef1 = numpy.stack((main_df['EFD5']).to_numpy())
ef2 = numpy.stack((main_df['EFD10']).to_numpy())  # ears free day 10

results = dict()
for m_idx, measure in enumerate(measures):
    if m_idx == 0: alternative = 'less'
    elif m_idx > 0: alternative = 'less'
    results[measure] = dict()
    results[measure]['ef0 mean'] = numpy.round(numpy.nanmean(ef0[:, m_idx]), 2)
    results[measure]['ef0 se'] = numpy.round(scipy.stats.sem(ef0[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['ef1 mean'] = numpy.round(numpy.nanmean(ef1[:, m_idx]), 2)
    results[measure]['ef1 se'] = numpy.round(scipy.stats.sem(ef1[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['ef2 mean'] = numpy.round(numpy.nanmean(ef2[:, m_idx]), 2)
    results[measure]['ef2 se'] = numpy.round(scipy.stats.sem(ef2[:, m_idx], nan_policy='omit', axis=0), 2)
    results[measure]['friedman p'] = scipy.stats.friedmanchisquare(ef0[:, m_idx], ef1[:, m_idx], ef2[:, m_idx])
    results[measure]['wilcox p efm1'] = scipy.stats.wilcoxon(ef0[:, m_idx], ef1[:, m_idx], alternative=alternative, nan_policy='omit')
    results[measure]['wilcox p efm2'] = scipy.stats.wilcoxon(ef0[:, m_idx], ef2[:, m_idx], alternative=alternative, nan_policy='omit')
    results[measure]['wilcox p m1m2'] = scipy.stats.wilcoxon(ef1[:, m_idx], ef2[:, m_idx], alternative=alternative, nan_policy='omit')

""" compare VSI dissimilarity to increase in free ears vertical RMSE -- no relation """
main_df = get_df.main_dataframe(path, processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300))
efm1_vsi_dissimilarity = numpy.stack((main_df['EF M1 VSI dissimilarity']).to_numpy())
efm2_vsi_dissimilarity = numpy.stack((main_df['EF M2 VSI dissimilarity']).to_numpy())
ef0rmse = numpy.stack((main_df['EFD0']).to_numpy())[:, 1]
ef1rmse = numpy.stack((main_df['EFD5']).to_numpy())[:, 1]
ef2rmse = numpy.stack((main_df['EFD10']).to_numpy())[:, 1] # ears free day 10
ef01diff = ef1rmse - ef0rmse
ef02diff = ef2rmse - ef0rmse