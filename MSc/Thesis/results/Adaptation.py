import MSc.analysis.statistics.stats_df as stats_df
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
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
main_df = get_df.main_dataframe(path, processed_hrtf=True)

""" adaptation comparing first vs last test with molds - one tailed wilcoxon signed rank test"""
efd0 = numpy.stack((main_df['EFD0']).to_numpy())  # ears free day 0
efd5 = numpy.stack((main_df['EFD5']).to_numpy())  # ears free day 5
m1d0 = numpy.stack((main_df['M1D0']).to_numpy())  # molds 1 day 0
m1d5 = numpy.stack((main_df['M1D5']).to_numpy())  # molds 1 day 0
m2d0 = numpy.stack((main_df['M2D0']).to_numpy())  # molds 2 day 0
m2d5 = numpy.stack((main_df['M2D5']).to_numpy())  # molds 2 day 5
m1_results = {}
for m_idx, measure in enumerate(measures):
    if m_idx == 0: # EG
        alternative = 'less' # test for EG increase
    else: alternative = 'greater'  # test for RMSE SD decrease
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
# normalize gain in elevation performance
m1gain_normalized = m1gain / m1drop
m2gain_normalized = m2gain / m2drop
adaptation_rate = {}
for m_idx, measure in enumerate(measures):
    adaptation_rate[measure] = dict()
    adaptation_rate[measure]['M1 mean'] = numpy.round(numpy.nanmean(m1gain_normalized[:, m_idx], axis=0), 2)
    adaptation_rate[measure]['M1 SD'] = numpy.round(scipy.stats.sem(m1gain_normalized[:, m_idx], nan_policy='omit', axis=0), 2)
    adaptation_rate[measure]['M2 mean'] = numpy.round(numpy.nanmean(m2gain_normalized[:, m_idx], axis=0), 2)
    adaptation_rate[measure]['M2 SD'] = numpy.round(scipy.stats.sem(m2gain_normalized[:, m_idx], nan_policy='omit', axis=0), 2)
    adaptation_rate[measure]['difference p'] = scipy.stats.wilcoxon(m1gain_normalized[:, m_idx], m2gain_normalized[:, m_idx],
                                                    alternative='less', nan_policy='omit')[1]
    adaptation_rate[measure]['correlation R'], adaptation_rate[measure]['correlation p']  = \
    scipy.stats.spearmanr(m1gain_normalized[:, m_idx], m2gain_normalized[:, m_idx], nan_policy='omit')

""" difference USO vs pink noise test: difference between conditions? """
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


""" misc """
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
plt.figure()
plt.scatter(ef01diff, efm1_vsi_dissimilarity)
R, p_val = scipy.stats.spearmanr(ef01diff, efm1_vsi_dissimilarity, nan_policy='omit')

""" VSI / VSI dissimilarity and learning rate """
main_df = get_df.main_dataframe(path, processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 13500), vsi_dis_bw=(5700, 13500))
m1drop = numpy.stack((main_df['M1 drop']).to_numpy())
m1gain = numpy.stack((main_df['M1 gain']).to_numpy())
m2drop = numpy.stack((main_df['M2 drop']).to_numpy())
m2gain = numpy.stack((main_df['M2 gain']).to_numpy())
m1gain_normalized = m1gain / m1drop
m2gain_normalized = m2gain / m2drop
m1_vsi = numpy.stack((main_df['M1 VSI']).to_numpy())
m2_vsi = numpy.stack((main_df['M2 VSI']).to_numpy())
# efm1_vsi_dissimilarity = numpy.stack((main_df['EF M1 VSI dissimilarity']).to_numpy())
# efm2_vsi_dissimilarity = numpy.stack((main_df['EF M2 VSI dissimilarity']).to_numpy())
# m1m2_vsi_dissimilarity = numpy.stack((main_df['M1 M2 VSI dissimilarity']).to_numpy())
# ---- M1 ---- #
plt.figure()
plt.scatter(m1gain_normalized[:, 1], m1_vsi)
R, p_val = scipy.stats.spearmanr(m1gain_normalized[:, 1], m1_vsi, nan_policy='omit')
# ---- M2 ---- #
plt.figure()
plt.scatter(m2gain_normalized[:, 0], m2_vsi)
R, p_val = scipy.stats.spearmanr(m2gain_normalized[:, 0], m2_vsi, nan_policy='omit')

