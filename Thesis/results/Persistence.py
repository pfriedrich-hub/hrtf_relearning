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
main_df = get_df.main_dataframe(path, processed_hrtf=True)
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
m1d5 = numpy.stack((main_df['M1D5']).to_numpy())  # molds 1 day 5
m1d10 = numpy.stack((main_df['M1D10']).to_numpy())  # molds 1 day 10
m2d5 = numpy.stack((main_df['M2D5']).to_numpy())  # molds 2 day 5
m2d10 = numpy.stack((main_df['M2D10']).to_numpy())  # molds 2 day 10

""" test persistence for each mold (one tailed wilcoxon for performance decrease from day 5 to day 10) """
persistence_results = {'M1': dict(), 'M2': dict()}
for m_idx, measure in enumerate(measures):
    if m_idx in [0, 2]:  # test for EG decrease
        alternative = 'greater'
    else: alternative = 'less'  # test for rmse increase
    persistence_results['M1'][measure] = dict()
    persistence_results['M1'][measure]['D5 mean'] = numpy.round(numpy.nanmean(m1d5[:, m_idx], axis=0), 2)
    persistence_results['M1'][measure]['D5 se'] = numpy.round(scipy.stats.sem(m1d5[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_results['M1'][measure]['D10 mean'] = numpy.round(numpy.nanmean(m1d10[:, m_idx], axis=0), 2)
    persistence_results['M1'][measure]['D10 se'] = numpy.round(scipy.stats.sem(m1d10[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_results['M1'][measure]['p'] = scipy.stats.wilcoxon(m1d5[:, m_idx], m1d10[:, m_idx], alternative=alternative, nan_policy='omit')[1]
    persistence_results['M2'][measure] = dict()
    persistence_results['M2'][measure]['D5 mean'] = numpy.round(numpy.nanmean(m2d5[:, m_idx], axis=0), 2)
    persistence_results['M2'][measure]['D5 se'] = numpy.round(scipy.stats.sem(m2d5[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_results['M2'][measure]['D10 mean'] = numpy.round(numpy.nanmean(m2d10[:, m_idx], axis=0), 2)
    persistence_results['M2'][measure]['D10 se'] = numpy.round(scipy.stats.sem(m2d10[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_results['M2'][measure]['p'] = \
        scipy.stats.wilcoxon(m2d5[:, m_idx], m2d10[:, m_idx], alternative=alternative, nan_policy='omit')[1]

""" test differences in persistence (one tailed wilcoxon for difference in decrease m1 vs m2)"""
m1_pers = m1d10 - m1d5
m2_pers = m2d10 - m2d5
persistence_difference = dict()
for m_idx, measure in enumerate(measures):
    if m_idx == 0:  # test for EG decrease
        alternative = 'less'
    else: alternative = 'greater'  # test for rmse increase
    persistence_difference[measure] = dict()
    persistence_difference[measure]['M1 mean'] = numpy.round(numpy.nanmean(m1_pers[:, m_idx], axis=0), 2)
    persistence_difference[measure]['M1 se'] = numpy.round(scipy.stats.sem(m1_pers[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_difference[measure]['M2 mean'] = numpy.round(numpy.nanmean(m2_pers[:, m_idx], axis=0), 2)
    persistence_difference[measure]['M2 se'] = numpy.round(scipy.stats.sem(m2_pers[:, m_idx], nan_policy='omit', axis=0), 2)
    persistence_difference[measure]['p'] = \
        scipy.stats.wilcoxon(m1_pers[:, m_idx], m2_pers[:, m_idx], alternative=alternative, nan_policy='omit')[1]
