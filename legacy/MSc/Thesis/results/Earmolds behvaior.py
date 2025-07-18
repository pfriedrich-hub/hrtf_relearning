from pathlib import Path
import scipy.stats
import pandas
import numpy
import legacy as get_df

pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'legacy' / 'MSc' / 'data' / 'experiment' / 'master'
# hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)

""" I Behavioral effect of the molds """
""" initial mold effect on localization performance - one sided wilcoxon signed rank test """
# ef = numpy.stack(pandas.concat((main_df['EFD0'], main_df['EFD5'])).to_numpy())  # ears free
# md0 = numpy.stack(pandas.concat((main_df['M1D0'], main_df['M2D0'])).to_numpy())  # molds 1 and 2 on day 0
efd0 = numpy.stack((main_df['EFD0']).to_numpy())  # ears free day 0
efd5 = numpy.stack((main_df['EFD5']).to_numpy())  # ears free day 5
m1d0 = numpy.stack((main_df['M1D0']).to_numpy())  # molds 1 day 0
m2d0 = numpy.stack((main_df['M2D0']).to_numpy())  # molds 2 day 0

results_m1 = {}
results_m1['EF mean'] = numpy.round(numpy.nanmean(efd0, axis=0), 2)
results_m1['EF SE'] = numpy.round(scipy.stats.sem(efd0, nan_policy='omit', axis=0), 2)
results_m1['M1 mean'] = numpy.round(numpy.nanmean(m1d0, axis=0), 2)
results_m1['M1 SE'] = numpy.round(scipy.stats.sem(m1d0, nan_policy='omit', axis=0), 2)
results_m1['EG p'] = scipy.stats.wilcoxon(efd0[:, 0], m1d0[:, 0], alternative='greater', nan_policy='omit')
results_m1['RMSE ele p'] = scipy.stats.wilcoxon(efd0[:, 1], m1d0[:, 1], alternative='less', nan_policy='omit')
results_m1['SD ele p'] = scipy.stats.wilcoxon(efd0[:, 2], m1d0[:, 2], alternative='less', nan_policy='omit')
results_m1['AG p'] = scipy.stats.wilcoxon(efd0[:, 3], m1d0[:, 3], alternative='less', nan_policy='omit')
results_m1['RMSE az p'] = scipy.stats.wilcoxon(efd0[:, 4], m1d0[:, 4], alternative='less', nan_policy='omit')
results_m1['SD az p'] = scipy.stats.wilcoxon(efd0[:, 5], m1d0[:, 5], alternative='less', nan_policy='omit')

results_m2 = {}
results_m2['EF mean'] = numpy.round(numpy.nanmean(efd5, axis=0), 2)
results_m2['EF SE'] = numpy.round(scipy.stats.sem(efd5, nan_policy='omit', axis=0), 2)
results_m2['M2 mean'] = numpy.round(numpy.nanmean(m2d0, axis=0), 2)
results_m2['M2 SE'] = numpy.round(scipy.stats.sem(m2d0, nan_policy='omit', axis=0), 2)
results_m2['EG p'] = scipy.stats.wilcoxon(efd5[:, 0], m2d0[:, 0], alternative='greater', nan_policy='omit')
results_m2['RMSE ele p'] = scipy.stats.wilcoxon(efd5[:, 1], m2d0[:, 1], alternative='less', nan_policy='omit')
results_m2['SD ele p'] = scipy.stats.wilcoxon(efd5[:, 2], m2d0[:, 2], alternative='less', nan_policy='omit')
results_m2['AG p'] = scipy.stats.wilcoxon(efd5[:, 3], m2d0[:, 3], alternative='less', nan_policy='omit')
results_m2['RMSE az p'] = scipy.stats.wilcoxon(efd5[:, 4], m2d0[:, 4], alternative='less', nan_policy='omit')
results_m2['SD az p'] = scipy.stats.wilcoxon(efd5[:, 5], m2d0[:, 5], alternative='less', nan_policy='omit')

""" test for difference in initial behavioral effect between m1 and m2"""
m1drop = numpy.stack((main_df['M1 drop']).to_numpy())
m2drop = numpy.stack((main_df['M2 drop']).to_numpy())
# check: test relation between initial drop and gain  - smaller disruption -> smaller gain
x = m1drop[:, 2]
y = m2drop[:, 2]
results_m1_m2 = scipy.stats.wilcoxon(x, y, alternative='two-sided', nan_policy='omit')
m1_mean = numpy.round(numpy.nanmean(x), 2)
m2_mean = numpy.round(numpy.nanmean(y), 2)
m1_se = numpy.round(scipy.stats.sem(x, nan_policy='omit', axis=0), 2)
m2_se = numpy.round(scipy.stats.sem(y, nan_policy='omit', axis=0), 2)


