import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.statistics.stats_df as stats_df
import analysis.plot.plot_spectral_behavior_stats as stats_plot
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.build_dataframe as build_df
import analysis.plot.learning as learn
import analysis.plot.elevation_learning as ele_learn
import analysis.plot.azimuth_learning as az_learn
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
hrtf_df = build_df.get_hrtf_df(path=path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)



""" I Behavioral effect of the molds """
""" initial mold effect on localization performance - one sided wilcoxon signed rank test """
ef = numpy.stack(pandas.concat((main_df['EFD0'], main_df['EFD5'])).to_numpy())  # ears free
md0 = numpy.stack(pandas.concat((main_df['M1D0'], main_df['M2D0'])).to_numpy())  # molds 1 and 2 on day 0
results = {}
results['EF mean'] = numpy.round(numpy.nanmean(ef, axis=0), 2)
results['EF SE'] = numpy.round(scipy.stats.sem(ef, nan_policy='omit', axis=0), 2)
results['M12 mean'] = numpy.round(numpy.nanmean(md0, axis=0), 2)
results['M12 SE'] = numpy.round(scipy.stats.sem(md0, nan_policy='omit', axis=0), 2)
results['RMSE ele p'] = scipy.stats.wilcoxon(ef[:, 1], md0[:, 1], alternative='less', nan_policy='omit')[1]
results['SD ele p'] = scipy.stats.wilcoxon(ef[:, 2], md0[:, 2], alternative='less', nan_policy='omit')[1]
results['EG p'] = scipy.stats.wilcoxon(ef[:, 0], md0[:, 0], alternative='greater', nan_policy='omit')[1]
results['RMSE az p'] = scipy.stats.wilcoxon(ef[:, 3], md0[:, 3], alternative='less', nan_policy='omit')[1]
results['SD az p'] = scipy.stats.wilcoxon(ef[:, 4], md0[:, 4], alternative='less', nan_policy='omit')[1]

""" Fig 1 - learning plot """
""" overall learning """
fig, axis = learn.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'])
""" elevation """
fig, axis = ele_learn.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'])
""" azimuth """
fig, axis = az_learn.learning_plot_azimuth(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'])



""" II Acoustic effect of the molds """
""" mean and SE of spectral change thresholds """
_, thresholds = stats_df.spectral_change_p(main_df, threshold=None, bandwidth=(500, 16000))
mean = numpy.mean(thresholds)
se = scipy.stats.sem(thresholds)

""" VSI across conditions """
bandwidth = (4000, 16000)   #
bandwidth = (5300, 11700)

# single_band = (3700, 12900)  # (Middlebrooks 1999)
vsis = dict()
spectral_strengths = dict()
for condition in conditions:
    vsis[condition] = dict()
    spectral_strengths[condition] = dict()
    vsis[condition]['data'] = []
    spectral_strengths[condition]['data'] = []
    for hrtf in list(hrtf_df[hrtf_df['condition'] == condition]['hrtf']):
        vsis[condition]['data'].append(hrtf_analysis.vsi(hrtf, bandwidth, ear_idx=[0, 1]))
        spectral_strengths[condition]['data'].append(hrtf_analysis.spectral_strength(hrtf, bandwidth, ear='both'))
    spectral_strengths[condition]['mean'] = numpy.mean(spectral_strengths[condition]['data'])
    spectral_strengths[condition]['SE'] = scipy.stats.sem(spectral_strengths[condition]['data'])
    vsis[condition]['mean'] = numpy.mean(vsis[condition]['data'])
    vsis[condition]['SE'] = scipy.stats.sem(vsis[condition]['data'])
    print(f"{condition}")
    print(f"vsi mean {vsis[condition]['mean']}, vsi se {vsis[condition]['SE']}")
    print(f"spectral strength mean {spectral_strengths[condition]['mean']}, spectral strength se"
          f" {spectral_strengths[condition]['SE']} \n")

""" Fig 2 - spectral change probability """
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
bandwidth = (4000, 16000)
threshold = None  # calculate threshold as rms between participants free ears dtfs
hrtf_plot.plot_spectral_change_p(main_df, 0, threshold, bandwidth, axes[0], False)
hrtf_plot.plot_spectral_change_p(main_df, 1, threshold, bandwidth, axes[1], False)
hrtf_plot.plot_spectral_change_p(main_df, 2, threshold, bandwidth, axes[2], True)
axes[0].set_title('EF M1')
axes[1].set_title('EF M2')
axes[2].set_title('M1 M2')



