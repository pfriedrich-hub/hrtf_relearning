import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)

path = Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']
localization_dataframe = loc_analysis.get_localization_dataframe(path, w2_exclude)
hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=True)
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)

# ----- VSI ------'
# ole_test parameters
# plot vsi across overlapping octave bands
# bands = [(3500, 7000), (4200, 8300), (4900, 9900), (5900, 11800), (7000, 14000)]
# bands = [(3500, 7000), (4200, 8500), (5100, 10200), (6200, 12400), (7500, 15000)]
# bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
# bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500)]
# non overlapping
bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
# bands = [(3500, 5000), (5000, 7200), (7200, 10400), (10400, 15000)]

n_bins=None  # no big difference on the average
equalize=False  # results in implausible vsi?
condition = 'Earmolds Week 1'
hrtf_plot.plot_hrtf_overview(hrtf_dataframe, condition, bands, n_bins, equalize)
hrtf_analysis.mean_vsi_across_bands(hrtf_dataframe, condition, bands, n_bins, equalize, show=True)
hrtf_plot.plot_average(hrtf_dataframe, condition=condition, equalize=True, kind='image')  #todo fix copy error

subject = 'jl'
# ole_test Ears Free performance / Ears Free VSI correlation
measure = 'RMSE ele'
loc_data = []
vsi_data = []
for subject in hrtf_stats.iterrows():
    [data] = list(subject[1]['EFD0'][measure])
    loc_data.append(data)
    vsi = hrtf_analysis.vsi(hrtf=subject[1]['EF hrtf'], bandwidth=(4000, 16000), n_bins=None, equalize=True)
    vsi_data.append(vsi)
print(scipy.stats.spearmanr(vsi_data, loc_data, axis=0, nan_policy='propagate', alternative='two-sided'))
plt.scatter(vsi_data, loc_data)

# ole_test Ears Free performance / Ears Free VSI correlation across non-overlapping 1/2-octave bands
bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
n_bins=None # no big difference on the average
equalize=False  # results in implausible vsi?
measure = 'EG'
loc_data = []
vsi_data = []
for subject in hrtf_stats.iterrows():
    [data] = list(subject[1]['EFD0'][measure])
    loc_data.append(data)
    vsi_bands = hrtf_analysis.vsi_across_bands(subject[1]['EF hrtf'], bands=bands, n_bins=n_bins, equalize=equalize)
    vsi_data.append(vsi_bands)
vsi_data = numpy.asarray(vsi_data)
for idx, band in enumerate(bands):
    corr = scipy.stats.spearmanr(vsi_data[:, idx], loc_data, axis=0, nan_policy='propagate', alternative='two-sided')
    print(f'{band} kHz {corr}')

# correlation of elevation RMSE / VSI in 5.7-8 kHz band: -0.5, p=0.067 #todo bonferroni


