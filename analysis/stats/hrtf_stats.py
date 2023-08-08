import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.localization_plot as loc_plot
import analysis.hrtf_plot as hrtf_plot
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)

path = Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']
loc_df = loc_analysis.get_localization_dataframe(path, w2_exclude)
hrtf_df = hrtf_analysis.get_hrtf_df(path, processed=True)

# common dataframe for behavior and hrtf data
hrtf_stats = pandas.DataFrame({'subject': [],
                               'EF hrtf': [], 'EFD0': [], 'EFD5': [],
                               'M1 hrtf': [], 'M1D0': [], 'M1D5': [],
                               'M2 hrtf': [], 'M2D0': [], 'M2D5': []})
for subject in hrtf_df['subject'].unique():
    # subject data
    subj_hrtfs = hrtf_df[hrtf_df['subject']==subject]
    subj_loc = loc_df[loc_df['subject']==subject]
    # ears free
    ef = subj_hrtfs[subj_hrtfs['condition']=='Ears Free']['hrtf'].item()
    efd0 = subj_loc[subj_loc['condition']=='Ears Free'][subj_loc['adaptation day']==0][subj_loc.columns[6:]]
    efd5 = subj_loc[subj_loc['condition']=='Ears Free'][subj_loc['adaptation day']==1][subj_loc.columns[6:]]
    # m1
    try:
        m1 = subj_hrtfs[hrtf_df['condition']=='Earmolds Week 1']['hrtf'].item()
        m1d0 = subj_loc[subj_loc['condition']=='Earmolds Week 1'][subj_loc['adaptation day']==0][subj_loc.columns[6:]]
        m1d5 = subj_loc[subj_loc['condition']=='Earmolds Week 1'][subj_loc['adaptation day']==5][subj_loc.columns[6:]]
    except ValueError:
        m1, m1d0, m1d5 = None, None, None
    # m2
    if subject not in w2_exclude:
        m2 = subj_hrtfs[hrtf_df['condition']=='Earmolds Week 2']['hrtf'].item()
        m2d0 = subj_loc[subj_loc['condition']=='Earmolds Week 2'][subj_loc['adaptation day']==0][subj_loc.columns[6:]]
        m2d5 = subj_loc[subj_loc['condition']=='Earmolds Week 2'][subj_loc['adaptation day']==5][subj_loc.columns[6:]]
    else: m2, m2d0, m2d5 = None, None, None
    new_row = [subject,
               ef, efd0, efd5,
               m1, m1d0, m1d5,
               m2, m2d0, m2d5]
    # hrtf_stats.to_csv('/Users/paulfriedrich/Desktop/hrtf_relearning/data/hrtf_stats.csv')
    hrtf_stats.loc[len(hrtf_stats)] = new_row

# ----- VSI ------'
# test parameters
# plot vsi across overlapping octave bands
# bands = [(3500, 7000), (4200, 8300), (4900, 9900), (5900, 11800), (7000, 14000)]
# bands = [(3500, 7000), (4200, 8500), (5100, 10200), (6200, 12400), (7500, 15000)]
bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
# bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500)]

n_bins=None # no big difference on the average
equalize=False  # results in implausible vsi?
# hrtf_plot.plot_hrtf_overview(hrtf_df, bands, n_bins, equalize)
hrtf_analysis.mean_vsi_across_bands(hrtf_df, 'Ears Free', bands, n_bins, equalize, show=True)
# hrtf_plot.plot_average(hrtf_df, condition='Ears Free', equalize=True, kind='image')  #todo fix copy error


# test Ears Free performance / Ears Free VSI correlation
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

# test Ears Free performance / Ears Free VSI correlation across non-overlapping 1/2-octave bands
bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
n_bins=300 # no big difference on the average
equalize=True  # results in implausible vsi?
measure = 'RMSE ele'
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

# --- test correlation mold induced drop with VSI dissimilarity in 5.7-8 kHz band --- #
measure = 'RMSE ele'
bandwidth = (5700, 8000)
loc_data = []
vsi_dis = []
for subject in hrtf_stats.iterrows():
    if subject != 'svm':
        [efd0] = list(subject[1]['EFD0'][measure])
        [m1d0] = list(subject[1]['M1D0'][measure])
        drop = numpy.diff((efd0, m1d0))
        loc_data.append(drop)
        vsi_dissimilarity = hrtf_analysis.vsi_dissimilarity(subject[1]['EF hrtf'], subject[1]['M1 hrtf'], bandwidth)
        vsi = hrtf_analysis.vsi(hrtf=subject[1]['EF hrtf'], bandwidth=(4000, 16000), n_bins=None, equalize=True)
        vsi_dis.append(vsi)
print(scipy.stats.spearmanr(vsi_data, loc_data, axis=0, nan_policy='propagate', alternative='two-sided'))
plt.scatter(vsi_data, loc_data)
