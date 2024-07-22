import MSc.analysis.localization_analysis as loc_analysis
import MSc.analysis.hrtf_analysis as hrtf_analysis
from MSc import misc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']  # these subjects did not complete Week 2 of the experiment
localization_dataframe = loc_analysis.get_localization_dataframe(path, w2_exclude)


""" ole_test correlation mold induced drop day 0 with EF / M1 VSI dissimilarity """
# I --- VSI dissimilarity (Trapeau & Schönwiesner 2015) --- #
measure = 'RMSE ele'
bandwidth = (4000, 16000)

exclude = ['svm']
n_bins=None # no big difference on the average
equalize=False  # results in implausible vsi?

hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=True)
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)
loc_data = []
vsi_dissimilarity = []
exclude = ['svm', 'sm'] # svm missing M1 hrtf measurement, sm
for subject_data in hrtf_stats.iterrows():
    if subject_data[1]['subject'] != 'svm':
        [efd0] = list(subject_data[1]['EFD0'][measure])
        [m1d0] = list(subject_data[1]['M1D0'][measure])
        # drop = numpy.diff((efd0, m1d0))
        drop = numpy.divide(efd0, m1d0)
        loc_data.append(drop)
        vsi_dis = hrtf_analysis.vsi_dissimilarity(subject_data[1]['EF hrtf'], subject_data[1]['M1 hrtf'], bandwidth,
                                                  n_bins, equalize)
        vsi_dissimilarity.append(vsi_dis)
print(scipy.stats.spearmanr(vsi_dissimilarity, loc_data, axis=0, nan_policy='propagate', alternative='two-sided'))
plt.scatter(vsi_dissimilarity, loc_data)

# II --- ole_test spectral difference across bands (Middlebrooks 1999)  --- #
measure = 'EG'
# bandwidth = (3700, 12900)
bands = MSc.misc.octave_spacing.overlapping_bands()[0]
exclude = ['svm']

hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=False, exclude=exclude)
hrtf_dataframe = hrtf_analysis.process_hrtfs(hrtf_dataframe, filter=None, baseline=True, write=False)
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)
m1_effect = []
spectral_difference = []
spectral_strength_difference = []
for bandwidth in bands:
    for subject in hrtf_stats.subject:
        subject_data = hrtf_stats[hrtf_stats['subject'] == subject]
        efd0 = subject_data['EFD0'].iloc[0][measure].iloc[0]
        m1d0 = subject_data['M1D0'].iloc[0][measure].iloc[0]
        # m1_effect.append(numpy.divide(efd0, m1d0))
        m1_effect.append(numpy.diff((efd0, m1d0)))
        hrtf_ef = subject_data['EF hrtf'].iloc[0]
        hrtf_m1 = subject_data['M1 hrtf'].iloc[0]
        # --- spectral difference
        spectral_difference.append(hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth))
        # --- spectral strength difference
        spectral_strength_diff = numpy.diff((hrtf_analysis.spectral_strength(hrtf_ef, bandwidth), \
                                             hrtf_analysis.spectral_strength(hrtf_m1, bandwidth)))
        spectral_strength_difference.append(spectral_strength_diff)

        print(f'bandwidth: {bandwidth}  subject: {subject} efd0 {efd0} m1d0 {m1d0} m1_effect {numpy.diff((efd0, m1d0))}'
              f' spectral str diff {spectral_strength_diff} '
              f'spectral diff {hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth)}')

        # fig, axes = plt.subplots(1,2)
        # hrtf_ef.plot_tf([6], axis=axes[0])
        # hrtf_m1.plot_tf([6], axis=axes[1])

    print(scipy.stats.spearmanr(spectral_strength_difference, m1_effect, axis=0, nan_policy='propagate', alternative='two-sided'))
    plt.figure()
    plt.scatter(spectral_strength_difference, m1_effect)




# ole_test d1 drop and spectral difference
import MSc.analysis.hrtf_analysis as hrtf_an
import MSc.analysis.plot.hrtf_plot as hrtf_pl
import MSc.analysis.localization_analysis as loc_an
import scipy
import MSc.misc.octave_spacing
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'

hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_df[hrtf_df['subject'] != 'svm']  # remove svm (missing hrtf_m1)
loc_df = loc_an.get_localization_dataframe()
octave_bands = MSc.misc.octave_spacing.overlapping_bands()[0]

show = False
for bandwidth in octave_bands:
    coords = []
    plt.figure()
    plt.title(f'bandwidth: {bandwidth} Hz')
    for subject in hrtf_df['subject'].unique():
        # loc data
        efd0 = loc_df[loc_df['condition'] == 'Ears Free'][loc_df['adaptation day'] == 0] \
            [loc_df['subject'] == subject]['sequence'].values[0]
        m1d0 = loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 0] \
            [loc_df['subject'] == subject]['sequence'].values[0]
        # hrtf diff
        try:
            hrtf_ef = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Ears Free']['hrtf'].values[0]
            hrtf_m1 = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Earmolds Week 1']['hrtf'].values[
                0]
            hrtf_diff = hrtf_an.hrtf_difference(hrtf_ef, hrtf_m1)
        except:
            continue
        # plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        loc_ef = loc_an.localization_accuracy(efd0, show=True, binned=True, axis=axes[0, 0])
        loc_m1 = loc_an.localization_accuracy(m1d0, show=True, binned=True, axis=axes[1, 0])
        fig.suptitle(subject)
        hrtf_pl.hrtf_image(hrtf_diff, axis=axes[0, 1], z_min=-20, z_max=20)
        if not show:
            plt.close()
        # ole_test params
        d1_drop = numpy.abs(numpy.array(loc_ef[0:2]) - numpy.array(loc_m1[0:2]))
        spectral_str = hrtf_an.spectral_strength(hrtf_diff, bandwidth)
        # spectral_str = hrtf_an.spectral_difference(hrtf_ef, hrtf_m1, bandwidth)
        coeff = spectral_str / d1_drop
        coords.append([spectral_str, d1_drop[1]])  # d1_drop idx: EG == 0, RMSE == 1, VAR == 2
    # plot scatter of spectral diff across bands and d1 drop
    coords = numpy.asarray(coords)
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, s in enumerate(list(hrtf_df['subject'].unique())):
        plt.annotate(s, (coords[i, 0], coords[i, 1]))
    corr_stats = scipy.stats.spearmanr(coords)
    plt.suptitle(f'correlation {corr_stats[0]}  pval {corr_stats[1]}')


# ole_test ef performance and spectral strength
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_df[hrtf_df['subject'] != 'svm']  # remove svm since there is no hrtf_m1
loc_df = loc_an.get_localization_dataframe()
octave_bands = MSc.misc.octave_spacing.overlapping_bands()[0]

for bandwidth in octave_bands:
    coords = []
    plt.figure()
    plt.title(f'bandwidth: {bandwidth} Hz')
    for subject in hrtf_df['subject'].unique():
        # loc data
        efd0 = loc_df[loc_df['condition'] == 'Ears Free'][loc_df['adaptation day'] == 0] \
            [loc_df['subject'] == subject]['sequence'].values[0]
        # hrtf diff
        try:
            hrtf_ef = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Ears Free']['hrtf'].values[0]
        except:
            continue
        # ole_test params
        spectral_str = hrtf_an.spectral_strength(hrtf_ef, bandwidth=bandwidth)
        loc_ef = loc_an.localization_accuracy(efd0, show=False, binned=True)
        coords.append([spectral_str, loc_ef[1]])
    # plot scatter of spectral diff across bands and d1 drop
    coords = numpy.asarray(coords)
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, s in enumerate(list(hrtf_df['subject'].unique())):
        plt.annotate(s, (coords[i, 0], coords[i, 1]))
    corr_stats = scipy.stats.spearmanr(coords)
    plt.suptitle(f'correlation {corr_stats[0]}  pval {corr_stats[1]}')

