import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
import misc.octave_spacing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
path = Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']  # these subjects did not complete Week 2 of the experiment
localization_dataframe = loc_analysis.get_localization_dataframe(path, w2_exclude)


"""  --- test spectral difference (Middlebrooks 1999) across bands correlation with behavior ---  """

""" I spectral difference free ears / mold 1 and day 0 drop in performance """

# plots and statistics
show_individual = False
bands = misc.octave_spacing.overlapping_bands()[0]
show_scatter = True
measure = 'RMSE ele'
measure_idx = measures.index(measure)
exclude = ['svm']
hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=False, exclude=exclude)

# hrtf_dataframe = hrtf_analysis.process_hrtfs(hrtf_dataframe, filter=None, baseline=True, write=False) # baseline hrtfs
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)
hrtf_stats['D0 drop'] = ''
hrtf_stats['EF VSI'] = ''
hrtf_stats['EF M1 VSI dissimilarity'] = ''
hrtf_stats['EF spectral strength'] = ''
hrtf_stats['EF M1 spectral difference'] = ''
for bandwidth in bands:
    for subject_id, row in hrtf_stats.iterrows():
        # behavioral changes after first mold insertion
        hrtf_stats.loc[subject_id]['D0 drop'] \
            = hrtf_stats.iloc[subject_id]['M1D0'][measure_idx] - hrtf_stats.iloc[subject_id]['EFD0'][measure_idx]
        hrtf_ef = hrtf_stats.iloc[subject_id]['EF hrtf']
        hrtf_m1 = hrtf_stats.iloc[subject_id]['M1 hrtf']
        hrtf_stats.loc[subject_id]['EF VSI'] = hrtf_analysis.vsi(hrtf_ef, bandwidth)
        hrtf_stats.loc[subject_id]['EF M1 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1, bandwidth)
        hrtf_stats.loc[subject_id]['EF spectral strength'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth)
        hrtf_stats.loc[subject_id]['EF M1 spectral difference'] \
            = hrtf_analysis.spectral_difference(hrtf_stats.iloc[subject_id]['EF hrtf'],
                                                hrtf_stats.iloc[subject_id]['M1 hrtf'], bandwidth)

        if show_individual:
            efd0 = localization_dataframe[localization_dataframe['condition'] == 'Ears Free'] \
            [localization_dataframe['adaptation day'] == 0] \
            [localization_dataframe['subject'] == row['subject']]['sequence'].values[0]
            m1d0 = localization_dataframe[localization_dataframe['condition'] == 'Earmolds Week 1'] \
                [localization_dataframe['adaptation day'] == 0] \
                [localization_dataframe['subject'] == row['subject']]['sequence'].values[0]
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            fig.suptitle(row['subject'])
            loc_analysis.localization_accuracy(efd0, show=True, binned=True, axis=axes[0, 0])
            loc_analysis.localization_accuracy(m1d0, show=True, binned=True, axis=axes[1, 0])
            hrtf_plot.hrtf_image(hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1), axis=axes[0, 1], z_min=-30, z_max=30)

    if show_scatter:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].scatter(hrtf_stats['EFD0'], hrtf_stats['EF VSI'])
        axes[0, 0].set_title('Ears Free')
        axes[0, 0].set_ylabel('VSI')
        axes[0, 0].set_xlabel(measure)

        for subject_id, row in hrtf_stats.iterrows():
            axes[0, 0].scatter(row['EFD0'], row['EF VSI'])
            axes[0, 0].set_title('Ears Free')
            axes[0, 0].set_ylabel('VSI')
            axes[0, 0].set_xlabel(measure)

            axes[1, 0].scatter(row['D0 drop'], row['EF M1 VSI dissimilarity'])
            axes[1, 0].set_title('Ears Free vs Molds')
            axes[1, 0].set_ylabel('VSI dissimilarity')
            axes[1, 0].set_xlabel(measure)

            axes[0, 1].scatter(row['EFD0'], row['EF spectral strength'])
            axes[0, 1].set_title('Ears Free')
            axes[0, 1].set_ylabel('spectral strength')
            axes[0, 1].set_xlabel(measure)

            axes[1, 1].scatter(row['D0 drop'], row['EF M1 spectral difference'])
            axes[1, 1].set_title('Ears Free vs Molds')
            axes[1, 1].set_ylabel('spectral difference')
            axes[1, 1].set_xlabel(measure)

            fig.suptitle(f'bandwidth {bandwidth}')





# test d1 drop and spectral difference
import analysis.hrtf_analysis as hrtf_an
import analysis.plot.hrtf_plot as hrtf_pl
import analysis.localization_analysis as loc_an
import scipy
import misc.octave_spacing
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'

hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_df[hrtf_df['subject'] != 'svm']  # remove svm (missing hrtf_m1)
loc_df = loc_an.get_localization_dataframe()
octave_bands = misc.octave_spacing.non_overlapping_bands()[0]

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
        # test params
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


# test ef performance and spectral strength
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_df[hrtf_df['subject'] != 'svm']  # remove svm since there is no hrtf_m1
loc_df = loc_an.get_localization_dataframe()
octave_bands = misc.octave_spacing.overlapping_bands()[0]

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
        # test params
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

