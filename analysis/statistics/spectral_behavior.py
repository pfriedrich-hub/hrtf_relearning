import analysis.statistics.stats_df as stats_df
import analysis.get_dataframe as get_df
from analysis.plot import plot_spectral_behavior_stats as stats_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from matplotlib import pyplot as plt

"""  --- test spectral difference (Middlebrooks 1999) and VSI (Trapeau, Schönwiesner 2015)
                         correlation with behavior across bands ---  """
# bandwidth = (5700, 8000)
# bandwidth = (5700, 11300)  # 2015, clearer relation between spectral features in this band and behavior
bandwidth = (3700, 12900)  # 1999, 3700 may include spectral variance due to low freq artifacts

main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=bandwidth)

# ears free vsi d0
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.ef_vsi(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.ef_vsi(main_df, 'EG', axis=axes[0, 1])
stats_plot.ef_spstr(main_df, 'RMSE ele', axis=axes[1, 0])
stats_plot.ef_spstr(main_df, 'EG', axis=axes[1, 1])
fig.suptitle('Ears free baseline')

# m1 / ears free vsi dissimilarity, d0 drop
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
stats_plot.d0dr_vsi_dis(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d0dr_vsi_dis(main_df, 'EG', axis=axes[0, 1])
stats_plot.d5ga_vsi_dis(main_df, 'RMSE ele', axis=axes[0, 2])
stats_plot.d5ga_vsi_dis(main_df, 'EG', axis=axes[0, 3])
stats_plot.d0dr_sp_dif(main_df, 'RMSE ele', axis=axes[1, 0])
stats_plot.d0dr_sp_dif(main_df, 'EG', axis=axes[1, 1])
stats_plot.d5ga_sp_dif(main_df, 'RMSE ele', axis=axes[1, 2])
stats_plot.d5ga_sp_dif(main_df, 'EG', axis=axes[1, 3])
axes[1, 0].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[1, 2].set_xlabel('RMSE')
axes[1, 3].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('VSI dissimilarity')
axes[1, 0].set_ylabel('spectral difference')
fig.text(.22, .92, 'Ears Free / M1 difference day 0', fontsize=12)
fig.text(.61, .92, 'Ears Free / M1 difference day 5', fontsize=12)

# m2 / ears free vsi dissimilarity d5 drop
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
stats_plot.d5dr_vsi_dis(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d5dr_vsi_dis(main_df, 'EG', axis=axes[0, 1])
stats_plot.d10ga_vsi_dis(main_df, 'RMSE ele', axis=axes[0, 2])
stats_plot.d10ga_vsi_dis(main_df, 'EG', axis=axes[0, 3])
stats_plot.d5dr_sp_dif(main_df, 'RMSE ele', axis=axes[1, 0])
stats_plot.d5dr_sp_dif(main_df, 'EG', axis=axes[1, 1])
stats_plot.d10ga_sp_dif(main_df, 'RMSE ele', axis=axes[1, 2])
stats_plot.d10ga_sp_dif(main_df, 'EG', axis=axes[1, 3])
axes[1, 0].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[1, 2].set_xlabel('RMSE')
axes[1, 3].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('VSI dissimilarity')
axes[1, 0].set_ylabel('spectral difference')
fig.text(.22, .92, 'Ears Free / M2 difference day 0', fontsize=12)
fig.text(.61, .92, 'Ears Free / M2 difference day 5', fontsize=12)

# m1 / m2 vsi dissimilarity d5 drop
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
stats_plot.d5dr_vsi_dis_m1m2(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d5dr_vsi_dis_m1m2(main_df, 'EG', axis=axes[0, 1])
stats_plot.d10ga_vsi_dis_m1m2(main_df, 'RMSE ele', axis=axes[0, 2])
stats_plot.d10ga_vsi_dis_m1m2(main_df, 'EG', axis=axes[0, 3])
stats_plot.d5dr_sp_dif_m1m2(main_df, 'RMSE ele', axis=axes[1, 0])
stats_plot.d5dr_sp_dif_m1m2(main_df, 'EG', axis=axes[1, 1])
stats_plot.d10ga_sp_dif_m1m2(main_df, 'RMSE ele', axis=axes[1, 2])
stats_plot.d10ga_sp_dif_m1m2(main_df, 'EG', axis=axes[1, 3])
axes[1, 0].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[1, 2].set_xlabel('RMSE')
axes[1, 3].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('VSI dissimilarity')
axes[1, 0].set_ylabel('spectral difference')
fig.text(.22, .92, 'M1/M2 drop vs M1/M2 difference', fontsize=12)
fig.text(.61, .92, 'M1/M2 gain vs M1/M2 difference', fontsize=12)


"""
# setting
measure = 'RMSE ele'
# measure = 'EG'
# bands = misc.octave_spacing.non_overlapping_bands()[0] # (2015)
# bands = misc.octave_spacing.overlapping_bands()[0]
# bands = [(5700, 11300)] # also works,
bands = [(3700, 12900)] # (1999) - seems like a solid choice

path = Path.cwd() / 'data' / 'experiment' / 'master'
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
measure_idx = measures.index(measure)
# w2_exclude=['cs', 'lm', 'lk']  # these subjects did not complete Week 2 of the experiment
localization_dataframe = loc_analysis.get_localization_dataframe(path)
# exclude = ['svm']  # missing M1 hrtf
hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=False)
# process hrtfs
hrtf_dataframe = hrtf_analysis.process_hrtfs(hrtf_dataframe, filter='erb', baseline=True, write=False)  # baseline hrtfs
# scatter plots and statistics


for bandwidth in bands:


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    stats_plot.ef_vsi(main_df, measure, axis=axes[0, 0])
    stats_plot.ef_spstr(main_df, measure, axis=axes[0, 1])
    stats_plot.d0dr_vsi_dis(main_df, measure, axis=axes[1, 0])
    stats_plot.d0dr_sp_dif(main_df, measure, axis=axes[1, 1])


    fig.suptitle(f'measure {measure}, bandwidth {bandwidth}')


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

"""