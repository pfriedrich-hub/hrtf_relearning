import analysis.hrtf_analysis as hrtf_analysis
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.localization_analysis as loc_analysis
import analysis.create_dataframe as create_df
import analysis.plot.localization_plot as loc_plot
import misc.octave_spacing
from pathlib import Path
import numpy
from matplotlib import pyplot as plt
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

hrtf_df = hrtf_analysis.get_hrtf_df(path=data_path, processed=True)

""" HRTF features """
# raw HRTF
hrtf_df = hrtf_analysis.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_analysis.process_hrtfs(hrtf_df, filter=None, baseline=True, dfe=True, write=False)  # baseline for hrtf image
hrtf_plot.hrtf_overwiev(hrtf_df, to_plot='average', dfe=True, n_bins=None)
# mean vsi / spectral str across bands
condition = conditions[1]
bands = misc.octave_spacing.overlapping_bands()[0]
hrtf_analysis.mean_vsi_across_bands(hrtf_df, condition=condition, bands=bands, show=True)
hrtf_analysis.mean_spectral_strength_across_bands(hrtf_df, condition, bands=bands, show=True)

""" adaptation """
# loc_plot.learning_plot(to_plot='average')
# loc_plot.localization_plot(to_plot='average')



""" ---- plot hrtf image, spectral strength and vsi across bands ---- """
# set conditions and bands
condition = conditions[2]
# bands = misc.octave_spacing.overlapping_bands()[0]
bands = misc.octave_spacing.non_overlapping_bands()[0]
# bands = [(6000, 12000)]
for subject in hrtf_df['subject'].unique():
    hrtf = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == condition]['hrtf'].values[0]
    fig, axis = plt.subplots(3, 1, figsize=(7,9))
    # image
    # hrtf_pl.hrtf_image(hrtf, bandwidth=(numpy.min(bands), numpy.max(bands)), n_bins=None, axis=axis[0], z_min=-30, z_max=30, cbar=True)
    # axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=-37.5, ymax=37.5, color='black')
    # waterfall
    hrtf.plot_tf(hrtf.cone_sources(0), axis=axis[0], xlim=(numpy.min(bands), numpy.max(bands)))
    axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=axis[0].get_ylim()[0], ymax=axis[0].get_ylim()[1],
                   color='black')
    axis[0].set_xticks(numpy.asarray(bands).flatten())
    axis[0].set_xticklabels(numpy.asarray(bands).flatten())
    hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=True, axis=axis[1])
    hrtf_analysis.vsi_across_bands(hrtf, bands, show=True, axis=axis[2])
    fig.suptitle(subject)

"""plot localization accuracy day 0 ears free and mold 1, difference spectrum and correlation matrix"""
# bandwidth = (5700, 11300) # 2015, works *slightly besser for vsi
# bandwidth = (3700, 12900) # (1999)
bandwidth = (4000, 16000)
loc_df = loc_analysis.get_localization_dataframe(path)
main_df = create_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')
main_df = create_df.add_hrtf_stats(main_df, bandwidth=bandwidth)
for subject in main_df['subject'].unique():
    hrtf_ef = main_df[main_df['subject'] == subject]['EF hrtf'].values[0]
    hrtf_m1 = main_df[main_df['subject'] == subject]['M1 hrtf'].values[0]
    efd0 = loc_df[loc_df['condition'] == 'Ears Free'] \
    [loc_df['adaptation day'] == 0] \
    [loc_df['subject'] == subject]['sequence'].values[0]
    m1d0 = loc_df[loc_df['condition'] == 'Earmolds Week 1'] \
        [loc_df['adaptation day'] == 0] \
        [loc_df['subject'] == subject]['sequence'].values[0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(subject)
    loc_analysis.localization_accuracy(efd0, show=True, binned=True, axis=axes[0, 0])
    loc_analysis.localization_accuracy(m1d0, show=True, binned=True, axis=axes[1, 0])
    # plot difference spectrum
    hrtf_plot.hrtf_image(hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1), bandwidth, axis=axes[0, 1], z_min=-30, z_max=30)
    axes[0, 1].set_title('Difference Spectrum')
    axes[0, 1].set_xlim(bandwidth)
    # plot auto - cross correlation distance matrix
    correlation_mtx = hrtf_analysis.hrtf_correlation(hrtf_ef, hrtf_m1, bandwidth)
    autocorrelation_mtx = hrtf_analysis.hrtf_correlation(hrtf_ef, hrtf_ef, bandwidth)
    distance_mtx = (autocorrelation_mtx - correlation_mtx)
    hrtf_plot.plot_correlation_matrix(distance_mtx, axis=axes[1, 1], tiles=True)


"""plot localization accuracy day 0 ears free and mold 2, difference spectrum and correlation matrix"""
# bandwidth = (5700, 11300) # 2015, works *slightly besser for vsi
# bandwidth = (3700, 12900) # (1999)
bandwidth = (4000, 16000)
loc_df = loc_analysis.get_localization_dataframe(path)
main_df = create_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')
main_df = create_df.add_hrtf_stats(main_df, bandwidth=bandwidth)
for subject in main_df['subject'].unique():
    hrtf_ef = main_df[main_df['subject'] == subject]['EF hrtf'].values[0]
    hrtf_m2 = main_df[main_df['subject'] == subject]['M2 hrtf'].values[0]
    efd0 = loc_df[loc_df['condition'] == 'Ears Free'] \
    [loc_df['adaptation day'] == 2] \
    [loc_df['subject'] == subject]['sequence'].values[0]
    m2d0 = loc_df[loc_df['condition'] == 'Earmolds Week 2'] \
        [loc_df['adaptation day'] == 0] \
        [loc_df['subject'] == subject]['sequence'].values[0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(subject)
    loc_analysis.localization_accuracy(efd0, show=True, binned=True, axis=axes[0, 0])
    loc_analysis.localization_accuracy(m2d0, show=True, binned=True, axis=axes[1, 0])
    # plot difference spectrum
    hrtf_plot.hrtf_image(hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m2), bandwidth, axis=axes[0, 1], z_min=-30, z_max=30)
    axes[0, 1].set_title('Difference Spectrum')
    axes[0, 1].set_xlim(bandwidth)
    # plot auto - cross correlation distance matrix
    correlation_mtx = hrtf_analysis.hrtf_correlation(hrtf_ef, hrtf_m2, bandwidth)
    autocorrelation_mtx = hrtf_analysis.hrtf_correlation(hrtf_ef, hrtf_ef, bandwidth)
    distance_mtx = (autocorrelation_mtx - correlation_mtx)
    hrtf_plot.plot_correlation_matrix(distance_mtx, axis=axes[1, 1], tiles=True)



"""# process all"""
# hrtf_df = hrtf_analysis.get_hrtf_df(path=data_path, processed=False)
# hrtf_df = hrtf_analysis.process_hrtfs(hrtf_df, filter='erb', baseline=True, write=False)
# hrtf_df = hrtf_analysis.process_hrtfs(hrtf_df, filter='scepstral', baseline=True, write=True)

# plot all subjects hrtf + vsi
# hrtf_df = hrtf_analysis.get_hrtf_df(path=data_path, processed=False)
# hrtf_pl.subj_hrtf_vsi(hrtf_df, to_plot='all', condition='Ears Free', bands=None)
# hrtf_pl.subj_hrtf_vsi_dis(hrtf_df, to_plot='all', conditions=('Ears Free', 'Earmolds Week 1'), bands=None)
