import old.MSc.analysis.processing.hrtf_processing as hrtf_processing
import old.MSc.analysis.hrtf_analysis as hrtf_analysis
import old.MSc.analysis.localization_analysis as loc_analysis
import old.MSc.analysis.plot.elevation_gain_learning as ele_learning
import old.MSc.analysis.plot.hrtf_plot as hrtf_plot
import old.MSc.analysis.build_dataframe as get_df
import old.MSc.analysis.statistics.stats_df as stats_df
import old.MSc.analysis.build_dataframe as build_df
from pathlib import Path
import numpy
from matplotlib import pyplot as plt
path = Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


""" HRTF features """
# process HRTFs
hrtf_df = build_df.get_hrtf_df(path, processed=True)
hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter=None, baseline=True, dfe=True, write=False)  # baseline for hrtf image

# plot overview
hrtf_plot.hrtf_overwiev(hrtf_df, to_plot='average', dfe=False, n_bins=None)

# get average hrtf


# mean vsi / spectral str across bands
bands = MSc.misc.octave_spacing.overlapping_bands()[0]
bands = MSc.misc.octave_spacing.non_overlapping_bands()[0]
lb = numpy.arange(4000,18000,2000)
bands = [(lb[i], lb[i+1]) for i in range(len(lb)-1)]
bands = MSc.misc.octave_spacing.overlapping_bands()[-1]

condition = conditions[0]
hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition=condition, bands=bands, ear_idx=[0, 1])
hrtf_analysis.mean_spectral_strength_across_bands(hrtf_df, condition, bands=bands, show=True, ear='both')
hrtf_analysis.mean_vsi_dissimilarity_across_bands(hrtf_df, conditions=('Ears Free', 'Earmolds Week 1'), ear_idx=[0, 1],
                                                  bands=bands, show=True)

""" adaptation """
for subject in hrtf_df['subject'].unique():
    if subject not in ['cs', 'lm', 'lk', 'lw']:
        # fig = loc_plot.response_evolution(to_plot=subject)[0]
        fig, axis = ele_learning.learning_plot(to_plot=subject)
        fig.suptitle(subject)

""" ---- plot hrtf image, spectral strength and vsi across bands ---- """
# set conditions and bands
condition = conditions[1]
bands = MSc.misc.octave_spacing.overlapping_bands()[0]
# bands = misc.octave_spacing.non_overlapping_bands()[0]
# bands = [(6000, 12000)]
ear = 'left'
ear_idx = [0]
for subject in hrtf_df['subject'].unique():
    hrtf = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == condition]['hrtf'].values[0]
    fig, axis = hrtf_plot.hrtf_image(hrtf, chan=0)
    axis.set_title(subject)

    fig, axis = plt.subplots(3, 1, figsize=(7,9))
    # image
    # hrtf_pl.hrtf_image(hrtf, bandwidth=(numpy.min(bands), numpy.max(bands)), n_bins=None, axis=axis[0], z_min=-30, z_max=30, cbar=True)
    # axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=-37.5, ymax=37.5, color='black')
    # waterfall
    hrtf.plot_tf(hrtf.cone_sources(0), axis=axis[0], xlim=(numpy.min(bands), numpy.max(bands)), ear=ear)
    axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=axis[0].get_ylim()[0], ymax=axis[0].get_ylim()[1],
                   color='black')
    axis[0].set_xticks(numpy.asarray(bands).flatten())
    axis[0].set_xticklabels(numpy.asarray(bands).flatten())
    hrtf_analysis.spectral_strength_across_bands(hrtf, bands, show=True, axis=axis[1], ear=ear)
    hrtf_analysis.vsi_across_bands(hrtf, bands, show=True, axis=axis[2], ear_idx=ear_idx)
    fig.suptitle(subject)

"""plot localization accuracy day 0 ears free and mold 1, difference spectrum and correlation matrix"""
# bandwidth = (5700, 11300) # 2015, works *slightly besser for vsi
# bandwidth = (3700, 12900) # (1999)
bandwidth = (4000, 16000)
loc_df = loc_analysis.get_localization_dataframe(path)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=bandwidth)
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
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=bandwidth)
for subject in main_df['subject'].unique():
    try:
        hrtf_ef = main_df[main_df['subject'] == subject]['EF hrtf'].values[0]
        hrtf_m2 = main_df[main_df['subject'] == subject]['M2 hrtf'].values[0]
        efd0 = loc_df[loc_df['condition'] == 'Ears Free'] \
        [loc_df['adaptation day'] == 1] \
        [loc_df['subject'] == subject]['sequence'].values[0]
        m2d0 = loc_df[loc_df['condition'] == 'Earmolds Week 2'] \
            [loc_df['adaptation day'] == 0] \
            [loc_df['subject'] == subject]['sequence'].values[0]
    except IndexError: continue
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


""" ---- plot subject hrtf images lef and right across conditions --- """
# hrtf_df = hrtf_processing.get_hrtf_df(path=path, processed=True)
hrtf_df = build_df.get_hrtf_df(path, processed=True)
# hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', bandwidth=(4000, 16000), baseline=False, dfe=True, write=False)
bands = MSc.misc.octave_spacing.non_overlapping_bands()[0]
from old.MSc.analysis.plot.hrtf_plot import l_r_image as lrimage
for subject in hrtf_df['subject'].unique():
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].set_title('left')
    axes[0, 1].set_title('right')
    fig.suptitle(subject)
    try:
        hrtf_ef = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Ears Free']['hrtf'].values[0]
        lrimage(hrtf_ef, figsize=(15, 7), axes=[axes[0, 0], axes[0, 1]])
        # hrtf_ef.plot_tf(hrtf_ef.cone_sources(0), axis=axes[0, 0], xlim=(numpy.min(bands), numpy.max(bands)), ear='left')
        # hrtf_ef.plot_tf(hrtf_ef.cone_sources(0), axis=axes[0, 1], xlim=(numpy.min(bands), numpy.max(bands)), ear='right')
    except: continue
    try:
        hrtf_m1 = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Earmolds Week 1']['hrtf'].values[0]
        lrimage(hrtf_m1, figsize=(15, 7), axes=[axes[1, 0], axes[1, 1]])
        # hrtf_m1.plot_tf(hrtf_m1.cone_sources(0), axis=axes[1, 0], xlim=(numpy.min(bands), numpy.max(bands)), ear='left')
        # hrtf_m1.plot_tf(hrtf_m1.cone_sources(0), axis=axes[1, 1], xlim=(numpy.min(bands), numpy.max(bands)), ear='right')
    except: continue
    try:
        hrtf_m2 = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == 'Earmolds Week 2']['hrtf'].values[0]
        lrimage(hrtf_m2, figsize=(15, 7), axes=[axes[2, 0], axes[2, 1]])
        # hrtf_m2.plot_tf(hrtf_m2.cone_sources(0), axis=axes[2, 0], xlim=(numpy.min(bands), numpy.max(bands)), ear='left')
        # hrtf_m2.plot_tf(hrtf_m2.cone_sources(0), axis=axes[2, 1], xlim=(numpy.min(bands), numpy.max(bands)), ear='right')
    except: continue


"""# process all"""
hrtf_df = hrtf_processing.get_hrtf_df(path=path, processed=False)
hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', baseline=True, dfe=True, write=True)
# hrtf_df = hrtf_analysis.process_hrtfs(hrtf_df, filter='scepstral', baseline=True, write=True)

# plot all subjects hrtf + vsi
# hrtf_df = hrtf_analysis.get_hrtf_df(path=data_path, processed=False)
# hrtf_pl.subj_hrtf_vsi(hrtf_df, to_plot='all', condition='Ears Free', bands=None)
# hrtf_pl.subj_hrtf_vsi_dis(hrtf_df, to_plot='all', conditions=('Ears Free', 'Earmolds Week 1'), bands=None)

