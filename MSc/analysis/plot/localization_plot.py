import MSc.analysis.build_dataframe as build_df
import numpy
import slab
from copy import deepcopy
from matplotlib import pyplot as plt
from pathlib import Path
from MSc.misc.unit_conversion import cm2in

def response_evolution(to_plot='average', axis=None, figsize=(14,8), path=Path.cwd() / 'data' / 'experiment' / 'master'):
    """ plot localization free, 1st vs last day of molds and persistence """
    w2_exclude = ['cs', 'lm', 'lk']
    loc_df = build_df.get_localization_dataframe(path)
    # cs and lm did not go into w2, lk did not learn in w2
    if not to_plot == 'average':
        subjects = [to_plot]
    else:
        subjects = list(loc_df['subject'].unique())
    sequence = slab.Trialsequence()
    sequence.this_n = 1
    # m1
    efd0, m1d0, m1d5, m1d10 = deepcopy(sequence), deepcopy(sequence), deepcopy(sequence), deepcopy(sequence)
    efd0.data, m1d0.data, m1d5.data, m1d10.data = [], [], [], []
    # m2
    efd5, m2d0, m2d5, m2d10 = deepcopy(sequence), deepcopy(sequence), deepcopy(sequence), deepcopy(sequence)
    efd5.data, m2d0.data, m2d5.data, m2d10.data = [], [], [], []
    # ef
    efd10 = deepcopy(sequence)
    efd10.data = []
    # fetch cross subject localization data across subjects
    for subject in subjects:
        # ears free day 0
        efd0.data.extend(loc_df[loc_df['condition'] =='Ears Free'][loc_df['adaptation day']
                                                    == 0][loc_df['subject']==subject]['sequence'].values[0].data)
        # m 1 day 0
        m1d0.data.extend(loc_df[loc_df['condition'] =='Earmolds Week 1'][loc_df['adaptation day']
                                                    == 0][loc_df['subject']==subject]['sequence'].values[0].data)
        # m 1 day 5
        m1d5.data.extend(loc_df[loc_df['condition'] =='Earmolds Week 1'][loc_df['adaptation day']
                                                    == 5][loc_df['subject']==subject]['sequence'].values[0].data)
        # m1 day 10
        m1d10.data.extend(loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day']
                                         == 6][loc_df['subject'] == subject]['sequence'].values[0].data)
        if not (subject in w2_exclude):
            # ears free day 5
            if loc_df[loc_df['condition'] =='Ears Free'][loc_df['adaptation day']
                                        == 1][loc_df['subject']==subject]['sequence'].values[0].n_remaining != 132:
                efd5.data.extend(loc_df[loc_df['condition'] =='Ears Free'][loc_df['adaptation day']
                                                    == 1][loc_df['subject']==subject]['sequence'].values[0].data)
            # m 2 day 0
            m2d0.data.extend(loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation day']
                                                    == 0][loc_df['subject']==subject]['sequence'].values[0].data)
            # m 2 day 5
            m2d5.data.extend(loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation day']
                                                    == 5][loc_df['subject']==subject]['sequence'].values[0].data)
            # m2 day 10
            m2d10.data.extend(loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation day']
                                                    == 6][loc_df['subject']==subject]['sequence'].values[0].data)
            if loc_df[loc_df['condition'] =='Ears Free'][loc_df['adaptation day']
                                        == 2][loc_df['subject']==subject]['sequence'].values[0].n_remaining != 132:
                efd10.data.extend(loc_df[loc_df['condition'] =='Ears Free'][loc_df['adaptation day']
                                                    == 2][loc_df['subject']==subject]['sequence'].values[0].data)

    # plot:
    in_width = cm2in(figsize[0])
    in_height = cm2in(figsize[1])
    if not axis:
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(in_width, in_height),
                                 subplot_kw=dict(box_aspect=1), layout='constrained')
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.05, wspace=0.05)
    fig.supxlabel("Response azimuth (degrees)", size=10)
    fig.supylabel("Response elevation (degrees)", size=10)
    # M1
    plot_response_pattern(efd0, axes[0,0], show_single_responses=False, labels=False)
    plot_response_pattern(m1d0, axes[0,1], show_single_responses=False, labels=False)
    plot_response_pattern(m1d5, axes[0,2], show_single_responses=False, labels=False)
    plot_response_pattern(m1d10, axes[0,3], show_single_responses=False, labels=False)
    # M2
    plot_response_pattern(efd5, axes[1,0], show_single_responses=False, labels=False)
    plot_response_pattern(m2d0, axes[1,1], show_single_responses=False, labels=False)
    plot_response_pattern(m2d5, axes[1,2], show_single_responses=False, labels=False)
    plot_response_pattern(m2d10, axes[1,3], show_single_responses=False, labels=False)
    # plot_response_pattern(efd10, axes[2,2], show_single_responses=False, labels=False)

    axes[0,0].set_title('Free ears')
    # axes[0,1].set_title('Mold insertion (day 0)')
    axes[0,1].set_title('Mold insertion')
    axes[0,2].set_title('Adaptation')
    # axes[0,2].set_title('Adaptation (day 5)')
    axes[0,3].set_title('Persistence')
    # axes[0,3].set_title('Adaptation persistence')
    axes[0,3].set_ylabel('Earmolds 1')
    axes[0,3].yaxis.set_label_position("right")
    axes[1,3].set_ylabel('Earmolds 2')
    axes[1,3].yaxis.set_label_position("right")
    return fig, axis

def plot_response_pattern(sequence, axis=None, show_single_responses=True, labels=True):
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    elevations = numpy.unique(loc_data[:, 1, 1])
    azimuths = numpy.unique(loc_data[:, 1, 0])
    # targets[:, 1] = loc_data[:, 1, 1]  # target elevations
    # responses[:, 1] == loc_data[:, 0, 1]  # percieved elevations
    # targets[:, 0] = loc_data[:, 1, 0]
    # responses[:, 0] = loc_data[:, 0, 0]
    # mean perceived location for each target speaker
    i = 0
    mean_loc = numpy.zeros((45, 2, 2))
    for az_id, azimuth in enumerate(numpy.unique(targets[:, 0])):
        for ele_id, elevation in enumerate(numpy.unique(targets[:, 1])):
            [perceived_targets] = loc_data[numpy.where(numpy.logical_and(loc_data[:, 1, 1] == elevation,
                                                                         loc_data[:, 1, 0] == azimuth)), 0]
            if perceived_targets.size != 0:
                mean_perceived = numpy.mean(perceived_targets, axis=0)
                mean_loc[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1
    # divide target space in 16 half overlapping sectors and get mean response for each sector
    mean_loc_binned = numpy.empty((0, 2, 2))
    for a in range(6):
        for e in range(6):
            tar_bin = loc_data[numpy.logical_or(loc_data[:, 1, 0] == azimuths[a],
                                                loc_data[:, 1, 0] == azimuths[a + 1])]
            tar_bin = tar_bin[numpy.logical_or(tar_bin[:, 1, 1] == elevations[e],
                                               tar_bin[:, 1, 1] == elevations[e + 1])]
            tar_bin[:, 1] = numpy.array((numpy.mean([azimuths[a], azimuths[a + 1]]),
                                         numpy.mean([elevations[e], elevations[e + 1]])))
            mean_tar_bin = numpy.mean(tar_bin, axis=0).T
            mean_tar_bin[:, [0, 1]] = mean_tar_bin[:, [1, 0]]
            mean_loc_binned = numpy.concatenate((mean_loc_binned, [mean_tar_bin]))
    azimuths = numpy.unique(mean_loc_binned[:, 0, 0])
    elevations = numpy.unique(mean_loc_binned[:, 1, 0])
    mean_loc = mean_loc_binned
    # plot
    if not axis:
        fig, axis = plt.subplots(1, 1)
    else:
        fig = axis.get_figure()
    if show_single_responses:
        axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
    axis.scatter(mean_loc[:, 0, 1], mean_loc[:, 1, 1], color='black', s=3)
    for az in azimuths:  # plot hlines between target locations
        [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 0]
        [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 0]
        axis.plot(x, y, color='0.6', linewidth=0.3, zorder=-1)
    for ele in elevations: # plot vlines
        [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 0]
        [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 0]
        axis.plot(x, y, color='0.6', linewidth=0.3, zorder=-1)
    for az in azimuths:  # plot lines between mean perceived locations for each target
        [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 1]
        [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 1]
        axis.plot(x, y, color='black', linewidth=0.6)
    for ele in elevations:
        [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 1]
        [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 1]
        axis.plot(x, y, color='black', linewidth=0.6)
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True, reset=False,
                     width=1, length=1)
    x_ticks = numpy.linspace(-45, 45, 3).astype('int')
    y_ticks = numpy.linspace(-30, 30, 3).astype('int')
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_ylim(-35, 35)
    axis.set_xlim(-50, 50)
    axis.set_xticklabels(x_ticks)
    axis.set_yticklabels(y_ticks)
    try:
        axis.get_xticklabels()[0].set_horizontalalignment('left')
        axis.get_xticklabels()[-1].set_horizontalalignment('right')
    except IndexError:
        pass
    if labels:
        axis.set_ylabel('Response azimuth (degrees)')
        axis.set_xlabel('Response elevation (degrees)')
    plt.show()
    return fig, axis

    # # overall
    # fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=[6, 8])
    # efd0.data.extend(efd5.data)
    # m1d0.data.extend(m2d0.data)
    # m1d5.data.extend(m2d5.data)
    # loc_analysis.localization_accuracy(efd0, True, 2, binned, axes[0], True)
    # loc_analysis.localization_accuracy(m1d0, True, 2, binned, axes[1], True)
    # loc_analysis.localization_accuracy(m1d5, True, 2, binned, axes[2], True)
    # fig.suptitle('M1/M2 %s' % to_plot)


    # fig.text(0.5, 0.05, 'Response azimuth (deg)', ha='center', size=12)
    # fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical', size=12)
    # axis[0, 0].set_xticks(axis[0, 0].get_xticks().astype('int'))
    # yticks = axis[0, 0].get_yticks()  # [1:-1]
    # axis[0, 0].set_yticks(yticks.astype('int'))
    # # for idx, i in enumerate(range(2, 10, 2)):
    # #     fig.text(i/10, 0.95, subjects[idx])
    # c = ['Ears Free\nDay1', 'Mold 1\nDay 1', 'Mold 1\nDay 6', 'Mold 2\nDay 1',
    #      'Mold 2\nDay 6', 'Mold 1\nDay 11', 'Mold 2\nDay 16']
    # for idx, i in enumerate(numpy.linspace(0.87, 0.15, 7)):
    #     fig.text(0.03, i, f'{c[idx]}', size=13)


# """ Plot mean VSI across bands """
# vsi_list = numpy.asarray(vsi_list)
# fig, axis = plt.subplots()
# axis.plot(numpy.mean(vsi_list, axis=0), c='k')
# axis.set_xticks([0, 1, 2, 3, 4])
# bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
# labels = [item.get_text() for item in axis.get_xticklabels()]
# for idx, band in enumerate(bandwidths / 1000):
#     labels[idx] = '%.1f - %.1f' % (band[0], band[1])
# err = scipy.statistics.sem(vsi_list, axis=0)
# axis.errorbar(axis.get_xticks(), numpy.mean(vsi_list, axis=0), capsize=3,
#               yerr=err, fmt="o", c='0.6', elinewidth=0.5, markersize=3)
# axis.set_xticklabels(labels)
# axis.set_xlabel('Frequency bands (kHz)')
# axis.set_ylabel('VSI')

#
# # plot individual hrtf of participants
# for subj_idx, subj_hrtf in enumerate(hrtf_list):
#     fig, axis = plt.subplots()
#     subj_hrtf.plot_tf(n_bins=300, kind='image', xlim=freq_range, sourceidx=hrtf.cone_sources(0), axis=axis)
#     axis.set_title(file_list[subj_idx])
#
#
# """ compare free vs mold """
# fig, axis = plt.subplots(2, 2)
# # hrtf_free, hrtf_mold = get_hrtfs(data_dir)
# src_free = hrtf_free.cone_sources(0, full_cone=True)
# src_mold = hrtf_mold.cone_sources(0, full_cone=True)
# # waterfall and vsi
# plot_vsi(hrtf_free, src_free, n_bins, axis=axis[0, 0])
# plot_vsi(hrtf_mold, src_mold, n_bins, axis=axis[0, 1])
# hrtf_free.plot_tf(src_free, n_bins=n_bins, kind='waterfall', axis=axis[1, 0])
# hrtf_mold.plot_tf(src_mold, n_bins=n_bins, kind='waterfall', axis=axis[1, 1])
# axis[0, 0].set_title('ears free')
# axis[0, 1].set_title('mold')
#
# # cross correlation
# # plot_hrtf_correlation(hrtf_free, hrtf_mold, src)
#
# """ Plot DTF correlation """
# from hrtf_analysis import dtf_correlation
# def plot_correlation(hrtf_free, hrtf_mold, sources):
#     # compare heatmap of hrtf free and with mold
#     fig, axis = plt.subplots(2, 2, sharey=True)
#     hrtf_free.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 0])
#     hrtf_mold.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 1])
#     fig.text(0.3, 0.9, 'Ear Free', ha='center')
#     fig.text(0.7, 0.9, 'With Mold', ha='center')
#     # plot hrtf autocorrelation free
#     corr_mtx, cbar_1 = dtf_correlation(hrtf_free, hrtf_free, show=True, bandwidth=None,
#                                          n_bins=96, axis=axis[1, 0])
#     # plot hrtf correlation free vs mold
#     cross_corr_mtx, cbar_2 = dtf_correlation(hrtf_free, hrtf_mold, show=True, bandwidth=None,
#                                                n_bins=96, axis=axis[1, 1])
#     fig.text(0.3, 0.5, 'Autocorrelation Ear Free', ha='center')
#     fig.text(0.7, 0.5, 'Correlation Free vs. Mold', ha='center')
#     cbar_1.remove()
