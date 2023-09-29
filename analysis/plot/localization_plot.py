import analysis.localization_analysis as loc_analysis
import numpy
import slab
from copy import deepcopy
from matplotlib import pyplot as plt
from pathlib import Path

### ----- LOCALIZATION ----- ####
def learning_plot(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude = ['cs', 'lm', 'lk']):
    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """
    localization_dict = loc_analysis.get_localization_dictionary(path=path)
    # localization_dict = deepcopy(localization_dictionary)
    if not to_plot == 'average':
        subjects = [to_plot]
    else:
        subjects = list(localization_dict['Ears Free'].keys())
    localization_dict = loc_analysis.get_localization_data(localization_dict, subjects, w2_exclude)
    days = numpy.arange(1, 13)  # days of measurement
    days[-1] = 16
    # means ears free / mold1 / mold2
    ef = numpy.nanmean(localization_dict['Ears Free']['data'], axis=0)
    m1 = numpy.nanmean(localization_dict['Earmolds Week 1']['data'], axis=0)
    m2 = numpy.nanmean(localization_dict['Earmolds Week 2']['data'], axis=0)
    # optionally delete nan for plt to interpolate points in learning curve
    if len(subjects) == 1:
        if numpy.isnan(ef).any():
            ef = numpy.delete(ef, numpy.where(numpy.isnan(ef))[0][0], numpy.where(numpy.isnan(ef))[1][0])
        if numpy.isnan(m1).any():
            m1 = numpy.delete(m1, numpy.where(numpy.isnan(m1))[0][0] ,numpy.where(numpy.isnan(m1))[1][0])
        if numpy.isnan(m2).any():
            m2 = numpy.delete(m2, numpy.where(numpy.isnan(m2))[0][0] ,numpy.where(numpy.isnan(m2))[1][0])
    # ----- plot ----- #
    labels = ['RMSE', 'SD']
    colors = [0.3, 0.7]
    color = colors[0]
    label = None
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for i, ax_id in enumerate([0, 1, 1]):
        if i >= 1:
            label = labels[i-1]
            color = colors[i-1]
        # EG # week 1
        axes[ax_id].plot([1, 1.05], [ef[0, i], m1[0, i]], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
        axes[ax_id].plot([1.05, 2, 3, 4, 5, 5.95], m1[:6, i], c=str(color-0.2), linewidth=1.5, label=label)  # week 1 learning curve
        axes[ax_id].plot([5.95, 6], [m1[5, i], ef[1, i]], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
        # week 2
        axes[ax_id].plot([6, 6.05], [ef[1, i], m2[0, i]], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
        axes[ax_id].plot([6.05,  7,  8,  9, 10, 10.95], m2[:6, i], c=str(color+0.2), linewidth=1.5)  # week 2 learning curve
        axes[ax_id].plot([10.95, 11], [m2[5, i], ef[2, i]], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axes[ax_id].plot([5.95, 11.05], m1[-2:, i], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)
        # mold 2 adaptation persistence
        axes[ax_id].plot([10.95, 16], m2[-2:, i], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)
        # error bars
        if len(subjects) > 1:
            axes[ax_id].errorbar([1, 6, 11], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                           fmt="o", c=str(color), elinewidth=0.8, markersize=5, fillstyle='none')  # error bar ears free
            axes[ax_id].errorbar([1.05, 2, 3, 4, 5, 5.95, 11.05], m1[:7, i], capsize=3, yerr=localization_dict
                            ['Earmolds Week 1']['SE'][:7, i], fmt="o", c=str(color-0.2), elinewidth=0.8, markersize=5)  # err m1
            axes[ax_id].errorbar([6.05,  7,  8,  9, 10, 10.95, 16], m2[:7, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE'][:7, i],
                           fmt="o", c=str(color+0.2), elinewidth=0.8, markersize=5)  # err m2
            # axes[ax_id].errorbar([6.1,  7,  8,  9, 10, 11.1], m2[:6, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE']
            #                     [:6, i], fmt="o", c=color, elinewidth=0.8, markersize=5)

    axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))
    axes[0].set_xticks(days)
    axes[0].set_ylabel('Elevation Gain')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Elevation in degrees')
    axes[1].legend()

    fig.text(0.3, 0.8, f'n={len(subjects)}', ha='center', size=10)
    fig.text(0.5, 0.8, f'n={len(subjects) - len(w2_exclude)}', ha='center', size=10)
    w1 = ' '.join([str(item) for item in subjects])
    w2 = subjects
    for subj in w2_exclude:
        if subj in subjects:
            w2.remove(subj)
    w2 = ' '.join([str(item) for item in w2])
    plt.suptitle(f'w1: {w1}, w2: {w2}')
    plt.show()
    return axes
    #
    # # save as scalable vector graphics
    # fig.savefig(Path.cwd() / 'data' / 'presentation' / 'learning_plot.svg', format='svg')

def localization_plot(to_plot='average', binned=True, path=Path.cwd() / 'data' / 'experiment' / 'master'):
    """ plot localization free, 1st vs last day of molds and persistence """
    loc_df = loc_analysis.get_localization_dataframe(path=path)
    # loc_df = deepcopy(loc_df)
    w2_exclude = ['cs', 'lm', 'lk']
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
    # plot:
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=[8, 8])
    # M1
    loc_analysis.localization_accuracy(efd0, True, 2, binned, axes[0, 0], False)
    loc_analysis.localization_accuracy(m1d0, True, 2, binned, axes[1, 0], False)
    loc_analysis.localization_accuracy(m1d5, True, 2, binned, axes[2, 0], False)
    # loc_analysis.localization_accuracy(m1d10, True, 2, binned, axes[3, 0], False)
    # M2
    loc_analysis.localization_accuracy(efd5, True, 2, binned, axes[0, 1], False)
    loc_analysis.localization_accuracy(m2d0, True, 2, binned, axes[1, 1], False)
    loc_analysis.localization_accuracy(m2d5, True, 2, binned, axes[2, 1], False)
    # loc_analysis.localization_accuracy(m2d10, True, 2, binned, axes[3, 1], False)
    fig.text(0.29, 0.95, 'Mold 1', size=13)
    fig.text(0.71, 0.95, 'Mold 2', size=13)
    fig.text(0.03, 0.75, 'Day 0 Ears Free', size=13, rotation=90)
    fig.text(0.03, 0.45, 'Day 0', size=13, rotation=90)
    fig.text(0.03, 0.18, 'Day 5', size=13, rotation=90)
    # fig.text(0.03, 0.18, 'Day 10', size=13, rotation=90)
    fig.suptitle(to_plot)

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
