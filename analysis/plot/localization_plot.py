import analysis.localization_analysis as loc_analysis
import numpy
import slab
from copy import deepcopy
from matplotlib import pyplot as plt
from pathlib import Path

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
