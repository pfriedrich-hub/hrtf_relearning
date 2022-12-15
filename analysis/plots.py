

# """ Plot mean VSI across bands """
# vsi_list = numpy.asarray(vsi_list)
# fig, axis = plt.subplots()
# axis.plot(numpy.mean(vsi_list, axis=0), c='k')
# axis.set_xticks([0, 1, 2, 3, 4])
# bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
# labels = [item.get_text() for item in axis.get_xticklabels()]
# for idx, band in enumerate(bandwidths / 1000):
#     labels[idx] = '%.1f - %.1f' % (band[0], band[1])
# err = scipy.stats.sem(vsi_list, axis=0)
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
