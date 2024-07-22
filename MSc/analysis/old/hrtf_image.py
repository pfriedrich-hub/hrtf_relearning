from pathlib import Path
import MSc.analysis.hrtf_analysis as hrtf_analysis


def hrtf_images(hrtf_df, n_bins, bandwidth, title=None, plot='image'):
    """
    Plots HRTFs and HRTF differences

    """
    # input is a dictionary with keys = condition and value = HRTF, 3 conditions
    df = copy.deepcopy(hrtf_df)
    conditions = df['condition'].unique()
    # conditions = list(dict.keys())
    diff_conditions = ['Difference Ears Free - Mold 1',
                       'Difference Ears Free - Mold 2', 'Difference Mold 1 - Mold 2']
    corr_conditions = ['Correlation Ears Free - Mold 1',
                       'Correlation Ears Free - Mold 2', 'Correlation Mold 1 - Mold 2']
    compare = [['Ears Free', 'Earmolds Week 1'], ['Ears Free', 'Earmolds Week 2'],
               ['Earmolds Week 1', 'Earmolds Week 2']]
    # 0° az cone sources
    # src_idx = dict[conditions[0]].cone_sources(0)
    src_idx = df['hrtf'][0].cone_sources(0)
    # get difference HRTFs
    dict = {}
    dict['difference'], dict['min'], dict['max'] = {}, [], []
    for i in range(3):
        dict['difference'][diff_conditions[i]] = hrtf_difference(dict[compare[i][0]], dict[compare[i][1]])
    # get min and max values for img cbar scaling etc
    frequencies = dict[conditions[0]][0].frequencies
    frequencies = numpy.linspace(0, frequencies[-1], n_bins)
    freqidx = numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])
    for condition in conditions:
        dict['min'].append(dict[condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].min())
        dict['max'].append(dict[condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].max())
    for condition in diff_conditions:
        dict['min'].append(dict['difference'][condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].min())
        dict['max'].append(dict['difference'][condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].max())
    z_min = numpy.floor(numpy.min(dict['min'])) - 1
    z_max = numpy.ceil(numpy.max(dict['max']))
    title_list = [['Ears Free', 'Week 1 Molds', 'Week 2 Molds'], diff_conditions, corr_conditions]

    # plot
    if plot == 'image':
        fig, axis = plt.subplots(2, 3, sharey=True, figsize=(13, 8))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
        cbar = False
        for i in range(3):
            if i == 2:
                cbar = True
            # plot HRTF
            hrtf_image(dict[conditions[i]], n_bins=n_bins,
                                          bandwidth=bandwidth, axis=axis[0, i], z_min=z_min, z_max=z_max, cbar=cbar)
            axis[0, i].set_title(title_list[0][i])
            # plot HRTF differences
            hrtf_image(dict['difference'][diff_conditions[i]], n_bins=n_bins,
                                          bandwidth=bandwidth, axis=axis[1, i], z_min=z_min, z_max=z_max, cbar=cbar)
            axis[1, i].set_title(title_list[1][i])
        fig.text(0.5, 0.04, 'Frequency (kHz)', ha='center', size=13)
        fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)
        if title:
            fig.suptitle(title)
        return fig, axis

    elif plot == 'correlation': # compute and plot HRTF correlation
        correlation = []
        fig, axis = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
        cbar = False
        for i in range(3):
            if i == 2:
                cbar = True
            correlation.append(hrtf_correlation(dict[compare[i][0]], dict[compare[i][1]], show=True, axis=axis[i],
                                                bandwidth=bandwidth, cbar=cbar, n_bins=n_bins))
            axis[i].set_title(title_list[2][i])
            vsi_dis = vsi_dissimilarity(dict[compare[i][0]], dict[compare[i][1]], bandwidth)
            axis[i].text(-30, 30, 'VSI dissimilarity: %.2f' % vsi_dis, size=10)
        fig.text(0.5, 0.02, 'Elevation (degrees)', ha='center', size=13)
        fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)
        if title:
            fig.suptitle(title)
        return fig, axis
    else:
        print('plot argument can be "image" or "correlation"')



""" Plot Images and Differences of HRTFs """
to_plot = 'average'  # 'average' or subject id
exclude = []  # to exclude from group average
n_bins = 300
bandwidth = (4000, 16000)
path = Path.cwd() / 'final_data' / 'experiment' / 'master'
subject_list = [subj.name for subj in list(path.iterdir())]
subject_list = [subj for subj in subject_list if subj not in exclude]
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

"""
read sofa files and process HRTFs :
smoothing applies lp filter at 1500 Hz
baselining subtracts mean power across DTFs within bandwidth and cuts DTFs at the specified bandwidth 
dfe applies diffuse field equalization 
"""

hrtf_dict = hrtf_analysis.get_hrtf_dict(path, subject_list, conditions, smoothe=True,
                                        baseline=True, bandwidth=bandwidth, dfe=True)

"""    plot  """
plot_dict = {}
for c in conditions:
    plot_dict[c] = hrtf_dict[c][to_plot]
hrtf_analysis.hrtf_images(plot_dict, n_bins=n_bins, bandwidth=bandwidth, plot='image')


"""

# hrtf_analysis.mean_vsi_across_bands(hrtf_dict, show=True)

lst = [subj for subj in subject_list if subj not in exclude]

  # plot subject hrtf / diff / corr
# for s in lst:
#     to_plot = s
#     plot_dict = {}
#     for c in conditions:
#         plot_dict[c] = hrtf_dict[c][to_plot]
#     hrtf_analysis.hrtf_images(plot_dict, n_bins, bandwidth=(4000, 16000), title=('participant id: %s' % s))



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hrtf.plot_tf(sources


#------ smoothe hrtf and write to file ------#
hrtf_dict = hrtf_analysis.get_hrtfs(path, subject_list, conditions, smoothe=False,
                                    baseline=False, bandwidth=bandwidth, dfe=False)
import copy
subject_dir_list = list(path.iterdir())
for condition in hrtf_dict.keys():
    for subj_idx, subject_path in enumerate(subject_dir_list):
        if subject_path.name in hrtf_dict[condition].keys():
            print('processing %s %s' %(subject_path.name, condition))
            hrtf_out = copy.deepcopy(hrtf_dict[condition][subject_path.name])
            hrtf_out = hrtf_analysis.smoothe_hrtf(hrtf_out, high_cutoff=1500)
            # hrtf_out = hrtf_out.diffuse_field_equalization()
            # hrtf_out = hrtf_analysis.baseline_hrtf(hrtf_out, bandwidth=(3000, 17000))
            hrtf_dict[condition][subject_path.name] = hrtf_out
            
hrtf_analysis.write_processed_hrtf(hrtf_dict, path, dir_name='processed_hrtf')

"""
