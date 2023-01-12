from pathlib import Path
import analysis.hrtf_analysis as hrtf_analysis

""" Plot Images and Differences of HRTFs """
to_plot = 'nn'  # 'average' or subject id
exclude = []  # to exclude from group average
n_bins = 4884
bandwidth = (4000, 16000)
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
subject_list = [subj.name for subj in list(path.iterdir())]
subject_list = [subj for subj in subject_list if subj not in exclude]
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

"""
read sofa files and process HRTFs :
smoothing applies lp filter at 1500 Hz
baselining subtracts mean power across DTFs within bandwidth and cuts DTFs at the specified bandwidth 
dfe applies diffuse field equalization 
"""

hrtf_dict = hrtf_analysis.get_hrtfs(path, subject_list, conditions, smoothe=True,
                                    baseline=True, bandwidth=bandwidth, dfe=True)

"""    plot  """
plot_dict = {}
for c in conditions:
    plot_dict[c] = hrtf_dict[c][to_plot]
hrtf_analysis.hrtf_images(plot_dict, n_bins=n_bins, bandwidth=bandwidth, plot='image')


"""
# hrtf_analysis.mean_vsi_across_bands(hrtf_dict, show=True)

lst = [subj for subj in subject_list if subj not in exclude]

  # plot subject hrtfs / diff / corr
# for s in lst:
#     to_plot = s
#     plot_dict = {}
#     for c in conditions:
#         plot_dict[c] = hrtf_dict[c][to_plot]
#     hrtf_analysis.hrtf_images(plot_dict, n_bins, bandwidth=(4000, 16000), title=('participant id: %s' % s))



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hrtf.plot_tf(sources
"""