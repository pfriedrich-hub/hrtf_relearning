from pathlib import Path
import analysis.hrtf_analysis as hrtf_analysis
import numpy

""" Plot Images, Differences, Correlation of HRTFs """
exclude = []
n_bins = 150
bandwidth = (4000, 16000)
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
subject_list = [subj.name for subj in list(path.iterdir())]
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

# read sofa files
hrtf_dict = hrtf_analysis.get_hrtfs(path, conditions, processed=True)
# baseline HRTFs and return average for each condition
for condition in conditions:
    for subj in subject_list:
        # get HRTFs from one condition
        if not subj in hrtf_dict[condition].keys():
            print('%s hrtf in condition %s is missing' % (subj, condition))
            exclude.append(subj)
        else:
            hrtf = hrtf_dict[condition][subj]
            # process HRTFs
            # hrtf = hrtf_analysis.smoothe_hrtf(hrtf, high_cutoff=1500)
            hrtf = hrtf_analysis.baseline_hrtf(hrtf, bandwidth=bandwidth)
            hrtf = hrtf.diffuse_field_equalization()
            hrtf_dict[condition][subj] = hrtf
    # average HRTFs of the same condition
    hrtf_dict[condition]['average'] = hrtf_analysis.average_hrtf(list(hrtf_dict[condition].values()))

lst = [subj for subj in subject_list if subj not in exclude]

to_plot = 'average'  # 'average' or subject id
plot_dict = {}
for c in conditions:
    plot_dict[c] = hrtf_dict[c][to_plot]
hrtf_analysis.hrtf_images(plot_dict, n_bins, bandwidth=(4000, 16000), show_tf=False, show_corr=True)

