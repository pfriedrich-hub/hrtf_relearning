from pathlib import Path
import analysis.hrtf_analysis as hrtf_analysis


""" Plot Images, Differences, Correlation of HRTFs """

path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
subject_list = [subj.name for subj in list(path.iterdir())]
conditions = ['ears_free', 'earmolds', 'earmolds_1']
n_bins = 150
bandwidth = (4000, 16000)
# load hrtfs
hrtf_dict = hrtf_analysis.get_hrtfs(path, conditions)
# baseline HRTFs and return average for each condition
exclude = []
for condition in conditions:
    for subj in subject_list:
        # get HRTFs from one condition
        if not subj in hrtf_dict[condition].keys():
            print('%s hrtf in condition %s is missing' % (subj, condition))
            exclude.append(subj)
        else:
            hrtf = hrtf_dict[condition][subj]
            # process HRTFs
            hrtf = hrtf_analysis.baseline_hrtf(hrtf, bandwidth=bandwidth)
            hrtf = hrtf.diffuse_field_equalization()
            hrtf_dict[condition][subj] = hrtf
    # average HRTFs of the same condition
    hrtf_dict[condition]['average'] = hrtf_analysis.average_hrtf(list(hrtf_dict[condition].values()))

# plot subject hrtfs / diff / corr
lst = [ele for ele in subject_list if ele not in exclude]
for s in lst:
    to_plot = s
    plot_dict = {}
    for c in conditions:
        plot_dict[c] = hrtf_dict[c][to_plot]
    hrtf_analysis.hrtf_plots(plot_dict, n_bins, bandwidth=(4000, 16000), title=('participant id: %s' % s))

to_plot = s
plot_dict = {}
for c in conditions:
    plot_dict[c] = hrtf_dict[c][to_plot]
hrtf_analysis.hrtf_plots(plot_dict, n_bins, bandwidth=(4000, 16000))


