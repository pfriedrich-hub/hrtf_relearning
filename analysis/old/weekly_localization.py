import analysis.localization_analysis as localization
import matplotlib.pyplot as plt
import numpy

""" -------  plot localization accuracy of all participants --------- """

plot_condition = 'Earmolds Week 1'
bracket = 'bracket_1'

# conditions = ['Ears Free', 'Earmolds Week 1'] #, 'Earmolds Week 2']

loc_dict = localization.get_localization_dictionary()
subjects = ['lku']
subjects = list(loc_dict['Ears Free'].keys())
# for ex in exclude: subjects.remove(ex)

fig, axis = plt.subplots(6, len(subjects), sharex=True, sharey=True, figsize=(15, 8))
for s_id, subj in enumerate(subjects):
    for f_id, file in enumerate(loc_dict[plot_condition][subj]):
        if f_id < 6:
            localization.localization_accuracy(loc_dict[plot_condition][subj][f_id], show=True,
                                               plot_dim=2, binned=True, axis=axis[f_id, s_id])
fig.text(0.5, 0.07, 'Response azimuth (deg)', ha='center')
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical')
axis[0, 0].set_xticks(numpy.array((-50, 0, 50)))
axis[0, 0].set_yticks(numpy.array((-30, 0, 30)))
for sub_idx, i in enumerate(range(2, 10, 2)):
    fig.text(i/10, 0.95, subjects[sub_idx])
fig.text(0.5, 0.97, plot_condition, ha='center', size=15)

# save as scalable vector graphics
# fig.savefig(Path.cwd() / 'data' / 'experiment' / 'images' / 'bracket_1' / 'prelim_localization_accuracy.svg', format='svg')
