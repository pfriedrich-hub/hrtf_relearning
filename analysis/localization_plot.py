import analysis.localization_analysis as localization
from pathlib import Path
import matplotlib.pyplot as plt

""" -------  plot localization accuracy of all participants --------- """

plot_condition = 'earmolds'
exclude = []
w2_exclude = []
bracket = 'bracket_1'

conditions = ['ears_free', 'earmolds', 'earmolds_1']
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
loc_dict = localization.get_localization_data(path, conditions)
subjects = list(loc_dict['ears_free'].keys())
for ex in exclude: subjects.remove(ex)

fig, axis = plt.subplots(6, len(subjects), sharex=True, sharey=True)
for s_id, subj in enumerate(subjects):
    for f_id, file in enumerate(loc_dict[plot_condition][subj]):
        if f_id < 6:
            localization.localization_accuracy(loc_dict[plot_condition][subj][f_id], show=True,
                                               plot_dim=2, binned=True, axis=axis[f_id, s_id])
fig.text(0.5, 0.07, 'Response azimuth (deg)', ha='center')
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical')
axis[0, 0].set_xticks(axis[0, 0].get_xticks().astype('int'))
for idx, i in enumerate(range(2, 10, 2)):
    fig.text(i/10, 0.95, subjects[idx])
