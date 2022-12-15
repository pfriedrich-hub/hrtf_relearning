import analysis.localization_analysis as localization
from pathlib import Path
import matplotlib.pyplot as plt
import numpy

""" -------  plot localization accuracy of all participants --------- """

plot_condition = 'earmolds_1'
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
for sub_idx, i in enumerate(range(2, 10, 2)):
    fig.text(i/10, 0.95, subjects[sub_idx])




""" -------  plot localization free, 1st vs last day of molds --------- """
exclude = []
w2_exclude = []
bracket = 'bracket_1'

conditions = ['ears_free', 'earmolds', 'earmolds_1']
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
loc_dict = localization.get_localization_data(path, conditions)
subjects = list(loc_dict['ears_free'].keys())
for ex in exclude: subjects.remove(ex)

fig, axis = plt.subplots(6, len(subjects), sharex=True, sharey=True, figsize=[15, 10])
fig.subplots_adjust(left=None, bottom=0.1, right=0.96, top=0.96, wspace=0.05, hspace=0.1)

for s_id, subj in enumerate(subjects):
    ax = axis[0, s_id]
    localization.localization_accuracy(loc_dict['ears_free'][subj][0], show=True,
                                       plot_dim=2, binned=True, axis=ax)  # ears free D1
    eg = ax.get_title()[-4:]
    ax.set_title(label=f'EG {eg}', y=0.8, size=10)
    for ax_id, i in zip([0, 1], [0, 5]):
        ax = axis[ax_id+1, s_id]
        localization.localization_accuracy(loc_dict['earmolds'][subj][i], show=True,
                                           plot_dim=2, binned=True, axis=ax)  # M1 D1 / D6
        eg = ax.get_title()[-4:]
        ax.set_title(label=f'EG {eg}', y=0.8, size=10)
    if len(loc_dict['earmolds_1'][subj]) >= 5:
        for ax_id, i in zip([1, 2], [0, 5]):
            ax = axis[ax_id+2, s_id]
            localization.localization_accuracy(loc_dict['earmolds_1'][subj][i], show=True,
                                               plot_dim=2, binned=True, axis=ax)  # M2 D1 / D6
            eg = ax.get_title()[-4:]
            ax.set_title(label=f'EG {eg}', y=0.8, size=10)
        localization.localization_accuracy(loc_dict['earmolds'][subj][-1], show=True,
                                           plot_dim=2, binned=True, axis=axis[5, s_id])  # M1 D11
        eg = axis[5, s_id].get_title()[-4:]
        axis[5, s_id].set_title(label=f'EG {eg}', y=0.8, size=10)
    # for ax_id, i in zip([0, 1], [0, 5]):
    #     localization.localization_accuracy(loc_dict['earmolds'][subj][i], show=True,
    #                                        plot_dim=2, binned=True, axis=axis[ax_id+1, s_id])  # M2 D16
fig.text(0.5, 0.05, 'Response azimuth (deg)', ha='center', size=12)
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical', size=12)
axis[0, 0].set_xticks(axis[0, 0].get_xticks().astype('int'))
yticks = axis[0, 0].get_yticks()#[1:-1]
axis[0, 0].set_yticks(yticks.astype('int'))
# for idx, i in enumerate(range(2, 10, 2)):
#     fig.text(i/10, 0.95, subjects[idx])
c = ['Ears Free\nDay1', 'Mold 1\nDay 1', 'Mold 1\nDay 6', 'Mold 2\nDay 1', 'Mold 2\nDay 6', 'Mold 1\nDay 11']
for idx, i in enumerate(numpy.linspace(0.87, 0.15, 6)):
    fig.text(0.03, i, f'{c[idx]}', size=13)

