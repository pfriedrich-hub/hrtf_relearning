import analysis.localization_analysis as localization
from pathlib import Path
import matplotlib.pyplot as plt
import numpy

""" -------  plot localization free, 1st vs last day of molds --------- """
bracket = 'bracket_1'

conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

path = Path.cwd() / 'final_data' / 'experiment' / bracket
loc_dict = localization.get_localization_data(path, conditions)
subjects = list(loc_dict['Ears Free'].keys())

fig, axis = plt.subplots(7, len(subjects), sharex=True, sharey=True, figsize=[15, 8])
fig.subplots_adjust(left=None, bottom=0.1, right=0.96, top=0.96, wspace=0.05, hspace=0.1)

for s_id, subj in enumerate(subjects):
    ax = axis[0, s_id]
    localization.localization_accuracy(loc_dict['Ears Free'][subj][0], show=True,
                                       plot_dim=2, binned=True, axis=ax)  # ears free D1
    eg = ax.get_title()[-4:]
    ax.set_title(label=f'EG {eg}', y=0.8, size=10)
    for ax_id, i in zip([0, 1], [0, 5]):
        ax = axis[ax_id+1, s_id]
        localization.localization_accuracy(loc_dict['Earmolds Week 1'][subj][i], show=True,
                                           plot_dim=2, binned=True, axis=ax)  # M1 D1 / D6
        eg = ax.get_title()[-4:]
        ax.set_title(label=f'EG {eg}', y=0.8, size=10)
    if len(loc_dict['Earmolds Week 2'][subj]) >= 5:
        for ax_id, i in zip([1, 2], [0, 5]):
            ax = axis[ax_id+2, s_id]
            localization.localization_accuracy(loc_dict['Earmolds Week 2'][subj][i], show=True,
                                               plot_dim=2, binned=True, axis=ax)  # M2 D1 / D6
            eg = ax.get_title()[-4:]
            ax.set_title(label=f'EG {eg}', y=0.8, size=10)
        localization.localization_accuracy(loc_dict['Earmolds Week 1'][subj][-1], show=True,
                                           plot_dim=2, binned=True, axis=axis[5, s_id])  # M1 D11
        eg = axis[5, s_id].get_title()[-4:]
        axis[5, s_id].set_title(label=f'EG {eg}', y=0.8, size=10)
        localization.localization_accuracy(loc_dict['Earmolds Week 2'][subj][-1], show=True,
                                           plot_dim=2, binned=True, axis=axis[6, s_id])  # M2 D16
        eg = axis[6, s_id].get_title()[-4:]
        axis[6, s_id].set_title(label=f'EG {eg}', y=0.8, size=10)

fig.text(0.5, 0.05, 'Response azimuth (deg)', ha='center', size=12)
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical', size=12)
axis[0, 0].set_xticks(axis[0, 0].get_xticks().astype('int'))
yticks = axis[0, 0].get_yticks()#[1:-1]
axis[0, 0].set_yticks(yticks.astype('int'))
# for idx, i in enumerate(range(2, 10, 2)):
#     fig.text(i/10, 0.95, subjects[idx])
c = ['Ears Free\nDay1', 'Mold 1\nDay 1', 'Mold 1\nDay 6', 'Mold 2\nDay 1',
     'Mold 2\nDay 6', 'Mold 1\nDay 11', 'Mold 2\nDay 16']
for idx, i in enumerate(numpy.linspace(0.87, 0.15, 7)):
    fig.text(0.03, i, f'{c[idx]}', size=13)

# save as scalable vector graphics
# fig.savefig(Path.cwd() / 'final_data' / 'experiment' / 'images' / 'bracket_1' / 'prelim_localization_accuracy.svg', format='svg')
