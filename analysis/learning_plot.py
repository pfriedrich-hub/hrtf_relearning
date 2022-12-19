from pathlib import Path
import analysis.localization_analysis as localization
import numpy
import slab
from matplotlib import pyplot as plt
import scipy

""" -------  plot group averaged learning curve ------ """

exclude = ['cs', 'lw']
w2_exclude = []
bracket = 'bracket_1'
conditions = ['ears_free', 'earmolds', 'earmolds_1']
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
loc_dict = localization.get_localization_data(path, conditions)
subjects = list(loc_dict['ears_free'].keys())
for ex in exclude: subjects.remove(ex)
for condition in conditions:
    loc_dict[condition]['data'] = numpy.zeros((len(subjects), 7, 3))  # subject x days x eg/rmse/sd
    loc_dict[condition]['SE'] = numpy.zeros((7, 3))  # SE for each measure days x eg/rmse/sd
    for s, subject in enumerate(subjects):
        sequence_list = loc_dict[condition][subject]
        for idx, sequence in enumerate(sequence_list):
            loc_dict[condition]['data'][s, idx] = localization.localization_accuracy(sequence, show=False)
            if s+1 == len(subjects):
                loc_dict[condition]['SE'][idx] = scipy.stats.sem(loc_dict[condition]['data'][:, idx], axis=0)
            # standard mean error (sd / sqrt(n)): (sd = sqrt(var), var = mean distance from sample mean)

ex_idx = [subjects.index(ex) for ex in w2_exclude]  # remove w2 excludes
loc_dict['earmolds_1']['data'] = numpy.delete(loc_dict['earmolds_1']['data'], ex_idx, axis=0)
loc_dict['ears_free']['data'] = numpy.delete(loc_dict['ears_free']['data'], ex_idx, axis=0)

days = numpy.arange(1, 13)  # days of measurement
days[-1] = 16
ef = numpy.mean(loc_dict['ears_free']['data'], axis=0)  # means
m1 = numpy.mean(loc_dict['earmolds']['data'], axis=0)
m2 = numpy.mean(loc_dict['earmolds_1']['data'], axis=0)

labels = ['RMSE', 'SD']
colors = ['k', '0.6']
color = colors[0]
label = None
fig, axes = plt.subplots(2, 1)
for i, ax_id in enumerate([0, 1, 1]):
    if i >= 1:
        label = labels[i-1]
        color = colors[i-1]
    # EG # week 1
    axes[ax_id].plot([1, 1], [ef[0, i], m1[0, i]], c=color, ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
    axes[ax_id].plot(days[:6], m1[:6, i], c=color, linewidth=1.5, label=label)  # week 1 learning curve
    axes[ax_id].plot([6, 6], [m1[5, i], ef[1, i]], c=color, ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
    # week 2
    axes[ax_id].plot([6, 6], [ef[1, i], m2[0, i]], c=color, ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
    axes[ax_id].plot(days[5:11], m2[:6, i], c=color, linewidth=1.5)  # week 2 learning curve
    axes[ax_id].plot([11, 11], [m2[5, i], ef[2, i]], c=color, ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
    # mold 1 adaptation persistence
    axes[ax_id].plot([6, 11], m1[-2:, i], c=color, ls=(0, (5, 10)), lw=0.8)
    # mold 2 adaptation persistence
    axes[ax_id].plot(days[-2:], m2[-2:, i], c=color, ls=(0, (5, 10)), lw=0.8)
    # error bars
    axes[ax_id].errorbar([1, 6, 11], ef[:3, i], capsize=3, yerr=loc_dict['ears_free']['SE'][:3, i],
                   fmt="o", c=color, elinewidth=0.8, markersize=5, fillstyle='none')  # error bar ears free
    axes[ax_id].errorbar(numpy.append(days[:6], 11), m1[:7, i], capsize=3, yerr=loc_dict['earmolds']['SE'][:7, i],  # error bar mold 2
                   fmt="o", c=color, elinewidth=0.8, markersize=5)
    # axes[ax_id].errorbar(days[5:], m2[:7, i], capsize=3, yerr=loc_dict['earmolds_1']['SE'][:7, i],  # error bar mold 2
    #                fmt="o", c=color, elinewidth=0.8, markersize=5)
    axes[ax_id].errorbar(days[5:-1], m2[:6, i], capsize=3, yerr=loc_dict['earmolds_1']['SE'][:6, i],  # error bar mold 2
                   fmt="o", c=color, elinewidth=0.8, markersize=5)

axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))
axes[0].set_xticks(days)
axes[0].set_ylabel('Elevation Gain')
axes[1].set_xlabel('Days')
axes[1].set_ylabel('Elevation in degrees')
axes[1].legend()

fig.text(0.3, 0.8, f'n={len(subjects)}', ha='center', size=10)
fig.text(0.5, 0.8, f'n={len(subjects) - len(w2_exclude)}', ha='center', size=10)
subjects = ' '.join([str(item) for item in subjects])
w2_exclude = ' '.join([str(item) for item in w2_exclude])
plt.suptitle(f'w1: {subjects}, w2 excluded: {w2_exclude}')
plt.show()

# save as scalable vector graphics
# fig.savefig(Path.cwd() / 'data' / 'experiment' / 'images' / 'bracket_1' / 'prelim_localization_accuracy.svg', format='svg')
