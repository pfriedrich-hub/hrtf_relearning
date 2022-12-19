from pathlib import Path
import analysis.localization_analysis as localization
import numpy
import slab
from matplotlib import pyplot as plt
import scipy

""" -------  plot group averaged learning curve ------ """

w2_exclude = ['cs']
bracket = 'bracket_1'
conditions = ['ears_free', 'earmolds', 'earmolds_1']
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
loc_dict = localization.get_localization_data(path, conditions)
subjects = list(loc_dict['ears_free'].keys())
for condition in conditions:
    loc_dict[condition]['data'] = numpy.zeros((len(subjects), 7, 3))  # subject x days x eg/rmse/sd
    loc_dict[condition]['SE'] = numpy.zeros((7, 3))  # SE for each measure days x eg/rmse/sd
    for s, subject in enumerate(subjects):
        if not (subject in w2_exclude and condition == 'earmolds_1'):
            sequence_list = loc_dict[condition][subject]
            for idx, sequence in enumerate(sequence_list):
                loc_dict[condition]['data'][s, idx] = localization.localization_accuracy(sequence, show=False)
                if s+1 == len(subjects):
                    loc_dict[condition]['SE'][idx] = scipy.stats.sem(loc_dict[condition]['data'][:, idx], axis=0)
                # standard mean error (sd / sqrt(n)): (sd = sqrt(var), var = mean distance from sample mean)

ex_idx = [subjects.index(ex) for ex in w2_exclude]  # remove w2 excludes
loc_dict['earmolds_1']['data'] = numpy.delete(loc_dict['earmolds_1']['data'], ex_idx, axis=0)

days = numpy.arange(1, 13) # days of measurement
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
    axes[ax_id].errorbar(days[5:], m2[:7, i], capsize=3, yerr=loc_dict['earmolds_1']['SE'][:7, i],  # error bar mold 2
                   fmt="o", c=color, elinewidth=0.8, markersize=5)

axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))
axes[0].set_xticks(days)
axes[0].set_ylabel('Elevation Gain')
axes[1].set_xlabel('Days')
axes[1].set_ylabel('Elevation in degrees')
axes[1].legend()
plt.show()


"""
# RMSE & SD
# week 1
 # week 1 learning curve
colors = ['k', '0.6']
labels = ['RMSE', 'SD']
for i in range(1, 3):
    ax[1].plot(w1[:2], numpy.mean(w1_data[:, i, :2], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
    ax[1].plot(w1[1:-1], numpy.mean(w1_data[:, i, 1:-1], axis=0), c=colors[i-1], linewidth=1.5, label=labels[i-1])  # week 1 learning curve
    ax[1].plot(w1[-2:], numpy.mean(w1_data[:, i, -2:], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
    # week 2
    ax[1].plot(w2[:2], numpy.mean(w2_data[:, i, :2], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
    ax[1].plot(w2[1:-3], numpy.mean(w2_data[:, i, 1:-3], axis=0), c=colors[i-1], linewidth=1.5)  # week 2 learning curve
    ax[1].plot(w2[-3:-1], numpy.mean(w2_data[:, i, -3:-1], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
    # mold 1 adaptation persistence
    ax[1].plot([w1[-1], w2[-2]], [numpy.mean(w1_data[:, i, -2], axis=0), numpy.mean(w2_data[:, i, -2], axis=0)],
               c=colors[i-1], ls=(0, (5, 10)), lw=0.8)
    # mold 2 adaptation persistence
    ax[1].plot(w2[-2:], numpy.mean(w2_data[:, i, -2:], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)
    # error bars
    ax[1].errorbar(w1[0], numpy.mean(w1_data[:, i, 0], axis=0), capsize=3, yerr=w1_err[i, 0],
                   fmt="o", c=colors[i-1], elinewidth=0.8, markersize=5, fillstyle='none')
    ax[1].errorbar(w1[1:-1], numpy.mean(w1_data[:, i, 1:-1], axis=0), capsize=3, yerr=w1_err[i, 1:-1],
                   fmt="o", c=colors[i-1], elinewidth=0.8, markersize=5)
    ax[1].errorbar(w1[-1], numpy.mean(w1_data[:, i, -1], axis=0), capsize=3, yerr=w1_err[i, -1],
                   fmt="o", c=colors[i-1], elinewidth=0.8, markersize=5, fillstyle='none')
    ax[1].errorbar(w2[1:-1], numpy.mean(w2_data[:, i, 1:-1], axis=0), capsize=3, yerr=w2_err[i, 1:-1],
                   fmt="o", c=colors[i-1], elinewidth=0.8, markersize=5)
    ax[1].errorbar(w2[-1:], numpy.mean(w2_data[:, i, -1], axis=0), capsize=3, yerr=w2_err[i, -1],
                   fmt="o", c=colors[i-1], elinewidth=0.8, markersize=5)







# plot Elevation Gain, RMSE and response variability across experiment

n = 17  # number of measurements per participant
subject_paths = list((Path.cwd() / 'data' / 'experiment' / bracket).iterdir())
condition_list = ['ears_free', 'earmolds', 'earmolds_1']
subj_list = []
data = numpy.zeros((len(subject_paths), 3, n))  # subject_id x measurement (eg, rmse, sd) x n datapoints
for subj_idx, subject_path in enumerate(subject_paths):
    # print('\n' + subject_path.name)
    subj_list.append(subject_path.name)
    for condition in condition_list:
        # print(condition)
        subject_dir = subject_path / condition
        f_idx = 0
        for file_name in sorted(list(subject_dir.iterdir())):
            if file_name.is_file() and not file_name.suffix == '.sofa':
                # print(file_name.name)
                sequence = slab.Trialsequence(conditions=45, n_reps=1)
                sequence.load_pickle(file_name=file_name)
                g, r, s = localization.localization_accuracy(sequence, show=False)
                if condition == 'ears_free' and f_idx == 0:
                    n_idx = 0
                elif condition == 'earmolds':
                    n_idx = f_idx + 1
                elif condition == 'ears_free' and f_idx == 1:
                    n_idx = 7  # 7th datapoint (ears free in between the two molds)
                elif condition == 'earmolds_1':
                    n_idx = f_idx + 8
                data[subj_idx, 0, n_idx], data[subj_idx, 1, n_idx], data[subj_idx, 2, n_idx] = g, r, s
                f_idx += 1

                w1_data = data[:, :, :len(w1)]
                excl_idx = [subj_list.index(excl) for excl in w2_exclude]
                w2_data = numpy.delete(data[:, :, -len(w2):], excl_idx, axis=0)  # exclude participants from week 2
                # exclude a participant from week 1 here:
                w1_data = numpy.delete(w1_data, excl_idx, axis=0)

                # standard mean error (sd / sqrt(n)): (sd = sqrt(var), var = mean distance from sample mean)
                w1_err, w2_err = scipy.stats.sem(w1_data, axis=0), scipy.stats.sem(w2_data, axis=0)
"""