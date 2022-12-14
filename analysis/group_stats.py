import analysis.hrtf_analysis as hrtf_analysis
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
import numpy
import slab
from matplotlib import pyplot as plt
import scipy


""" plot mean learning curve """
w2_exclude = ['cs']

# plot Elevation Gain, RMSE and response variability across experiment
n = 17  # number of measurements per participant
subject_paths = list((Path.cwd() / 'data' / 'experiment' / 'bracket_1').iterdir())
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
                g, r, s = localization_accuracy(sequence, show=False)
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

w1 = [1, 1, 2, 3, 4, 5, 6, 6]  # week 1
w2 = [6, 6, 7, 8, 9, 10, 11, 11, 11, 16]  # week 2
days = w1 + w2
w1_data = data[:, :, :len(w1)]
excl_idx = [subj_list.index(excl) for excl in w2_exclude]
w2_data = numpy.delete(data[:, :, -len(w2):], excl_idx, axis=0)  # exclude participants from week 2
# exclude a participant from week 1 here:
w1_data = numpy.delete(w1_data, excl_idx, axis=0)

# standard mean error (sd / sqrt(n)): (sd = sqrt(var), var = mean distance from sample mean)
w1_err, w2_err = scipy.stats.sem(w1_data, axis=0), scipy.stats.sem(w2_data, axis=0)

fig, ax = plt.subplots(2, 1, sharex=True)
# EG # week 1
ax[0].plot(w1[:2], numpy.mean(w1_data[:, 0, :2], axis=0), c='k', ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
ax[0].plot(w1[1:-1], numpy.mean(w1_data[:, 0, 1:-1], axis=0), c='k', linewidth=1.5)  # week 1 learning curve
ax[0].plot(w1[-2:], numpy.mean(w1_data[:, 0, -2:], axis=0), c='k', ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
# week 2
ax[0].plot(w2[:2], numpy.mean(w2_data[:, 0, :2], axis=0), c='k', ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
ax[0].plot(w2[1:-3], numpy.mean(w2_data[:, 0, 1:-3], axis=0), c='k', linewidth=1.5)  # week 2 learning curve
ax[0].plot(w2[-3:-1], numpy.mean(w2_data[:, 0, -3:-1], axis=0), c='k', ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
# mold 1 adaptation persistence
ax[0].plot([w1[-1], w2[-2]], [numpy.mean(w1_data[:, 0, -2], axis=0), numpy.mean(w2_data[:, 0, -2], axis=0)],
           c='k', ls=(0, (5, 10)), lw=0.8)
# mold 2 adaptation persistence
ax[0].plot(w2[-2:], numpy.mean(w2_data[:, 0, -2:], axis=0), c='k', ls=(0, (5, 10)), lw=0.8)
# error bars
ax[0].errorbar(w1[0], numpy.mean(w1_data[:, 0, 0], axis=0), capsize=3, yerr=w1_err[0, 0],
               fmt="o", c='k', elinewidth=0.8, markersize=5, fillstyle='none')
ax[0].errorbar(w1[1:-1], numpy.mean(w1_data[:, 0, 1:-1], axis=0), capsize=3, yerr=w1_err[0, 1:-1],
               fmt="o", c='k', elinewidth=0.8, markersize=5)
ax[0].errorbar(w1[-1], numpy.mean(w1_data[:, 0, -1], axis=0), capsize=3, yerr=w1_err[0, -1],
               fmt="o", c='k', elinewidth=0.8, markersize=5, fillstyle='none')
ax[0].errorbar(w2[1:-1], numpy.mean(w2_data[:, 0, 1:-1], axis=0), capsize=3, yerr=w2_err[0, 1:-1],
               fmt="o", c='k', elinewidth=0.8, markersize=5)
ax[0].errorbar(w2[-1:], numpy.mean(w2_data[:, 0, -1], axis=0), capsize=3, yerr=w2_err[0, -1],
               fmt="o", c='k', elinewidth=0.8, markersize=5)

# RMSE & SD
# week 1
 # week 1 learning curve
colors = ['k', '0.6']
labels = ['MSE', 'SD']
for i in range(1, 3):
    ax[1].plot(w1[:2], numpy.mean(w1_data[:, i, :2], axis=0), c=colors[i-1], ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
    ax[1].plot(w1[1:-1], numpy.mean(w1_data[:, i, 1:-1], axis=0), c='k', linewidth=1.5, label=labels[i])  # week 1 learning curve
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

ax[0].set_yticks(numpy.arange(0, 1.2, 0.2))
ax[0].set_xticks(days)
ax[0].set_xlabel('Days')
ax[0].set_ylabel('Elevation Gain')
ax[1].set_xlabel('Days')
ax[1].set_ylabel('Elevation in degrees')
ax[1].legend()

# plot participants EG
# fig, ax = plt.subplots(1, 1, sharex=True)
# for i in range(ele_gain.shape[0]):
#     ax.scatter(days, ele_gain[i, :], label='participant %i'% i)
#     ax.plot(days, ele_gain[i, :])
# ax.set_xticks(days)
# ax.set_ylabel('Elevation Gain')
# ax.set_xlabel('Days')
# ax.set_yticks(numpy.arange(0, 1.2, 0.1))
# ax.legend()