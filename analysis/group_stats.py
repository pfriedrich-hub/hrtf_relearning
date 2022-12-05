import analysis.hrtf_analysis as hrtf_analysis
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
import numpy
import os

# plot Elevation Gain, RMSE and response variability across experiment
# get files
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1'

def plot_learning(data_dir, group_stats):
    # get localization accuracy data (if group analysis, get data from subject subdirectories,
    if group_stats:
        subject_list = next(os.walk(data_dir))[1]
    else:
        subject_list = [data_dir.name]
        data_dir = data_dir.parent

    ele_gain = numpy.zeros((len(subject_list), 6))  # for now, 6 measurements per participant
    rmse = numpy.zeros((len(subject_list), 6))
    sd = numpy.zeros((len(subject_list), 6))
    for subj_idx, subject_folder in enumerate(subject_list):
        subj_files = os.listdir(str(data_dir / subject_folder))
        subj_files = [fname for fname in subj_files if not fname.endswith('.sofa')]  # remove sofa files from list
        if '.DS_Store' in subj_files: subj_files.remove('.DS_Store')
        subj_files.sort()
        for file_idx, subj_file in enumerate(subj_files):
            ele_gain[subj_idx, file_idx], rmse[subj_idx, file_idx], sd[subj_idx, file_idx] \
                = localization_accuracy.localization_accuracy(data_dir / subject_folder / subj_file, show=False)
    # plot participants EG
    fig, ax = plt.subplots(1, 3, sharex=True)
    days = numpy.arange(0, duration+1)
    days[0] = 1
    for i in range(ele_gain.shape[0]):
        ax[0].scatter(days, ele_gain[i, :], label='participant %i'% i)
        ax[0].plot(days, ele_gain[i, :])
    ax[0].set_xticks(numpy.arange(1,duration+1))
    ax[0].set_ylabel('Elevation Gain')
    ax[0].set_xlabel('Days')
    ax[0].set_yticks(numpy.arange(0,1.2,0.1))
    ax[0].legend()
    # EG
    ax[1].plot(days, numpy.mean(ele_gain, axis=0), c='k', linewidth=0.5)
    ax[1].set_yticks(numpy.arange(0,1.2,0.1))
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel('Elevation Gain')
    ax[1].set_title('Elevation Gain')
    ax[2].plot(days, numpy.mean(rmse, axis=0), c='k', linewidth=0.5, label='RMSE')    # RMSE
    ax[2].plot(days, numpy.mean(sd, axis=0), c='0.6', linewidth=0.5, label='SD')     # SD
    # ax[2].set_yticks(numpy.arange(0,23,2))
    ax[2].set_xlabel('Days')
    ax[2].set_title('RMSE')
    ax[2].legend()
    ax[2].set_ylabel('Elevation in degrees')
    if group_stats:
        ax[1].errorbar(days, numpy.mean(ele_gain, axis=0), capsize=3, yerr=numpy.abs(numpy.diff(ele_gain, axis=0)),
                       fmt="o", c='k', elinewidth=0.5, markersize=3)
        ax[2].errorbar(days, numpy.mean(rmse, axis=0), capsize=3, yerr=numpy.abs(numpy.diff(rmse, axis=0)),
                       fmt="o", c='k', elinewidth=0.5, markersize=3)