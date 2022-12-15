from analysis.localization_analysis import localization_accuracy
from pathlib import Path
import matplotlib.pyplot as plt
import slab
""" -------  plot localization accuracy of all participants ------ """
# get path for each subject data folder
subject_dir_list = list((Path.cwd() / 'data' / 'experiment' / 'bracket_1').iterdir())
condition = 'earmolds_1'
fig, axis = plt.subplots(6, len(subject_dir_list), sharex=True, sharey=True)

for subj_idx, subject_path in enumerate(subject_dir_list):
    subject_dir = subject_path / condition
    file_idx = 0
    for file_name in sorted(list(subject_dir.iterdir())):
        if file_name.is_file() and not file_name.suffix == '.sofa':
            sequence = slab.Trialsequence(conditions=45, n_reps=1)
            sequence.load_pickle(file_name=file_name)
            elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=2,
                                        binned=True, axis=axis[file_idx, subj_idx])
            axis[file_idx, 0].set_ylabel('Day %i' % int(file_idx+1))
            axis[file_idx, 0].yaxis.set_label_coords(-0.5, 0.5)
            file_idx += 1
fig.text(0.5, 0.07, 'Response azimuth (deg)', ha='center')
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical')
axis[0,0].set_xticks(axis[0,0].get_xticks().astype('int'))
for idx, i in enumerate(range(2, 10, 2)):
    fig.text(i/10, 0.95, subject_dir_list[idx].name)
