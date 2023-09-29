import slab
from pathlib import Path
from analysis.localization_analysis import localization_accuracy

file_name = 'localization_mh_Earmolds Week 2_16.08'

for path in Path.cwd().glob("**/"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_path)

# plot
from matplotlib import pyplot as plt
fig, axis = plt.subplots(1, 1)
elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2,
 binned=True, axis=axis)
axis.set_xlabel('Response Azimuth (degrees)')
axis.set_ylabel('Response Elevation (degrees)')
fig.suptitle(file_name)