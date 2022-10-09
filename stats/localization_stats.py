import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

data_dir = Path.cwd() / 'data' / 'localization_data' / 'pilot'
subject_id = 'varvara_mold_1_07.10'

def localization_accuracy(subject_id, show=True):
    # calculate elevation gain
    sequence = slab.Trialsequence(conditions=47, n_reps=1)
    sequence.load_pickle(file_name=data_dir / subject_id)
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    target_elevations = loc_data[:, 1, 1]  # target elevations
    perceived_elevations = loc_data[:, 0, 1]  # percieved elevations
    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
    # below_ids = numpy.where(loc_data[:, 1, 1] < 0)
    elevation_gain, n = scipy.stats.linregress(target_elevations, perceived_elevations)[:2]
    rmse = numpy.sqrt(numpy.square(numpy.subtract(target_elevations, perceived_elevations)).mean())
    sd = numpy.mean([numpy.std(perceived_elevations[numpy.where(target_elevations == target)])
                for target in numpy.unique(target_elevations)])
    if show:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylim(-60, 60)
        axis.set_xlim(-60, 60)
        axis.set_xlabel('target elevations')
        axis.set_ylabel('perceived elevations')
        # scatter plot with regression line (elevation gain)
        axis.scatter(target_elevations[left_ids], perceived_elevations[left_ids], s=10, c='red', label='left')
        axis.scatter(target_elevations[right_ids], perceived_elevations[right_ids], s=10, c='blue', label='right')
        axis.scatter(target_elevations[mid_ids], perceived_elevations[mid_ids], s=10, c='black', label='middle')
        x = numpy.arange(-55, 56)
        y = elevation_gain * x + n
        axis.plot(x, y, c='grey', linewidth=0.6, label='elevation gain %.2f' % elevation_gain)
        plt.legend()
        axis.set_title(str(subject_id))
        plt.show()
    return elevation_gain, rmse, sd

if __name__ == "__main__":
    elevation_gain, rmse, sd = localization_accuracy(subject_id, show=True)

"""
# for azimuth:
az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
bads_idx = numpy.where(az_y == None)
az_y = numpy.array(numpy.delete(az_y, bads_idx), dtype=numpy.float)
az_x = numpy.array(numpy.delete(az_x, bads_idx), dtype=numpy.float)
azimuth_gain = scipy.stats.linregress(az_x, az_y)[0]
"""