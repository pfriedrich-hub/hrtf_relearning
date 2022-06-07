import scipy
import slab
from pathlib import Path
data_dir = Path.cwd() / 'data'
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

# elevation gain
subj_id = '001_05_Jun'
sequence = slab.Trialsequence(conditions=47, n_reps=1)
sequence.load_pickle(file_name=data_dir / 'localization_data' / subj_id)
loc_data = numpy.asarray(sequence.data)
loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)

ele_x = loc_data[:, 1, 1]  # target elevations
ele_y = loc_data[:, 0, 1]  # percieved elevations
plt.scatter(ele_x, ele_y)
elevation_gain = scipy.stats.linregress(ele_x, ele_y)[0]

az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
azimuth_gain = scipy.stats.linregress(az_x, az_y)[0]

#  DTF correlation matrix
data_dir = Path.cwd() / 'data'
filename = 'kemar_fflab.sofa'
hrtf_1 = slab.HRTF(data_dir / 'hrtfs' / filename)
hrtf_2 = slab.HRTF(data_dir / 'hrtfs' / filename)

sources = hrtf_1.cone_sources(0)
tfs_1 = hrtf_1.tfs_from_sources(sources, n_bins=200)
tfs_2 = hrtf_2.tfs_from_sources(sources, n_bins=200)

n_sources = len(sources)
corr_mtx = numpy.zeros((n_sources, n_sources))
for i in range(n_sources):
    for j in range(n_sources):
        corr_mtx[i, j] = numpy.corrcoef(tfs_1[:, i], tfs_2[:, j])[1, 0]

# plot correlation matrix
fig, axis = plt.subplots()
contour = axis.contourf(hrtf_1.sources[sources, 1], hrtf_2.sources[sources, 1], corr_mtx,
                        cmap='hot', levels=10)
ax, _ = matplotlib.colorbar.make_axes(plt.gca())
cbar = matplotlib.colorbar.ColorbarBase(ax, cmap='hot', ticks=numpy.arange(0,1.1,.1),
                       norm=matplotlib.colors.Normalize(vmin=0, vmax=1), label='Correlation Coefficient')
axis.set_ylabel('Elevation (degrees)')
axis.set_xlabel('Elevation (degrees)')

