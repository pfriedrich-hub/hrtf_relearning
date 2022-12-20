import slab
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy
import statsmodels

"""  DTF correlation matrix  """
data_dir = Path.cwd() / 'data'
filename1 = 'kemar_free.sofa'
filename2 = 'kemar_mold_1.sofa'
hrtf_1 = slab.HRTF(data_dir / 'hrtfs' / filename1)
hrtf_2 = slab.HRTF(data_dir / 'hrtfs' / filename1)

sources = hrtf_1.cone_sources(0)
tfs_1 = hrtf_1.tfs_from_sources(sources, n_bins=200)
tfs_2 = hrtf_2.tfs_from_sources(sources, n_bins=200)

n_sources = len(sources)
corr_mtx = numpy.zeros((n_sources, n_sources))
for i in range(n_sources):
    for j in range(n_sources):
        corr_mtx[i, j] = numpy.corrcoef(tfs_1[:, i], tfs_2[:, j])[1, 0]

"""  plot correlation matrix  """
fig, axis = plt.subplots()
contour = axis.contourf(hrtf_1.sources.vertical_polar[sources, 1], hrtf_2.sources.vertical_polar[sources, 1], corr_mtx,
                        cmap=None, levels=10)
ax, _ = matplotlib.colorbar.make_axes(plt.gca())
cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=None, ticks=numpy.arange(0, 1.1, .1),
                       norm=matplotlib.colors.Normalize(vmin=0, vmax=1), label='Correlation Coefficient')
axis.set_ylabel('HRTF 1 Elevation (degrees)')
axis.set_xlabel('HRTF 2 Elevation (degrees)')


""" power analysis """
from statsmodels.stats.power import tt_solve_power
import numpy
# power: (= 1 - beta, fehler 2. art) - wahrscheinlichkeit die nullhypothese richtigerweise zu verwerfen
# alpha: (fehler 1. art) - wahrscheinlichkeit, die nullhypothese fälschlicherweise zu verwerfen
# effect_size: cohen's d
x1 = 0
x2 = 0
std1 = 0
std2 = 0
d = (numpy.mean(x1) - numpy.mean(x2)) / numpy.sqrt(((std1**2)+(std2**2))/2)
# calculate N
N = tt_solve_power(power=0,effect_size=0,alpha=0)


# remove invalid values, this is redundant for meta motion head tracking
# bads_idx = numpy.where(ele_y == None)
# ele_y = numpy.array(numpy.delete(ele_y, bads_idx), dtype='float')
# ele_x = numpy.array(numpy.delete(ele_x, bads_idx), dtype='float')




