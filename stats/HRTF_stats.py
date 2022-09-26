import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import numpy

def dtf_correlation(hrtf_1, hrtf_2, show=False, bandwidth=None, n_bins=96, axis=None, cbar=None):
    # get sources and dtfs
    sources = hrtf_1.cone_sources(0)
    dtf = hrtf_1.tfs_from_sources(sources, n_bins)
    dtf_2 = hrtf_2.tfs_from_sources(sources, n_bins)
    if bandwidth:  # cap dtf to bandwidth
        frequencies = numpy.linspace(0, hrtf_1[0].frequencies[-1], n_bins)
        dtf = dtf[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
        dtf_2 = dtf_2[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
    # calculate correlation coefficients
    n_sources = len(sources)
    corr_mtx = numpy.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr_mtx[i, j] = numpy.corrcoef(dtf[:, i], dtf_2[:, j])[1, 0]
    # plot correlation matrix
    if show:
        if axis is None:
            fig, axis = plt.subplots()
        else:
            fig = axis.figure
        levels = numpy.linspace(-1, 1, 11)
        contour = axis.contourf(hrtf_1.sources.vertical_polar[sources, 1],
                                hrtf_2.sources.vertical_polar[sources, 1], corr_mtx,
                                cmap='viridis', levels=levels)
        axis.set_xticks(numpy.arange(-37.5, 38, 12.5))
        axis.set_yticks(numpy.arange(-37.5, 38, 12.5))
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=numpy.linspace(-1, 1, 11),
                     label='Correlation Coefficient')
        axis.set_ylabel('Elevation (degrees)')
        axis.set_xlabel('Elevation (degrees)')
        plt.show()
    return corr_mtx, cbar

def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth):
    # get correlation matrices
    correlation_free_v_mold = dtf_correlation(hrtf_1, hrtf_2, bandwidth=bandwidth)
    autocorrelation_free = dtf_correlation(hrtf_1, hrtf_1, bandwidth=bandwidth)
    # VSI dissimilarity: euclidean distance between the matrices
    vsi_dissimilarity = numpy.linalg.norm(correlation_free_v_mold - autocorrelation_free)
    return vsi_dissimilarity




"""
# if __name__ == "__main__":
    # plot correlation_matrix
    dtf_correlation(hrtf_1, hrtf_2, show=True)
    # vsi dissimilarity
    vsi_dissimilarity = vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth=(5700, 11300))
    print('VSI Dissimilarity; %s and %s: %2f \n' % (sofa_1, sofa_2, vsi_dissimilarity))
    
    
data_dir = Path.cwd() / 'data' / 'hrtfs' / 'pilot'
sofa_1 = 'hannah_ears_free_23.09.sofa'
sofa_2 = 'hannah_mold_1.1_12_Sep.sofa'
hrtf_1 = slab.HRTF(data_dir / sofa_1)
hrtf_2 = slab.HRTF(data_dir / sofa_2)
hrtf_1 = slab.HRTF.kemar()
"""