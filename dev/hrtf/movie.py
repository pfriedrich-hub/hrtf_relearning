import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from misc.average import hrtf_average
from misc.feature_p import feature_p


def movie(hrtf_list, azimuth_range, elevation_range, ear='both', interval=500, map='feature_p', kind='image', save=None):
    global data, fig, ax, frequencies, azimuths, elevations, settings
    settings = {'map': map, 'kind': kind}
    # plot features for sources in range 0 / +50 azimuth across elevations
    # if axis == 'azimuth':
    source_idx = hrtf_list[0].get_source_idx(azimuth=azimuth_range, elevation=elevation_range)
    azimuths = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 0])
    elevations = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 1])
    bandwidth = (1000, 18000)
    data = []
    for i, az in enumerate(azimuths):
        print(f'Azimuth: {az}')
        source_idx = hrtf_list[0].get_source_idx(azimuth=az,elevation=(elevations.min(), elevations.max()))
        # sort by ascending elevation
        sources = hrtf_list[0].sources.vertical_polar[source_idx]
        sorting_idx = numpy.argsort(sources, axis=0, kind=None, order=None)[:,1]
        source_idx = source_idx[sorting_idx]
        if settings['map'] == 'feature_p':
            map, frequencies = feature_p(hrtf_list, source_idx, thresholds=None, bandwidth=bandwidth, ear=ear)
            map = map.reshape(map.shape[:2])
        elif settings['map'] == 'average':
            frequencies = hrtf_list[0][0].tf(show=False)[0]
            freq_idx = numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])
            frequencies = frequencies[freq_idx]
            map = hrtf_average(hrtf_list).tfs_from_sources(source_idx, n_bins=None, ear=ear)
            map = map.reshape(map.shape[:2])[:, freq_idx]
        data.append(map)

    fig, ax = plt.subplots(figsize=(10,5))

    if settings['kind'] == 'image':
        global cbar_axis, cbar_levels, cbar_ticks
        z_min = numpy.floor(numpy.min(data))
        z_max = numpy.ceil(numpy.max(data))
        cbar_levels = numpy.linspace(z_min, z_max, 50)  # set levels:contour 50:10
        cbar_ticks = numpy.arange(z_min, z_max, 6)[1:]
        cax_pos = list(ax.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[2] = cax_pos[2] * 0.06  # cbar width in fractions of axis width
        cax_pos[0] = 0.91
        cbar_axis = fig.add_axes(cax_pos)
        im = ax.contourf(frequencies, elevations, data[0], levels=cbar_levels)
        cbar = fig.colorbar(im, cbar_axis, orientation='vertical', ticks=cbar_ticks)
        if settings['map'] == 'feature_p':
            cbar_axis.set_title('p')
        elif settings['map'] == 'average':
            cbar_axis.set_title('dB')

    ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=interval, blit=False)
    if save:
        writervideo = animation.FFMpegWriter(fps=interval / 1000)
        ani.save(Path.cwd() / 'data' / 'animations' / str(save+'.mp4'), writer=writervideo)
    plt.show()

def init():
    im = plot(data=data[0])
    return (im,)

def animate(i):
    im = plot(data=data[i])
    ax.set_title(f'Azimuth: {azimuths[i]}')
    return (im,)

def plot(data):
    fig.axes[0].clear()
    if settings['kind'] == 'image':
        im = ax.contourf(frequencies, elevations, data, levels=cbar_levels)
    elif settings['kind'] == 'waterfall':
        linesep = 20
        vlines = numpy.arange(0, len(data)) * linesep
        for idx, filter in enumerate(data):
            im = ax.plot(frequencies, filter + vlines[idx], linewidth=0.75, color='0.0', alpha=0.7)
        ticks = vlines[::2]  # plot every second elevation
        labels = elevations
        # plot every third elevation label, omit comma to save space
        labels = labels[::2].astype(int)
        ax.set(yticks=ticks, yticklabels=labels)
        ax.grid(visible=True, axis='y', which='both', linewidth=0.25)
        ax.plot([frequencies[0] + 500, frequencies[0] + 500], [vlines[-1] + 10, vlines[-1] +
                                                   10 + linesep], linewidth=1, color='0.0', alpha=0.9)
        ax.text(x=frequencies[0] + 600, y=vlines[-1] + 10 + linesep / 2,
                  s=str(linesep) + 'dB', va='center', ha='left', fontsize=6, alpha=0.7)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title(f'Azimuth: {azimuths[0]}')
    return im


# ----- plot aachen database

# database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'aachen_database'
# hrtf_list = [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]
# source_idx = hrtf_list[0].get_source_idx(azimuth_range=(0, 50), elevation_range=(-45, 45))
# movie(hrtf_list, source_idx, map='feature_p', kind='image')

# ----- plot kemar
#
# hrtf_list = [slab.HRTF.kemar()]
# movie(hrtf_list, azimuth_range=(0,50), elevation_range=(-20,20), map='average', kind='image')
#

"""

        # plot average
        fig, axis = plt.subplots()
        contour = axis.contourf(frequencies, azimuths, map)
        plt.title(ele)


    fig, ax = plt.subplots()

    ax.set_xlim((0, 50))
    ax.set_ylim((0, 50))

    im = ax.imshow(data[0,:,:])

def init():
    im.set_data(data[0,:,:])
    return (im,)

# animation function. This is called sequentially
def animate(i):
    data_slice = data[i,:,:]
    im.set_data(data_slice)
    return (im,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=20, blit=True)

    HTML(anim.to_html5_video())"""