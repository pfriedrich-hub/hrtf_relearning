import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from hrtf.processing.average import hrtf_average
from hrtf.analysis.feature_p import feature_p


def hrtf_animation(hrtf, azimuth_range=(-180,180), elevation_range=(-60,60), ear='left', interval=100,
                   map='feature_p', kind='image', filename=None, write=None, show=True, figsize=(5,5)):
    global data, fig, ax, frequencies, azimuths, elevations, settings
    settings = {'map': map, 'kind': kind}
    # plots features for sources in range 0 / +50 azimuth across elevations
    # if axis == 'azimuth':
    if type(hrtf).__name__ == 'HRTF':  # convert to list if a single hrtf is given
        hrtf_list = [hrtf]
    elif type(hrtf ) == list:
        hrtf_list = hrtf
    else:
        raise ValueError('hrtf must be a HRTF object or a list of HRTF objects')
    if not len(hrtf_list) == 0:
        source_idx = hrtf_list[0].get_source_idx(azimuth=azimuth_range, elevation=elevation_range)
        sources = hrtf_list[0].sources.vertical_polar
    else:
        raise ValueError('hrtf list empty')
    azimuths = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 0])
    elevations = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 1])
    bandwidth = (1000, 18000)
    data = []
    for i, az in enumerate(azimuths):
        print(f'Azimuth: {az}')
        # _src_idx = numpy.where(sources[source_idx, 0] == az)
        source_idx = hrtf_list[0].get_source_idx(azimuth=az,elevation=(elevations.min(), elevations.max()), tolerance=.03)
        # sort by ascending elevation
        sources = hrtf_list[0].sources.vertical_polar[source_idx]
        sorting_idx = numpy.argsort(sources, axis=0, kind=None, order=None)[:,1]
        source_idx = numpy.array(source_idx)[sorting_idx]
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

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(filename)
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
    if write:
        writervideo = animation.FFMpegWriter(fps= int(1000/interval))
        ani.save(Path.cwd() / 'data' / 'animations' / str(filename + '.mp4'), writer=writervideo)
    if show:
        plt.show()
    else:
        plt.close()

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
        ticks = vlines[::2]  # plots every second elevation
        labels = elevations
        # plots every third elevation label, omit comma to save space
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


# ----- plots aachen database
# import slab
# database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'aachen_database'
# hrtf_list = [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]
# movie(hrtf_list, azimuth_range=(-180, 180), elevation_range=(-60,60), ear='left', interval=150, map='feature_p',
#       kind='image', save=Path.cwd() / 'data' / 'animations' / 'aachen_full.mp4')

# ----- plots single hrtf
# import slab
# database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'aachen_database'
# hrtf_list = [slab.HRTF(list(database_path.glob('*.sofa'))[0])]
# movie(hrtf_list, azimuth_range=(-180, 180), elevation_range=(-60,60), ear='left', interval=150, map='average',
#       kind='image', save=Path.cwd() / 'data' / 'animations' / 'aachen_01.mp4')

# ----- plots kemar
#
# hrtf_list = [slab.HRTF.kemar()]
# movie(hrtf_list, azimuth_range=(0,50), elevation_range=(-20,20), map='average', kind='image')
#

"""

        # plots average
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