import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import slab
from pathlib import Path
from dev.hrtf.average import hrtf_average
import dev.hrtf.binary_feature_map as feature_map

def movie(hrtf_list, azimuth_range, elevation_range, interval=500, map='feature_p', kind='image'):
    global data, ax, frequencies, azimuths, elevations, settings
    settings = {'map': map, 'kind': kind}
    # plot features for sources in range 0 / +50 azimuth across elevations
    # if axis == 'azimuth':
    source_idx = hrtf_list[0].get_source_idx(azimuth_range=azimuth_range, elevation_range=elevation_range)
    azimuths = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 0])
    elevations = numpy.unique(hrtf_list[0].sources.vertical_polar[source_idx, 1])
    data = []
    for az in azimuths:
        print(az)
        source_idx = hrtf_list[0].get_source_idx(azimuth_range=az,elevation_range=(elevations.min(), elevations.max()))
        # sort by ascending elevation
        sources = hrtf_list[0].sources.vertical_polar[source_idx]
        sorting_idx = numpy.argsort(sources, axis=0, kind=None, order=None)[:,1]
        source_idx = source_idx[sorting_idx]
        if settings['map'] == 'feature_p':
            map, frequencies = feature_map.feature_p(hrtf_list, source_idx, thresholds=None, bandwidth=(1000,18000))
        elif settings['map'] == 'average':
            frequencies = hrtf_list[0][0].tf(show=False)[0]
            map = hrtf_average(hrtf_list).tfs_from_sources(source_idx, n_bins=None, ear='right')
            map = map.reshape(map.shape[:2])
        data.append(map)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=len(data), interval=interval, blit=False)
    plt.show()

def init():
    # im = ax.contourf(frequencies, elevations, data[0,:,:])
    # im = ax.contourf(frequencies, elevations, data[0])
    im = plot(kind=settings['kind'], data=data[0])
    ax.set_title(f'Azimuth: {azimuths[0]}')
    return (im,)

def animate(i):
    ax.clear()
    # im = ax.contourf(frequencies, elevations, data[i,:,:])
    # im = ax.contourf(frequencies, elevations, data[i])
    im = plot(kind=settings['kind'], data=data[i])
    ax.set_title(f'Azimuth: {azimuths[i]}')
    return (im,)

def plot(kind, data):
    if kind == 'image':
        im = ax.contourf(frequencies, elevations, data)
    elif kind == 'waterfall':
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