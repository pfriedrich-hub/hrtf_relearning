import numpy
import slab
from sklearn import linear_model
reg = linear_model.LinearRegression()


def add_feature(tf, freq_bins, mu, sigma, scaling):
    """
    add a scaled inverse gaussian 'feature' to a dtf
    """
    notch = (1 / (sigma * numpy.sqrt(2 * numpy.pi))) * (numpy.exp(-0.5 * ((freq_bins - mu) / sigma) ** 2)) * scaling
    notch_idx = numpy.where(numpy.logical_and(freq_bins > int(mu - sigma * 4), freq_bins < int(mu + sigma * 4)))
    tf[notch_idx] += notch[notch_idx]
    return tf

def linear_notch_position(azimuth, elevation, x, y):
    """
    linear notch from y1 kHz at x[0, 0]° Azimuth and x[0, 1]° Elevation
     to y2 kHz at x[1, 0]° Azimuth and x[1, 1]° Elevation
    """
    x = numpy.asarray(x).T
    y = numpy.asarray(y).reshape(-1, 1)
    if x[0, 1] <= elevation <= x[1, 1] or x[0, 0] <= azimuth <= x[1, 0]:
        if y[0] == y[1]:
            mu = y[0]
        else:
            reg.fit(x, y)
            mu = reg.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0, 0]
    else:
        mu = 0 + numpy.finfo(float).eps
    return mu


def linear_notch_width(azimuth, elevation, x, y):
    """
    linear notch with y1 width at x[0, 0]° Azimuth and x[0, 1]° Elevation
     to y2 width at x[1, 0]° Azimuth and x[1, 1]° Elevation.
     x (list of tuples):
     y (list, tuple): Width, expressed in standard deviations
    """
    x = numpy.asarray(x).T
    y = numpy.asarray(y).reshape(-1, 1)
    if x[0, 1] <= elevation <= x[1, 1] or x[0, 0] <= azimuth <= x[1, 0]:
        if y[0] == y[1]:
            sigma = y[0]
        else:
            reg.fit(x, y)
            sigma = reg.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0, 0]
    else:
        sigma = 0 + numpy.finfo(float).eps
    return sigma

def linear_scaling_factor(azimuth, elevation, x, y):
    """
    linear notch from y1 kHz at x[0, 0]° Azimuth and x[0, 1]° Elevation
     to y2 kHz at x[1, 0]° Azimuth and x[1, 1]° Elevation
    """
    x = numpy.asarray(x).T
    y = numpy.asarray(y).reshape(-1, 1)
    if x[0, 1] <= elevation <= x[1, 1] and x[0, 0] <= azimuth <= x[1, 0]:
        if y[0] == y[1]:
            scaling = y[0]
        else:
            reg.fit(x, y)
            scaling = reg.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0, 0]
    else:
        scaling = 0 + numpy.finfo(float).eps  # avoid log10(0) error
    return scaling

def make_hrtf(n_azimuths, azimuth_range, n_elevations, elevation_range, n_bins=256):
    azimuths = numpy.linspace(azimuth_range[1], azimuth_range[0], n_azimuths)
    elevations = numpy.linspace(elevation_range[1], elevation_range[0], n_elevations)
    distances = numpy.ones(n_azimuths * n_elevations) * 1.4
    sources = numpy.array(numpy.meshgrid(azimuths, elevations)).T.reshape(n_azimuths*n_elevations, 2)
    sources = numpy.column_stack((sources, distances))

    dtfs = numpy.zeros((len(sources), n_bins, 2))
    freq_bins = numpy.linspace(20, 20e3, n_bins)
    source_idx = 0
    for az_idx, azimuth in enumerate(azimuths):
        for ele_idx, elevation in enumerate(elevations):
            tf = numpy.ones(n_bins)   # blank dtf

            # peak 1
            mu = linear_notch_position(azimuth, elevation, x=[(0,50), (-40,40)], y=(3.5e3, 4.5e3))  # y=mu freqs, x=coordinates
            s = linear_notch_width(azimuth, elevation, x=[(0,50), (-20,20)], y=(800, 1800))
            sf = linear_scaling_factor(azimuth, elevation, x=[(0,50), (-40,40)], y=(s * 6, s * 8))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

            # notch 1
            mu = linear_notch_position(azimuth, elevation, x=[(-10,10), (-40,40)], y=(6e3, 10e3))
            s = linear_notch_width(azimuth, elevation, x=[(0,50), (-40,40)], y=(1200, 500))
            sf = linear_scaling_factor(azimuth, elevation, x=[(0,50), (-40,40)], y=(s * -1.8, s * -2.4))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

            # peak 2
            mu = linear_notch_position(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(10e3, 12e3))
            s = linear_notch_width(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(1000, 200))
            sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(s * 6, s * .5))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

            # notch 2        # todo use nonlinear width here
            mu = linear_notch_position(azimuth, elevation, x=[(0, 10), (-40, 25)], y=(15e3, 13e3))
            s = linear_notch_width(azimuth, elevation, x=[(0, 50), (40, -40)], y=(800, 300))
            sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-40, 25)], y=(s * -2, s * -2.3))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian

            # peak 3
            mu = linear_notch_position(azimuth, elevation, x=[(0, 10), (-30, 40)], y=(18.5e3, 14.5e3))
            s = linear_notch_width(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(800, 400))
            sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(s * 2.2, s * 1.5))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)


            tf += numpy.finfo(float).eps  # avoid log10(0) error

            dtfs[source_idx, :, 0] = tf
            dtfs[source_idx, :, 1] = tf
            source_idx += 1
    return slab.HRTF(data=dtfs, samplerate=44.1e3, datatype='TF', sources=sources)
#
# hrtf = make_hrtf(n_azimuths=44, azimuth_range=(-90, 90), n_elevations=8, elevation_range=(-40, 40), n_bins=256)
# movie([hrtf], (0,50), (-40,40), interval=25, map='average', kind='waterfall', save='hrtf_1')

# hrtf.plot_sources(hrtf.cone_sources(0))  # todo fix
# hrtf.plot_tf(hrtf.cone_sources(0))  # todo check if notch moves correctly from 0 to 50 az
# todo mirror hrtf for sources in the back and see if glitches disappear
