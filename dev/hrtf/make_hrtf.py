""" make a hrtf from gaussians """
import matplotlib.pyplot as plt
import numpy
import scipy
import slab

def add_feature(tf, freq_bins, mu, sigma, scaling):
    """
    add a scaled inverse gaussian 'notch' to a dtf
    """
    notch = (1 / (sigma * numpy.sqrt(2 * numpy.pi))) * (numpy.exp(-0.5 * ((freq_bins - mu) / sigma) ** 2)) * scaling
    notch_idx = numpy.where(numpy.logical_and(freq_bins > int(mu - sigma * 4), freq_bins < int(mu + sigma * 4)))
    tf[notch_idx] += notch[notch_idx]
    return tf

def linear_notch_position(elevation, x, y):
    """
    linearly moving notch from x1 kHz at y1° Elevation to x2 kHz at y2° Elevation
    """
    if y[0] <= elevation <= y[1]:
        if x[0] == x[1]:
            mu = x[0]
        else:
            m, n = scipy.stats.linregress(x, y)[:2]
            mu = (elevation - n) / m
    else:
        mu = 0
    return mu

def linear_notch_width(elevation, x, y):
    if y[0] <= elevation <= y[1]:
        if x[0] == x[1]:
            sigma = x[0]
        else:
            m, n = scipy.stats.linregress(x, y)[:2]
            sigma = (elevation - n) / m
    else:
        sigma = 0
    return sigma

def linear_scaling_factor(elevation, x, y):
    if y[0] <= elevation <= y[1]:
        if x[0] == x[1]:
            sf = x[0]
        else:
            m, n = scipy.stats.linregress(x, y)[:2]
            sf = (elevation - n) / m
    else:
        sf = 0
    return sf

def make_hrtf(n_elevations=10, elevation_range=(-40, 40), n_bins=256):
    elevations = numpy.linspace(elevation_range[1], elevation_range[0], n_elevations)
    azimuths = numpy.zeros(n_elevations)
    distances = numpy.ones(n_elevations) * 1.4
    sources = numpy.column_stack((azimuths, elevations, distances))
    dtfs = numpy.zeros((n_elevations, n_bins, 2))
    freq_bins = numpy.linspace(20, 20e3, n_bins)
    for ele_idx, elevation in enumerate(elevations):
        tf = numpy.ones(n_bins)   # blank dtf

        # peak 1
        ele_range = (-40, 40)
        mu = linear_notch_position(elevation, x=(3e3, 5e3), y=ele_range)  # x=freqs, y=elevations
        s = linear_notch_width(elevation, x=(500, 1200), y=ele_range)  # sd (filter bins)
        sf = linear_scaling_factor(elevation, x=(s * 2, s * 3), y=ele_range)
        tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian notch

        # notch 1
        ele_range = (-40, 40)
        mu = linear_notch_position(elevation, x=(6e3, 11e3), y=ele_range)  # x=freqs, y=elevations
        s = linear_notch_width(elevation, x=(500, 1000), y=ele_range)  # sd (filter bins)
        sf = linear_scaling_factor(elevation, x=(s * -1.7, s * -2.2), y=ele_range)
        tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian notch

        # peak 2
        ele_range = (-40, 5)
        mu = linear_notch_position(elevation, x=(10e3, 12e3), y=ele_range)  # x=freqs, y=elevations
        s = linear_notch_width(elevation, x=(1000, 300), y=ele_range)  # sd (filter bins)
        sf = linear_scaling_factor(elevation, x=(s * 2.5, s * 1.5), y=ele_range)
        tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian notch

        # notch 2        # todo use nonlinear width here
        ele_range = (-40, 25)
        mu = linear_notch_position(elevation, x=(15e3, 13e3), y=ele_range)  # x=freqs, y=elevations
        s = linear_notch_width(elevation, x=(500, 500), y=ele_range)  # sd (filter bins)
        sf = linear_scaling_factor(elevation, x=(s * -2, s * -1.5), y=ele_range)
        tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian notch

        # peak 3
        ele_range = (-30, 40)
        mu = linear_notch_position(elevation, x=(19e3, 15e3), y=ele_range)  # x=freq range, y=elevation range
        s = linear_notch_width(elevation, x=(300, 500), y=ele_range)  # sd (filter bins) x=width range
        sf = linear_scaling_factor(elevation, x=(s * 1.5, s * 2.2), y=ele_range)  # x=scale range
        tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian notch

        tf += numpy.finfo(float).eps  # avoid log10(0) error
        dtfs[ele_idx, :, 0] = tf
        dtfs[ele_idx, :, 1] = tf
    return slab.HRTF(data=dtfs, samplerate=40e3, datatype='TF', sources=sources)

hrtf = make_hrtf(n_elevations=10, elevation_range=(-40, 40), n_bins=256)
hrtf.plot_tf(hrtf.cone_sources(0))



