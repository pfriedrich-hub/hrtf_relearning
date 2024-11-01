import matplotlib.pyplot as plt
import numpy
import slab
from pathlib import Path
from sklearn.linear_model import LinearRegression
from dev.hrtf.movie import movie

def add_feature(tf, freq_bins, mu, sigma, scaling):
    """
    add a scaled inverse gaussian 'feature' to a dtf
    """
    notch = (1 / (sigma * numpy.sqrt(2 * numpy.pi))) * (numpy.exp(-0.5 * ((freq_bins - mu) / sigma) ** 2)) * scaling
    notch_idx = numpy.where(numpy.logical_and(freq_bins > int(mu - sigma * 4), freq_bins < int(mu + sigma * 4)))
    tf[notch_idx] += notch[notch_idx]
    return tf

def linear_notch_position(azimuth, elevation, X1, X2, Y):
    """
    linear notch from Y[0] kHz at X1[0]° Azimuth and X2[0]° Elevation
     to Y[1] kHz at X1[1]° Azimuth and X2[1]° Elevation
    """
    # linear regression
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    data = numpy.array((azimuth, elevation))
    mu = model.predict(data.reshape(1, -1))
    return float(mu)

def linear_notch_width(azimuth, elevation, X1, X2, Y):
    """
    linear notch width from sigma = Y[0] at X1[0]° Azimuth and X2[0]° Elevation
     to sigma = Y[1] at X1[1]° Azimuth and X2[1]° Elevation
    """
    # linear regression
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    data = numpy.array((azimuth, elevation))
    sigma = model.predict(data.reshape(1, -1))
    if sigma <= 0:  # avoid negative notch width
        sigma = numpy.finfo(float).eps
    return float(sigma)

def linear_scaling_factor(azimuth, elevation, X1, X2, Y):
    """
    linear scaling factor Y[0] at X1[0]° Azimuth and X2[0]° Elevation
     to Y[1] at X1[1]° Azimuth and X2[1]° Elevation
    """
    # linear regression
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    data = numpy.array((azimuth, elevation))
    scaling = model.predict(data.reshape(1, -1))
    return float(scaling)

def make_hrtf(n_bins=256):
    azimuths = numpy.linspace(-90, 90, 72)
    elevations = numpy.linspace(-60, 60, 25)
    n_sources = len(azimuths) * len(elevations)
    distance = 1
    sources = numpy.zeros((n_sources, 3))
    # sources = numpy.array(numpy.meshgrid(azimuths, elevations)).T.reshape(n_sources, 2)
    # sources = numpy.column_stack((sources, distances))

    dtfs = numpy.zeros((n_sources, n_bins, 2))
    freq_bins = numpy.linspace(20, 20e3, n_bins)
    source_idx = 0
    for az_idx, azimuth in enumerate(azimuths):
        for ele_idx, elevation in enumerate(elevations):
            tf = numpy.ones(n_bins)   # blank dtf

            # ---- left ear ---- #

            # peak 1
            # increasing width and scaling from -60 to 50° az
            mu = linear_notch_position(azimuth, elevation, X1=(0, 0), X2=(-60, 60), Y=(3.5e3, 4.5e3))
            sigma = linear_notch_width(azimuth, elevation, X1=(-60, 60), X2=(-60, 60), Y=(800, 2000))
            scaling = linear_scaling_factor(azimuth, elevation, X1=(-60, 60), X2=(0, 0), Y=(sigma * 6, sigma * 8))
            tf = add_feature(tf, freq_bins, mu, sigma, scaling)

            # # notch 1
            # mu = linear_notch_position(azimuth, elevation, x=[(-10,10), (-40,40)], y=(6e3, 10e3))
            # s = linear_notch_width(azimuth, elevation, x=[(0,50), (-40,40)], y=(1200, 500))
            # sf = linear_scaling_factor(azimuth, elevation, x=[(0,50), (-40,40)], y=(s * -1.8, s * -2.4))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)
            #
            # # peak 2
            # mu = linear_notch_position(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(10e3, 12e3))
            # s = linear_notch_width(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(1000, 200))
            # sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-40, 10)], y=(s * 6, s * .5))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)
            #
            # # notch 2        # todo use nonlinear width here
            # mu = linear_notch_position(azimuth, elevation, x=[(0, 10), (-40, 25)], y=(15e3, 13e3))
            # s = linear_notch_width(azimuth, elevation, x=[(0, 50), (40, -40)], y=(800, 300))
            # sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-40, 25)], y=(s * -2, s * -2.3))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)  # add gaussian
            #
            # # peak 3
            # mu = linear_notch_position(azimuth, elevation, x=[(0, 10), (-30, 40)], y=(18.5e3, 14.5e3))
            # s = linear_notch_width(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(800, 400))
            # sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(s * 2.2, s * 1.5))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)


            tf += numpy.finfo(float).eps  # avoid log10(0) error
            dtfs[source_idx, :, 0] = tf

            # --- right ear
            dtfs[source_idx, :, 1] = tf

            # --- source coords
            sources[source_idx] = numpy.array((azimuth, elevation, distance))
            source_idx += 1

    return slab.HRTF(data=dtfs, samplerate=44e3, datatype='TF', sources=sources)

hrtf = make_hrtf()
movie([hrtf], (-180,180), (-60,60), ear='left', interval=100, map='average', kind='image', save='hrtf_1.1')