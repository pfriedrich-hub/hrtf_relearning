import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy
import slab
import scipy
from pathlib import Path
from sklearn.linear_model import LinearRegression
from hrtf.analysis.animation import hrtf_animation
from hrtf.processing.tf2ir import tf2ir
from hrtf.processing.add_interaural import *
filename = 'single_notch'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

show=False
write=True

def make_hrtf(n_bins=256):
    n_azimuths = 37
    n_elevations = 25
    azimuths = numpy.linspace(-90, 90, n_azimuths)
    elevations = numpy.linspace(-60, 60, n_elevations)
    n_sources = n_azimuths * n_elevations
    distance = 1
    sources = numpy.array(numpy.meshgrid(azimuths, elevations)).T.reshape(n_sources, 2)
    sources = numpy.column_stack((sources, numpy.ones(n_sources) * distance))

    ext_tf = externalization_tf(hrtf=None, az=0, n_bins=n_bins)
    dtfs = []
    freq_bins = numpy.linspace(20, 20e3, n_bins)
    for az_idx, azimuth in enumerate(azimuths):
        dtfs_at_az = numpy.zeros((len(elevations), n_bins))
        for ele_idx, elevation in enumerate(elevations):
            tf = numpy.ones(n_bins)   # blank dtf

            # peak 1
            # increasing width and scaling from -60 to 50° az
            # mu = linear_notch_position(azimuth, elevation, X1=(0, 0), X2=(-60, 60), Y=(3.5e3, 4.5e3))
            # sigma = linear_notch_width(azimuth, elevation, X1=(-60, 60), X2=(-60, 60), Y=(800, 2000))
            # scaling = linear_scaling_factor(azimuth, elevation, X1=(-60, 60), X2=(0, 0), Y=(sigma * 6, sigma * 8))
            # tf = add_feature(tf, freq_bins, mu, sigma, scaling)

            # # notch 1
            # mu = linear_notch_position(azimuth, elevation, X1=(-20,20), X2=(-60,60), Y=(6e3, 10e3))
            # s = linear_notch_width(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(500, 500))
            # sf = linear_scaling_factor(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(s * -2.2, s * -2.2))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)
            # azimuth-invariant notch
            mu = linear_notch_position(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(6e3, 10e3))
            s = linear_notch_width(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(500, 500))
            sf = linear_scaling_factor(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(s * -2.2, s * -2.2))
            tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

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

            # mu = linear_notch_position(azimuth, elevation, X1=(-20,20), X2=(-60,60), Y=(10e3, 14e3))
            # s = linear_notch_width(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(500, 500))
            # sf = linear_scaling_factor(azimuth, elevation, X1=(0,0), X2=(-60,60), Y=(s * -2.2, s * -2.2))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

            #
            # # peak 3
            # mu = linear_notch_position(azimuth, elevation, x=[(0, 10), (-30, 40)], y=(18.5e3, 14.5e3))
            # s = linear_notch_width(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(800, 400))
            # sf = linear_scaling_factor(azimuth, elevation, x=[(0, 50), (-30, 40)], y=(s * 2.2, s * 1.5))
            # tf = add_feature(tf, freq_bins=freq_bins, mu=mu, sigma=s, scaling=sf)

            tf += numpy.finfo(float).eps  # avoid log10(0) error

            #add ext_tf with ild
            ext_tf = externalization_tf(hrtf=None, az=azimuth, n_bins=n_bins)
            tf += ext_tf  # add externalization tf
            dtfs_at_az[ele_idx, :] = tf
        dtfs.append(dtfs_at_az)

    # left ear
    dtfs_l = numpy.asarray(dtfs).reshape(n_azimuths * n_elevations, n_bins)
    # right ear: mirror DTFs on the sagittal plane, DTFs at 90° az will be equal to -90° for the other ear
    dtfs_r = numpy.asarray(list(reversed(dtfs))).reshape(n_azimuths * n_elevations, n_bins)
    dtfs = numpy.stack((dtfs_l, dtfs_r), axis=2)

    sources[sources[:, 0] < 0, 0] = sources[sources[:, 0] < 0, 0] + 360  # convert sources to sofa convention (0, 360)°
    return slab.HRTF(data=dtfs, samplerate=44.1e3, datatype='TF', sources=sources)

def externalization_tf(hrtf=None, az=0, n_bins=256):
    """
    Get a low-res version of a DTF from 0° az and 0° elevation from a recorded HRTF to externalize a synthetic HRTF
    """
    if not hrtf:
        hrtf = slab.HRTF.kemar()  # load KEMAR as default
    idx_frontal = hrtf.get_source_idx(az, 0)[0]
    ir_data = hrtf.data[idx_frontal].channel(0).data
    # get low-res version of HRTF spectrum
    tf_data = numpy.abs(scipy.signal.freqz(ir_data, worN=12, fs=hrtf.samplerate))[1]
    tf_data[0] = 1  # avoids low-freq attenuation in KEMAR HRTF
    tf_data = scipy.signal.resample(tf_data, n_bins)
    return tf_data

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

# make
hrtf = make_hrtf()

# plots
# hrtf_animation([hrtf], (-180,180), (-60,60), 'left', 100,
#                'average', 'image', filename+'_L', write, show)
# hrtf_animation([hrtf], (-180,180), (-60,60), 'left', 100,
#                'average', 'image', filename+'_R', write, show)

# convert to hrir, add interaural differences and save to sofa
if write:
    hrir = tf2ir(hrtf)
    hrir = add_itd(hrir)
    hrir.write_sofa(sofa_path / str(filename+'.sofa'))