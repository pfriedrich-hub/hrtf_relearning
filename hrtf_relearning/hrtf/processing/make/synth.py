import matplotlib as mpl
mpl.use('Qt5Agg')
from pathlib import Path
from sklearn.linear_model import LinearRegression
from hrtf_relearning.hrtf.processing.tf2ir import *
from hrtf_relearning.hrtf.processing.make.add_interaural import *
filename = 'single_notch'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

show=False
write=True

def make_hrtf(n_bins=128):
    n_azimuths = 37
    n_elevations = 25
    azimuths = numpy.linspace(-90, 90, n_azimuths)
    elevations = numpy.linspace(-60, 60, n_elevations)
    n_sources = n_azimuths * n_elevations
    distance = 1
    sources = numpy.array(numpy.meshgrid(azimuths, elevations)).T.reshape(n_sources, 2)
    sources = numpy.column_stack((sources, numpy.ones(n_sources) * distance))
    kemar = slab.HRTF.kemar()  # get kemar for ils
    dtfs = []
    freq_bins = numpy.linspace(20, 20e3, n_bins)
    for az_idx, azimuth in enumerate(azimuths):
        dtfs_at_az = numpy.zeros((len(elevations), n_bins))
        for ele_idx, elevation in enumerate(elevations):
            tf = slab.Filter(numpy.ones(shape=(1, n_bins)), samplerate=44.1e3, fir='TF')
            # tf = add_ils(tf, azimuth, template_hrtf=kemar)

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

            # tf.data += numpy.finfo(float).eps  # avoid log10(0) error
            dtfs_at_az[ele_idx, :] = tf.data.flatten()
        dtfs.append(dtfs_at_az)
    # left ear
    dtfs_l = numpy.asarray(dtfs).reshape(n_azimuths * n_elevations, n_bins)
    # right ear: mirror DTFs on the sagittal plane, DTFs at 90° az will be equal to -90° for the other ear
    dtfs_r = numpy.asarray(list(reversed(dtfs))).reshape(n_azimuths * n_elevations, n_bins)
    dtfs = numpy.stack((dtfs_l, dtfs_r), axis=2)

    sources[sources[:, 0] < 0, 0] = sources[sources[:, 0] < 0, 0] + 360  # convert sources to sofa convention (0, 360)°
    return slab.HRTF(data=dtfs, samplerate=44.1e3, datatype='TF', sources=sources)

def add_feature(tf, freq_bins, mu, sigma, scaling):
    """
    add a scaled inverse gaussian 'feature' to a dtf
    """
    notch = (1 / (sigma * numpy.sqrt(2 * numpy.pi))) * (numpy.exp(-0.5 * ((freq_bins - mu) / sigma) ** 2)) * scaling
    notch_idx = numpy.where(numpy.logical_and(freq_bins > int(mu - sigma * 4), freq_bins < int(mu + sigma * 4)))
    tf.data[notch_idx, 0] += notch[notch_idx]
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
    mu = model.predict(data.reshape(1, -1))[0]
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
    sigma = model.predict(data.reshape(1, -1))[0]
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
    scaling = model.predict(data.reshape(1, -1))[0]
    return float(scaling)

# # make
hrtf = make_hrtf()
hrir = hrtf2hrir(hrtf)
hrir = add_itd(hrir)
hrir = add_ild(hrir)


# # plots
# hrtf_animation(hrtf=[hrtf], azimuth_range=(-180,180), elevation_range=(-60,60), ear='left', interval=100,
#                map='average', kind='waterfall', filename=filename+'_L', write=write, show=show, figsize=(7,5))
# hrtf_animation([hrtf], (-180,180), (-60,60), 'right', 100,
#                'average', 'waterfall', filename+'R', write, show, figsize=(7,5))

# convert to hrir, add interaural differences and save to sofa
if write:
    hrir.write_sofa(sofa_path / str(filename+'.sofa'))