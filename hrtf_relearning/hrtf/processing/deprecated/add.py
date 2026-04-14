import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import copy
import numpy
from sklearn.linear_model import LinearRegression
from hrtf_relearning import PATH
hrtf_dir = PATH / 'data' /'hrtf'/'sofa'
import slab

sub_id = 'JP'


def add_spectral_notch(hrtf):
    out = copy.deepcopy(hrtf)

    for filt, source in zip(out, out.sources.vertical_polar):
        azimuth, elevation = float(source[0]), float(source[1])

        ir = numpy.asarray(filt.data)              # (samples, channels) = (256, 2)
        n_samples, n_channels = ir.shape
        fs = filt.samplerate

        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / fs)
        spec = numpy.fft.rfft(ir, axis=0)          # (freq_bins, channels)

        mu = linear_notch_position(azimuth, elevation, X1=(0, 0), X2=(-60, 60), Y=(6e3, 12e3))
        sigma = linear_notch_width(azimuth, elevation, X1=(0, 0), X2=(-60, 60), Y=(300, 300))

        # interpret "scaling factor" as depth in dB (positive)
        depth_db = abs(linear_scaling_factor(
            azimuth, elevation,
            X1=(0, 0), X2=(-60, 60),
            Y=(15.0, 15.0)   # choose what you want
        ))

        notch_db = -depth_db * numpy.exp(-0.5 * ((freqs - mu) / sigma) ** 2)
        notch_lin = 10 ** (notch_db / 20.0)

        # keep endpoints untouched (optional)
        notch_lin[0] = 1.0
        notch_lin[-1] = 1.0

        # limit extreme attenuation (optional safety)
        notch_lin = numpy.clip(notch_lin, 10 ** (-80 / 20.0), 1.0)

        spec *= notch_lin[:, None]
        ir_new = numpy.fft.irfft(spec, n=n_samples, axis=0)

        filt.data = ir_new

    return out


def linear_notch_position(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    mu = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(mu)

def linear_notch_width(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    sigma = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(max(sigma, numpy.finfo(float).eps))

def linear_scaling_factor(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    scaling = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(scaling)

def plot(hrtf, hrtf_modified, kind='image'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    hrtf.plot_tf(hrtf.cone_sources(0), kind=kind, axis=ax[0])
    hrtf_modified.plot_tf(hrtf.cone_sources(0), kind=kind, axis=ax[1])
    ax[0].set_title('original')
    ax[1].set_title('modified')
    plt.show(block=False)
    plt.pause(0.1)  # give Qt time to draw
    return fig

if __name__ == '__main__':
    hrtf = slab.HRTF(hrtf_dir / str(sub_id + '.sofa'))
    hrtf_modified = add_spectral_notch(hrtf)
    # fig = plot(hrtf, hrtf_modified, 'image')
    if input('press enter to save'):
        # fig.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + '_modified.png'))
        hrtf_modified.write_sofa(hrtf_dir / str(sub_id + '_notch.sofa'))