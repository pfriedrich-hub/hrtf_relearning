"""Quick listening test for a recorded / modified HRTF — "fly-over".

No head tracker, no game: a virtual source is moved along the measured source
grid while a continuous stimulus plays, so you can hear what an HRTF does.
Head position is assumed fixed and facing straight ahead (0, 0).

Run cell by cell (# %%): config -> setup -> one of the fly-over cells (repeat
as often as you like) -> stop.

Coordinates are given in *world* convention (like training/localization
targets): azimuth positive = right, elevation positive = up. They are mapped
to the SOFA convention (az 0..360, counter-clockwise) and snapped to the
nearest measured source before being sent to pyBinSim.
"""
import logging
import time

import matplotlib
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
import matplotlib.pyplot as plt
import numpy
import slab
from pythonosc import udp_client

from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.hrtf.binsim.stream import start_stream
from hrtf_relearning.utils import paths

logging.getLogger().setLevel('INFO')

# ------------------------------ CONFIG ------------------------------

SUBJECT_ID = 'GLK'
HRIR_NAME = 'GLK'          # SOFA in data/hrtf/sofa/<SUBJECT_ID>/, e.g. 'GLK_shift'
EAR = None                 # None = binaural, 'left'/'right' = monaural
HP = 'DT990'

GAIN = .2                  # /pyBinSimLoudness
DWELL = .06                # s per measured direction (sets fly-over speed)
STIM = 'noise'             # 'noise' (continuous pink noise) or 'pulses'
PLOT = 'TF'                # 'TF', 'IR' or None — live plot of the active filter

hrir_settings = dict(
    name=HRIR_NAME,
    subject_id=SUBJECT_ID,
    ear=EAR,
    other_ear='flat',
    reverb=True,
    drr=20,
    hp_filter=True,
    hp=HP,
    convolution='cpu',     # 'cuda' on the rig — torch cuda is not available on macOS
    storage='cpu',
)

# --------------------------- HELPERS --------------------------------

def _world_az(sofa_az):
    """SOFA az (0..360, ccw) -> signed world az (negative left, positive right)."""
    return ((-numpy.asarray(sofa_az) + 180) % 360) - 180


def _nearest(values, value):
    values = numpy.unique(values)
    return values[numpy.argmin(numpy.abs(values - value))]


def horizontal_path(el=0., az_range=(-50, 50)):
    """Indices of measured sources on one elevation ring, left -> right."""
    sources = hrir.sources.vertical_polar
    el = _nearest(sources[:, 1], el)
    idx = numpy.where(sources[:, 1] == el)[0]
    az = _world_az(sources[idx, 0])
    idx = idx[(az >= az_range[0]) & (az <= az_range[1])]
    return idx[numpy.argsort(_world_az(sources[idx, 0]))]


def vertical_path(az=0., el_range=(-37.5, 37.5)):
    """Indices of measured sources on one azimuth column, bottom -> top."""
    sources = hrir.sources.vertical_polar
    az = _nearest(_world_az(sources[:, 0]), az)
    idx = numpy.where(numpy.isclose(_world_az(sources[:, 0]), az))[0]
    el = sources[idx, 1]
    idx = idx[(el >= el_range[0]) & (el <= el_range[1])]
    return idx[numpy.argsort(sources[idx, 1])]


def raster_path(az_range=(-50, 50), el_range=(-37.5, 37.5)):
    """Boustrophedon sweep over the whole frontal grid, bottom row first."""
    sources = hrir.sources.vertical_polar
    els = numpy.unique(sources[:, 1])
    els = els[(els >= el_range[0]) & (els <= el_range[1])]
    path = []
    for i, el in enumerate(els):
        row = horizontal_path(el, az_range)
        path.append(row if i % 2 == 0 else row[::-1])
    return numpy.concatenate(path)


def fly(path, dwell=None, loops=1, pingpong=False, stim=None, gain=None, plot=None):
    """Move the virtual source along `path` (source indices) and play.

    dwell : seconds spent at each measured direction.
    loops : number of repetitions of the whole path.
    pingpong : append the reversed path, so the source flies back.
    """
    dwell = DWELL if dwell is None else dwell
    stim = STIM if stim is None else stim
    gain = GAIN if gain is None else gain
    plot = PLOT if plot is None else plot

    path = numpy.asarray(path)
    if pingpong:
        path = numpy.concatenate([path, path[::-1]])
    path = numpy.tile(path, loops)
    if not len(path):
        raise ValueError('empty path — check the az/el ranges against the measured grid')

    sources = hrir.sources.vertical_polar
    duration = len(path) * dwell + .5
    soundfile = _write_stim(duration, stim)

    fig = ax = None
    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.ion()
        plt.show(block=False)

    _set_filter(sources[path[0]])
    osc_play.send_message('/pyBinSimLoudness', gain)
    osc_play.send_message('/pyBinSimFile', str(soundfile))
    try:
        for idx in path:
            t0 = time.time()
            _set_filter(sources[idx])
            logging.debug('az %.1f el %.1f', _world_az(sources[idx, 0]), sources[idx, 1])
            if plot:
                _draw(ax, idx, plot)
                plt.pause(max(.001, dwell - (time.time() - t0)))
            else:
                time.sleep(max(0, dwell - (time.time() - t0)))
    except KeyboardInterrupt:
        logging.info('interrupted')
    finally:
        osc_play.send_message('/pyBinSimLoudness', 0)
    return fig


def _set_filter(source):
    """Send one measured direction to the pyBinSim ds convolver."""
    osc_filter.send_message('/pyBinSim_ds_Filter',
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             float(source[0]), float(source[1]), 0,
                             0, 0, 0])


def _write_stim(duration, stim='noise', level=80):
    sound_dir = paths.BINSIM_DIR / hrir.name / 'sounds'
    if stim == 'noise':
        sound = slab.Sound.pinknoise(duration, level=level)
    elif stim == 'pulses':
        pulse = slab.Sound.pinknoise(.1, level=level).ramp(duration=.01)
        gap = slab.Sound.silence(.05)
        sound = slab.Sound.sequence(*[s for _ in range(int(numpy.ceil(duration / .15)))
                                      for s in (pulse, gap)])
    else:
        raise ValueError("stim must be 'noise' or 'pulses'")
    sound = sound.ramp(duration=.05)
    sound.write(sound_dir / 'flyover.wav')
    return sound_dir / 'flyover.wav'


def _draw(ax, idx, kind='TF'):
    sources = hrir.sources.vertical_polar
    ax.cla()
    if kind == 'TF':
        hrir[idx].channel(0).tf(show=True, axis=ax)
        hrir[idx].channel(1).tf(show=True, axis=ax)
        ax.lines[0].set_label('left')
        ax.lines[1].set_label('right')
    else:
        times = numpy.linspace(0, hrir[idx].n_samples / hrir.samplerate, hrir[idx].n_samples)
        ax.plot(times, hrir[idx].data[:, 0], label='left')
        ax.plot(times, hrir[idx].data[:, 1], label='right')
    ax.legend(loc='upper right')
    ax.set_title(f'{hrir.name} | az {_world_az(sources[idx, 0]):.1f}°  '
                 f'el {sources[idx, 1]:.1f}°')
    ax.grid(True, which='both', linestyle=':', linewidth=.6)


# %% ---------------------------- SETUP -------------------------------
# build the pyBinSim database and start the audio stream (run once)

hrir = hrtf2binsim(hrir_settings, overwrite=True)
slab.set_default_samplerate(hrir.samplerate)

osc_filter = udp_client.SimpleUDPClient('127.0.0.1', 10000)   # /pyBinSim_ds_Filter
osc_play = udp_client.SimpleUDPClient('127.0.0.1', 10003)     # loudness / soundfile

# pyBinSim runs in its own interpreter (not multiprocessing): under 'spawn' a
# worker would re-execute this script and start a second stream.
binsim_proc = start_stream(hrir.name, 'test')
osc_play.send_message('/pyBinSimLoudness', 0)

# %% ------------------------ HORIZONTAL ------------------------------
# left -> right and back, at ear level

fly(horizontal_path(el=0), pingpong=True)

# %% ------------------------- VERTICAL -------------------------------
# bottom -> top in the median plane (the elevation cue to listen for)

fly(vertical_path(az=0), dwell=.15, pingpong=True)

# %% -------------------------- RASTER --------------------------------
# whole frontal field, row by row

fly(raster_path())

# %% ------------------------- SINGLE SPOT ----------------------------
# park the source at one direction and keep it there

fly(horizontal_path(el=0, az_range=(0, 0)), dwell=3, stim='pulses')

# %% --------------------------- STOP ---------------------------------

osc_play.send_message('/pyBinSimLoudness', 0)
binsim_proc.terminate()
binsim_proc.wait()
plt.close('all')
