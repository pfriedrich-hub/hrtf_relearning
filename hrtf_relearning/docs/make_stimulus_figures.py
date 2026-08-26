"""
make_stimulus_figures.py

Regenerates the localization-stimulus figures in stimulus_synthesis.md.
Everything is drawn from the real synthesis code -- nothing here reimplements
it, so the figures cannot drift away from what the experiment plays.

    python -m hrtf_relearning.docs.make_stimulus_figures

Writes stimulus_synthesis_{1,2,3}.png next to this file. Set SUBJECT / EAR to
draw the DTF reference from a different listener.

Panels
------
1  synthesis      the burst train, example envelopes, and where the envelope
                  puts its energy in ripple density against the cue
2  across trials  what actually varies from trial to trial, per condition
3  consequence    why below-band variation is separable and in-band is not
"""
from pathlib import Path

import numpy
import slab
from matplotlib import pyplot as plt
from scipy.fftpack import dct, idct

from hrtf_relearning.experiment.localization.localization_helpers import spectral_metrics as sm
from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise, shape_from_coefficients,
    SHAPE_FLO, SHAPE_FHI, SHAPE_N, RIPPLE_TILT_MAX, RIPPLE_CUE_MAX, DURATION)

SUBJECT, EAR = 'AS', 'left'
SAMPLERATE = 48828
RMS_TILT, RMS_CUE = 3.0, 2.0       # the two ripple conditions of the protocol
N_TOKENS = 200                     # for the across-trial statistics
N_SHOWN = 25                       # tokens drawn as individual lines
OUT_DIR = Path(__file__).parent

# categorical slots 1-3, validated all-pairs for colour-vision deficiency;
# the DTF cue is deliberately NOT a categorical slot -- it is the reference the
# stimulus is measured against, so it stays ink-coloured
NOISE, BELOW, IN_BAND = '#2a78d6', '#eb6834', '#1baf7a'
CUE, INK, MUTED = '#0b0b0b', '#52514e', '#b8b7b2'
CONDITIONS = (('noise', NOISE, {}),
              ('below-band', BELOW, {'rms_tilt': RMS_TILT, 'rms_cue': 0.0}),
              ('in-band', IN_BAND, {'rms_tilt': RMS_TILT, 'rms_cue': RMS_CUE}))

SHAPE_F, SHAPE_DENSITY = sm.log_axis(SHAPE_FLO, SHAPE_FHI, SHAPE_N)


def _style(axis, xlabel=None, ylabel=None, title=None, logx=True):
    """Recessive frame: no top/right spine, thin grid under the data."""
    if logx:
        axis.set_xscale('log')
        axis.set_xticks([500, 1000, 2000, 4000, 8000, 16000])
        axis.set_xticklabels(['0.5', '1', '2', '4', '8', '16'])
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK, length=3)
    axis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
    axis.set_axisbelow(True)
    if xlabel:
        axis.set_xlabel(xlabel, color=INK)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK)
    if title:
        axis.set_title(title, color=INK, fontsize=10, loc='left')


def draw_tokens(n=N_TOKENS, **stim_kwargs):
    """n tokens of one condition -> (envelopes dB, coefficient arrays, spectra dB).

    envelopes is empty for the noise condition, which has no imposed envelope.
    """
    envelopes, coeffs, spectra = [], [], []
    for _ in range(n):
        if stim_kwargs:
            sound, params = make_rippled_pinknoise(**stim_kwargs)
            coefficients = numpy.zeros(SHAPE_N)
            coefficients[:len(params['coeffs'])] = params['coeffs']
            coeffs.append(coefficients)
            envelopes.append(shape_from_coefficients(coefficients)[1])
        else:
            sound = make_gapped_pinknoise()
        spectra.append(sm.log_spectrum(sound.data[:, 0], sound.samplerate, SHAPE_F))
    return (numpy.array(envelopes), numpy.array(coeffs), numpy.array(spectra))


def dtf_on_shape_axis(subject=SUBJECT, ear=EAR):
    """DTF log spectra across midline elevation, on the stimulus's own axis.

    The envelope is defined on SHAPE_FLO..SHAPE_FHI, so anything overlaying the
    two has to put the DTF on that axis rather than the 3.5-16 kHz measurement
    axis -- ripple density is defined per axis.
    """
    hrtf = sm.load_hrtf(subject, subject)
    channel = 0 if ear == 'left' else 1
    idx = sm.column_indices(hrtf, azimuth=0.0)
    spectra = numpy.array([sm.log_spectrum(hrtf[i].data[:, channel],
                                           hrtf.samplerate, SHAPE_F) for i in idx])
    return spectra, hrtf.sources.vertical_polar[idx, 1]


def cue_band_shading(axis):
    axis.axvspan(RIPPLE_TILT_MAX, RIPPLE_CUE_MAX, color=CUE, alpha=0.06, lw=0)
    axis.axvline(RIPPLE_TILT_MAX, color=MUTED, lw=0.8, ls='--')
    axis.axvline(RIPPLE_CUE_MAX, color=MUTED, lw=0.8, ls='--')


# --------------------------------------------------------------------------
def figure_1(data, dtf):
    """The burst train, two sets of example envelopes, and the density split."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (below_env, below_c, _), (in_env, in_c, _) = data['below-band'], data['in-band']

    # (a) one token in the time domain -- ink, not a condition colour: the burst
    # train is identical in every condition, only its spectrum differs
    axis = axes[0, 0]
    sound, _ = make_rippled_pinknoise(rms_tilt=RMS_TILT, rms_cue=RMS_CUE)
    time = numpy.arange(sound.n_samples) / sound.samplerate * 1000
    axis.plot(time, sound.data[:, 0] / numpy.abs(sound.data).max(),
              color=INK, lw=0.4)
    for start in (25, 75, 125, 175):
        axis.axvspan(start, start + 25, color=MUTED, alpha=0.28, lw=0)
    axis.set_xlim(0, 1000 * DURATION)
    axis.set_ylim(-1.15, 1.35)
    axis.text(4, 1.14, '5 bursts of 25 ms', color=INK, fontsize=8)
    axis.text(132, 1.14, '4 gaps of 25 ms (shaded)', color=INK, fontsize=8)
    _style(axis, 'time (ms)', 'amplitude (normalised)',
           'a  one trial: 225 ms gapped pinknoise', logx=False)

    # (b, c) example envelopes, same y-scale so the two conditions compare
    limit = 1.05 * max(numpy.abs(below_env[:N_SHOWN]).max(),
                       numpy.abs(in_env[:N_SHOWN]).max())
    for axis, envelopes, colour, label, note in (
            (axes[0, 1], below_env, BELOW, 'below-band',
             f'rms_tilt={RMS_TILT:g}, rms_cue=0'),
            (axes[1, 0], in_env, IN_BAND, 'in-band',
             f'rms_tilt={RMS_TILT:g}, rms_cue={RMS_CUE:g}')):
        for envelope in envelopes[:N_SHOWN]:
            axis.plot(SHAPE_F, envelope, color=colour, lw=0.7, alpha=0.35)
        axis.plot(SHAPE_F, envelopes[0], color=colour, lw=2.0)
        axis.axhline(0, color=MUTED, lw=0.8)
        axis.set_ylim(-limit, limit)
        axis.text(0.98, 0.05, note, transform=axis.transAxes, ha='right',
                  color=INK, fontsize=8)
        axis.text(0.03, 0.94, label, transform=axis.transAxes, color=colour,
                  fontsize=10, fontweight='bold', va='top')
        _style(axis, 'frequency (kHz)', 'envelope (dB)',
               f'{"b" if colour == BELOW else "c"}  {N_SHOWN} trial envelopes '
               f'(one bold)')

    # (d) where each condition puts its energy in ripple density, against the cue
    axis = axes[1, 1]
    cue_band_shading(axis)
    axis.plot(SHAPE_DENSITY, dct(dtf, type=2, norm='ortho', axis=1).std(0),
              color=CUE, lw=2.0, label='DTF cue (across elevation)')
    # in-band first, below-band on top: they are identical under 0.5 rip/oct
    # (same rms_tilt) and the upper trace would otherwise hide the lower one
    axis.plot(SHAPE_DENSITY, numpy.abs(in_c).std(0), color=IN_BAND, lw=2.6,
              label='in-band source')
    axis.plot(SHAPE_DENSITY, numpy.abs(below_c).std(0), color=BELOW, lw=1.6,
              ls=(0, (4, 2)), label='below-band source')
    axis.set_xlim(0, 3)
    axis.text(1.25, axis.get_ylim()[1] * 0.93, 'cue band', color=INK,
              fontsize=8, ha='center')
    axis.legend(frameon=False, labelcolor=INK, fontsize=8, loc='upper right')
    _style(axis, 'ripple density (ripples / octave)', 'SD (dB per coefficient)',
           'd  the manipulation, in ripple density', logx=False)

    fig.tight_layout()
    return fig


def figure_2(data, dtf):
    """What varies across trials, per condition, against the cue it must not bury."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    # each token minus the across-token MEAN: the pink -3 dB/oct slope is common
    # to every trial, so leaving it in would hide the only thing this panel is
    # about, which is how much the tokens differ from one another
    centred = {label: data[label][2] - data[label][2].mean(0) for label in data}
    limit = 1.05 * max(numpy.abs(s[:N_SHOWN]).max() for s in centred.values())

    for axis, (label, colour, _) in zip(axes[:3], CONDITIONS):
        for spectrum in centred[label][:N_SHOWN]:
            axis.plot(SHAPE_F, spectrum, color=colour, lw=0.6, alpha=0.35)
        axis.axhline(0, color=MUTED, lw=0.8)
        axis.set_ylim(-limit, limit)
        axis.text(0.04, 0.94, label, transform=axis.transAxes, color=colour,
                  fontsize=10, fontweight='bold', va='top')
        _style(axis, 'frequency (kHz)',
               'level re across-trial mean (dB)' if label == 'noise' else None,
               f'{"abc"[list(data).index(label)]}  {label}: {N_SHOWN} tokens, '
               f'each re their mean')

    axis = axes[3]
    for label, colour, _ in CONDITIONS:
        axis.plot(SHAPE_F, data[label][2].std(0), color=colour, lw=2.0, label=label)
        axis.text(SHAPE_F[-1], data[label][2].std(0)[-1], f' {label}', color=colour,
                  fontsize=8, va='center')
    axis.plot(SHAPE_F, dtf.std(0), color=CUE, lw=2.0, label='DTF cue')
    axis.text(SHAPE_F[-1], dtf.std(0)[-1], ' DTF cue', color=CUE, fontsize=8, va='center')
    axis.set_xlim(SHAPE_FLO, 1.9 * SHAPE_FHI)
    _style(axis, 'frequency (kHz)', 'SD (dB)',
           'd  across-trial spread vs the cue')

    fig.tight_layout()
    return fig


def cue_band_filter(spectra):
    """Keep only 0.5-2 ripples/oct -- what a band-selective read-out would see."""
    coefficients = dct(numpy.asarray(spectra), type=2, norm='ortho', axis=-1)
    keep = (SHAPE_DENSITY >= RIPPLE_TILT_MAX) & (SHAPE_DENSITY < RIPPLE_CUE_MAX)
    return idct(coefficients * keep, type=2, norm='ortho', axis=-1)


def figure_3(dtf, elevations, n_draws=10, separation=25.0):
    """Received spectrum = DTF + source envelope, two elevations, both conditions.

    Top row is what arrives at the eardrum; bottom row is the same after
    band-limiting to the cue band, i.e. what a read-out that simply filters by
    ripple density would work with. Below-band source variation vanishes in
    that filter and the two elevations separate cleanly; in-band variation
    survives it, and the two elevations overlap. That is the whole argument for
    why only the in-band condition requires inference.

    Colour here encodes a signed quantity (below vs above the horizon), not a
    category, so it uses the warm/cool diverging pair rather than the
    categorical order of figures 1 and 2.
    """
    low = int(numpy.argmin(numpy.abs(elevations + separation)))
    high = int(numpy.argmin(numpy.abs(elevations - separation)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    columns = (('below-band source', {'rms_tilt': RMS_TILT, 'rms_cue': 0.0}),
               ('in-band source', {'rms_tilt': RMS_TILT, 'rms_cue': RMS_CUE}))

    for column, (label, kwargs) in enumerate(columns):
        received = {}
        for index in (low, high):
            draws = []
            for _ in range(n_draws):
                _, params = make_rippled_pinknoise(**kwargs)
                coefficients = numpy.zeros(SHAPE_N)
                coefficients[:len(params['coeffs'])] = params['coeffs']
                draws.append(dtf[index] + shape_from_coefficients(coefficients)[1])
            received[index] = numpy.array(draws)

        for row, transform in enumerate((lambda x: x, cue_band_filter)):
            axis = axes[row, column]
            for index, colour in ((low, '#2a78d6'), (high, '#eb6834')):
                for spectrum in transform(received[index]):
                    axis.plot(SHAPE_F, spectrum, color=colour, lw=0.8, alpha=0.45)
                reference = transform(dtf[index][None, :])[0]
                axis.plot(SHAPE_F, reference, color=colour, lw=2.6)
                axis.text(SHAPE_F[-1], reference[-1],
                          f'  {elevations[index]:+.0f}°', color=colour,
                          fontsize=9, fontweight='bold', va='center')
            axis.set_xlim(SHAPE_FLO, 1.5 * SHAPE_FHI)
            axis.axhline(0, color=MUTED, lw=0.8)
            stage = 'received' if row == 0 else 'cue band only (0.5-2 rip/oct)'
            _style(axis, 'frequency (kHz)' if row else None,
                   f'level re mean (dB)' if column == 0 else None,
                   f'{"ab"[column] if row == 0 else "cd"[column]}  {label}, {stage}')
            if row == 0 and column == 0:
                axis.text(0.02, 0.06, f'bold = DTF alone, thin = {n_draws} source draws',
                          transform=axis.transAxes, color=INK, fontsize=8)
            if row == 1 and column == 0:
                axis.text(0.02, 0.06, f'all {n_draws} draws lie under the DTF curves',
                          transform=axis.transAxes, color=INK, fontsize=8)

    for row in range(2):                       # one y-scale per row, so the two
        limit = max(abs(v) for axis in axes[row]  # columns are directly comparable
                    for v in axis.get_ylim())
        for axis in axes[row]:
            axis.set_ylim(-limit, limit)
    fig.tight_layout()
    return fig


def main():
    slab.set_default_samplerate(SAMPLERATE)
    dtf, elevations = dtf_on_shape_axis()
    data = {label: draw_tokens(**kwargs) for label, _, kwargs in CONDITIONS}

    for number, fig in enumerate((figure_1(data, dtf), figure_2(data, dtf),
                                 figure_3(dtf, elevations)), start=1):
        path = OUT_DIR / f'stimulus_synthesis_{number}.png'
        fig.savefig(path, dpi=160, facecolor='white')
        print(f'-> {path}')

    # the numbers quoted in stimulus_synthesis.md, recomputed
    print(f'\n{"condition":12s} {"across-trial SD":>16s} {"cue:source below":>18s} '
          f'{"cue:source in band":>19s}')
    tilt_band = (0.2, RIPPLE_TILT_MAX)
    cue_below = sm.ripple_rms(dtf, tilt_band, SHAPE_DENSITY)
    cue_in = sm.ripple_rms(dtf, (RIPPLE_TILT_MAX, RIPPLE_CUE_MAX), SHAPE_DENSITY)
    for label, _, _ in CONDITIONS:
        spectra = data[label][2]
        below = sm.ripple_rms(spectra, tilt_band, SHAPE_DENSITY)
        inside = sm.ripple_rms(spectra, (RIPPLE_TILT_MAX, RIPPLE_CUE_MAX), SHAPE_DENSITY)
        print(f'{label:12s} {spectra.std(0).mean():13.2f} dB '
              f'{cue_below / below:16.2f}:1 {cue_in / inside:17.2f}:1')


if __name__ == '__main__':
    main()
