"""
fig_stimulus_variation.py — how much the localization stimulus varies from
trial to trial, old vs new, measured against the elevation cue it has to spare.

WHY A SUMMARY AND NOT FIVE TOKENS. Overlaying a handful of spectra shows that
the new stimulus is not the old one and nothing else: the eye cannot tell
whether the variation sits at a spectral scale the auditory system discounts or
at the scale the pinna notch lives on, and that distinction is the entire
design. Both panels here are statistics over MANY tokens, so what is drawn is
the distribution the listener is sampled from, not a sample from it.

    A  SD across tokens of the 1/6-octave log spectrum, per frequency.
       "How different does the next trial sound?", in the units the cue is in.
       The DTF cue (SD across ELEVATION of the same measure, at the tested
       azimuth) is drawn on the same axis: the source variation has to be
       comparable to it or template matching still solves the task.

    B  the same token sets decomposed by ripple density (rms per DCT
       coefficient of the log spectrum on a log-frequency axis). This is the
       panel that carries the argument: the new variation is deliberately
       confined BELOW 0.5 ripples/oct — broad colouration, which the auditory
       system discounts — while the 0.5–2 ripples/oct band, where the elevation
       cue lives, is left as bare as it was under plain noise.

The single number in the corner is the ratio of the two inside the cue band.
Plain noise sits near 9:1 (no source variation at all — absolute-spectrum
template matching is sufficient), a fully randomised USO near 1:1 (cue buried).
The design target is ~3:1: enough source variation that the absolute spectrum
is uninformative, not so much that the cue is eroded.

Defaults reproduce the production stimulus: `STIM='ripple'`,
`STIM_SETTINGS={'rms_tilt': 3}` in learning_transfer.py, measured against the
composite SOFA the participant is tested on. `rms_tilt` was settled on AS
(2026-08-18) — it is the only setting whose polar error is indistinguishable
from noise on the dome. See docs/stimulus_spectral_variation.md.

Run this file directly; it saves to ``results/<id>/plots/acoustic/``.
"""

from pathlib import Path

import numpy
import slab
from scipy.fftpack import dct
from matplotlib import pyplot as plt

from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise)
from hrtf_relearning.utils import paths

# --- configuration -----------------------------------------------------------
SUBJECT_ID = 'AS'
SOFA_NAME = 'AS_donor_GS_env4_left'   # the HRTF the subject is tested on
EAR = 'left'                          # ear whose DTF carries the trained cue
AZIMUTH = -20.0                       # experiment convention (negative = left)
RMS_TILT = 3.0                        # production value, learning_transfer.STIM_SETTINGS
RMS_CUE = 0.0                         # 0 by design — see the module docstring
N_TOKENS = 240
SEED = 20260821                       # figure is deterministic; change to resample
SHOW = False

# --- analysis axis -----------------------------------------------------------
# Same definitions as protocols/dev/stimulus_check.py, kept here rather than
# imported: that file is a cell script whose top level plays sounds, so
# importing it would run the audition cells.
FLO, FHI, NPTS = 3500.0, 16000.0, 192
LOGF = numpy.logspace(numpy.log10(FLO), numpy.log10(FHI), NPTS)
NOCT = numpy.log2(FHI / FLO)
BW = 2 ** (1 / 12)                    # ±1/12 oct → 1/6-octave window (~1 ERB at 8 kHz)
DENSITY = numpy.arange(NPTS) / (2 * NOCT)
CUE_BAND = (0.5, 2.0)                 # ripples/oct — where the pinna notch lives
TILT_BAND = (0.2, 0.5)                # ripples/oct — where the new variation is put

COLOURS = {'cue': '#1b1b1f', 'noise': '#1b6ca8', 'ripple': '#2e8b57'}


def log_spectrum(x, samplerate):
    """1/6-octave-smoothed log power spectrum on LOGF, mean removed (shape only)."""
    x = numpy.asarray(x, dtype=float).squeeze()
    power = numpy.abs(numpy.fft.rfft(x)) ** 2
    freqs = numpy.fft.rfftfreq(len(x), 1 / samplerate)
    cumulative = numpy.concatenate([[0], numpy.cumsum(power)])
    lo = numpy.searchsorted(freqs, LOGF / BW)
    hi = numpy.searchsorted(freqs, LOGF * BW)
    band = (cumulative[hi] - cumulative[lo]) / numpy.maximum(hi - lo, 1)
    spectrum = 10 * numpy.log10(band + 1e-20)
    return spectrum - spectrum.mean()


def density_rms(spectra):
    """rms per DCT coefficient over a set of log spectra — the density profile."""
    return dct(numpy.asarray(spectra), type=2, norm='ortho', axis=1).std(0)


def band_rms(spectra, band):
    """That profile collapsed to one number inside a ripple-density band."""
    coeffs = density_rms(spectra)
    selected = (DENSITY >= band[0]) & (DENSITY < band[1])
    return float(numpy.sqrt((coeffs[selected] ** 2).mean()))


def cue_spectra(subject_id=SUBJECT_ID, sofa_name=SOFA_NAME, ear=EAR,
                azimuth=AZIMUTH):
    """Log spectra of one DTF azimuth column across elevation — the cue itself."""
    hrtf = slab.HRTF(str(paths.SOFA_DIR / subject_id / f'{sofa_name}.sofa'))
    sources = hrtf.sources.vertical_polar
    sofa_azimuth = numpy.mod(-azimuth, 360)      # the experiment negates azimuth
    grid = sources[:, 0]
    nearest = grid[numpy.argmin(numpy.abs(((grid - sofa_azimuth + 180) % 360) - 180))]
    selected = ((numpy.abs(((grid - nearest + 180) % 360) - 180) < 1)
                & (numpy.abs(sources[:, 1]) <= 35))
    channel = 0 if ear == 'left' else 1
    return numpy.array([log_spectrum(hrtf[i].data[:, channel], hrtf.samplerate)
                        for i in numpy.flatnonzero(selected)])


def token_spectra(kind, n=N_TOKENS, seed=SEED, rms_tilt=RMS_TILT, rms_cue=RMS_CUE):
    """Log spectra of ``n`` fresh tokens of one stimulus kind."""
    rng = numpy.random.default_rng(seed)
    spectra = []
    for _ in range(n):
        if kind == 'noise':
            sound = make_gapped_pinknoise()
        elif kind == 'ripple':
            sound = make_rippled_pinknoise(
                rms_tilt=rms_tilt, rms_cue=rms_cue,
                seed=int(rng.integers(0, 2 ** 31 - 1)))[0]
        else:
            raise ValueError(f'unknown stimulus kind {kind!r}')
        spectra.append(log_spectrum(sound.data[:, 0], sound.samplerate))
    return numpy.asarray(spectra)


def summarise(cue, sets):
    """The numbers the figure annotates, as a dict of dicts."""
    cue_band = band_rms(cue, CUE_BAND)
    out = {'cue': {'pointwise': float(cue.std(0).mean()),
                   'cue_band': cue_band,
                   'tilt_band': band_rms(cue, TILT_BAND)}}
    for name, spectra in sets.items():
        source_cue = band_rms(spectra, CUE_BAND)
        out[name] = {'pointwise': float(spectra.std(0).mean()),
                     'cue_band': source_cue,
                     'tilt_band': band_rms(spectra, TILT_BAND),
                     'ratio': cue_band / source_cue}
    return out


def figure(subject_id=SUBJECT_ID, sofa_name=SOFA_NAME, ear=EAR, azimuth=AZIMUTH,
           rms_tilt=RMS_TILT, rms_cue=RMS_CUE, n_tokens=N_TOKENS, seed=SEED,
           show=SHOW):
    cue = cue_spectra(subject_id, sofa_name, ear, azimuth)
    sets = {
        'noise': token_spectra('noise', n_tokens, seed),
        'ripple': token_spectra('ripple', n_tokens, seed, rms_tilt, rms_cue),
    }
    stats = summarise(cue, sets)
    labels = {'noise': 'old — gapped pinknoise',
              'ripple': f'new — rippled pinknoise (rms_tilt={rms_tilt:g})'}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=200)

    # --- A: where in FREQUENCY the variation sits ---------------------------
    # Both curves are an SD of the same quantity, but over different things:
    # the cue varies across ELEVATION (one source spectrum, many directions),
    # the stimuli across TRIALS (one direction, many source spectra). That is
    # the comparison — a listener cannot separate the two from one token.
    axis = axes[0]
    axis.semilogx(LOGF, cue.std(0), color=COLOURS['cue'], lw=2.2,
                  label=f'DTF elevation cue — across elevation '
                        f'({ear} ear, az {azimuth:+.0f}°)')
    for name, spectra in sets.items():
        axis.semilogx(LOGF, spectra.std(0), color=COLOURS[name], lw=1.7,
                      label=f'{labels[name]} — across trials')
    axis.set(xlabel='Frequency [kHz]', ylabel='SD of 1/6-oct log spectrum [dB]',
             xlim=(FLO, FHI))
    # headroom for the legend, which otherwise sits on the cue peak
    axis.set_ylim(0, 1.35 * max(cue.std(0).max(),
                                max(s.std(0).max() for s in sets.values())))
    ticks = [4000, 6000, 8000, 10000, 12000, 16000]
    axis.set_xticks(ticks, [f'{t / 1000:g}' for t in ticks], minor=False)
    axis.set_xticks([], minor=True)
    axis.set_title('A   how much the source spectrum moves', loc='left', fontsize=10)
    axis.legend(fontsize=7.5, frameon=False, loc='upper left')
    axis.grid(alpha=0.2, which='major', lw=0.5)

    # --- B: at what SCALE it sits — the panel that makes the argument -------
    axis = axes[1]
    axis.axvspan(*TILT_BAND, color=COLOURS['ripple'], alpha=0.07, lw=0)
    axis.axvspan(*CUE_BAND, color=COLOURS['cue'], alpha=0.07, lw=0)
    axis.plot(DENSITY[1:], density_rms(cue)[1:], color=COLOURS['cue'], lw=2.2)
    for name, spectra in sets.items():
        axis.plot(DENSITY[1:], density_rms(spectra)[1:], color=COLOURS[name], lw=1.7)
    axis.set(xlabel='Ripple density [ripples/octave]',
             ylabel='rms per DCT coefficient [dB]',
             xlim=(0, 4.5), ylim=(0, None))
    axis.set_title('B   at what spectral scale', loc='left', fontsize=10)
    axis.grid(alpha=0.2, lw=0.5)
    top = axis.get_ylim()[1]
    axis.text(TILT_BAND[0] + 0.02, top * 0.42, 'source band\n< 0.5 rip/oct',
              ha='left', va='top', fontsize=7.5, color=COLOURS['ripple'])
    axis.text(numpy.mean(CUE_BAND), top * 0.99, 'cue band\n0.5–2 rip/oct',
              ha='center', va='top', fontsize=7.5, color=COLOURS['cue'])

    lines = ['cue : source inside the cue band']
    lines += [f'  {labels[name]:<42s}{stats[name]["ratio"]:5.1f} : 1'
              for name in sets]
    lines += [f'  {"acoustic screen floor":<42s}{"≥ 3":>5s} : 1']
    axis.text(0.98, 0.55, '\n'.join(lines), transform=axis.transAxes,
              ha='right', va='top', fontsize=7, family='monospace',
              bbox=dict(fc='white', ec='#bbbbbb', lw=0.6, pad=3))

    fig.suptitle(f'{subject_id} {sofa_name} — across-trial spectral variation '
                 f'vs the elevation cue  ({n_tokens} tokens per stimulus)',
                 fontsize=11)
    fig.tight_layout()
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return fig, stats


if __name__ == '__main__':
    fig, stats = figure()
    print(f'{"":24s} {"point-wise SD":>13s} {"tilt band":>10s} {"cue band":>9s} '
          f'{"cue:source":>11s}')
    for name, row in stats.items():
        ratio = f'{row["ratio"]:9.1f}:1' if 'ratio' in row else ''
        print(f'{name:24s} {row["pointwise"]:12.2f}dB {row["tilt_band"]:9.2f}dB '
              f'{row["cue_band"]:8.2f}dB {ratio:>11s}')

    out_dir = paths.subject_acoustic_dir(SUBJECT_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        out = Path(out_dir) / f'{SUBJECT_ID}_stimulus_variation.{suffix}'
        fig.savefig(out, bbox_inches='tight')
        print(f'wrote {out}')
