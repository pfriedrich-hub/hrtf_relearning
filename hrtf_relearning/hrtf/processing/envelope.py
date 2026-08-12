"""
envelope.py — give the non-listening ear the coarse spectral envelope instead of
a flat spectrum.

WHY. In the monaural learning paradigm the other ear is silenced as a cue source
by :func:`hrtf_relearning.hrtf.processing.flatten.flatten_dtf`, which replaces
its whole IR with a single delta at the onset: ITD and broadband ILD survive, all
spectral shape is gone. That is maximally clean as a cue manipulation, but a
delta has no pinna, no ear-canal resonance and no head shadow colouration, so the
two ears no longer look like they belong to the same head. Externalization
collapses toward the middle of the skull, and a percept that is not externalized
is a poor teacher — there is little reason for the system to recalibrate a
spatial map for a sound that is not out in space.

WHAT THIS DOES INSTEAD. The other ear keeps its own DTF *envelope* and loses only
the fine detail:

    log|H|  =  envelope (first n_keep cosine coeffs)  +  detail
    other ear  ->  envelope only

The split is the same cepstral one used on the trained ear by
:mod:`hrtf_relearning.hrtf.modify.shift_spectral_detail` (Kulkarni & Colburn
1998, Nature 396:747) — the same ``n_keep`` (M) that defines what the shift holds
fixed defines what the untrained ear keeps. So the manipulation is symmetric in
its own terms: on the trained ear the detail is transported, on the untrained ear
the detail is removed, and both ears carry the identical, untouched envelope.

Magnitude-only: the ORIGINAL PHASE is kept, so onset timing and ITD are exactly
as measured. Broadband energy is rescaled to the original per-direction value
(``match_level='energy'``), so broadband ILD is exactly preserved too — the same
two binaural cues flatten_dtf preserves.

WHAT LEAKS. The envelope is direction-dependent (mostly head shadow and the broad
concha resonance), so unlike a flat ear the untrained ear is not literally
cue-free: it retains slow spectral shape that co-varies with azimuth, and to a
much smaller extent with elevation. That is the deliberate trade — with
``n_keep=4`` the envelope has ~2 extrema across the whole spectrum and cannot
resolve pinna notches (0.5-2 ripples/octave), so it supports externalization
without supplying a usable elevation cue. Lower ``n_keep`` if you want it closer
to flat; check what is left with the ``__main__`` QC block below, which plots the
residual detail that the processed ear retains.

Usage (via the binsim pipeline, not by hand)::

    hrir_settings = dict(..., ear='left', other_ear='envelope', env_n_keep=4)

See :func:`hrtf_relearning.hrtf.binsim.hrtf2binsim.hrtf2binsim`, and the pilot
protocol experiment/protocols/learning_transfer/learning_transfer.py.
"""

import copy
import logging

import numpy

# numpy-only helper; importing it (rather than re-deriving the smoother) is what
# guarantees the untrained ear's envelope is the SAME object the shift holds
# fixed on the trained ear.
from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude

logger = logging.getLogger(__name__)

DEFAULT_N_KEEP = 4


def envelope_spectrum(mag, n_keep=DEFAULT_N_KEEP):
    """Coarse envelope of a one-sided magnitude spectrum (linear, not dB).

    Thin wrapper over
    :func:`hrtf_relearning.hrtf.modify.shift_spectral_detail.smooth_magnitude`
    that accepts a 1-D spectrum as well as ``(n_bins, n_channels)``.
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim == 1:
        return smooth_magnitude(mag[:, None], n_keep=int(n_keep))[:, 0]
    return smooth_magnitude(mag, n_keep=int(n_keep))


def envelope_dtf(hrir, ear='left', n_keep=DEFAULT_N_KEEP, match_level='energy'):
    """Replace the OTHER ear's DTF with its coarse spectral envelope.

    Drop-in alternative to
    :func:`hrtf_relearning.hrtf.processing.flatten.flatten_dtf`: same signature
    convention (``ear`` names the ear that is KEPT intact — the listening /
    trained ear — and the opposite ear is processed), same preserved binaural
    cues (ITD via the original phase, broadband ILD via ``match_level``), but the
    processed ear keeps its coarse spectral shape instead of becoming a delta.

    Parameters
    ----------
    hrir : slab.HRTF
        Input HRIR (time domain, FIR). Not modified — a deep copy is returned.
    ear : {'left', 'right', 'both'}, default 'left'
        The ear to KEEP — the other ear is reduced to its envelope. ``'both'``
        keeps neither and smooths BOTH ears, which is not a monaural condition
        at all: it is the binaural spectral-detail control (Kulkarni & Colburn
        1998), where the whole HRTF is replaced by its coarse envelope and the
        listener has no fine structure at either ear.
    n_keep : int, default 4
        Cosine coefficients kept for the envelope (M in Kulkarni & Colburn
        1998). Same value as the trained ear's ``envelope_n_keep``. Lower ->
        smoother, closer to flat; higher -> more spectral detail survives on the
        untrained ear.
    match_level : {'energy', None}, default 'energy'
        ``'energy'`` rescales each processed IR so its L2 energy equals the
        original's, making the per-direction broadband ILD exact. ``None``
        leaves the envelope's own level (already close, since the DC cosine
        coefficient carries the mean log-magnitude).

    Returns
    -------
    slab.HRTF
        Deep copy with one ear reduced to its envelope.
    """
    if ear not in ('left', 'right', 'both'):
        raise ValueError(f"ear must be 'left', 'right' or 'both', got {ear!r}")
    if int(n_keep) < 1:
        raise ValueError(f'n_keep must be >= 1, got {n_keep}')
    if match_level not in ('energy', None):
        raise ValueError("match_level must be 'energy' or None")

    out = copy.deepcopy(hrir)
    if ear == 'both':
        channels = (0, 1)
    else:
        channels = (1 if ear == 'left' else 0,)   # the ear being processed
    logger.debug('Envelope-only DTF for the %s ear(s) (n_keep=%d)',
                 'both' if ear == 'both' else ('right' if channels[0] else 'left'),
                 int(n_keep))

    eps = numpy.finfo(float).tiny
    for source_idx in range(out.n_sources):
        for channel in channels:
            ir = numpy.asarray(out[source_idx].data[:, channel], dtype=float)
            n_samples = ir.size
            spectrum = numpy.fft.rfft(ir)
            mag = numpy.abs(spectrum)
            if mag.max() <= 0:
                continue
            mag_env = envelope_spectrum(mag, n_keep=int(n_keep))
            # magnitude-only edit: original phase kept -> onset / ITD untouched
            ir_env = numpy.fft.irfft(mag_env * numpy.exp(1j * numpy.angle(spectrum)),
                                     n=n_samples)
            if match_level == 'energy':
                e_original = float(numpy.linalg.norm(ir, ord=2))
                e_env = float(numpy.linalg.norm(ir_env, ord=2))
                if e_env > eps:
                    ir_env *= e_original / e_env
            out[source_idx].data[:, channel] = ir_env
    return out


def residual_detail_db(hrir, processed, ear='left', band=(4000, 16000),
                       reference_n_keep=DEFAULT_N_KEEP):
    """RMS of the structure finer than a ``reference_n_keep`` envelope, in dB.

    Direction-averaged, inside ``band``, measured on the ear that was processed
    (i.e. NOT ``ear``). A native ear sits at several dB; a flat ear at 0.

    Note the reference is fixed, so this is trivially 0 for an ear processed
    with ``n_keep <= reference_n_keep`` — it answers "is anything left above the
    coarse scale", not "does the ear still carry a cue". For the latter use
    :func:`direction_variation_db`: an envelope can be featureless per direction
    and still vary systematically between directions.

    Returns
    -------
    (native_rms_db, processed_rms_db) : tuple of float
    """
    other_idx = 1 if ear == 'left' else 0
    eps = numpy.finfo(float).tiny

    def _rms(hrtf_obj):
        values = []
        for source_idx in range(hrtf_obj.n_sources):
            ir = numpy.asarray(hrtf_obj[source_idx].data[:, other_idx], dtype=float)
            freqs = numpy.fft.rfftfreq(ir.size, d=1.0 / hrtf_obj[source_idx].samplerate)
            mag = numpy.abs(numpy.fft.rfft(ir))
            log_mag = 20.0 * numpy.log10(numpy.maximum(mag, eps))
            env = 20.0 * numpy.log10(numpy.maximum(
                envelope_spectrum(mag, n_keep=int(reference_n_keep)), eps))
            in_band = (freqs >= band[0]) & (freqs <= band[1])
            values.append(numpy.sqrt(numpy.mean((log_mag - env)[in_band] ** 2)))
        return float(numpy.mean(values))

    return _rms(hrir), _rms(processed)


def direction_variation_db(hrtf, ear='left', band=(4000, 16000), bin_deg=5.0):
    """How much the processed ear's spectrum still changes with direction.

    The question the envelope ear has to answer: it must stay plausible (so
    externalization survives) without handing back an elevation cue. Per-bin SD
    of the log-magnitude, averaged over the band, computed two ways:

    * ``elevation`` — across directions WITHIN an azimuth column, i.e. the part
      that could be read as an elevation cue. This is what should drop.
    * ``azimuth`` — across directions WITHIN an elevation row, mostly head
      shadow. This is expected to survive and is what keeps the ear plausible.

    Both are measured on the ear that was processed (NOT ``ear``). Requires
    ``hrtf.sources.vertical_polar`` (slab.HRTF).

    Returns
    -------
    (elevation_sd_db, azimuth_sd_db) : tuple of float
    """
    other_idx = 1 if ear == 'left' else 0
    eps = numpy.finfo(float).tiny

    sources = numpy.asarray(hrtf.sources.vertical_polar, dtype=float)
    azimuth = numpy.mod(sources[:, 0] + 180.0, 360.0) - 180.0
    elevation = sources[:, 1]

    spectra = []
    for source_idx in range(hrtf.n_sources):
        ir = numpy.asarray(hrtf[source_idx].data[:, other_idx], dtype=float)
        freqs = numpy.fft.rfftfreq(ir.size, d=1.0 / hrtf[source_idx].samplerate)
        mag = numpy.abs(numpy.fft.rfft(ir))
        in_band = (freqs >= band[0]) & (freqs <= band[1])
        spectra.append(20.0 * numpy.log10(numpy.maximum(mag, eps))[in_band])
    spectra = numpy.array(spectra)

    def _spread(group):
        group = numpy.round(numpy.asarray(group) / bin_deg) * bin_deg
        values = [numpy.mean(numpy.std(spectra[group == g], axis=0))
                  for g in numpy.unique(group) if numpy.sum(group == g) > 2]
        return float(numpy.mean(values)) if values else float('nan')

    # vary elevation at fixed azimuth -> group BY azimuth, and vice versa
    return _spread(azimuth), _spread(elevation)


# ---------------------------------------------------------------------------
# QC preview — run this file directly
# ---------------------------------------------------------------------------

SUB_ID = 'CO'          # subject with a measured <id>.sofa in data/hrtf/sofa/<id>/
SOFA_NAME = None       # None -> '<SUB_ID>.sofa'; e.g. f'{SUB_ID}_shift' for the modified set
KEEP_EAR = 'left'      # ear kept intact (= the trained ear); the OTHER one is processed
N_KEEP = DEFAULT_N_KEEP
PLOT_KIND = 'image'    # 'image' | 'waterfall' | 'surface'

if __name__ == '__main__':
    import matplotlib
    from hrtf_relearning.utils.mpl_backend import use_interactive
    use_interactive()
    import slab
    from hrtf_relearning.utils import paths
    from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
    from hrtf_relearning.hrtf.processing.flatten import flatten_dtf

    sofa_dir = paths.SOFA_DIR / SUB_ID
    sofa_path = sofa_dir / f'{SOFA_NAME or SUB_ID}.sofa'
    if not sofa_path.exists():
        raise FileNotFoundError(
            f'no SOFA at {sofa_path} — check SUB_ID / SOFA_NAME in the config block')

    hrtf = slab.HRTF(str(sofa_path))
    hrtf.name = sofa_path.stem
    print(f'loaded {sofa_path.name}')

    processed_ear = 'right' if KEEP_EAR == 'left' else 'left'
    hrtf_env = envelope_dtf(hrtf, ear=KEEP_EAR, n_keep=N_KEEP)
    hrtf_flat = flatten_dtf(hrtf, ear=KEEP_EAR)

    native_db, env_db = residual_detail_db(hrtf, hrtf_env, ear=KEEP_EAR)
    _, flat_db = residual_detail_db(hrtf, hrtf_flat, ear=KEEP_EAR)
    print(f'{processed_ear} ear, 4-16 kHz')
    print(f'  detail RMS (finer than a {DEFAULT_N_KEEP}-coeff envelope):  '
          f'native {native_db:.2f} dB | envelope(n_keep={N_KEEP}) {env_db:.2f} dB '
          f'| flat {flat_db:.2f} dB')
    for label, h in (('native', hrtf), (f'envelope(n_keep={N_KEEP})', hrtf_env),
                     ('flat', hrtf_flat)):
        el_sd, az_sd = direction_variation_db(h, ear=KEEP_EAR)
        print(f'  spectral SD {label:>22}:  across elevation {el_sd:5.2f} dB  |  '
              f'across azimuth {az_sd:5.2f} dB')
    print('  -> elevation SD is the cue that must go; azimuth SD should survive')

    # before/after for the ear that is being reduced: everything sharp should be
    # gone, the broad shape and its azimuth dependence should remain.
    fig = plot_ears(hrtf, hrtf_env, suptitle=f'{SUB_ID}  envelope n_keep={N_KEEP}')
    plot_dir = paths.subject_acoustic_dir(SUB_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / f'{hrtf.name}_env{N_KEEP}_{processed_ear}_ear.png'
    fig.savefig(out_png, bbox_inches='tight')
    print(f'wrote {out_png}')
