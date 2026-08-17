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

#: Band the envelope is fitted over when frequencies are supplied. Below the
#: lower edge there are no pinna cues to remove and the highpass plus the
#: low-frequency extrapolation are steep enough to drag any fit; above the
#: upper edge is the anti-alias rolloff. Outside it the measured response is
#: kept, so the untrained ear's low-frequency level -- and therefore the ILD --
#: is the listener's own.
ENVELOPE_BAND = (700.0, 18000.0)


def _erb_rate(frequencies):
    """Glasberg & Moore (1990) ERB-rate: Hz -> ERB number."""
    return 21.4 * numpy.log10(1.0 + 0.00437 * numpy.asarray(frequencies, dtype=float))


def _erb_envelope(mag, n_keep, freqs, band, skirt_octaves=(1.0, 0.5)):
    """``n_keep``-term cosine fit to log-magnitude on an ERB-rate axis.

    WHY NOT THE LINEAR AXIS. :func:`...shift_spectral_detail.smooth_magnitude`
    puts the cosine basis on the FFT bin index, i.e. linear in Hz. At
    ``n_keep=4`` the finest term is one half-cycle over 8.1 kHz, so below about
    2 kHz all four basis functions are still sitting at their k=0 values and the
    envelope cannot vary there at all. 200-700 Hz is 1.6% of that axis against
    40% for 8-18 kHz, so least squares sets the low end to whatever best serves
    the top of the band -- measured on this cohort, 15 dB above the true DTF at
    DC, which put the untrained ear's 200-2000 Hz ILD 9-12 dB away from native
    and outside the range of any real direction in the listener's own HRTF.

    On an ERB-rate axis the same four terms give 0.7-2 kHz a quarter of the
    fit instead of a twentieth, which also matches the 0.5-2 ripples/octave
    scale the split is meant to sit at rather better than the linear axis did.

    The least-squares solve uses only the first ``n_keep`` basis COLUMNS on the
    in-band ROWS -- restricting rows alone leaves the system underdetermined.
    """
    lo, hi = float(band[0]), float(band[1])
    log_mag = numpy.log(numpy.maximum(mag, numpy.finfo(float).tiny))

    rate = (_erb_rate(freqs) - _erb_rate(lo)) / (_erb_rate(hi) - _erb_rate(lo))
    basis = numpy.cos(numpy.pi * numpy.clip(rate, 0.0, 1.0)[:, None]
                      * numpy.arange(int(n_keep))[None, :])
    in_band = (freqs >= lo) & (freqs <= hi)
    if int(in_band.sum()) < int(n_keep):
        raise ValueError(f'only {int(in_band.sum())} bins inside {band}, '
                         f'need at least n_keep={n_keep}')

    # raised-cosine crossfade in log frequency: envelope in band, measured out
    freq = numpy.maximum(numpy.asarray(freqs, dtype=float), 1e-6)
    weight = ((freq >= lo) & (freq <= hi)).astype(float)
    low_edge = lo * 2.0 ** -skirt_octaves[0]
    rising = (freq > low_edge) & (freq < lo)
    weight[rising] = 0.5 - 0.5 * numpy.cos(
        numpy.pi * numpy.log2(freq[rising] / low_edge) / skirt_octaves[0])
    falling = (freq > hi) & (freq < hi * 2.0 ** skirt_octaves[1])
    weight[falling] = 0.5 + 0.5 * numpy.cos(
        numpy.pi * numpy.log2(freq[falling] / hi) / skirt_octaves[1])

    out = numpy.empty_like(log_mag)
    for channel in range(mag.shape[1]):
        coeffs, *_ = numpy.linalg.lstsq(basis[in_band], log_mag[in_band, channel],
                                        rcond=None)
        out[:, channel] = (weight * (basis @ coeffs)
                           + (1.0 - weight) * log_mag[:, channel])
    return numpy.exp(out)


def envelope_spectrum(mag, n_keep=DEFAULT_N_KEEP, freqs=None, band=ENVELOPE_BAND):
    """Coarse envelope of a one-sided magnitude spectrum (linear, not dB).

    With ``freqs`` the fit runs on an ERB-rate axis over ``band`` and the
    measured response is kept outside it (:func:`_erb_envelope`). Without
    ``freqs`` this is the historical linear-axis, full-range smoother
    :func:`hrtf_relearning.hrtf.modify.shift_spectral_detail.smooth_magnitude`
    -- kept so pre-2026-08 builds stay reproducible, NOT because it is
    equivalent. Accepts a 1-D spectrum as well as ``(n_bins, n_channels)``.
    """
    mag = numpy.asarray(mag, dtype=float)
    squeeze = mag.ndim == 1
    if squeeze:
        mag = mag[:, None]
    if freqs is None:
        out = smooth_magnitude(mag, n_keep=int(n_keep))
    else:
        out = _erb_envelope(mag, int(n_keep), numpy.asarray(freqs, dtype=float), band)
    return out[:, 0] if squeeze else out


def envelope_dtf(hrir, ear='left', n_keep=DEFAULT_N_KEEP, match_level='energy',
                 band=ENVELOPE_BAND, elevation_average=True):
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

        NB broadband energy being exact says nothing about the ILD inside any
        given band -- it read exactly 0.00 dB while the 200-2000 Hz ILD was
        10 dB out. Check bands, not the broadband number; see
        :func:`hrtf_relearning.hrtf.processing.midline.qc_midline`.
    band : (float, float), default ``ENVELOPE_BAND``
        Band the envelope is fitted over; the measured response is kept
        outside it. See :func:`_erb_envelope`.
    elevation_average : bool, default True
        Replace each direction's envelope by the mean over elevation at that
        azimuth, so one spectral shape serves every elevation.

        This is what actually removes the cue. Smoothing in FREQUENCY only ever
        approximated it, and had to be traded against level accuracy because
        both come out of the same n_keep coefficients -- measured on this
        cohort, a per-direction envelope still carried 2.30 dB of
        elevation-varying structure in the 5.7-11.3 kHz band against 4.66 dB
        for the unprocessed ear, i.e. about half the cue. Averaging removes it
        by construction: identical shapes cannot encode elevation.

        Grouping by azimuth means only elevation dependence goes. On the az=0
        arc that is every direction; on a full set the head shadow survives,
        which is what keeps the ear plausible enough to externalize.

        ``match_level`` is applied after, per direction, so broadband ILD is
        still exact even though the shape no longer varies.

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
    logger.debug('Envelope-only DTF for the %s ear(s) (n_keep=%d, elevation_average=%s)',
                 'both' if ear == 'both' else ('right' if channels[0] else 'left'),
                 int(n_keep), bool(elevation_average))

    eps = numpy.finfo(float).tiny
    n_sources = out.n_sources

    # pass 1 -- envelope per direction, kept in log magnitude so the averaging
    # below is a mean of log spectra (i.e. a geometric mean of magnitudes)
    log_envelope = {}
    for source_idx in range(n_sources):
        freqs = numpy.fft.rfftfreq(
            hrir[source_idx].data.shape[0],
            d=1.0 / float(hrir[source_idx].samplerate))
        for channel in channels:
            mag = numpy.abs(numpy.fft.rfft(
                numpy.asarray(hrir[source_idx].data[:, channel], dtype=float)))
            if mag.max() <= 0:
                continue
            log_envelope[(source_idx, channel)] = numpy.log(numpy.maximum(
                envelope_spectrum(mag, n_keep=int(n_keep), freqs=freqs, band=band),
                eps))

    # pass 2 -- replace each envelope by the mean over elevation AT ITS AZIMUTH.
    # Grouping by azimuth is what makes this safe to call on a full set as well
    # as on the az=0 arc: it removes the elevation dependence and nothing else,
    # so the head shadow (which is azimuth dependence) is untouched.
    if elevation_average:
        azimuth = numpy.round(numpy.mod(
            numpy.asarray(hrir.sources.vertical_polar[:, 0], dtype=float), 360.0), 3)
        for value in numpy.unique(azimuth):
            group = numpy.where(azimuth == value)[0]
            for channel in channels:
                stack = [log_envelope[(i, channel)] for i in group
                         if (i, channel) in log_envelope]
                if not stack:
                    continue
                mean_log = numpy.mean(numpy.asarray(stack), axis=0)
                for i in group:
                    if (i, channel) in log_envelope:
                        log_envelope[(i, channel)] = mean_log

    # pass 3 -- back to the time domain, per-direction level restored
    for (source_idx, channel), log_env in log_envelope.items():
        ir = numpy.asarray(hrir[source_idx].data[:, channel], dtype=float)
        spectrum = numpy.fft.rfft(ir)
        # magnitude-only edit: original phase kept -> onset / ITD untouched
        ir_env = numpy.fft.irfft(
            numpy.exp(log_env) * numpy.exp(1j * numpy.angle(spectrum)), n=ir.size)
        if match_level == 'energy':
            # applied AFTER the averaging, so each direction keeps its own
            # broadband level and the per-direction ILD stays exact even though
            # every elevation now shares one spectral shape
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
