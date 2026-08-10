"""
native.py — leave the non-listening ear completely alone.

Third option for what the other ear receives in a "monaural" condition, next to
:func:`hrtf_relearning.hrtf.processing.flatten.flatten_dtf` (delta impulse) and
:func:`hrtf_relearning.hrtf.processing.envelope.envelope_dtf` (coarse envelope):

    flat      other ear = one delta at the onset      no spectral shape
    envelope  other ear = its own n_keep envelope     coarse shape only
    native    other ear = its own UNMODIFIED DTF      everything  <- this file

The cue manipulation (e.g. the ERB shift) is applied to a whole SOFA, i.e. to
both ears. This module takes the other ear's channel back from the NATIVE SOFA,
so the delivered stimulus carries the modification on the listening ear only —
"shift one side, leave the other alone". Equivalent to building a one-sided
modified SOFA, but done at binsim build time so a single modified SOFA can serve
all three delivery modes.

READ THIS BEFORE USING IT AS A TRAINING CONDITION. This is not a monaural
condition and does not behave like one:

* The other ear keeps a COMPLETE, VERIDICAL elevation cue. Measured in the
  trained hemifield it carries as much elevation-dependent spectral variation as
  the listening ear itself, attenuated only by head shadow (a few dB at +-35
  deg). The listener can localize from it.
* So an improvement across training days can be reweighting toward the intact
  ear rather than relearning the modified cue on the trained ear. The two are
  indistinguishable unless you also probe with the other ear reduced (flat or
  envelope) — see the probe phases in
  experiment/protocols/learning_transfer/learning_transfer.py.
* And the untrained ear is no longer naive at test time: it was stimulated with
  its own veridical cues throughout training.

Its clean use is DIAGNOSTIC: it is the externalization ceiling. Both ears are
fully natural except for the modified band on one side, so if externalization is
still poor here, the ear reduction is not what is breaking it.

Usage (via the binsim pipeline)::

    hrir_settings = dict(..., ear='left', other_ear='native')   # native_sofa
                                                                # defaults to
                                                                # subject_id
"""

import copy
import logging

import numpy

logger = logging.getLogger(__name__)


def source_index_map(hrir, native, tol=1e-3):
    """Map each source of ``hrir`` onto the matching source of ``native``.

    The modified SOFA is built from the native one, so the grids normally match
    row for row; this checks that rather than assuming it, and falls back to
    matching by position. Raises if any source has no counterpart — silently
    pairing the wrong directions would put one ear's cue at the wrong place.
    """
    a = numpy.asarray(hrir.sources.vertical_polar, dtype=float)
    b = numpy.asarray(native.sources.vertical_polar, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f'source grids differ in size: {a.shape[0]} vs {b.shape[0]} — the '
            f'native SOFA must be the one the modified set was built from')

    if numpy.allclose(a, b, atol=tol):
        return numpy.arange(a.shape[0])

    # azimuth is circular; compare on the wrapped difference
    logger.info('source order differs between the two SOFAs — matching by position')
    index = numpy.empty(a.shape[0], dtype=int)
    for i, (az, el, _r) in enumerate(a):
        d_az = numpy.abs((b[:, 0] - az + 180.0) % 360.0 - 180.0)
        d_el = numpy.abs(b[:, 1] - el)
        j = int(numpy.argmin(numpy.hypot(d_az, d_el)))
        if numpy.hypot(d_az[j], d_el[j]) > 0.5:
            raise ValueError(
                f'no matching native source for (az={az:.1f}, el={el:.1f}) — '
                f'closest is {numpy.hypot(d_az[j], d_el[j]):.2f} deg away')
        index[i] = j
    return index


def native_dtf(hrir, native, ear='left'):
    """Restore the OTHER ear's channel from the native (unmodified) HRIR.

    Same signature convention as ``flatten_dtf`` / ``envelope_dtf``: ``ear``
    names the ear that KEEPS the modified cue (the listening / trained ear); the
    opposite ear is taken from ``native`` untouched, so it carries its own full
    DTF including ITD, ILD and all spectral detail.

    Parameters
    ----------
    hrir : slab.HRTF
        The MODIFIED set (e.g. the ERB-shifted one). Not modified in place.
    native : slab.HRTF
        The subject's unmodified measured HRIR, same source grid and taps.
    ear : {'left', 'right'}, default 'left'
        The ear that keeps the modification.

    Returns
    -------
    slab.HRTF
        Deep copy: ``ear`` from ``hrir``, the other ear from ``native``.
    """
    if ear not in ('left', 'right'):
        raise ValueError(f"ear must be 'left' or 'right', got {ear!r}")

    out = copy.deepcopy(hrir)
    other_idx = 1 if ear == 'left' else 0
    index = source_index_map(hrir, native)

    n_taps = out[0].data.shape[0]
    if native[0].data.shape[0] != n_taps:
        raise ValueError(
            f'tap count differs: modified {n_taps} vs native '
            f'{native[0].data.shape[0]} — cannot splice channels')
    if not numpy.isclose(float(out[0].samplerate), float(native[0].samplerate)):
        raise ValueError('samplerate differs between the modified and native sets')

    logger.debug('Restoring native DTF on the %s ear',
                 'right' if other_idx else 'left')
    for source_idx in range(out.n_sources):
        out[source_idx].data[:, other_idx] = \
            native[index[source_idx]].data[:, other_idx]
    return out


def interaural_cue_conflict_db(composite, native, ear='left', band=(5700, 11300)):
    """How far the two ears now disagree, in dB, inside ``band``.

    With a native other ear the modification survives only as an INTERAURAL
    difference: the listening ear's spectrum has moved inside ``band`` while the
    other ear's has not. This returns the direction-averaged RMS change of the
    interaural log-magnitude difference (composite vs native), i.e. how large a
    spectral-ILD anomaly the manipulation now creates. Zero would mean the
    manipulation is interaurally invisible; large values mean the two ears
    report different elevations in that band.
    """
    eps = numpy.finfo(float).tiny

    def _profile(hrtf_obj):
        rows = []
        for source_idx in range(hrtf_obj.n_sources):
            data = numpy.asarray(hrtf_obj[source_idx].data, dtype=float)
            freqs = numpy.fft.rfftfreq(data.shape[0],
                                       d=1.0 / hrtf_obj[source_idx].samplerate)
            spec = 20.0 * numpy.log10(
                numpy.maximum(numpy.abs(numpy.fft.rfft(data, axis=0)), eps))
            in_band = (freqs >= band[0]) & (freqs <= band[1])
            rows.append((spec[:, 0] - spec[:, 1])[in_band])
        return numpy.array(rows)

    delta = _profile(composite) - _profile(native)
    return float(numpy.mean(numpy.sqrt(numpy.mean(delta ** 2, axis=1))))
