"""
donor_selection.py — pick the donor for a `donor_detail` composite.

Scores every candidate donor by what the LISTENER WOULD ACTUALLY HEAR: the
composite (own n_keep envelope + donor detail) against the listener's own
unmodified HRTF, on the median-plane arc, in Trapeau's peak-VSI octave.

Metrics, all from one pair of correlation matrices
--------------------------------------------------
``vsi_dissimilarity``
    Trapeau et al. (2016): RMS distance between the own-vs-composite
    cross-correlation matrix and the own autocorrelation matrix. In their data
    it predicted both the initial loss of elevation performance (RMSE R=0.52,
    EG R=-0.53, n=30) and how much of that loss was still there after six days
    of adaptation (26% of the variance). THE TARGET IS MID-RANGE, NOT MAXIMUM:
    they found stronger disruption meant less or slower improvement, while Van
    Wanrooij & Van Opstal (2005) found too-similar cues are absorbed as a bias
    and never relearned. Use the between-participant distribution
    (:func:`pairwise_reference`) as the scale — that is exactly the yardstick
    Trapeau used to argue their molds were physiologically plausible.

``vsi``
    Of the composite. Must stay near the listener's own. Trapeau's molds moved
    VSI from 0.76 to 0.81, i.e. they changed the cue without removing
    information. A candidate that lowers VSI is a degraded map, not a new one.

``i_sim`` and ``peak_r``
    Van Wanrooij & Van Opstal's similarity index: for each composite DTF, the SD
    (resp. maximum) of its correlations with all of the listener's own DTFs.
    Deliberately blind to WHERE the match sits, so it answers the question VSI
    dissimilarity cannot — is there still a consistent match the old map can use?

``ridge_slope``
    Where the best match sits, regressed on true elevation. ~+1 means the old
    map reads the composite as a coherent elevation (offset by ``ridge_bias``)
    and can absorb the manipulation as a bias; ~0 means it cannot. This is the
    guard against the one failure mode VSI dissimilarity is blind to: a
    correlation ridge that MOVED and one that is GONE score the same there.
    For real human donors the ridge normally collapses, but check rather than
    assume.

``detail_strength`` (see :func:`detail_strength`)
    Not a pairing score — a property of ONE recording. The SD across directions
    of its own in-band spectral detail, in dB. This is how strong the cue being
    handed over actually is, and it is what ranks the pool and breaks ties
    inside the tolerance band. It is NOT VSI; see the pool comment at the bottom
    of this module for why VSI turned out to be the wrong quantity for this job.

Method
------
Scoring follows the paper: the composite is built on the FFT grid (that is what
the manipulation actually does) and then passed through the Middlebrooks (1999)
filter bank before correlating — triangular filters spaced 0.0286 octaves (2%
frequency steps), 36 points per octave, NOT a gammatone/ERB bank and not linear
FFT bins. ``resolution='fft'`` is kept for comparison but weights the top of the
band far more heavily, because linear bins pile up there.

The one part of their method this rig cannot reproduce is the DTF
normalisation: they divide each transfer function by the grand-average RMS over
48 directions uniformly spaced 22.5 deg apart including the rear field. Only the
az=0 arc is measured here, so use ``normalize='detail'`` for cross-subject work
and ``'none'`` within a subject (see :mod:`hrtf_relearning.hrtf.analysis.vsi`).

Cost: the cepstral split is done once per subject-ear on the median-plane arc
only (~19 spectra), then every pairing is an addition, one filter-bank pass and
two correlation matrices. A full cohort scan is seconds, not minutes. Because
correlation is invariant to a per-direction constant, the energy matching that
:func:`~hrtf_relearning.hrtf.modify.donor_detail.donor_detail_dtf` applies does
not change any of these scores — what is ranked here is what gets delivered.
"""

import logging

import numpy

from hrtf_relearning.hrtf.analysis import vsi as vsi_module
from hrtf_relearning.hrtf.modify.donor_detail import DEFAULT_N_KEEP, split_log_magnitude

logger = logging.getLogger(__name__)

EARS = ('left', 'right')

# VSI dissimilarity correlated with the behavioural effect of the molds in the
# 5.7-11.3 kHz OCTAVE band (Trapeau et al. 2016, Figs 5B/5C). That is a
# different band from the one where plain VSI tracks RMSE (the 5657-8000
# half-octave, vsi.BEHAVIOURAL_BAND) — different metric, different band.
DEFAULT_BAND = (5657, 11314)
DEFAULT_RESOLUTION = 'filterbank'

# ---------------------------------------------------------------------------
# Protocol constants — FIXED ACROSS PARTICIPANTS.
# The donor is the only thing that varies per subject; everything else here is
# the same for everyone, so the manipulation is one sentence in the methods.
# ---------------------------------------------------------------------------
N_KEEP = 4                    # envelope coefficients kept (Kulkarni & Colburn 1998)
TARGET_DISSIMILARITY = 0.40   # aim, in this cohort's between-subject distribution
MAX_RIDGE_SLOPE = 0.5         # above this the old map can read the composite as
                              # a coherent elevation and absorb it as a bias
TOLERANCE = 0.05              # half-width of the band around the target

# WHY THE TARGET MOVED FROM 0.50 TO 0.40 (2026-08-13). The rule has always been
# "the median of this cohort's between-subject distribution" — 0.50 was that
# median when it could only be estimated from 7 listeners (21 pairs: median
# 0.55). Measured over all 20 conforming recordings (190 pairs) the
# distribution is min 0.157 | Q1 0.334 | median 0.403 | Q3 0.526 | max 0.942,
# so 0.50 had drifted to roughly the 72nd percentile — a stronger perturbation
# than intended, in the direction Trapeau found gives slower adaptation. 0.40
# restores the stated rationale rather than changing it. On the 11 listeners
# recorded so far this moves 2/11 in-band picks to 5/11 with no loss of donor
# cue strength (median detail SD 3.0 dB either way).
#
# WHY A TOLERANCE BAND INSTEAD OF PLAIN ARGMIN. With a pool this size,
# min |dissimilarity - target| is decided by differences of 0.01-0.02, which is
# well inside the measurement noise of the metric — it selects on noise. The
# band says "close enough to target" and then picks on a quantity that is
# actually meaningful (donor detail strength). It also gives the in-session
# swap a principled 2nd and 3rd choice instead of an arbitrary one.

# VSI IS REPORTED BUT DOES NOT GATE ANYTHING HERE. Trapeau's VSI is defined on
# diffuse-field-normalised DTFs; these recordings sample only the az=0 arc, so no
# diffuse-field average exists and neither available proxy is that quantity.
# Measured in the 5657-11314 band, ear-averaged, this cohort spreads 0.24-0.93
# (SD 0.23) on raw transfer functions and 0.19-1.01 (SD 0.27) with the
# per-direction envelope removed, against Trapeau's 0.76 +/- 0.02 on 30 properly
# normalised listeners. A quantity that unstable cannot decide eligibility, so
# the rule uses only the ridge slope, which asks a question about correlation
# STRUCTURE rather than about absolute cue strength. (The ridge is not immune to
# the missing normalisation either — a large shared direction-independent
# component pulls every correlation toward 1 and makes the argmax noisier — but
# it biases toward noise, not toward a wrong answer.)


# ---------------------------------------------------------------------------
# correlation helpers (numpy only, so they can be tested without slab)
# ---------------------------------------------------------------------------

def _correlate(a, b):
    """Pearson correlation between every row of ``a`` and every row of ``b``."""
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / numpy.maximum(numpy.linalg.norm(a, axis=1, keepdims=True), 1e-30)
    b = b / numpy.maximum(numpy.linalg.norm(b, axis=1, keepdims=True), 1e-30)
    return a @ b.T


def matrix_metrics(own, candidate, elevations):
    """All five readouts for one ear, from two (n_elevation, n_bin) dB matrices.

    ``own`` is the reference (the listener's unmodified DTFs), ``candidate`` the
    manipulated set. Rows of the cross matrix are own elevations, columns are
    candidate elevations — so column j is "what the old map makes of the
    manipulated DTF for elevation j".
    """
    cross = _correlate(own, candidate)
    auto = _correlate(own, own)
    n = len(own)
    candidate_auto = _correlate(candidate, candidate)

    best = numpy.argmax(cross, axis=0)
    ridge = numpy.asarray(elevations, dtype=float)[best]
    slope, bias = numpy.polyfit(numpy.asarray(elevations, dtype=float), ridge, 1)

    return {
        'vsi': 1.0 - (candidate_auto.sum() - n) / (n * (n - 1)),
        'vsi_dissimilarity': float(numpy.sqrt(numpy.mean((cross - auto) ** 2))),
        'i_sim': float(numpy.mean(numpy.std(cross, axis=0))),
        'peak_r': float(numpy.mean(cross.max(axis=0))),
        'ridge_slope': float(slope),
        'ridge_bias': float(bias),
    }


# ---------------------------------------------------------------------------
# per-HRTF median-plane cache
# ---------------------------------------------------------------------------

def median_plane_split(hrtf, n_keep=DEFAULT_N_KEEP):
    """Envelope / detail / full dB spectra on the median-plane arc, per ear.

    Returns ``{'freqs', 'elevations', 'left': {...}, 'right': {...}}`` where each
    ear holds ``full``, ``envelope`` and ``detail`` as (n_elevation, n_bin) dB.
    Compute this ONCE per subject-ear and reuse it across every pairing — it is
    the only expensive step (one least-squares cosine fit per direction).

    Duplicate elevations are dropped via
    :func:`~hrtf_relearning.hrtf.analysis.vsi.median_plane_sources` — some
    recordings store the az=0 arc twice.
    """
    freqs, _ = hrtf[0].tf(show=False)
    sources = vsi_module.median_plane_sources(hrtf)
    elevations = numpy.asarray(hrtf.sources.vertical_polar[sources, 1], dtype=float)

    out = {'freqs': numpy.asarray(freqs, dtype=float), 'elevations': elevations}
    for ear in EARS:
        # full spectrum, unnormalised dB: the split IS the normalisation here
        full = vsi_module.dtf_matrix(hrtf, ear, sources=sources,
                                     bandwidth=(freqs[0], freqs[-1]),
                                     normalize='none')
        mag = 10.0 ** (numpy.asarray(full, dtype=float).T / 20.0)   # (n_bin, n_dir)
        envelope_db, detail_db = split_log_magnitude(mag, n_keep=n_keep)
        out[ear] = {'full': numpy.asarray(full, dtype=float),
                    'envelope': envelope_db.T, 'detail': detail_db.T}
    return out


def _reduce(matrix, freqs, bandwidth, resolution=DEFAULT_RESOLUTION,
            octave_spacing=vsi_module.OCTAVE_SPACING):
    """Band-limit a full-spectrum dB matrix, the paper's way by default.

    ``'filterbank'`` runs the Middlebrooks (1999) 2%-spaced triangular bank
    (36 points/octave, even in log frequency); ``'fft'`` just slices the linear
    bins, which over-weights the top of the band.
    """
    if resolution == 'filterbank':
        return vsi_module.filterbank_levels(freqs, matrix, bandwidth, octave_spacing)
    if resolution != 'fft':
        raise ValueError(f"resolution must be 'filterbank' or 'fft', got {resolution!r}")
    idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    return matrix[:, idx]


def score_pair(own_split, donor_split, bandwidth=DEFAULT_BAND, donor_ear=None,
               resolution=DEFAULT_RESOLUTION):
    """Score the composite built from ``own_split`` and ``donor_split``.

    Returns the ear-averaged metrics plus the per-ear values. ``donor_ear``
    ``None`` takes the donor's matching side for each of the listener's ears.
    """
    freqs = own_split['freqs']
    elevations = own_split['elevations']
    if len(donor_split['elevations']) != len(elevations):
        raise ValueError(
            f"median-plane grids differ: {len(elevations)} vs "
            f"{len(donor_split['elevations'])} elevations")

    per_ear = {}
    for ear in EARS:
        source_ear = ear if donor_ear is None else donor_ear
        composite = own_split[ear]['envelope'] + donor_split[source_ear]['detail']
        per_ear[ear] = matrix_metrics(
            _reduce(own_split[ear]['full'], freqs, bandwidth, resolution),
            _reduce(composite, freqs, bandwidth, resolution),
            elevations)

    keys = per_ear['left'].keys()
    scores = {k: float(numpy.mean([per_ear[e][k] for e in EARS])) for k in keys}
    scores['per_ear'] = per_ear
    return scores


# ---------------------------------------------------------------------------
# cohort scan
# ---------------------------------------------------------------------------

def rank_donors(subject_hrtf, donors, n_keep=DEFAULT_N_KEEP,
                bandwidth=DEFAULT_BAND, donor_ear=None,
                resolution=DEFAULT_RESOLUTION):
    """Rank candidate donors for one listener.

    Parameters
    ----------
    subject_hrtf : slab.HRTF
        The listener's own unmodified HRIR — the reference every candidate is
        scored against.
    donors : dict of {name: slab.HRTF}
        Candidates. Anything with a matching median-plane grid.
    donor_ear : {'left', 'right', None}
        Which of the donor's ears supplies the detail. ``None`` = same side.

    Returns
    -------
    list of dict
        One row per donor, sorted by ``vsi_dissimilarity`` ASCENDING. Read it
        with the target in mind: the useful candidates are mid-distribution,
        not the extremes (see the module docstring).
    """
    own_split = median_plane_split(subject_hrtf, n_keep=n_keep)
    rows = []
    for name, donor in donors.items():
        try:
            donor_split = median_plane_split(donor, n_keep=n_keep)
            scores = score_pair(own_split, donor_split, bandwidth=bandwidth,
                                donor_ear=donor_ear, resolution=resolution)
            strength = detail_strength(donor_split, bandwidth, resolution)
        except Exception as exc:
            logger.warning('skipping donor %s: %s', name, exc)
            continue
        scores.pop('per_ear')
        rows.append({'donor': name, 'donor_strength': strength, **scores})
    return sorted(rows, key=lambda row: row['vsi_dissimilarity'])


def pairwise_reference(hrtfs, n_keep=DEFAULT_N_KEEP, bandwidth=DEFAULT_BAND,
                       resolution=DEFAULT_RESOLUTION):
    """VSI dissimilarity between every pair of listeners' own (unmodified) HRTFs.

    This is Trapeau's plausibility yardstick: they argued their molds were
    physiologically reasonable because free-vs-mold dissimilarity fell inside
    this distribution. Use it to place a candidate composite — a donor near the
    median of this spread is a typical between-person difference, which is the
    magnitude their six-day adaptation was measured at.

    Returns ``(values, {(a, b): value})``.
    """
    splits = {name: median_plane_split(hrtf, n_keep=n_keep)
              for name, hrtf in hrtfs.items()}
    names = list(splits)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            try:
                per_ear = []
                for ear in EARS:
                    per_ear.append(matrix_metrics(
                        _reduce(splits[a][ear]['full'], splits[a]['freqs'],
                                bandwidth, resolution),
                        _reduce(splits[b][ear]['full'], splits[b]['freqs'],
                                bandwidth, resolution),
                        splits[a]['elevations'])['vsi_dissimilarity'])
                pairs[(a, b)] = float(numpy.mean(per_ear))
            except Exception as exc:
                logger.warning('skipping pair %s/%s: %s', a, b, exc)
    return numpy.array(sorted(pairs.values())), pairs


# ---------------------------------------------------------------------------
# Van Wanrooij & Van Opstal (2005) I_sim — J Neurosci 25(22):5413-5424
#
# THE REASON THIS IS HERE. Trapeau's VSI needs DTFs normalised by a grand
# average over directions distributed AROUND the listener, which this rig
# cannot produce: only the az=0 arc is measured. Van Opstal's DTF (their Eq. 1)
# is normalised by the grand average across the elevations OF THE ARC, which is
# exactly what is measured here. So this is the published measure whose
# normalisation is actually available with this geometry, and it comes from the
# monaural perturbation study this experiment is modelled on.
#
# Their band is 4-20 kHz. Use 4-18 kHz here: the recordings are equalised
# against the reference only up to 18 kHz (record_hrir, inversion_range_hz).
#
# ANCHORS, and the scale correction they imply. Their Fig 7A gives I_sim = 0.5
# for a normal listener's own left ear against their own right, and Fig 7C
# gives 0.3 for their molds. Measured here with the same definition: own left
# vs own right = 0.37 (n=31, range 0.18-0.51), between listeners same ear =
# 0.25 (465 pairs). Both come out ~0.74x their values, which is expected --
# I_sim is an SD across correlations and therefore scales with the elevation
# sampling (19 elevations over +-37.5 deg here against their 25 over a wider
# arc). Because the ratio is the same at both anchors, their numbers can be
# carried over by scaling rather than used raw. DO NOT compare a raw I_sim
# computed here against 0.3 or 0.5 without that correction.
# ---------------------------------------------------------------------------

VANOPSTAL_BAND = (4000.0, 18000.0)


def vanopstal_dtfs(hrtf, bandwidth=VANOPSTAL_BAND):
    """DTFs per their Eq. 1: spectrum / grand average across the arc elevations.

    Returns ``({'left': (n_el, n_bin) dB, 'right': ...}, elevations)``. The
    normalisation is the whole point -- it removes the component common to all
    directions (microphone, canal, any residual room), which is what makes a
    correlation between two DTF sets interpretable. Without it, correlations
    are dominated by that shared component.
    """
    sources = vsi_module.median_plane_sources(hrtf)
    elevations = numpy.asarray(hrtf.sources.vertical_polar[sources, 1], dtype=float)
    n_taps = hrtf[0].data.shape[0]
    freqs = numpy.fft.rfftfreq(n_taps, 1.0 / float(hrtf[0].samplerate))
    in_band = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])

    out = {}
    for channel, ear in ((0, 'left'), (1, 'right')):
        magnitude = numpy.stack([
            numpy.abs(numpy.fft.rfft(
                numpy.asarray(hrtf[index].data, dtype=float)[:, channel], n_taps))
            for index in sources])
        grand = magnitude.mean(axis=0, keepdims=True)
        ratio = magnitude / numpy.maximum(grand, 1e-20)
        out[ear] = 20.0 * numpy.log10(numpy.maximum(ratio, 1e-12))[:, in_band]
    return out, elevations


def isim(own_dtfs, candidate_dtfs):
    """Their similarity index, for one ear.

    ``C(e1, e2)`` is the correlation between the own DTF at ``e1`` and the
    candidate DTF at ``e2``. For each candidate elevation, take the SD of its
    correlations with all own DTFs; ``I_sim`` is the mean of those SDs.

    High means the candidate still discriminates elevation the way the
    listener's own ear does (correlation concentrated on the diagonal); low
    means no elevation is well matched by any. Deliberately blind to WHERE the
    match sits, so it does not distinguish a map that moved from one that
    vanished -- pair it with ``matrix_metrics(...)['ridge_slope']`` if that
    distinction matters.

    In their data I_sim predicted acute pre-adaptation elevation gain with
    r^2 = 0.94 (n=6), which is why it is the natural quantity to target: it is
    a proxy for the acute degradation the manipulation is supposed to produce.
    """
    return float(numpy.mean(numpy.std(_correlate(own_dtfs, candidate_dtfs), axis=0)))


def composite_isim(own_hrtf, donor_hrtf, bandwidth=VANOPSTAL_BAND):
    """I_sim of the own-envelope + donor-detail composite against the own ears.

    Ear-averaged. Because the cepstral split is linear in log magnitude, the
    composite's detail is the donor's detail, so this is very close to scoring
    the donor directly; it is computed on the actual composite anyway so that
    what is reported is what is delivered.
    """
    own, _ = vanopstal_dtfs(own_hrtf, bandwidth)
    donor, _ = vanopstal_dtfs(donor_hrtf, bandwidth)
    values = []
    for ear in EARS:
        # own envelope + donor detail, in the normalised dB domain
        own_env, _ = split_log_magnitude(10.0 ** (own[ear].T / 20.0), n_keep=N_KEEP)
        _, donor_det = split_log_magnitude(10.0 ** (donor[ear].T / 20.0), n_keep=N_KEEP)
        values.append(isim(own[ear], (own_env + donor_det).T))
    return float(numpy.mean(values))


def detail_strength(split, bandwidth=DEFAULT_BAND, resolution=DEFAULT_RESOLUTION):
    """How strong ONE recording's own elevation cue is, in dB. Ear-averaged.

    The SD across directions of the in-band spectral DETAIL — i.e. of exactly
    the quantity :func:`~hrtf_relearning.hrtf.modify.donor_detail.donor_detail_dtf`
    transplants. High means the donor's fine structure both is deep AND changes
    with elevation; a donor scoring low is handing over almost nothing, which is
    a cue REMOVAL dressed up as a cue replacement.

    Two components have to be there and only their product is a cue:
    depth alone is not enough (a deep but elevation-INDEPENDENT ripple is
    timbre, not a cue — pilot/AGV has 6.7 dB of detail and 2.1 dB of SD), and
    elevation dependence of a shallow spectrum is not enough either. Taking the
    SD across directions of the detail requires both.

    Computed on the per-direction detail, so unlike VSI it does not depend on
    the diffuse-field normalisation this rig cannot produce — the envelope,
    which is where the un-normalised common component lives, has already been
    subtracted direction by direction.
    """
    values = []
    for ear in EARS:
        detail = _reduce(split[ear]['detail'], split['freqs'], bandwidth, resolution)
        values.append(float(numpy.mean(numpy.std(numpy.asarray(detail, dtype=float),
                                                 axis=0))))
    return float(numpy.mean(values))


def own_vsi(split, bandwidth=DEFAULT_BAND, resolution=DEFAULT_RESOLUTION):
    """VSI of an unmodified set, in the same band/resolution as the scores."""
    values = [matrix_metrics(_reduce(split[ear]['full'], split['freqs'], bandwidth, resolution),
                             _reduce(split[ear]['full'], split['freqs'], bandwidth, resolution),
                             split['elevations'])['vsi'] for ear in EARS]
    return float(numpy.mean(values))


def shortlist(subject_hrtf, candidates, target=TARGET_DISSIMILARITY,
              n_keep=N_KEEP, bandwidth=DEFAULT_BAND,
              resolution=DEFAULT_RESOLUTION, donor_ear=None,
              max_ridge_slope=MAX_RIDGE_SLOPE, tolerance=TOLERANCE):
    """Every candidate, ranked best-first by the selection rule.

    ``shortlist()[0]`` is the donor :func:`select_donor` returns; ``[1]`` and
    ``[2]`` are the principled 2nd and 3rd choices the in-session swap uses when
    a participant cannot localize at all with the first one. The ordering is a
    RULE, not a preference, so a swap stays reportable: it moves down a list
    that was fixed before the participant heard anything.

    Three tiers, in order:

    ``'band'``
        Ridge collapsed (``ridge_slope <= max_ridge_slope``) AND dissimilarity
        within ``tolerance`` of ``target``. Ranked by ``donor_strength``
        DESCENDING — among donors that are all equally close to the intended
        perturbation size, hand over the strongest cue.
    ``'widened'``
        Ridge collapsed but outside the band. Ranked by distance to target
        ascending. This is the old rule, and it is what runs when the band is
        empty.
    ``'fallback'``
        Ridge did not collapse for any candidate. Ranked by ridge slope
        ascending. The composite may be partly absorbable as a bias; report it.

    Every row carries ``tier``, ``rank``, ``distance``, ``eligible``
    (ridge criterion), ``in_band``, ``donor_strength`` and the listener's
    ``own_vsi``, so the whole table can go in a supplement.
    """
    own_split = median_plane_split(subject_hrtf, n_keep=n_keep)
    reference_vsi = own_vsi(own_split, bandwidth, resolution)

    rows = rank_donors(subject_hrtf, candidates, n_keep=n_keep,
                       bandwidth=bandwidth, donor_ear=donor_ear,
                       resolution=resolution)
    if not rows:
        raise ValueError('no candidate donor could be scored')
    for row in rows:
        row['own_vsi'] = reference_vsi
        row['distance'] = abs(row['vsi_dissimilarity'] - target)
        row['eligible'] = bool(row['ridge_slope'] <= max_ridge_slope)
        row['in_band'] = bool(row['eligible'] and row['distance'] <= tolerance)

    band = [row for row in rows if row['in_band']]
    ridge_only = [row for row in rows if row['eligible'] and not row['in_band']]
    rejected = [row for row in rows if not row['eligible']]

    if band:
        ordered = (sorted(band, key=lambda r: -r['donor_strength'])
                   + sorted(ridge_only, key=lambda r: r['distance'])
                   + sorted(rejected, key=lambda r: r['ridge_slope']))
        tier = 'band'
    elif ridge_only:
        ordered = (sorted(ridge_only, key=lambda r: r['distance'])
                   + sorted(rejected, key=lambda r: r['ridge_slope']))
        tier = 'widened'
        logger.warning(
            'no candidate landed within %.2f of target %.2f — falling back to '
            'nearest-to-target among ridge-eligible donors', tolerance, target)
    else:
        ordered = sorted(rejected, key=lambda r: r['ridge_slope'])
        tier = 'fallback'
        logger.warning(
            'no candidate reached ridge_slope <= %.2f (best %.2f, donor %s) — '
            'falling back to the lowest slope', max_ridge_slope,
            ordered[0]['ridge_slope'], ordered[0]['donor'])

    for rank, row in enumerate(ordered):
        row['rank'] = rank
        # the tier the CHOSEN donor sits in, so one field says how the pick was
        # made; per-row membership is still readable from eligible/in_band
        row['tier'] = tier
        row['fallback'] = bool(tier == 'fallback')
    return ordered


def select_donor(subject_hrtf, candidates, target=TARGET_DISSIMILARITY,
                 n_keep=N_KEEP, bandwidth=DEFAULT_BAND,
                 resolution=DEFAULT_RESOLUTION, donor_ear=None,
                 max_ridge_slope=MAX_RIDGE_SLOPE, tolerance=TOLERANCE,
                 rank=0):
    """The per-subject donor choice — the ONLY thing that varies between subjects.

    Thin wrapper over :func:`shortlist`: returns ``(chosen_row, all_rows)`` with
    ``all_rows`` already in rule order. ``rank=0`` is the protocol choice;
    ``rank=1``/``rank=2`` are the documented alternates for an in-session swap
    (see :func:`shortlist` for what defines the order).

    Note ``vsi`` is diagnostic only — see the constants block above for why it
    does not gate anything, and the pool comment for why ``donor_strength``
    replaced it as the ranking quantity.
    """
    rows = shortlist(subject_hrtf, candidates, target=target, n_keep=n_keep,
                     bandwidth=bandwidth, resolution=resolution,
                     donor_ear=donor_ear, max_ridge_slope=max_ridge_slope,
                     tolerance=tolerance)
    if rank >= len(rows):
        raise IndexError(f'requested donor rank {rank} but only {len(rows)} '
                         f'candidates were scored')
    return rows[rank], rows


# ---------------------------------------------------------------------------
# The donor pool — EDIT THIS LIST.
#
# The target grid is 475 sources, 512 taps, 48828 Hz, 19 median-plane
# elevations from -38 to +38 deg. 30 of the 38 recordings on disk reach it:
# 20 natively, and 10 more through `conform_recording` (zero-padding 256 taps
# to 512, dropping the second az=0 arc where it was measured twice). The
# remaining 8 have a genuinely coarser source grid and would need spatial
# interpolation, which smooths the notches this manipulation transplants, so
# they stay out. Use 'pilot/NAME' to force the pilot copy when an id exists in
# both folders (AS, PF do).
#
# RANKED BY DETAIL STRENGTH, NOT BY VSI (changed 2026-08-13). The pool used to
# be the 8 highest-VSI recordings. That was wrong: measured over all 20, VSI is
# NEGATIVELY rank-correlated with how deep the spectral detail is (Spearman
# -0.40). AS tops the VSI table at 0.93 and has the SHALLOWEST detail of all 20
# (3.6 dB); GS scores 0.29 — 16th — on the deepest detail in the set (7.1 dB)
# and the 4th most elevation-dependent, and localizes well with her own HRTF.
# The reason is the one already documented above for why VSI cannot gate
# eligibility: without the diffuse-field normalisation this rig cannot produce,
# a large shared across-direction component drags every correlation toward 1,
# so a listener with a strong common resonance reads as "all my spectra look
# alike" no matter how much fine structure rides on top. `detail_strength`
# is computed after the per-direction envelope has been subtracted, so it does
# not have that failure mode — and it measures the exact quantity the donor
# hands over. VSI is still reported everywhere; it just no longer ranks anything.
#
# SIZE IS NOT THE BINDING CONSTRAINT — the ridge guard is. Admitting every
# recording raises in-band picks from 5/11 to 8/11, but only by selecting IM,
# AGV, LS and JF, whose detail SD is 1.6-2.2 dB against 2.7-3.9 for the pool.
# Those composites hand over almost nothing, i.e. they are a cue REMOVAL
# dressed up as a cue replacement. So the pool is a STRENGTH FLOOR (>= 2.66 dB,
# the weakest recording that was already in it), not a rank cut. Adding the
# five padded recordings that clear that floor takes the pool from 12 to 17 and
# buys one fewer fallback (2 -> 1) plus a much stronger donor for GLK and IR
# (MB at 3.9 dB, against 2.7 and 3.2 before) at no cost to the floor.
#
# DO NOT READ THE ORDER BELOW AS MEANINGFUL AT RANK GRANULARITY. Adjacent
# ranks differ by ~0.05 dB and THERE IS NO REPEATABILITY ESTIMATE to say
# whether that is a real difference: every subject's arc is measured once.
# (An earlier note here claimed a 0.42 dB test-retest from the six recordings
# that carry the arc twice. That was wrong — the second arc is the azimuth
# expansion's own processed copy, not a repeat measurement, so 0.42 dB is the
# fidelity of that pipeline step, not measurement noise. See
# `conform_recording`.) Until someone re-records a subject, treat only broad
# differences as real — JF at 1.6 vs FD at 3.7 is a distinction, rank 12 vs
# rank 13 is not. Hence a floor rather than a top-N.
#
# Elevation SD of in-band detail, dB, ear-averaged (`detail_strength`).
# '*' = reaches the grid only via conform_recording (was 256-tap):
#   MB* 3.9   FD  3.7   NKa 3.6   MSc 3.5   GS  3.3   SK* 3.3
#   CO  3.3   TS  3.3   AS  3.2   GLK 3.2   JP* 3.2   VD  3.0
#   PF  3.0   NK* 3.0   RK* 2.7   AH  2.7   FS  2.7
#   --------------------------------------------- floor 2.66 dB
#   SS  2.4   IR  2.3   UG  2.3   SW* 2.3   IM  2.2   PC  2.2
#   PFo* 2.2  LS  2.1   AGV 2.1   JR* 2.1   VG* 2.0   CZ* 1.7   JF  1.6
# ---------------------------------------------------------------------------
DONOR_POOL = ('pilot/MB', 'FD', 'pilot/NKa', 'pilot/MSc', 'GS', 'pilot/SK',
              'CO', 'TS', 'AS', 'GLK', 'pilot/JP', 'pilot/VD', 'PF',
              'pilot/NK', 'pilot/RK', 'pilot/AH', 'FS')

# what a recording must match to be usable without resampling
REQUIRED_TAPS = 512
REQUIRED_SAMPLERATE = 48828.0
REQUIRED_ELEVATIONS = 19


def conforms(hrtf, taps=REQUIRED_TAPS, samplerate=REQUIRED_SAMPLERATE,
             elevations=REQUIRED_ELEVATIONS):
    """True if a recording matches the current measurement setup.

    Composites are built sample-by-sample against the listener's own HRIR, so a
    donor with a different tap count, sample rate or median-plane grid cannot be
    used as it stands. Run :func:`conform_recording` first — it fixes the two
    differences that are lossless to fix (short tap count, duplicated az=0 arc)
    and leaves everything else to fail here.
    """
    try:
        return bool(hrtf[0].data.shape[0] == taps
                    and abs(float(hrtf[0].samplerate) - samplerate) < 50
                    and len(vsi_module.median_plane_sources(hrtf)) == elevations)
    except Exception:
        return False


def conform_recording(hrtf, taps=REQUIRED_TAPS, samplerate=REQUIRED_SAMPLERATE):
    """Bring a recording onto the current grid where that is LOSSLESS, else None.

    Two differences among the recordings on disk can be repaired exactly, and
    only those two are attempted:

    ``256 taps instead of 512``
        Zero-padded. Padding an impulse response interpolates its spectrum onto
        a finer frequency grid; it invents nothing. Whether 256 taps RESOLVE the
        detail is a separate question, and it was measured: truncating the
        512-tap recordings to 256 and padding back moves detail strength and
        notch depth by <= 0.01 dB (against a 2.7-3.7 dB spread across donors)
        and VSI dissimilarity by <= 0.001 (against a +-0.05 tolerance band).
        99.98% of the energy is in the first 256 samples and 191 Hz bin spacing
        is far finer than the in-band notches. The one quantity that does move
        is ``ridge_slope``, by up to 0.087 — it is an argmax statistic, so it
        steps rather than drifts. A candidate sitting within ~0.09 of the
        MAX_RIDGE_SLOPE cutoff can therefore flip on tap count alone; treat
        those as ties rather than as decisions.

    ``494 sources instead of 475``
        These files carry the az=0 arc twice, at 0 and at 360. The two copies
        are NOT two measurements. Only the median-plane arc is ever recorded;
        `record_hrir.expand_azimuths_with_binaural_cues` generates every other
        azimuth from it, and older versions of that function did not exclude
        360, so the second arc is the pipeline's own processed copy of the
        first. Current recordings have 475 sources because that function now
        skips 0/360 explicitly.

        The FIRST occurrence is kept, i.e. the measured arc, and the generated
        copy is discarded. That is deliberate: the generated one has been
        through the spherical-head ITD step, which time-shifts the RIGHT ear —
        measured against the arc it was derived from, the right ear moves by up
        to 11 samples and correlates at r 0.72-0.94, while the left ear stays
        at r 0.86-0.98 with no onset shift. Verified: after dedup the source
        set matches the 475-source grid exactly.

    Anything else — a different sample rate, a coarser source grid, a shorter
    median-plane arc — is NOT repaired, because the only repair available is
    interpolation, and interpolating an HRTF across direction smooths precisely
    the notches this manipulation transplants. Those return ``None``.
    """
    import slab

    if abs(float(hrtf[0].samplerate) - samplerate) >= 50:
        return None

    sources = numpy.asarray(hrtf.sources.vertical_polar, dtype=float)
    n_taps = hrtf[0].data.shape[0]
    if n_taps > taps:
        return None

    # duplicate rows: same (azimuth mod 360, elevation) to 0.1 deg
    key = numpy.stack([numpy.mod(numpy.round(sources[:, 0], 1), 360.0),
                       numpy.round(sources[:, 1], 1)], axis=1)
    seen, keep = set(), []
    for index, row in enumerate(map(tuple, key)):
        if row in seen:
            continue
        seen.add(row)
        keep.append(index)

    if len(keep) == hrtf.n_sources and n_taps == taps:
        return hrtf

    data = numpy.zeros((len(keep), 2, taps), dtype=float)
    for new_index, old_index in enumerate(keep):
        ir = numpy.asarray(hrtf[old_index].data, dtype=float)   # (taps, 2)
        data[new_index, :, :n_taps] = ir.T

    conformed = slab.HRTF(data, datatype='FIR', samplerate=hrtf[0].samplerate,
                          sources=sources[keep], listener=hrtf.listener)
    conformed.name = getattr(hrtf, 'name', None)
    logger.info('conformed %s: %d sources x %d taps -> %d x %d',
                conformed.name, hrtf.n_sources, n_taps, len(keep), taps)
    return conformed


def load_candidates(subject_id, pool=DONOR_POOL, sofa_dir=None, suffix='',
                    check_conformance=True):
    """Load the donor pool as ``{id: slab.HRTF}``.

    ``pool`` names the candidates explicitly — that is the point, the pool is a
    curated list rather than "whoever happens to be on disk". Names resolve
    against the participant folders first and the ``pilot`` folder second;
    prefix with ``pilot/`` to force the pilot copy. ``pool=None`` falls back to
    every conforming recording in both folders, which is useful for building the
    candidate gallery but not for running a subject.

    The listener's own id is dropped if it appears. Non-conforming recordings
    are skipped with a warning rather than silently.
    """
    import slab
    from hrtf_relearning.utils import paths

    sofa_dir = paths.SOFA_DIR if sofa_dir is None else sofa_dir
    pilot_dir = sofa_dir / 'pilot'

    def _path(name):
        if name.startswith('pilot/'):
            stem = name.split('/', 1)[1]
            return pilot_dir / stem / f'{stem}{suffix}.sofa'
        for base in (sofa_dir, pilot_dir):
            candidate = base / name / f'{name}{suffix}.sofa'
            if candidate.exists():
                return candidate
        return sofa_dir / name / f'{name}{suffix}.sofa'

    if pool is None:
        pool = []
        for base, prefix in ((sofa_dir, ''), (pilot_dir, 'pilot/')):
            if not base.exists():
                continue
            for folder in sorted(p for p in base.iterdir() if p.is_dir()):
                if folder.name in ('database', 'pilot'):
                    continue
                if (folder / f'{folder.name}{suffix}.sofa').exists():
                    pool.append(f'{prefix}{folder.name}')

    candidates = {}
    for name in pool:
        key = name.split('/')[-1]
        if key == subject_id or name == subject_id:
            continue
        path = _path(name)
        if not path.exists():
            logger.warning('donor %s not found at %s', name, path)
            continue
        try:
            hrtf = slab.HRTF(str(path))
        except Exception as exc:
            logger.warning('could not load donor %s: %s', name, exc)
            continue
        hrtf.name = name
        if check_conformance:
            hrtf = conform_recording(hrtf)
            if hrtf is None or not conforms(hrtf):
                logger.warning(
                    'donor %s does not match the current setup and cannot be '
                    'conformed losslessly — skipped', name)
                continue
        candidates[name] = hrtf
    return candidates


def report(rows, reference=None):
    """Print a donor ranking, with the between-subject distribution for scale.

    Rows are expected in :func:`shortlist` order, so the top line is the pick
    and the next two are the alternates an in-session swap would move to.
    """
    tier = rows[0].get('tier') if rows else None
    if tier == 'band':
        print(f'selection tier: BAND — within {TOLERANCE:.2f} of target '
              f'{TARGET_DISSIMILARITY:.2f}, ranked by donor cue strength')
    elif tier == 'widened':
        print(f'selection tier: WIDENED — nothing within {TOLERANCE:.2f} of '
              f'target {TARGET_DISSIMILARITY:.2f}; ranked by distance to target')
    elif tier == 'fallback':
        print(f'selection tier: FALLBACK — no candidate collapsed the ridge '
              f'(<= {MAX_RIDGE_SLOPE:.2f}); ranked by ridge slope. REPORT THIS.')
    print(f'{"":>4}{"donor":>14}  {"VSI-dis":>8} {"strength":>8} {"VSI":>6} '
          f'{"I_sim":>7} {"peak r":>7} {"ridge":>7} {"bias":>7}  {"":>5}')
    for row in rows:
        if 'eligible' not in row:
            mark = ''
        elif row.get('in_band'):
            mark = 'band'
        elif row['eligible']:
            mark = 'ok'
        else:
            mark = 'x'
        arrow = '-->' if row.get('rank') == 0 else (
            f'{row["rank"]}.' if 'rank' in row else '')
        print(f'{arrow:>4}{row["donor"]:>14}  {row["vsi_dissimilarity"]:8.3f} '
              f'{row.get("donor_strength", float("nan")):8.1f} {row["vsi"]:6.2f} '
              f'{row["i_sim"]:7.3f} {row["peak_r"]:7.2f} {row["ridge_slope"]:+7.2f} '
              f'{row["ridge_bias"]:+7.1f}  {mark:>5}')
    if reference is not None and len(reference):
        quartiles = numpy.percentile(reference, [25, 50, 75])
        print(f'\nbetween-subject VSI dissimilarity (n={len(reference)} pairs): '
              f'min {reference.min():.3f} | Q1 {quartiles[0]:.3f} | '
              f'median {quartiles[1]:.3f} | Q3 {quartiles[2]:.3f} | '
              f'max {reference.max():.3f}')
        print('target a donor near the middle of that spread — Trapeau found '
              'stronger disruption gave slower adaptation, Van Wanrooij found '
              'weaker disruption gave none')
