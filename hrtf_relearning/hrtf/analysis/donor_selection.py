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
TARGET_DISSIMILARITY = 0.50   # aim, in this cohort's between-subject distribution
MAX_RIDGE_SLOPE = 0.5         # above this the old map can read the composite as
                              # a coherent elevation and absorb it as a bias

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
        except Exception as exc:
            logger.warning('skipping donor %s: %s', name, exc)
            continue
        scores.pop('per_ear')
        rows.append({'donor': name, **scores})
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


def own_vsi(split, bandwidth=DEFAULT_BAND, resolution=DEFAULT_RESOLUTION):
    """VSI of an unmodified set, in the same band/resolution as the scores."""
    values = [matrix_metrics(_reduce(split[ear]['full'], split['freqs'], bandwidth, resolution),
                             _reduce(split[ear]['full'], split['freqs'], bandwidth, resolution),
                             split['elevations'])['vsi'] for ear in EARS]
    return float(numpy.mean(values))


def select_donor(subject_hrtf, candidates, target=TARGET_DISSIMILARITY,
                 n_keep=N_KEEP, bandwidth=DEFAULT_BAND,
                 resolution=DEFAULT_RESOLUTION, donor_ear=None,
                 max_ridge_slope=MAX_RIDGE_SLOPE):
    """The per-subject donor choice — the ONLY thing that varies between subjects.

    Every candidate composite (listener's envelope + that donor's detail, same
    fixed ``n_keep``) is scored against the listener's own HRTF. A candidate is
    eligible if its correlation ridge has collapsed
    (``ridge_slope <= max_ridge_slope``), i.e. the listener's existing map can no
    longer read the composite as a coherent elevation and absorb the change as a
    constant bias. Among eligible candidates the one whose VSI dissimilarity is
    closest to ``target`` wins.

    If nothing is eligible the candidate with the LOWEST ridge slope is returned
    with ``fallback=True`` rather than raising — the 0.5 cutoff has no external
    anchor, so treating 0.49 and 0.51 as categorically different would be false
    precision. Check ``fallback`` and report it.

    Returns ``(chosen_row, all_rows)``; every row carries ``distance``,
    ``eligible`` and the listener's ``own_vsi`` so the full table can go in a
    supplement. Note ``vsi`` is diagnostic only — see the constants block above
    for why it does not gate anything.
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
        row['fallback'] = False

    eligible = [row for row in rows if row['eligible']]
    if eligible:
        return min(eligible, key=lambda row: row['distance']), rows
    chosen = min(rows, key=lambda row: row['ridge_slope'])
    chosen['fallback'] = True
    logger.warning(
        'no candidate reached ridge_slope <= %.2f (best %.2f, donor %s) — '
        'falling back to the lowest slope', max_ridge_slope,
        chosen['ridge_slope'], chosen['donor'])
    return chosen, rows


# ---------------------------------------------------------------------------
# The donor pool — EDIT THIS LIST.
#
# Only recordings made with the current setup can be used as they stand: 475
# sources, 512 taps, 48828 Hz, 19 median-plane elevations from -38 to +38 deg.
# 16 of the 32 available recordings qualify (the rest are 256-tap or coarser
# grids and would need resampling/interpolation first). Use 'pilot/NAME' to
# force the pilot copy when an id exists in both folders (AS, PF do).
#
# SIZE MATTERS as much as quality. The ridge guard rejects most candidates for
# any given listener, so a small pool leaves the rule no choice: measured on the
# seven current listeners, a pool of the 4 best gives an eligible donor for only
# 4 of them and forces one (CO) onto a composite at 0.95, above the maximum of
# any real subject pair. A pool of 8 resolves all 7 with every pick inside the
# between-subject range (0.37-0.68). Going to all 16 also resolves, but then the
# rule starts selecting donors whose own cue is visibly weak (IM, AGV), which is
# the thing the pool is there to prevent. Eight is the compromise.
#
# Per-ear VSI of the qualifying 16 (left/right, 5657-11314 Hz, filter bank):
#   AS  .89/.97   PF  .77/.81   NKa .59/.63   GS  .61/.58   <- pool
#   AH  .60/.43   FD  .56/.42   VD  .39/.65   CO  .65/.29   <- pool
#   PC  .45/.41   SS  .39/.48   MSc .28/.39   UG  .26/.35
#   JF  .24/.24   LS  .18/.19   IM  .11/.24   AGV .11/.12
# ---------------------------------------------------------------------------
DONOR_POOL = ('AS', 'PF', 'pilot/NKa', 'GS', 'pilot/AH', 'FD', 'pilot/VD', 'CO')

# what a recording must match to be usable without resampling
REQUIRED_TAPS = 512
REQUIRED_SAMPLERATE = 48828.0
REQUIRED_ELEVATIONS = 19


def conforms(hrtf, taps=REQUIRED_TAPS, samplerate=REQUIRED_SAMPLERATE,
             elevations=REQUIRED_ELEVATIONS):
    """True if a recording matches the current measurement setup.

    Composites are built sample-by-sample against the listener's own HRIR, so a
    donor with a different tap count, sample rate or median-plane grid cannot be
    used as it stands — it would need resampling and elevation interpolation
    first. Checked explicitly rather than left to fail deep in the scoring.
    """
    try:
        return bool(hrtf[0].data.shape[0] == taps
                    and abs(float(hrtf[0].samplerate) - samplerate) < 50
                    and len(vsi_module.median_plane_sources(hrtf)) == elevations)
    except Exception:
        return False


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
        if check_conformance and not conforms(hrtf):
            logger.warning(
                'donor %s does not match the current setup (%d taps, %.0f Hz, '
                '%d median-plane elevations) — skipped', name,
                hrtf[0].data.shape[0], float(hrtf[0].samplerate),
                len(vsi_module.median_plane_sources(hrtf)))
            continue
        candidates[name] = hrtf
    return candidates


def report(rows, reference=None):
    """Print a donor ranking, with the between-subject distribution for scale."""
    print(f'{"donor":>14}  {"VSI-dis":>8} {"VSI":>6} {"I_sim":>7} {"peak r":>7} '
          f'{"ridge":>7} {"bias":>7}  {"":>3}')
    for row in rows:
        mark = ('' if 'eligible' not in row
                else ('ok ' if row['eligible'] else 'x  '))
        print(f'{row["donor"]:>14}  {row["vsi_dissimilarity"]:8.3f} {row["vsi"]:6.2f} '
              f'{row["i_sim"]:7.3f} {row["peak_r"]:7.2f} {row["ridge_slope"]:+7.2f} '
              f'{row["ridge_bias"]:+7.1f}  {mark:>3}')
    if reference is not None and len(reference):
        quartiles = numpy.percentile(reference, [25, 50, 75])
        print(f'\nbetween-subject VSI dissimilarity (n={len(reference)} pairs): '
              f'min {reference.min():.3f} | Q1 {quartiles[0]:.3f} | '
              f'median {quartiles[1]:.3f} | Q3 {quartiles[2]:.3f} | '
              f'max {reference.max():.3f}')
        print('target a donor near the middle of that spread — Trapeau found '
              'stronger disruption gave slower adaptation, Van Wanrooij found '
              'weaker disruption gave none')
