"""
vsi_rmse.py — VSI vs individual-HRTF localization accuracy across the database.

Replicates the Trapeau & Schönwiesner (2016) VSI-vs-performance relation on this
rig: for every participant who has BOTH a measured individual HRTF and a
localization test run with that (unmodified) HRTF, compute the VSI of the HRTF
and the elevation accuracy of the test, then regress one on the other.

What counts as a datapoint
--------------------------
HRTF : ``data/hrtf/sofa/<id>/<id>.sofa`` (or ``sofa/pilot/<id>/<id>.sofa``) —
    the recorded base, never a modified variant.
Run  : the FIRST finished localization run whose ``hrir`` equals the subject id,
    i.e. the pre-training baseline, so the score is not contaminated by
    adaptation to a modified HRTF. Free-field ``*_dome`` runs are excluded —
    those are real speakers, not the HRTF under test, even where the run carries
    the subject id in its ``hrir`` field. ``MIN_TRIALS`` additionally skips the
    short (n=34) midline screening tests that some sessions ran first.

Grid matching
-------------
Runs differ in grid (n = 34/51/75/150, elevation span ±31.2 / ±33.3, azimuth
span 0 or ±33). Targets are therefore restricted to a common elevation window
(``EL_LIMIT``; auto = the narrowest span present) before scoring, and two scores
are reported per subject:

``*_all``  every target inside the elevation window — more trials, but the
           azimuth span is not the same for all subjects.
``*_mid``  only near-midline targets (|az| <= ``AZ_LIMIT``) — fewer trials, but
           the same slice of space for everyone, and the slice VSI actually
           describes (VSI is a median-plane measure).

Normalisation
-------------
VSI is computed under every mode in ``vsi.NORMALIZATIONS`` so the choice can be
made on evidence rather than assertion — see the ``hrtf.analysis.vsi`` module
docstring for why ``arc_mean`` is expected to be degenerate here (it is included
as a control) and why ``detail`` is the candidate for cross-subject work.

Run this file directly, or cell by cell.
"""

# %% ---------------------------------------------------------------- config
import logging
import pickle
from pathlib import Path

import numpy
import scipy.stats

from hrtf_relearning.utils import paths
from hrtf_relearning.hrtf.analysis import vsi as vsi_module
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy)

BAND = vsi_module.BEHAVIOURAL_BAND   # 5657-8000 Hz — the half-octave band in
                          # which Trapeau found VSI vs vertical RMSE, NOT the
                          # peak-VSI octave (5657-11314). See vsi.py docstring.
RESOLUTION = 'filterbank'  # 2%-spaced triangular filters, as in the paper
N_KEEP = 4                # envelope coefficients for normalize='detail'
MIN_TRIALS = 50           # skip the short midline screening runs
EL_LIMIT = None           # elevation window (deg); None -> narrowest span present
AZ_LIMIT = 8.5            # |az| for the near-midline score
SD_FLOOR = 5.0            # ele_sd below this = near-constant responses -> suspect
OUT_DIR = paths.ANALYSIS_RESULTS_DIR / 'vsi_rmse'

logger = logging.getLogger(__name__)


# %% ------------------------------------------------------------- discovery
def find_participants():
    """(subject_id, sofa_path, pkl_path) for everyone with a base SOFA + pickle.

    Both cohorts are covered: active subjects (``sofa/<id>/<id>.sofa`` +
    ``results/<id>/<id>.pkl``) and the pilot archive (``sofa/pilot/<id>/`` +
    a pickle under ``results/pilot/``). A few ids exist in BOTH trees — these
    are not necessarily the same person (same initials; cf. demerge_subject.py),
    so the SOFA is always paired with the pickle from the SAME cohort and a
    warning is logged rather than silently crossing them. THE ACTIVE PAIR WINS:
    where an id has a current recording, that is the one being used in the
    experiment, and pairing it with a pilot namesake's data would score the
    wrong person (e.g. ``sofa/pilot/AS`` is a 63-source screening recording, not
    the AS being run).

    ``results/pilot/`` has TWO layouts and both must be searched: the original
    flat ``<id>.pkl`` and the per-subject ``<id>/<id>.pkl`` that later subjects
    were migrated into. Looking only for the flat form silently loses everyone
    who was migrated — CO, FD, GLK, GS, JF, SS among them. A
    ``<id>_corrected.pkl`` wins over ``<id>.pkl`` (some pilot pickles are
    damaged).
    """
    ids = {sofa.parent.name for sofa in paths.SOFA_DIR.rglob('*/*.sofa')
           if sofa.stem == sofa.parent.name}          # base recordings only
    pilot_results = paths.RESULTS_DIR / 'pilot'

    def _pilot_pkl(subject_id):
        """First existing pilot pickle, corrected before plain, nested or flat."""
        candidates = (
            pilot_results / subject_id / f'{subject_id}_corrected.pkl',
            pilot_results / subject_id / f'{subject_id}.pkl',
            pilot_results / f'{subject_id}_corrected.pkl',
            pilot_results / f'{subject_id}.pkl',
        )
        return next((p for p in candidates if p.exists()), candidates[-1])

    found = []
    for subject_id in sorted(ids):
        sofas = {'active': paths.SOFA_DIR / subject_id / f'{subject_id}.sofa',
                 'pilot': paths.SOFA_DIR / 'pilot' / subject_id / f'{subject_id}.sofa'}
        pkls = {'active': paths.subject_pkl(subject_id),
                'pilot': _pilot_pkl(subject_id)}
        have_sofa = [c for c in ('active', 'pilot') if sofas[c].exists()]
        have_pkl = [c for c in ('active', 'pilot') if pkls[c].exists()]

        if not have_sofa or not have_pkl:
            logger.info('%s: SOFA (%s) / pickle (%s) — skipped', subject_id,
                        '+'.join(have_sofa) or 'none', '+'.join(have_pkl) or 'none')
            continue

        # active wins on both sides: where an id has a current recording, that is
        # the one being run, and its results are the ones to score
        sofa_cohort, pkl_cohort = have_sofa[0], have_pkl[0]
        if len(have_sofa) > 1:
            logger.warning('%s: a recording exists in BOTH trees — using the '
                           'active one; confirm they are the same person',
                           subject_id)
            if sofa_cohort != pkl_cohort:
                # genuinely ambiguous AND crossed: the pickle may belong to a
                # different person with the same initials. Refuse rather than
                # score the wrong one.
                logger.warning('%s: two recordings but only a %s pickle — '
                               'cannot tell which person it belongs to, skipped',
                               subject_id, pkl_cohort)
                continue
        elif sofa_cohort != pkl_cohort:
            # only one recording exists, so there is nothing to confuse it with;
            # the pickle simply lives in the other tree (e.g. an active
            # recording whose results were filed under results/pilot/<id>/)
            logger.info('%s: %s recording paired with %s pickle (only one of '
                        'each exists)', subject_id, sofa_cohort, pkl_cohort)
        found.append((subject_id, sofas[sofa_cohort], pkls[pkl_cohort]))
    return found


def load_runs(pkl_path):
    """The ``localization`` dict of a subject pickle, or {} if unreadable."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f).get('localization', {}) or {}
    except Exception:
        logger.warning('%s: unreadable pickle — skipped', pkl_path.name)
        return {}


def baseline_run(runs, subject_id, min_trials=MIN_TRIALS):
    """First finished run made with the subject's own unmodified HRTF.

    Insertion order is chronological (filenames carry no year, so never sort
    them lexically — see Subject.localization_summary).
    """
    for key, seq in runs.items():
        if getattr(seq, 'hrir', None) != subject_id:
            continue
        if not getattr(seq, 'finished', False):
            continue
        if key.endswith('dome'):          # free field, not the HRTF under test
            continue
        data = getattr(seq, 'data', None)
        if not data or len(data) < min_trials:
            continue
        return key, seq
    return None, None


# %% -------------------------------------------------------------- scoring
class _Slice:
    """Minimal stand-in for a Trialsequence holding a subset of trials.

    Scored with the project's own ``localization_accuracy`` so the numbers are
    identical to those in the per-subject plots rather than a re-derivation.
    """

    def __init__(self, data):
        self.data = [list(d) for d in data]
        self.this_n = len(self.data) - 1
        self.n_remaining = 0


def targets_responses(seq):
    data = numpy.asarray(seq.data, dtype=float).reshape(-1, 2, 2)
    return data[:, 1], data[:, 0]      # targets, responses


def score(seq, el_limit, az_limit=None):
    """(elevation_gain, ele_rmse, ele_sd, n_trials) for the selected targets."""
    targets, _ = targets_responses(seq)
    keep = numpy.abs(targets[:, 1]) <= el_limit + 1e-6
    if az_limit is not None:
        keep &= numpy.abs(_signed_az(targets[:, 0])) <= az_limit + 1e-6
    if keep.sum() < 2:
        return numpy.nan, numpy.nan, numpy.nan, int(keep.sum())
    data = numpy.asarray(seq.data, dtype=float).reshape(-1, 2, 2)[keep]
    eg, ele_rmse, ele_sd, *_ = localization_accuracy(_Slice(data.reshape(len(data), 4)))
    return eg, ele_rmse, ele_sd, int(keep.sum())


def _signed_az(az):
    """Azimuths wrapped to [-180, 180) — some runs store 350 rather than -10."""
    return (numpy.asarray(az, dtype=float) + 180.0) % 360.0 - 180.0


# %% ----------------------------------------------------------------- main
def collect(min_trials=MIN_TRIALS, el_limit=EL_LIMIT, az_limit=AZ_LIMIT,
            band=BAND, n_keep=N_KEEP, resolution=RESOLUTION):
    """One row per participant: VSI under each normalisation + baseline accuracy."""
    import slab

    participants = []
    for subject_id, sofa, pkl in find_participants():
        runs = load_runs(pkl)
        key, seq = baseline_run(runs, subject_id, min_trials)
        if seq is None:
            logger.info('%s: no finished baseline run with the own HRTF — skipped',
                        subject_id)
            continue
        participants.append((subject_id, sofa, key, seq))

    if not participants:
        raise RuntimeError('no participants with both an HRTF and a baseline run')

    # common elevation window across the runs that survived
    if el_limit is None:
        el_limit = min(float(numpy.abs(targets_responses(seq)[0][:, 1]).max())
                       for _, _, _, seq in participants)
        logger.info('common elevation window: +/- %.1f deg', el_limit)

    rows = []
    for subject_id, sofa, key, seq in participants:
        hrtf = slab.HRTF(str(sofa))
        row = {'subject': subject_id, 'run': key,
               'n_trials': len(seq.data),
               'n_elevations': len(vsi_module.median_plane_sources(hrtf)),
               'el_limit': el_limit, 'band': band, 'resolution': resolution}
        for mode in vsi_module.NORMALIZATIONS:
            row[f'vsi_{mode}'] = vsi_module.vsi(hrtf, bandwidth=band,
                                                normalize=mode, n_keep=n_keep,
                                                resolution=resolution)
        for tag, az in (('all', None), ('mid', az_limit)):
            eg, rmse, sd, n = score(seq, el_limit, az)
            row.update({f'eg_{tag}': eg, f'ele_rmse_{tag}': rmse,
                        f'ele_sd_{tag}': sd, f'n_{tag}': n})
        # A run where the responses barely vary in elevation carries no
        # elevation information at all; its RMSE is then just the spread of the
        # targets around a constant answer, not a measure of the cue. Flagged
        # rather than dropped — see SD_FLOOR.
        row['suspect'] = bool(row['ele_sd_all'] < SD_FLOOR)
        rows.append(row)
        logger.info('%s: %s', subject_id, row)
    return rows


def regress(rows, mode, score_key='ele_rmse_mid'):
    """Pearson r, p and the fit of one VSI variant against one accuracy score."""
    x = numpy.array([r[f'vsi_{mode}'] for r in rows], dtype=float)
    y = numpy.array([r[score_key] for r in rows], dtype=float)
    ok = numpy.isfinite(x) & numpy.isfinite(y)
    if ok.sum() < 3:
        return None
    fit = scipy.stats.linregress(x[ok], y[ok])
    return {'mode': mode, 'score': score_key, 'n': int(ok.sum()),
            'r': float(fit.rvalue), 'p': float(fit.pvalue),
            'slope': float(fit.slope), 'intercept': float(fit.intercept),
            'vsi_mean': float(x[ok].mean()), 'vsi_sd': float(x[ok].std(ddof=1)),
            'vsi_min': float(x[ok].min()), 'vsi_max': float(x[ok].max())}


def band_sweep(rows, score_keys=('ele_rmse_all', 'eg_all'),
               bands=vsi_module.PAPER_OCTAVE_BANDS + vsi_module.PAPER_HALF_OCTAVE_BANDS,
               modes=('none', 'detail'), n_keep=N_KEEP, resolution=RESOLUTION):
    """Re-run the regression in each band — is the result band-dependent?

    The default ``BAND`` is only one of the five octave bands Trapeau &
    Schönwiesner report. Sweeping the rest is the honest check, but note that
    it is also a multiple-comparisons machine: len(bands) x len(modes) x
    however many score keys you try. Treat a lone significant cell as noise
    unless it survives in both normalisations with the SAME sign.
    """
    import slab
    sofa = {subject_id: path for subject_id, path, _ in find_participants()}
    out = []
    cache = {mode: {r['subject']: vsi_module.vsi_bands(
        slab.HRTF(str(sofa[r['subject']])), bands=bands, normalize=mode,
        n_keep=n_keep, resolution=resolution) for r in rows} for mode in modes}
    for band in bands:
        for mode in modes:
            for r in rows:
                r[f'vsi_{mode}'] = cache[mode][r['subject']][tuple(band)]
            for score_key in score_keys:
                stats = regress(rows, mode, score_key)
                if stats:
                    out.append({**stats, 'band': tuple(band)})
    return out


def write_csv(rows, path):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows, score_key='ele_rmse_mid', filepath=None):
    """Scatter + least-squares fit, one panel per normalisation."""
    from matplotlib import pyplot as plt
    modes = vsi_module.NORMALIZATIONS
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4), dpi=200)
    for ax, mode in zip(numpy.atleast_1d(axes), modes):
        x = numpy.array([r[f'vsi_{mode}'] for r in rows], dtype=float)
        y = numpy.array([r[score_key] for r in rows], dtype=float)
        ok = numpy.isfinite(x) & numpy.isfinite(y)
        suspect = numpy.array([bool(r.get('suspect')) for r in rows])
        ax.scatter(x[ok & ~suspect], y[ok & ~suspect], s=12,
                   facecolor='none', edgecolor='0.3')
        ax.scatter(x[ok & suspect], y[ok & suspect], s=14, marker='x',
                   color='0.65', linewidth=0.8)
        for xi, yi, r in zip(x, y, rows):
            if numpy.isfinite(xi) and numpy.isfinite(yi):
                ax.annotate(r['subject'], (xi, yi), fontsize=5,
                            xytext=(2, 2), textcoords='offset points', color='0.5')
        stats = regress(rows, mode, score_key)
        if stats:
            xs = numpy.linspace(x[ok].min(), x[ok].max(), 2)
            ax.plot(xs, stats['intercept'] + stats['slope'] * xs, 'k--', lw=0.8)
            ax.set_title(f"{mode}\nr = {stats['r']:.2f}, p = {stats['p']:.3f}, "
                         f"n = {stats['n']}", fontsize=8)
        ax.set_xlabel(f'VSI ({mode})', fontsize=8)
        ax.set_ylabel(f'{score_key} (deg)', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    if filepath:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)
    return fig


# %% ------------------------------------------------------------------ run
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rows = collect()
    csv_path = write_csv(rows, OUT_DIR / 'vsi_rmse.csv')
    print(f'\n{len(rows)} participants -> {csv_path}')

    clean = [r for r in rows if not r['suspect']]
    flagged = [r['subject'] for r in rows if r['suspect']]
    if flagged:
        print(f"near-constant responses (ele_sd < {SD_FLOOR}): {', '.join(flagged)}")

    for subset, label in ((rows, 'all'), (clean, f'ele_sd >= {SD_FLOOR}')):
        for score_key in ('ele_rmse_all', 'ele_rmse_mid', 'eg_all'):
            print(f'\n--- {score_key}  [{label}] ---')
            for mode in vsi_module.NORMALIZATIONS:
                s = regress(subset, mode, score_key)
                if s:
                    print(f"  {mode:<9} VSI {s['vsi_mean']:.3f} +/- {s['vsi_sd']:.3f} "
                          f"[{s['vsi_min']:.3f}, {s['vsi_max']:.3f}]   "
                          f"r = {s['r']:+.3f}  p = {s['p']:.4f}  n = {s['n']}")

    print(f"\n--- band sweep, resolution={RESOLUTION} "
          f"(octave bands first, then half-octave; {BAND} is the default) ---")
    default_vsi = [{k: v for k, v in r.items() if k.startswith('vsi_')}
                   for r in rows]
    for s in band_sweep(rows):
        print(f"    {str(s['band']):<16}{s['mode']:<8}{s['score']:<14}"
              f"VSI {s['vsi_mean']:.2f} +/- {s['vsi_sd']:.2f}   "
              f"r = {s['r']:+.2f}  p = {s['p']:.3f}")

    # band_sweep overwrites the vsi_* columns in place; put BAND back before
    # plotting so the figures show the module default, not the last band swept
    for row, saved in zip(rows, default_vsi):
        row.update(saved)
    for score_key in ('ele_rmse_all', 'ele_rmse_mid', 'eg_all'):
        plot(rows, score_key, OUT_DIR / f'vsi_vs_{score_key}.png')
