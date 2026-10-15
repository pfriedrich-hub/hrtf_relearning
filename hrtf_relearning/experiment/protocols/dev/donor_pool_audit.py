"""
donor_pool_audit.py — does DONOR_POOL still match what is on disk?

Run this after a recording week. It CHANGES NOTHING. It reads every base
recording and every results archive, applies the pool's own two qualification
rules, and prints

  1. one line per recording with its verdict,
  2. the ``DONOR_POOL`` tuple to paste into ``donor_selection.py``, and
  3. the ``TARGET_R_MATCH`` that pool implies, with the distribution behind it.

Then a human reviews and commits.

WHY THE POOL IS NOT DISCOVERED AT IMPORT TIME
---------------------------------------------
``TARGET_R_MATCH`` is the MEDIAN of the pool's own pairwise ``r_match``
distribution. A pool that grew by itself would therefore re-centre selection
every time a participant was recorded, silently, and no commit would record
which criterion a given shortlist was produced against. This is not
hypothetical: adding LGL/PA/SH moved the per-ear median 0.626 -> 0.573, more
than one ``TOLERANCE`` band. Keeping the pool a versioned constant is what makes
"re-derive SH's shortlist" answerable — ``git log donor_selection.py`` says
exactly what he was selected against. So this script proposes; it does not
apply.

THE TWO RULES, both taken from donor_selection
----------------------------------------------
acoustic
    The recording must match the current measurement grid as it stands or after
    :func:`~hrtf_relearning.hrtf.analysis.donor_selection.conform_recording`
    (512 taps, 48828 Hz, 19 median-plane elevations). Nothing else is repaired,
    because the only repair available is interpolation across direction and
    that smooths precisely the notches this manipulation transplants.

behavioural
    The owner must have localized with their OWN unmodified HRTF in the virtual
    environment at an elevation gain of at least ``MIN_DONOR_ELEVATION_GAIN``.
    Scored by ``vsi_rmse.baseline_run``'s rule — first FINISHED non-dome
    own-HRTF run of >= ``MIN_TRIALS`` trials — restricted to |el| <=
    ``EL_LIMIT``, which is the convention the pool comment's EG column is
    quoted at. Polar error is reported beside it; gain is a slope and can look
    healthy while absolute accuracy is poor.

Detail strength is reported for proposed additions and GATES NOTHING, which is
the current policy: measured against localization on the 17 candidates that
have both, it predicts it at Spearman +0.09. The historical 2.66 dB floor is
printed only so a borderline case (LGL came in at 2.69) is visible.

WHAT THIS SCRIPT WILL NOT DO
----------------------------
**Propose a removal.** A member whose EG now reads below the floor, or whose
recording no longer conforms, may have been admitted on a different run or a
different file than the one scored here, and dropping it would change every
shortlist already run. Both cases are flagged loudly and left to a human.
``GRANDFATHERED`` exists for members that cannot be re-derived at all; as of
2026-10-07 it is EMPTY, because all 15 reproduce (see its comment).

**Recognise a re-record.** LG and LGL are the same person, and no rule in the
data can tell you that — a self-donor pairing looks like a legitimately low
r_match, which is the one thing selection is hunting for. Every such case has
to be written into ``SUPERSEDED`` below BEFORE this audit means anything. If
you re-record someone under a new id, add it there in the same session.

Run cell by cell (# %%), or run the file.
"""

# %% ---------------------------------------------------------------- config
EL_LIMIT = 30.0          # elevation window for the EG column (pool convention)
MIN_TRIALS = 50          # baseline_run: skips the short midline screening blocks
PREFER = 'json'          # 'json' (append-only archive) or 'pkl' first
WRITE_CSV = True
STRENGTH_NOTE_FLOOR = 2.66   # reported, NOT applied — see the module docstring

#: Re-records: ``{dead id: the id that replaces it}``. A dead id must never
#: enter the pool. ``load_candidates`` drops the listener's own id by STRING, so
#: 'LG' in the pool makes LG a self-donor for LGL: a near-zero perturbation
#: that scores as a legitimately low r_match and would be SELECTED. Nothing in
#: the data reveals this; it has to be written down.
SUPERSEDED = {'LG': 'LGL'}

#: Qualified, conforming, and deliberately LEFT OUT, with the reason. Without
#: this the audit re-proposes the same recording every week and a considered
#: decision looks like an oversight. Delete an entry to let it through.
#: Empty as of 2026-10-13. AM sat here for one afternoon as a detail-space
#: outlier and was then added to the pool deliberately — keeping outliers in is
#: what lets a listener far from the pool centre be reached at all. So a low
#: mean r_match is NOT by itself a reason to hold a recording back; if one goes
#: in here, the reason has to be something other than "unusual".
HELD = {}

#: Not people. Manikin and rig-verification recordings live in the same folders
#: and would otherwise be reported as "needs one own-HRTF block" forever.
NON_HUMAN_PREFIXES = ('Kemar', 'kemar')

#: In the pool on numbers that cannot be re-derived from anything still on
#: disk. The audit cannot confirm these and must not propose dropping them.
#:
#: EMPTY as of 2026-10-07, and the two entries it used to hold are worth
#: recording. ``donor_selection``'s provenance comment calls pilot/SW (0.93) and
#: pilot/VD (0.89) "unverifiable -- no json, no pkl". They are not: both
#: reproduce to two decimals from the FLAT pilot pickles,
#: ``results/pilot/SW.pkl`` (0.934, n=128) and ``results/pilot/VD.pkl`` (0.889,
#: n=45). This is the third time the same lookup bug has hidden a pilot
#: archive -- AGV and AH were found the same way in 2026-09-17 -- so
#: `results_archives` below searches all four layouts deliberately.
GRANDFATHERED = ()

# %% --------------------------------------------------------------- imports
import csv
import datetime
import logging

import numpy

from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.experiment.analysis.database import vsi_rmse
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy, polar_error)
from hrtf_relearning.utils import paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

POOL = tuple(selection.DONOR_POOL)
FLOOR = selection.MIN_DONOR_ELEVATION_GAIN


class _Slice:
    """Minimal stand-in for a Trialsequence holding a subset of trials.

    Identical to ``vsi_rmse._Slice`` and kept local only so this script does
    not import a private name; the point of both is that the selected trials go
    through the project's own ``localization_accuracy`` / ``polar_error`` rather
    than a re-derivation, so the numbers match the per-subject plots.
    """

    def __init__(self, data):
        self.data = [list(d) for d in data]
        self.this_n = len(self.data) - 1
        self.n_remaining = 0


# %% ------------------------------------------------------------- discovery
def inventory():
    """Every base recording on disk as ``{pool name: sofa path}``.

    Pilot recordings are always named ``pilot/<id>`` even where the bare name
    would also resolve, because the bare form is ambiguous for the three ids
    that exist in both trees (AS, LS, PF) and those are not necessarily the
    same person — see ``vsi_rmse.find_participants``.
    """
    found = {}
    for base, prefix in ((paths.SOFA_DIR, ''), (paths.SOFA_DIR / 'pilot', 'pilot/')):
        if not base.exists():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            if folder.name in ('database', 'pilot'):
                continue
            sofa = folder / f'{folder.name}.sofa'
            if sofa.exists():
                found[f'{prefix}{folder.name}'] = sofa
    return found


def resolve_pool(recordings):
    """``{pool entry: inventory name or None}``, resolved as load_candidates does.

    A bare entry resolves against the participant folders FIRST and the pilot
    folder second, so the pool's 'AS' is ``sofa/AS/AS.sofa`` and NOT
    ``sofa/pilot/AS/AS.sofa`` — which is a 63-source screening recording of a
    different person with the same initials. Matching pool membership by stem
    gets this backwards and reports the pilot namesake as a broken pool member.
    """
    resolved = {}
    for entry in POOL:
        if entry in recordings:
            resolved[entry] = entry
        elif not entry.startswith('pilot/') and f'pilot/{entry}' in recordings:
            resolved[entry] = f'pilot/{entry}'
        else:
            resolved[entry] = None
    return resolved


def is_non_human(name):
    return name.split('/')[-1].startswith(NON_HUMAN_PREFIXES)


def conformance(path):
    """``(hrtf or None, note)`` — conformed where that is lossless, else None."""
    import slab
    try:
        hrtf = slab.HRTF(str(path))
    except Exception as exc:
        return None, f'unreadable ({type(exc).__name__})'
    sources, taps = hrtf.n_sources, hrtf[0].data.shape[0]
    rate = float(hrtf[0].samplerate)
    conformed = selection.conform_recording(hrtf)
    if conformed is None or not selection.conforms(conformed):
        return None, f'off grid: {sources} src / {taps} taps / {rate:.0f} Hz'
    if conformed is hrtf:
        return hrtf, ''
    return conformed, f'conformed from {sources} src / {taps} taps'


def results_archives(name, ambiguous_stems):
    """``([(kind, path), ...], note)`` — every archive for this id, best first.

    The JSON is preferred because it is append-only: a pickle can lose runs (see
    project_LS_pkl_incident_20260827), the JSON cannot. Both layouts of
    ``results/pilot/`` are searched — the flat ``<id>.json`` and the migrated
    ``<id>/<id>.json`` — plus ``pilot/backup/``, because looking only for the
    nested form silently loses AGV, AH, IM and UG.

    ALL of them are returned, in order, and ``qualify`` walks the list. A
    re-recorded participant gets a FRESH archive in the active tree that holds
    only day-1 screens, while the >= MIN_TRIALS block that qualified them sits
    in the older one — CO, re-recorded 2026-10-13, is exactly this. Returning
    only the first archive reports such a member as unqualified; walking the
    list qualifies them and says which archive did it.

    An id with a base recording in BOTH trees and no archive in its own tree is
    REFUSED rather than crossed: the archive may belong to a different person
    with the same initials.
    """
    stem = name.split('/')[-1]
    pilot = paths.RESULTS_DIR / 'pilot'
    if name.startswith('pilot/'):
        roots = [pilot / stem, pilot, pilot / 'backup']
    else:
        own = paths.RESULTS_DIR / stem
        if stem in ambiguous_stems and not any(
                (own / f'{stem}.{k}').exists() for k in ('json', 'pkl')):
            return [], 'two recordings, no archive in the active tree'
        roots = [own, pilot / stem, pilot, pilot / 'backup']

    found = []
    for kind in (('json', 'pkl') if PREFER == 'json' else ('pkl', 'json')):
        for root in roots:
            for filename in (f'{stem}_corrected.{kind}', f'{stem}.{kind}'):
                candidate = root / filename
                if candidate.exists():
                    found.append((kind, candidate))
    return found, ('' if found else 'no results archive')


def _label(path):
    """An archive path as it reads under ``results/``."""
    try:
        return str(path.relative_to(paths.RESULTS_DIR))
    except ValueError:
        return path.name


def qualify(name, ambiguous_stems):
    """``(scored or None, note)`` — the first archive that yields a baseline run.

    ``scored['from_fallback']`` is set when the qualifying run came from an
    archive other than the preferred one, which is the signal that the
    recording now in the pool is NOT the one the number was measured on.
    """
    archives, note = results_archives(name, ambiguous_stems)
    if not archives:
        return None, note
    for index, (kind, path) in enumerate(archives):
        scored = score_baseline(load_runs(kind, path), name.split('/')[-1])
        if scored is None:
            continue
        # relative to results/, so results/CO/CO.json and
        # results/pilot/CO/CO.json do not both print as 'CO/CO.json'
        where = _label(path)
        scored['source'] = f'{kind}:{where}'
        scored['from_fallback'] = '' if index == 0 else (
            f'qualified from {where}; the current archive '
            f'({_label(archives[0][1])}) has no >={MIN_TRIALS}-trial '
            f'own-HRTF block')
        return scored, note
    return None, (f'{len(archives)} archive(s) found, none with a finished '
                  f'non-dome own-HRTF run of >={MIN_TRIALS} trials')


def load_runs(kind, path):
    """The ``localization`` dict of an archive, or ``{}`` if unreadable."""
    if kind == 'pkl':
        return vsi_rmse.load_runs(path)
    from hrtf_relearning.experiment.analysis.subject.restore_from_json import (
        load_json_localization)
    try:
        return load_json_localization(path)[1] or {}
    except Exception as exc:
        logger.warning('%s: unreadable JSON (%s) — skipped', path.name, exc)
        return {}


# %% --------------------------------------------------------------- scoring
def score_baseline(runs, stem):
    """``dict`` for the first own-HRTF baseline run, or ``None`` if there is none.

    ``baseline_run`` supplies the rule (first FINISHED run whose ``hrir`` is the
    subject's own id, not a ``*_dome`` free-field block, >= ``MIN_TRIALS``
    trials). Insertion order is chronological — filenames carry no year, so
    never sort them lexically.
    """
    key, seq = vsi_rmse.baseline_run(runs, stem, MIN_TRIALS)
    if seq is None:
        return None
    data = numpy.asarray(seq.data, dtype=float).reshape(-1, 2, 2)
    keep = numpy.abs(data[:, 1, 1]) <= EL_LIMIT + 1e-6
    if keep.sum() < 2:
        return {'run': key, 'eg': numpy.nan, 'pe': numpy.nan,
                'quadrant': numpy.nan, 'n': int(keep.sum())}
    sliced = _Slice(data[keep].reshape(int(keep.sum()), 4))
    eg = localization_accuracy(sliced)[0]
    local_mean, _, quadrant = polar_error(sliced)
    return {'run': key, 'eg': float(eg), 'pe': local_mean,
            'quadrant': quadrant, 'n': int(keep.sum())}


def verdict(name, hrtf, note, scored, in_pool):
    """One of the fixed verdict strings, plus a comment."""
    stem = name.split('/')[-1]

    if is_non_human(name):
        return 'rig', 'manikin / verification recording'
    if stem in SUPERSEDED:
        return 'SUPERSEDED', f'same person as {SUPERSEDED[stem]} — keep out'
    if hrtf is None:
        return ('POOL-BROKEN' if in_pool else 'off grid'), note

    eg = None if scored is None else scored['eg']
    if in_pool:
        if eg is not None and not numpy.isnan(eg) and scored['from_fallback']:
            return 'POOL-requalify', scored['from_fallback']
        if eg is None or numpy.isnan(eg):
            if name in GRANDFATHERED:
                return 'POOL-legacy', 'qualified on data no longer on disk'
            return 'POOL-unverifiable', 'no own-HRTF baseline run found'
        if eg < FLOOR:
            return 'POOL-BELOW', (f'EG {eg:.2f} now reads below the {FLOOR:.1f} '
                                  f'floor — check which run admitted it')
        return 'POOL', note
    if eg is None or numpy.isnan(eg):
        return 'needs block', 'one own-HRTF localization block qualifies it'
    if eg < FLOOR:
        return 'below floor', f'EG {eg:.2f} < {FLOOR:.1f} — unproven, not unusable'
    if stem in HELD:
        return 'held', HELD[stem]
    return 'ADD', f'EG {eg:.2f} clears the floor'


def audit():
    """One row per base recording on disk, in verdict order."""
    recordings = inventory()
    stems = [n.split('/')[-1] for n in recordings]
    ambiguous = {s for s in stems if stems.count(s) > 1}
    if ambiguous:
        logger.warning('ids with a base recording in BOTH trees: %s — not '
                       'necessarily the same person', ', '.join(sorted(ambiguous)))

    resolved = resolve_pool(recordings)
    in_pool_names = {n for n in resolved.values() if n is not None}

    rows = []
    for name, sofa in recordings.items():
        hrtf, note = conformance(sofa)
        scored, source = None, ''
        if hrtf is not None and not is_non_human(name):
            scored, archive_note = qualify(name, ambiguous)
            source = scored['source'] if scored else archive_note
            if scored is None and archive_note:
                note = archive_note
        state, comment = verdict(name, hrtf, note, scored,
                                 in_pool=name in in_pool_names)
        rows.append({'name': name, 'verdict': state, 'comment': comment,
                     'eg': numpy.nan if scored is None else scored['eg'],
                     'pe': numpy.nan if scored is None else scored['pe'],
                     'quadrant': numpy.nan if scored is None else scored['quadrant'],
                     'n': 0 if scored is None else scored['n'],
                     'run': '' if scored is None else scored['run'],
                     'source': source, 'hrtf': hrtf})
    order = ['ADD', 'held', 'POOL-BELOW', 'POOL-BROKEN', 'POOL-unverifiable',
             'POOL-requalify', 'POOL',
             'POOL-legacy', 'needs block', 'below floor', 'off grid',
             'SUPERSEDED', 'rig']
    rows.sort(key=lambda r: (order.index(r['verdict']) if r['verdict'] in order
                             else len(order), r['name']))
    for row in rows:
        row['resolved_from'] = next((e for e, n in resolved.items()
                                     if n == row['name']), '')
    return rows, resolved


def report(rows, resolved):
    """Print the table and the two things to act on."""
    def _f(value, fmt='{:.2f}'):
        return '  -  ' if value is None or (isinstance(value, float)
                                            and numpy.isnan(value)) else fmt.format(value)

    print(f'\nDONOR POOL AUDIT   {datetime.date.today().isoformat()}')
    print(f'pool: {len(POOL)} recordings   floor: EG >= {FLOOR:.1f} at '
          f"|el| <= {EL_LIMIT:.0f} deg, >= {MIN_TRIALS} trials   target: "
          f'{selection.TARGET_R_MATCH:.3f}')
    print('=' * 104)
    print(f'{"recording":<20} {"verdict":<18} {"EG":>6} {"polar":>7} {"quad":>6} '
          f'{"n":>5}  comment')
    print('-' * 104)
    for row in rows:
        print(f'{row["name"]:<20} {row["verdict"]:<18} {_f(row["eg"]):>6} '
              f'{_f(row["pe"], "{:.1f}"):>7} {_f(row["quadrant"], "{:.2f}"):>6} '
              f'{row["n"]:>5}  {row["comment"]}')
    print('-' * 104)

    missing = [entry for entry, name in resolved.items() if name is None]
    if missing:
        print(f'IN THE POOL BUT NOT ON DISK: {", ".join(missing)}')

    for state, headline in (('POOL-BROKEN', 'POOL MEMBERS THAT NO LONGER LOAD OR CONFORM'),
                            ('POOL-BELOW', 'POOL MEMBERS WHOSE EG NOW READS BELOW THE FLOOR'),
                            ('POOL-requalify', 'POOL MEMBERS RE-RECORDED SINCE THEY QUALIFIED'),
                            ('SUPERSEDED', 'RE-RECORDS (must stay out of the pool)')):
        flagged = [r for r in rows if r['verdict'] == state]
        if flagged:
            print(f'\n{headline}:')
            for row in flagged:
                print(f'  {row["name"]:<16} {row["comment"]}')
            if state in ('POOL-BROKEN', 'POOL-BELOW'):
                print('  -> NOT removed automatically: dropping a member changes '
                      'every shortlist already run. Decide by hand.')
            if state == 'POOL-requalify':
                print('  -> The FILE in the pool is not the recording the number '
                      'was measured on. Still\n     a donor (same ears, and the '
                      'pool asks whether a cue is demonstrably usable),\n     but '
                      'one own-HRTF block of >=%d trials on the new recording '
                      'settles it.' % MIN_TRIALS)

    held = [r for r in rows if r['verdict'] == 'held']
    if held:
        print('\nQUALIFIED BUT HELD OUT ON PURPOSE:')
        for row in held:
            print(f'  {row["name"]:<16} {row["comment"]}')

    additions = [r['name'] for r in rows if r['verdict'] == 'ADD']
    if not additions:
        print('\nNo additions. The committed pool is current.')
    else:
        print(f'\n{len(additions)} ADDITION(S): {", ".join(additions)}')
    return additions


def proposed_pool(additions):
    """The current pool with the additions appended, in pool order."""
    return POOL + tuple(a for a in additions if a not in POOL)


# %% ------------------------------------------------- TARGET_R_MATCH for a pool
def target_for(pool):
    """``(values, stats)`` — the pairwise r_match distribution of a pool.

    Uses the recipe the ``TARGET_R_MATCH`` comment in ``donor_selection``
    prescribes verbatim, including reloading through ``load_candidates``, so the
    number here is derived exactly the way the committed one was. ``per_ear``
    follows ``PER_EAR_SELECTION``: a candidate is a donor EAR, and a person's
    own two ears are not a pair.
    """
    candidates = selection.load_candidates('__none__', pool=list(pool))
    values, _ = selection.pairwise_r_match(candidates)
    stats = {'n_recordings': len(candidates), 'n_pairs': len(values),
             'min': float(values.min()), 'q1': float(numpy.percentile(values, 25)),
             'median': float(numpy.median(values)),
             'q3': float(numpy.percentile(values, 75)),
             'max': float(values.max())}
    return values, stats


def _stat_line(label, stats):
    return (f'  {label:<12} n={stats["n_pairs"]:<4} | min {stats["min"]:+.3f} | '
            f'Q1 {stats["q1"]:.3f} | median {stats["median"]:.3f} | '
            f'Q3 {stats["q3"]:.3f} | max {stats["max"]:.3f}')


def paste_block(pool, stats, previous_stats=None):
    """Print the two edits to make in ``donor_selection.py``, ready to paste."""
    print('\n' + '=' * 104)
    print('TO APPLY — two edits in hrtf/analysis/donor_selection.py')
    print('=' * 104)
    print(f'\nTARGET_R_MATCH = {stats["median"]:.3f}')
    print(f'\n  per-ear  {"PER_EAR_SELECTION=True" if selection.PER_EAR_SELECTION else "per-person"}, '
          f'{stats["n_recordings"]} recordings')
    print(_stat_line('proposed', stats))
    if previous_stats is not None:
        print(_stat_line('committed', previous_stats))
    delta = stats['median'] - selection.TARGET_R_MATCH
    verdict_line = ('inside' if abs(delta) <= selection.TOLERANCE else 'OUTSIDE')
    print(f'\n  committed constant {selection.TARGET_R_MATCH:.3f}, proposed '
          f'{stats["median"]:.3f}, delta {delta:+.3f} — {verdict_line} one '
          f'TOLERANCE band ({selection.TOLERANCE:.2f}).')
    if abs(delta) > selection.TOLERANCE:
        print('  The band this defines does not overlap the correct one: '
              'selection is off-centre until this is committed.')
    print('\nDONOR_POOL = (' + ', '.join(f"'{n}'" for n in pool) + ')')
    print('\nAlso update the dated provenance comment above each constant — the '
          'EG column, the\ndistributions above, and what moved. That comment is '
          'the only record of why a\nshortlist run last month used a different '
          'number.')


# %% ------------------------------------------- additions: what they hand over
def describe(rows, names):
    """Detail strength and cue gradient, PER EAR, for the named recordings.

    Neither gates anything (see the module docstring). They are here so a
    borderline addition is visible rather than silent: a recording can hand over
    a deep, elevation-dependent and completely SCRAMBLED cue and score well on
    strength alone, because strength is invariant to permuting the elevation
    labels — ``cue_gradient`` is the term that catches that.
    """
    by_name = {r['name']: r for r in rows}
    print(f'\nWHAT THE ADDITIONS HAND OVER   (strength floor {STRENGTH_NOTE_FLOOR} dB '
          'is HISTORICAL — reported, not applied)')
    print('-' * 104)
    print(f'{"recording":<20} {"ear":<7} {"strength dB":>12} {"gradient":>10} '
          f'{"monoton.":>10} {"decodab.":>10}')
    out = []
    for name in names:
        hrtf = by_name[name]['hrtf']
        split = selection.median_plane_split(hrtf, n_keep=selection.N_KEEP)
        for ear in selection.EARS:
            strength = selection.detail_strength(split, ear=ear)
            cue = selection.cue_gradient(split, ear=ear)['per_ear'][ear]
            flag = '  <- under the historical floor' if strength < STRENGTH_NOTE_FLOOR else ''
            print(f'{name:<20} {ear:<7} {strength:>12.2f} {cue["gradient"]:>10.2f} '
                  f'{cue["monotonicity"]:>10.2f} {cue["decodability"]:>10.2f}{flag}')
            out.append({'name': name, 'ear': ear, 'strength': strength, **cue})
    return out


# %% ------------------------------------------------------------------- csv
def write_csv(rows, stats, pool):
    out_dir = paths.ANALYSIS_RESULTS_DIR / 'donor_pool_audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'donor_pool_audit_{datetime.date.today().isoformat()}.csv'
    fields = ['name', 'verdict', 'eg', 'pe', 'quadrant', 'n', 'run', 'source',
              'comment']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if isinstance(row.get(k), float)
                                 and numpy.isnan(row[k]) else row.get(k, ''))
                             for k in fields})
        writer.writerow({})
        writer.writerow({'name': 'proposed_pool', 'verdict': ' '.join(pool)})
        writer.writerow({'name': 'proposed_target', 'verdict': f'{stats["median"]:.3f}',
                         'n': stats['n_pairs']})
    print(f'\nwritten: {path}')
    return path


# %% ------------------------------------------------------------------- run
ROWS, RESOLVED = audit()
ADDITIONS = report(ROWS, RESOLVED)

# %% ------------------------------------------- what to commit, if anything
_, COMMITTED_STATS = target_for(POOL)
PROPOSED = proposed_pool(ADDITIONS)
STATS = COMMITTED_STATS

if PROPOSED == POOL:
    print('\nPool unchanged — checking the committed target is still right.')
    paste_block(POOL, COMMITTED_STATS)
else:
    _, STATS = target_for(PROPOSED)
    describe(ROWS, ADDITIONS)
    paste_block(PROPOSED, STATS, COMMITTED_STATS)

if WRITE_CSV:
    write_csv(ROWS, STATS, PROPOSED)
