"""
group_transfer.py — the final-day 2x2 (Ear x Side), across participants.

The learning curve (:mod:`group_learning`) answers "did they adapt". This one
answers "to what": the same composite filter is re-tested on the last day in
four cells, and where performance survives says whether what was learned lives
in the trained EAR or in a spatial map that both ears read.

    A   trained ear, trained side     what the curve's last point measures
    D   other ear,   other side       MAIN transfer test
    B   trained ear, other side
    C   other ear,   trained side

Mirroring swaps the two channels AND negates source azimuth, so it delivers the
trained ear's own filter to the untrained ear. **D is physically the identical
stimulus to A with left and right exchanged** — same filter, same sound, other
side of the head. That makes A vs D the one contrast that needs no assumptions,
and it is why the two are drawn adjacent here and the counterbalance cells B and
C sit to the right of a divider. B and C break the ear/side confound: without
them a drop from A to D could be either factor. Block order within a
participant is a balanced Williams square (project_exp1_transfer_counterbalance),
so the four cells are not confounded with test position.

READING IT. A high D means the adaptation is not ear-specific — the map moved.
A D that falls back to the day-1 line (drawn faint) means it did not transfer at
all and the learning is tied to the trained ear. Anything between is partial,
and B/C say which factor carried it.

The selection rules are `group_learning`'s, with one addition: only the four
runs that form the counterbalanced square are used (:func:`_latin_square`). So
block A here is the single A inside that sequence, NOT the day's mean of every
A — which is what the learning curve's last point is, and why the two differ
slightly. Participants are on different stimuli (FS/IR noise, AS ripple), which
is fine — every contrast drawn is WITHIN a participant.

Run this file directly; it saves to ``ANALYSIS_RESULTS_DIR``.
"""

from collections import OrderedDict

import numpy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from hrtf_relearning.experiment.analysis.localization.plot.elevation_learning import (
    extract_day, extract_seq_meta, transfer_block, trained_hemifield)
from hrtf_relearning.experiment.analysis.localization.plot import group_learning as gl
from hrtf_relearning.utils import paths

# --- configuration -----------------------------------------------------------
SUBJECT_IDS = gl.SUBJECT_IDS
COLOURS = gl.COLOURS
METRICS = gl.METRICS
SHOW = False

#: drawing order — the assumption-free contrast first, counterbalance after.
BLOCK_ORDER = ('A', 'D', 'B', 'C')
MAIN_BLOCKS = ('A', 'D')
#: Ticks stay bare letters — they are the protocol's own vocabulary, and four
#: two-word labels do not fit a third of a 17.5 cm figure without colliding.
#: The mapping is spelled out once in the footnote instead.
BLOCK_LABEL = {'A': 'A', 'D': 'D', 'B': 'B', 'C': 'C'}
BLOCK_MEANING = {
    'A': 'trained ear, trained side',
    'D': 'other ear, other side',
    'B': 'trained ear, other side',
    'C': 'other ear, trained side',
}

#: Sign convention for the headline contrast: POSITIVE always means "moving
#: away from the trained condition costs you this much". Gain is better when
#: high, the two error metrics when low, so one of them has to be flipped —
#: leaving both as a literal A-D would have the same finding print as +0.24 in
#: one panel and -4.2 in the next.
HIGHER_IS_BETTER = {'gain': True, 'polar': False, 'sd': False}


def _latin_square(candidates):
    """The four runs that form the counterbalanced square, in the order run.

    ``candidates`` is the last day's qualifying runs as ``(key, seq, block)``,
    chronological. The final day holds MORE block-A tests than the square does:
    the ordinary pre-training daily test comes before it, and a post-training
    re-test may come after (AS ran one on 21.08 at 10:35, after her square
    closed at 10:14). Averaging every A on the day would compare a mean of
    tests taken at three different points in the session against single B, C and
    D blocks taken inside a balanced sequence — which is exactly the
    order confound the Williams square exists to remove.

    So: take the LATEST window of four consecutive runs that is a permutation of
    A, B, C, D. That is the square by construction, and it picked the right
    four on all three participants (FS A-B-D-C, IR D-A-C-B, AS C-D-B-A — three
    different Williams orders, as intended). Falls back to the last occurrence
    of each letter if no such window exists, which is what an aborted or
    re-run block would leave behind; the note says which happened.
    """
    n = len(BLOCK_ORDER)
    for start in range(len(candidates) - n, -1, -1):
        window = candidates[start:start + n]
        if sorted(block for _k, _s, block in window) == sorted(BLOCK_ORDER):
            return window, 'complete Williams square'
    latest = {}
    for key, seq, block in candidates:
        latest[block] = (key, seq, block)
    found = [latest[b] for b in BLOCK_ORDER if b in latest]
    return found, (f'NO complete square on the day — using the last of each '
                   f'block ({len(found)}/{n} present)')


def transfer_cells(subject_id, aggregate=gl.AGGREGATE,
                   off_stimulus=gl.OFF_STIMULUS):
    """The four final-day cells for one participant.

    Returns a dict with one entry per metric mapping block letter -> value,
    plus 'day1_<metric>' (the first-day trained-condition value, the naive
    reference), 'last_day_<metric>' (the learning curve's last point, i.e. the
    day's MEAN of block A — not the same as the square's A), 'order', 'keys',
    'square_note' and 'excluded'.

    Qualifying is `group_learning`'s: modal HRIR (or its ``_mirrored`` twin,
    which is what C and D are recorded under), modal stimulus, and at least
    MIN_TRIALS_FRACTION of the modal trial count. Of those, only the four runs
    forming the counterbalanced square are kept — see :func:`_latin_square`.
    """
    runs = gl._runs(subject_id)
    if not runs:
        return None
    metas = [extract_seq_meta(seq, key) for key, seq in runs]
    field = trained_hemifield(metas)
    hrir = gl._curve_hrir(runs)
    stim = gl._curve_stim(runs)
    min_trials = gl.MIN_TRIALS_FRACTION * gl._curve_n_trials(runs)
    last_day = extract_day(runs[-1][0])
    # C and D are the same composite delivered mirrored, and the run is stored
    # under the mirrored SOFA name — accept both spellings, nothing else.
    accepted = {hrir, f'{hrir}_mirrored'}

    candidates, excluded = [], []
    for (key, seq), meta in zip(runs, metas):
        if extract_day(key) != last_day:
            continue
        if meta.get('hrir') not in accepted:
            continue
        if len(seq.data) < min_trials:
            excluded.append((key, f'{len(seq.data)} trials'))
            continue
        if (meta.get('stim') or '').lower() != stim:
            excluded.append((key, f"{meta.get('stim')} stimulus"))
            continue
        block = transfer_block({**meta, 'stim': None}, field)
        if block not in BLOCK_ORDER:
            continue
        candidates.append((key, seq, block))

    square, note = _latin_square(candidates)
    for key, _seq, block in candidates:
        if key not in {k for k, _s, _b in square}:
            excluded.append((key, f'block {block} outside the counterbalanced '
                                  f'square'))
    by_block = OrderedDict()
    for key, seq, block in square:
        by_block.setdefault(block, []).append(seq)

    series = gl.subject_series(subject_id, aggregate, off_stimulus)
    out = {'subject_id': subject_id, 'trained_field': field, 'stim': stim,
           'last_day': last_day, 'excluded': excluded, 'square_note': note,
           'order': [b for _k, _s, b in square],
           'keys': {b: k for k, _s, b in square},
           'n_tests': {b: len(v) for b, v in by_block.items()}}
    for name, (_label, extract) in METRICS.items():
        values = {}
        for block, seqs in by_block.items():
            scores = [extract(seq) for seq in seqs]
            scores = [v for v in scores if v is not None and numpy.isfinite(v)]
            if scores:
                values[block] = (scores[0] if aggregate == 'first' else
                                 scores[-1] if aggregate == 'last' else
                                 float(numpy.mean(scores)))
        out[name] = values
        out[f'day1_{name}'] = (float(series[name][0]) if series[name].size
                               else numpy.nan)
        # the learning curve's last point is the day's MEAN of block A; the
        # square's A is the single run inside the counterbalanced sequence, so
        # the two are close but need not be equal. Keep both.
        out[f'last_day_{name}'] = (float(series[name][-1]) if series[name].size
                                   else numpy.nan)
    return out


def _mean_sem(cells_list, name, block):
    values = [c[name][block] for c in cells_list if block in c[name]]
    values = [v for v in values if numpy.isfinite(v)]
    if not values:
        return numpy.nan, numpy.nan, 0
    sem = (numpy.std(values, ddof=1) / numpy.sqrt(len(values))
           if len(values) > 1 else numpy.nan)
    return float(numpy.mean(values)), sem, len(values)


def transfer_difference(cells_list, name, a='A', b='D'):
    """Paired within-participant cost of going from cell ``a`` to cell ``b``.

    Signed so that positive is always a cost (see HIGHER_IS_BETTER), and paired
    per participant before averaging — with three participants on two different
    stimuli, a difference of means would mix the between-subject spread into a
    within-subject contrast.
    """
    sign = 1.0 if HIGHER_IS_BETTER[name] else -1.0
    diffs = [sign * (c[name][a] - c[name][b]) for c in cells_list
             if a in c[name] and b in c[name]]
    diffs = [d for d in diffs if numpy.isfinite(d)]
    if not diffs:
        return numpy.nan, numpy.nan, 0
    sem = (numpy.std(diffs, ddof=1) / numpy.sqrt(len(diffs))
           if len(diffs) > 1 else numpy.nan)
    return float(numpy.mean(diffs)), sem, len(diffs)


def figure(subject_ids=SUBJECT_IDS, show=SHOW):
    cells_list = [c for c in (transfer_cells(sid) for sid in subject_ids) if c]

    fs = 8
    plt.rcParams.update({
        'xtick.labelsize': fs - 1.5, 'ytick.labelsize': fs,
        'axes.labelsize': fs, 'axes.titlesize': fs, 'lines.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 0, 'ytick.major.size': 2, 'axes.linewidth': 0.7,
        'axes.spines.right': False, 'axes.spines.top': False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17.5 / 2.54, 7.0 / 2.54),
                             constrained_layout=True, dpi=264)

    x_of = {block: i for i, block in enumerate(BLOCK_ORDER)}
    divider = len(MAIN_BLOCKS) - 0.5

    for panel, (name, (label, _extract)) in enumerate(METRICS.items()):
        axis = axes[panel]

        # naive reference: where the trained condition started on day 1
        day1 = [c[f'day1_{name}'] for c in cells_list]
        day1 = [v for v in day1 if numpy.isfinite(v)]
        if day1:
            axis.axhline(numpy.mean(day1), color='0.6', lw=0.7, ls=':', zorder=0)

        axis.axvline(divider, color='0.85', lw=0.6, zorder=-1)

        # individual participants — every contrast in this figure is within one
        for cells in cells_list:
            colour = COLOURS[cells['subject_id']]
            xs = [x_of[b] for b in BLOCK_ORDER if b in cells[name]]
            ys = [cells[name][b] for b in BLOCK_ORDER if b in cells[name]]
            axis.plot(xs, ys, '-o', ms=2.2, color=colour, alpha=0.5, lw=0.8,
                      zorder=2)

        # group mean
        xs, means, sems = [], [], []
        for block in BLOCK_ORDER:
            mean, sem, n = _mean_sem(cells_list, name, block)
            if n:
                xs.append(x_of[block]); means.append(mean); sems.append(sem)
        axis.errorbar(xs, means, yerr=sems, fmt='-o', ms=3.2, color='0',
                      ecolor='0', elinewidth=0.8, capsize=1.6, lw=1.3, zorder=3)

        axis.set_xticks(list(x_of.values()))
        axis.set_xticklabels([BLOCK_LABEL[b] for b in BLOCK_ORDER])
        axis.set_xlabel('final-day block')
        axis.set_xlim(-0.45, len(BLOCK_ORDER) - 0.55)
        axis.set_ylabel(label)
        if name == 'gain':
            axis.set_ylim(0, 1.02)
            axis.set_yticks(numpy.arange(0, 1.2, 0.2))
        else:
            axis.set_ylim(0, None)
        for y in axis.get_yticks():
            axis.axhline(y, color='0.93', lw=0.5, zorder=-2)

        # the headline contrast, stated rather than left to be eyeballed
        diff, diff_sem, n = transfer_difference(cells_list, name)
        if numpy.isfinite(diff):
            unit = '' if name == 'gain' else '°'
            precision = 2 if name == 'gain' else 1
            text = f'A→D costs {diff:.{precision}f}{unit}'
            if numpy.isfinite(diff_sem):
                text += f' ± {diff_sem:.{precision}f}'
            axis.set_title(f'{"ABC"[panel]}   {text}   (n={n})', loc='left',
                           fontsize=fs - 1)

    handles = [Line2D([0], [0], color=COLOURS[c['subject_id']], marker='o',
                      ms=2.2, lw=0.8, alpha=0.7, label=c['subject_id'])
               for c in cells_list]
    handles += [
        Line2D([0], [0], color='0', marker='o', ms=3.2, lw=1.3,
               label=f'mean ± SEM (n={len(cells_list)})'),
        Line2D([0], [0], color='0.6', lw=0.7, ls=':', label='day 1 (naive)'),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=5.5, loc='lower left',
                   handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3, ncol=2,
                   columnspacing=0.9)

    fig.suptitle('Final-day transfer: the trained filter delivered to the other '
                 'ear and the other side', fontsize=fs + 1)
    fig.text(0.5, -0.01,
             '   ·   '.join(f'{b} = {BLOCK_MEANING[b]}' for b in BLOCK_ORDER),
             ha='center', va='top', fontsize=5.5, color='0.35')
    fig.text(0.5, -0.045,
             'A and D are the same stimulus with left and right exchanged; B and '
             'C (right of the divider) break the ear / side confound.   '
             'Positive cost = worse than the trained condition.',
             ha='center', va='top', fontsize=5.5, color='0.35')
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return fig, cells_list


if __name__ == '__main__':
    fig, cells_list = figure()
    for cells in cells_list:
        print(f"\n{cells['subject_id']}  final day {cells['last_day']}  "
              f"stim={cells['stim']}  trained field={cells['trained_field']}")
        print(f"  order {'-'.join(cells['order'])}  ({cells['square_note']})")
        print(f"  curve day-4 mean gain {cells['last_day_gain']:.2f} vs "
              f"square A {cells['gain'].get('A', float('nan')):.2f}")
        print(f"  {'block':>5s} {'n':>2s} {'gain':>6s} {'polar':>7s} {'sd':>6s}")
        for block in BLOCK_ORDER:
            if block not in cells['gain']:
                print(f"  {block:>5s}  - (missing)")
                continue
            print(f"  {block:>5s} {cells['n_tests'].get(block, 0):2d} "
                  f"{cells['gain'][block]:6.2f} {cells['polar'][block]:7.2f} "
                  f"{cells['sd'][block]:6.2f}")
        for key, reason in cells['excluded']:
            print(f"    excluded {key}: {reason}")
    print('\ntransfer contrasts (paired, within participant):')
    for name in METRICS:
        for a, b in (('A', 'D'), ('A', 'B'), ('A', 'C')):
            diff, sem, n = transfer_difference(cells_list, name, a, b)
            print(f'  {name:>5s}  {a}→{b} costs {diff:+6.2f} ± '
                  f'{sem if numpy.isfinite(sem) else float("nan"):.2f}  (n={n})')

    out_dir = paths.ANALYSIS_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'group_transfer_' + '_'.join(c['subject_id'] for c in cells_list)
    for suffix in ('png', 'svg'):
        out = out_dir / f'{stem}.{suffix}'
        fig.savefig(out, bbox_inches='tight')
        print(f'\nwrote {out}')
