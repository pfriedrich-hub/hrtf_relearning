"""
learning_figures.py — the two group learning-curve figures built on group_learning.

  screen_figure()   for data collection. Post-training tests only. Participants
                    who have run the final 2x2 are the reference group (light
                    grey, mean ± SEM in black); anyone still mid-experiment is
                    drawn in colour on top, so a new subject reads directly
                    against the group as data accumulate.

  prepost_figure()  for the paper. Group means only, individual post-training
                    traces in light grey behind them. Filled = after that day's
                    training; open = the next morning's pre-training test,
                    drawn just left of its day. Dotted = overnight (post day
                    n-1 -> pre day n), thin grey = within-session (pre -> post).

Both use elevation gain and polar error, side by side — the two primary
metrics, read together. Day 1 is the baseline (block A of the day-1 2x2, before
any training) in both. The final-day 2x2 battery is never on the curve
(group_learning.EXCLUDE_FINAL_2X2); it is its own result.

Pre/post is decided per test by group_learning.test_phase: training logged
earlier that day, or an earlier same-day test on the composite >= 30 min before.

  transfer_screen_figure()   the final 2x2 per cell, day 1 vs final battery, same
                    colour logic as screen_figure (an active participant shows
                    day 1 only until the battery has run).

  transfer_figure() for the paper. Change per cell from day 1 to the final
                    battery, signed so that UP is always improvement (gain up,
                    polar error down); the 75-trial noise band is shaded.
                    Within-ear change scores: the foreign-composite
                    disadvantage cancels within each cell.

Cells: A trained ear/trained side, D other ear/other side (A mirrored: the same
filter, ears swapped — the main transfer test), B trained ear/other side,
C other ear/trained side. Day-1 cell = earliest run of that cell on the first
day carrying the composite; final cell = the battery as found by
group_learning.final_battery (block order from the sheet), except where
FINAL_A_OVERRIDE names an agreed substitute.

Add new participants to SUBJECT_IDS and run this file; all four figures are
saved to ANALYSIS_RESULTS_DIR / 'learning_curves'.
"""
import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from hrtf_relearning.experiment.analysis.localization.plot import group_learning as gl
from hrtf_relearning.experiment.analysis.localization.plot.elevation_learning import (
    extract_day, extract_seq_meta, trained_hemifield, transfer_block)
from hrtf_relearning.utils import paths

# --- configuration -----------------------------------------------------------
#: the learning-curve group. New participants go here as soon as they start.
SUBJECT_IDS = ('FS', 'IR', 'SH', 'PA', 'SK', 'DA', 'EM', 'MM', 'LP')
#: participants deliberately kept off the curve, with the reason (printed on run)
EXCLUDED = {
    'LGL': 'not counted as a learner — EG trend only, polar error flat (2026-10-22)',
    # flat-noise training cohort (Jul-Sep); FS and IR are in
    'AS': 'no learning: post-training dEG +.10, dPE 0.5 deg; day-1 impairment only +4.1',
    'GM': 'no learning: dEG -.01, dPE 0.3 deg (and her PE change is head pose)',
    'LS': 'borderline, kept out like LGL: dEG +.19, dPE 4.1 deg, both inside the noise band',
    'NR': 'aborted — no final 2x2',
    'IK': 'mixed-stimulus cohort; borderline like LGL/LS: dEG +.18, dPE 3.4 deg',
}
#: the paper figure uses only participants who have completed the final 2x2
PAPER_COMPLETE_ONLY = True
OUT_DIR = paths.ANALYSIS_RESULTS_DIR / 'learning_curves'
#: colours for participants still mid-experiment in the screen figure
ACTIVE_COLOURS = ('#c94f2e', '#1b6ca8', '#2e8b57', '#9467bd', '#d4a017', '#8c564b')
GREY = '0.72'
PRE_DX = 0.28               # pre-test drawn this far left of its day
PANELS = (('gain', 'Elevation gain'), ('polar', 'Polar error [°]'))
OWN_X, DOME_X = -0.25, -0.75  # reference columns, left of day 1
SHOW = False

# --- transfer (final 2x2) ----------------------------------------------------
#: drawing order: the assumption-free contrast first, the counterbalance cells after
CELLS = ('A', 'D', 'B', 'C')
CELL_MEANING = {'A': 'trained ear, trained side', 'D': 'other ear, other side',
                'B': 'trained ear, other side', 'C': 'other ear, trained side'}
#: agreed per-participant substitutes for a final-battery block (cell -> key prefix).
#: SH: told right after her 17:40 post-training test that the battery probes the
#: other ear/side; the battery's A (17:47, first block) took the uncertainty.
FINAL_OVERRIDE = {'SH': {'A': 'SH_01.10_17-40'}}
#: change-score noise band, two 75-trial blocks (behavioural noise floor)
NOISE_BAND = {'gain': 0.25, 'polar': 5.9}
HIGHER_IS_BETTER = {'gain': True, 'polar': False}


def _style(fs=8):
    plt.rcParams.update({
        'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
        'axes.titlesize': fs, 'lines.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.linewidth': 0.7,
        'axes.spines.right': False, 'axes.spines.top': False,
    })


def _frame(axis, name, max_day, refs=True):
    ticks = list(range(1, max_day + 1))
    labels = [str(d) for d in ticks]
    if refs:
        ticks = [DOME_X, OWN_X] + ticks
        labels = ['ear', 'own'] + labels
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_xlim((DOME_X if refs else 1) - 0.45, max_day + 0.35)
    axis.set_xlabel('Day')
    if name == 'gain':
        axis.set_ylim(0, 1.02)
        axis.set_yticks(numpy.arange(0, 1.2, 0.2))
    else:
        axis.set_ylim(0, None)
    for y in axis.get_yticks():
        axis.axhline(y, color='0.93', lw=0.5, zorder=-1)


def _trace(axis, series, name, colour, lw, ms, alpha, zorder):
    """One participant's curve; a segment touching an off-stimulus stand-in is dashed."""
    days, values, filled = series['days'], series[name], series['filled']
    for j in range(len(days) - 1):
        axis.plot(days[j:j + 2], values[j:j + 2], color=colour, lw=lw, alpha=alpha,
                  ls='--' if (filled[j] or filled[j + 1]) else '-', zorder=zorder)
    axis.plot(days[~filled], values[~filled], 'o', ms=ms, color=colour, alpha=alpha,
              ls='None', zorder=zorder)
    axis.plot(days[filled], values[filled], 'o', ms=ms, markerfacecolor='0.85',
              markeredgecolor=colour, alpha=alpha, ls='None', zorder=zorder)


def _mean(axis, series_list, name, x_shift=0.0, open_marker=False, line=True, zorder=4):
    """Group mean ± SEM. Days not every participant reached: no bar, half-filled."""
    if not series_list:
        return numpy.array([]), numpy.array([])
    days, means, sems, counts = gl.group_mean(series_list, name)
    ok = numpy.isfinite(means)
    days, means, sems, counts = days[ok], means[ok], sems[ok], counts[ok]
    x = days + x_shift
    if line:
        axis.plot(x, means, '-', color='0', lw=1.3, zorder=zorder - 1)
    full = counts == len(series_list)
    # a mean resting on a stand-in (another stimulus, or the battery's A) is
    # grey-faced, as in group_learning — it never claims more than its inputs
    impure = numpy.array([any(bool(s['filled'][s['days'] == d][0]) for s in series_list
                              if (s['days'] == d).any()) for d in days], dtype=bool)
    for mask, face in ((full & ~impure, 'white' if open_marker else '0'),
                       (full & impure, '0.75')):
        if mask.any():
            axis.errorbar(x[mask], means[mask], yerr=sems[mask], fmt='o', ms=3.4,
                          markerfacecolor=face, markeredgecolor='0', ecolor='0',
                          elinewidth=0.8, capsize=1.6, zorder=zorder)
    if (~full).any():
        axis.plot(x[~full], means[~full], 'o', ms=3.4, ls='None', zorder=zorder,
                  markerfacecolor='white' if open_marker else '0.6',
                  markeredgecolor='0')
    return days, means


def _references(axis, series_list, name, colour_of=None):
    """Free-field own ears ('ear') and own HRTF through AR ('own'), per participant + mean."""
    for x, prefix, marker in ((DOME_X, 'dome', 'o'), (OWN_X, 'own', 'D')):
        values = [(s, s[f'{prefix}_{name}']) for s in series_list]
        values = [(s, v) for s, v in values if numpy.isfinite(v)]
        for s, v in values:
            c = colour_of(s) if colour_of else GREY
            axis.plot([x], [v], marker=marker, ms=2.4, markerfacecolor='none',
                      markeredgecolor=c, ls='None', zorder=2)
        if values:
            axis.plot([x], [numpy.mean([v for _s, v in values])], marker=marker, ms=3.6,
                      markerfacecolor='none', markeredgecolor='0.2', mew=1.0,
                      ls='None', zorder=3)


def _report_exclusions():
    for sid, why in EXCLUDED.items():
        print(f'  {sid} excluded: {why}')


def _end_labels(axis, series_list, name, min_gap_pt=6):
    """Participant IDs at the end of their trace, nudged apart so none overlap."""
    ends = sorted(((s[name][-1], s['days'][-1], s['subject_id']) for s in series_list
                   if s['days'].size), key=lambda e: e[0])
    px_per_pt = axis.figure.dpi / 72
    to_px = lambda y: axis.transData.transform((0, y))[1]
    gap, placed = min_gap_pt * px_per_pt, []
    for y, x, sid in ends:
        offset = 0.0
        if placed and to_px(y) - placed[-1] < gap:
            offset = placed[-1] + gap - to_px(y)
        placed.append(to_px(y) + offset)
        axis.annotate(sid, (x, y), xytext=(3, offset / px_per_pt),
                      textcoords='offset points', fontsize=5, color='0.55', va='center')


def screen_figure(subject_ids=SUBJECT_IDS, show=SHOW):
    """Post-training curve; completed participants = grey reference group, active ones in colour."""
    series = [gl.subject_series(sid, tests='post') for sid in subject_ids]
    done = [s for s in series if s['complete']]
    active = [s for s in series if not s['complete']]
    colour = {s['subject_id']: ACTIVE_COLOURS[i % len(ACTIVE_COLOURS)]
              for i, s in enumerate(active)}
    colour_of = lambda s: colour.get(s['subject_id'], GREY)
    max_day = int(max(s['days'].max() for s in series))

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7 / 2.54),
                             constrained_layout=True, dpi=264)
    for axis, (name, label) in zip(axes, PANELS):
        _references(axis, series, name, colour_of)
        for s in done:
            _trace(axis, s, name, GREY, lw=0.8, ms=2.0, alpha=1.0, zorder=1)
        _mean(axis, done, name, zorder=4)
        for s in active:
            _trace(axis, s, name, colour[s['subject_id']], lw=1.4, ms=3.0, alpha=1.0,
                   zorder=5)
        _frame(axis, name, max_day)
        _end_labels(axis, done, name)   # after _frame: needs the final y-limits
        axis.set_ylabel(label)

    handles = [Line2D([], [], color='0', marker='o', ms=3.4, lw=1.3,
                      label=f'completed, mean ± SEM (n={len(done)})'),
               Line2D([], [], color=GREY, marker='o', ms=2.0, lw=0.8,
                      label='completed, individual')]
    handles += [Line2D([], [], color=colour[s['subject_id']], marker='o', ms=3.0,
                       lw=1.4, label=f"{s['subject_id']} (day {int(s['days'].max())})")
                for s in active]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc='lower right',
                   handletextpad=0.4, labelspacing=0.25)
    fig.suptitle('Learning screen — post-training tests (day 1 = baseline)', fontsize=9)
    fig.text(0.5, -0.02, "ear = free field, own ears · own = own HRTF through AR · "
             "grey face + dashed = stand-in: test on another stimulus, or the final 2x2's A where no separate post-test exists\n"
             "grey-faced mean = rests on a stand-in · light mean without bar = not every participant has that day",
             ha='center', va='top', fontsize=5.5, color='0.35')
    if show:
        plt.show(block=False)
    return fig, series


def prepost_figure(subject_ids=SUBJECT_IDS, complete_only=PAPER_COMPLETE_ONLY,
                   show=SHOW):
    """Post-training mean (filled) with next-morning pre-training mean (open)."""
    post = [gl.subject_series(sid, tests='post') for sid in subject_ids]
    keep = [s['subject_id'] for s in post if s['complete'] or not complete_only]
    post = [s for s in post if s['subject_id'] in keep]
    pre = [gl.subject_series(sid, tests='pre') for sid in keep]
    max_day = int(max(s['days'].max() for s in post + pre))

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7 / 2.54),
                             constrained_layout=True, dpi=264)
    for letter, axis, (name, label) in zip('AB', axes, PANELS):
        _references(axis, post, name)
        for s in post:
            _trace(axis, s, name, GREY, lw=0.7, ms=1.6, alpha=1.0, zorder=1)
        dp, mp = _mean(axis, post, name, zorder=5)
        # pre-tests from day 2 on (day 1 is the shared baseline, already drawn)
        pre_late = []
        for s in pre:
            late = s['days'] > 1
            s2 = dict(s)
            for k in ('days', 'filled', 'gain', 'polar', 'sd'):
                s2[k] = s[k][late]
            pre_late.append(s2)
        dq, mq = _mean(axis, [s for s in pre_late if s['days'].size], name,
                       x_shift=-PRE_DX, open_marker=True, line=False, zorder=6)
        post_at = dict(zip(dp, mp))
        for d, m in zip(dq, mq):
            if d - 1 in post_at:
                axis.plot([d - 1, d - PRE_DX], [post_at[d - 1], m], ':', color='0',
                          lw=0.9, zorder=4)
            if d in post_at:
                axis.plot([d - PRE_DX, d], [m, post_at[d]], '-', color='0.55', lw=0.8,
                          zorder=4)
        _frame(axis, name, max_day)
        axis.set_ylabel(label)
        axis.annotate(letter, xy=(-0.16, 1.0), xycoords='axes fraction', fontsize=8,
                      weight='bold')

    handles = [Line2D([], [], color='0', marker='o', ms=3.4, lw=1.3,
                      label='after training (mean ± SEM)'),
               Line2D([], [], color='0', marker='o', ms=3.4, markerfacecolor='white',
                      ls=':', lw=0.9, label='next day, before training'),
               Line2D([], [], color=GREY, lw=0.7, marker='o', ms=1.6,
                      label=f'individual, after training (n={len(post)})')]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc='lower right',
                   handletextpad=0.4, labelspacing=0.25)
    if show:
        plt.show(block=False)
    return fig, post, pre


# --- transfer -----------------------------------------------------------------

def transfer_cells(subject_id):
    """Day-1 and final-battery value per cell, and the change between them.

    Returns {'subject_id', 'complete', 'stim', 'day1': {metric: {cell: v}},
    'final': {...}, 'change': {...} (signed: positive = improvement),
    'stand_in': cells whose day-1 run is on another stimulus, 'keys'}.
    """
    runs = gl._runs(subject_id)
    metas = [extract_seq_meta(seq, key) for key, seq in runs]
    hrir, stim = gl._curve_hrir(runs), gl._curve_stim(runs)
    field = trained_hemifield(metas)
    min_trials = gl.MIN_TRIALS_FRACTION * gl._curve_n_trials(runs)
    family = {hrir, f'{hrir}_mirrored'}
    by_key = dict(runs)
    curve = [(key, seq, meta) for (key, seq), meta in zip(runs, metas)
             if meta.get('hrir') in family and len(seq.data) >= min_trials]
    first_day = extract_day(curve[0][0]) if curve else None

    day1_keys, stand_in = {}, set()
    for key, seq, meta in curve:
        if extract_day(key) != first_day:
            continue
        cell = transfer_block({**meta, 'stim': None}, field)
        if cell not in CELLS:
            continue
        on_stim = (meta.get('stim') or '').lower() == stim
        # earliest run of the cell; an on-stimulus run replaces an off-stimulus one
        if cell not in day1_keys or (on_stim and cell in stand_in):
            day1_keys[cell] = key
            (stand_in.discard if on_stim else stand_in.add)(cell)

    final_keys = gl.final_battery(runs, metas, hrir, field, subject_id)
    for cell, prefix in FINAL_OVERRIDE.get(subject_id, {}).items():
        match = [k for k in by_key if k.startswith(prefix)]
        if final_keys and match:
            final_keys[cell] = match[0]

    out = {'subject_id': subject_id, 'complete': bool(final_keys), 'stim': stim,
           'stand_in': stand_in, 'keys': {'day1': day1_keys, 'final': final_keys},
           'day1': {}, 'final': {}, 'change': {}}
    for name, _label in PANELS:
        extract = gl.METRICS[name][1]
        out['day1'][name] = {c: extract(by_key[k]) for c, k in day1_keys.items()}
        out['final'][name] = {c: extract(by_key[k]) for c, k in final_keys.items()}
        sign = 1.0 if HIGHER_IS_BETTER[name] else -1.0
        out['change'][name] = {c: sign * (out['final'][name][c] - out['day1'][name][c])
                               for c in CELLS
                               if c in out['final'][name] and c in out['day1'][name]}
    return out


def _cell_mean(values):
    values = [v for v in values if numpy.isfinite(v)]
    if not values:
        return numpy.nan, numpy.nan, 0
    sem = numpy.std(values, ddof=1) / numpy.sqrt(len(values)) if len(values) > 1 else numpy.nan
    return float(numpy.mean(values)), sem, len(values)


def _cell_axis(axis, ylabel, n_label=None):
    axis.set_xticks(range(len(CELLS)))
    axis.set_xticklabels([f'{c}\n{n_label[c]}' if n_label else c for c in CELLS])
    axis.set_xlim(-0.6, len(CELLS) - 0.4)
    axis.axvline(1.5, color='0.85', lw=0.6, zorder=-1)   # A D | B C
    axis.set_ylabel(ylabel)


def _transfer_footnote(fig, extra=''):
    text = ' · '.join(f'{c} = {CELL_MEANING[c]}' for c in CELLS)
    fig.text(0.5, -0.02, text + (f'\n{extra}' if extra else ''), ha='center',
             va='top', fontsize=5.5, color='0.35')


def transfer_screen_figure(subject_ids=SUBJECT_IDS, show=SHOW):
    """Day 1 vs final battery per cell; completed grey + mean, active in colour."""
    cells = [transfer_cells(sid) for sid in subject_ids]
    done = [c for c in cells if c['complete']]
    active = [c for c in cells if not c['complete']]
    colour = {c['subject_id']: ACTIVE_COLOURS[i % len(ACTIVE_COLOURS)]
              for i, c in enumerate(active)}
    dx = 0.2

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7 / 2.54),
                             constrained_layout=True, dpi=264)
    for axis, (name, label) in zip(axes, PANELS):
        for i, cell in enumerate(CELLS):
            for c in done:
                a, b = c['day1'][name].get(cell), c['final'][name].get(cell)
                if a is not None and b is not None:
                    axis.plot([i - dx, i + dx], [a, b], '-', color=GREY, lw=0.8, zorder=1)
                for x, v in ((i - dx, a), (i + dx, b)):
                    if v is not None:
                        face = '0.85' if (x < i and cell in c['stand_in']) else GREY
                        axis.plot([x], [v], 'o', ms=2.0, color=GREY,
                                  markerfacecolor=face, zorder=1)
            m1 = _cell_mean([c['day1'][name].get(cell, numpy.nan) for c in done])
            m2 = _cell_mean([c['final'][name].get(cell, numpy.nan) for c in done])
            if numpy.isfinite(m1[0]) and numpy.isfinite(m2[0]):
                axis.plot([i - dx, i + dx], [m1[0], m2[0]], '-', color='0', lw=1.3, zorder=3)
            for x, (m, e, n) in ((i - dx, m1), (i + dx, m2)):
                if numpy.isfinite(m):
                    full = n == len(done)
                    axis.errorbar([x], [m], yerr=[e] if full and numpy.isfinite(e) else None,
                                  fmt='o', ms=3.4, color='0',
                                  markerfacecolor='0' if full else 'white',
                                  elinewidth=0.8, capsize=1.6, zorder=4)
            for c in active:
                col = colour[c['subject_id']]
                a, b = c['day1'][name].get(cell), c['final'][name].get(cell)
                if a is not None and b is not None:
                    axis.plot([i - dx, i + dx], [a, b], '-', color=col, lw=1.4, zorder=5)
                for x, v in ((i - dx, a), (i + dx, b)):
                    if v is not None:
                        axis.plot([x], [v], 'o', ms=3.0, color=col, zorder=5)
        _cell_axis(axis, label)
        if name == 'gain':
            low = min([0.0] + [v for c in cells for d in (c['day1'], c['final'])
                               for v in d[name].values() if numpy.isfinite(v)])
            axis.set_ylim(min(0.0, low - 0.04), 1.02)
        else:
            axis.set_ylim(0, None)
        for y in axis.get_yticks():
            axis.axhline(y, color='0.93', lw=0.5, zorder=-1)
    handles = [Line2D([], [], color='0', marker='o', ms=3.4, lw=1.3,
                      label=f'completed, mean ± SEM (n={len(done)})'),
               Line2D([], [], color=GREY, marker='o', ms=2.0, lw=0.8,
                      label='completed, individual')]
    handles += [Line2D([], [], color=colour[c['subject_id']], marker='o', ms=3.0, lw=1.4,
                       label=f"{c['subject_id']} (day 1 only)" if not c['final']['gain']
                       else c['subject_id']) for c in active]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc='upper right',
                   handletextpad=0.4, labelspacing=0.25)
    fig.suptitle('Transfer screen — each cell: day 1 (left) vs final 2x2 (right)', fontsize=9)
    _transfer_footnote(fig, 'grey face = day-1 run on another stimulus · '
                            'open mean = not every completed participant has that cell')
    if show:
        plt.show(block=False)
    return fig, cells


def transfer_figure(subject_ids=SUBJECT_IDS, complete_only=PAPER_COMPLETE_ONLY,
                    show=SHOW):
    """Change per cell, day 1 -> final battery, positive = improvement."""
    cells = [transfer_cells(sid) for sid in subject_ids]
    cells = [c for c in cells if c['complete'] or not complete_only]

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7 / 2.54),
                             constrained_layout=True, dpi=264)
    labels = {'gain': 'Δ elevation gain', 'polar': 'Polar error reduction [°]'}
    for axis, (name, _label) in zip(axes, PANELS):
        band = NOISE_BAND[name]
        axis.add_patch(Rectangle((-0.6, -band), len(CELLS) + 0.2, 2 * band,
                                 color='0.94', lw=0, zorder=-2))
        axis.axhline(0, color='0.6', lw=0.7, zorder=-1)
        for c in cells:
            xs = [i for i, cell in enumerate(CELLS) if cell in c['change'][name]]
            ys = [c['change'][name][CELLS[i]] for i in xs]
            # connect only adjacent cells inside each pair (A-D, B-C)
            for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                if (x0, x1) in ((0, 1), (2, 3)):
                    axis.plot([x0, x1], [y0, y1], '-', color=GREY, lw=0.7, zorder=1)
            for x, y in zip(xs, ys):
                face = '0.85' if CELLS[x] in c['stand_in'] else GREY
                axis.plot([x], [y], 'o', ms=1.8, color=GREY, markerfacecolor=face, zorder=1)
        n_label = {}
        for i, cell in enumerate(CELLS):
            vals = [c['change'][name].get(cell, numpy.nan) for c in cells]
            m, e, n = _cell_mean(vals)
            n_label[cell] = f'n={n}'
            if not numpy.isfinite(m):
                continue
            impure = any(cell in c['stand_in'] for c in cells if cell in c['change'][name])
            axis.errorbar([i], [m], yerr=[e] if numpy.isfinite(e) else None, fmt='o',
                          ms=3.6, color='0', markerfacecolor='0.75' if impure else '0',
                          elinewidth=0.9, capsize=2, zorder=4)
        _cell_axis(axis, labels[name], n_label)
    handles = [Line2D([], [], color='0', marker='o', ms=3.6, ls='None',
                      label='mean ± SEM'),
               Line2D([], [], color=GREY, marker='o', ms=1.8, lw=0.7,
                      label=f'individual (n={len(cells)})'),
               Rectangle((0, 0), 1, 1, color='0.94', label='test-retest noise band')]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc='lower left',
                   handletextpad=0.4, labelspacing=0.25)
    _transfer_footnote(fig, 'change from day 1 to the final 2x2, up = improvement · '
                            'grey-faced = day-1 run on another stimulus')
    if show:
        plt.show(block=False)
    return fig, cells


def _print(series_list, title):
    print(f'\n{title}')
    for s in series_list:
        rows = ' '.join(f"d{int(d)} {g:.2f}/{p:.1f}"
                        for d, g, p in zip(s['days'], s['gain'], s['polar']))
        print(f"  {s['subject_id']:4s} {'done' if s['complete'] else 'active':6s} "
              f"{s['stim']:7s} EG/PE  {rows}")


if __name__ == '__main__':
    _report_exclusions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, series = screen_figure()
    _print(series, 'screen (post-training)')
    for ext in ('png', 'svg'):
        fig.savefig(OUT_DIR / f'learning_screen.{ext}', bbox_inches='tight')
    fig, post, pre = prepost_figure()
    _print(pre, 'paper, pre-training tests')
    for ext in ('png', 'svg'):
        fig.savefig(OUT_DIR / f'learning_prepost.{ext}', bbox_inches='tight')
    fig, cells = transfer_screen_figure()
    for ext in ('png', 'svg'):
        fig.savefig(OUT_DIR / f'transfer_screen.{ext}', bbox_inches='tight')
    fig, cells = transfer_figure()
    for ext in ('png', 'svg'):
        fig.savefig(OUT_DIR / f'transfer_change.{ext}', bbox_inches='tight')
    print('\ntransfer, change day 1 -> final (positive = improvement), EG / PE')
    for c in cells:
        print(f"  {c['subject_id']:4s} " + '  '.join(
            f"{cell}: {c['change']['gain'].get(cell, numpy.nan):+.2f} / "
            f"{c['change']['polar'].get(cell, numpy.nan):+5.1f}" for cell in CELLS))
    print(f'\nsaved to {OUT_DIR}')
