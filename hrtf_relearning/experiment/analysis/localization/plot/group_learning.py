"""
group_learning.py — one figure for the learning of several participants.

The group counterpart of :mod:`elevation_learning`, which draws one subject in
full detail. Here each participant contributes ONE point per training day and
the figure is about what they have in common, in the layout of Trapeau et al.
(2016) fig. 4:

    A (large)  elevation gain          the primary axis
    B          polar error, degrees    the primary metric (see polar_error)
    C          residual SD, degrees    precision, gain fitted out

Individual participants are drawn as faint coloured traces, the group mean
±SEM in black on top. With three participants the individual traces are the
data and the mean is a reading aid, not the other way round — which is why they
are labelled rather than anonymised to grey.

WHAT COUNTS AS A POINT ON THE CURVE. The trained condition — composite HRTF,
unmirrored, trained hemifield (block A of the final 2x2,
:func:`elevation_learning.transfer_block`) — narrowed by three per-participant
modal tests, each of which turned out to matter on this cohort:

  * modal HRIR. FS's day 1 ladders the other-ear treatment and n_keep
    (``_left_nat``, ``_left`` flat, ``_n8_left_env4``). Those are block A by
    every structural test and are not the trained condition.
  * modal TRIAL COUNT (>= MIN_TRIALS_FRACTION of it). FS's day 1 also holds
    10-trial externalization-ladder rungs on the curve's own HRIR and stimulus;
    one of them alone moved his day-1 gain from 0.42 to 0.32.
  * modal STIMULUS. The participants are not on the same one — FS and IR ran on
    noise, AS on ripple — and AS's day 1 holds one noise block among otherwise
    ripple blocks. Comparing days within a participant needs the stimulus held
    constant.

OFF-STIMULUS RUNS (``OFF_STIMULUS``). A trained-condition run on some other
stimulus is not simply dropped. Under the default ``'fill_gaps'`` it goes on
the curve only on a day that has NO on-stimulus test — the completeness case,
and nothing wider. That recovers IR's day 1, which is USO because his entire
first day was (a module-level STIM left at 'uso', see
project_stimulus_source_variation) and was never redone; he has since trained,
so it cannot be recovered any other way. It leaves FS's day-4 USO probe and
AS's day-1 noise block off the curve, since those days already hold the real
measurement.

Such a point is drawn grey-faced, the segments touching it are dashed, and a
group mean resting on one is grey-faced too. Read it as a lower bound: USO is
the harder stimulus, so IR's day 1 sits lower than a noise test that morning
would have, and the day-1-to-day-2 rise for him is therefore an overestimate.
``OFF_STIMULUS='never'`` reverts to a genuine gap.

Runs whose responses never moved are discarded up front
(:func:`localization_analysis.is_degenerate`) — IR's two 18.08 dome blocks are
both a frozen head tracker and score as a plausible-looking gain of exactly
0.00 rather than as an obvious failure.

Days are indexed per participant (day 1 = that participant's first session),
not by calendar date, since the three ran weeks apart. Where a day holds more
than one test of the trained condition (typically pre- and post-training) the
day's value is their mean; ``aggregate='first'`` or ``'last'`` picks one.

Run this file directly; it saves to ``ANALYSIS_RESULTS_DIR``.
"""

from collections import OrderedDict, Counter

import numpy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import hrtf_relearning as hr
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    _azimuth_span, is_degenerate, localization_accuracy, polar_error)
from hrtf_relearning.experiment.analysis.localization.plot.elevation_learning import (
    _is_analysable, condition_of_meta, extract_day, extract_seq_meta,
    parse_loc_key, transfer_block, trained_hemifield)
from hrtf_relearning.utils import paths

# --- configuration -----------------------------------------------------------
SUBJECT_IDS = ('FS', 'IR', 'AS')
COLOURS = {'FS': '#1b6ca8', 'IR': '#2e8b57', 'AS': '#c94f2e'}
AGGREGATE = 'mean'          # 'mean' | 'first' | 'last' within a day
MIN_TRIALS_FRACTION = 0.5   # of the modal test length — see _curve_n_trials
#: what to do with a trained-condition run on a stimulus other than the
#: participant's modal one. 'fill_gaps' puts it on the curve only where the day
#: has nothing else (IR's day 1); 'never' keeps it off; 'always' pools it in.
OFF_STIMULUS = 'fill_gaps'  # 'fill_gaps' | 'never' | 'always'
SHOW = False

#: (row label, extractor) for the three panels, in drawing order.
METRICS = OrderedDict((
    ('gain', ('Elevation gain', lambda seq: localization_accuracy(seq)[0])),
    ('polar', ('Polar error [°]', lambda seq: polar_error(seq)[0])),
    ('sd', ('SD [°]', lambda seq: localization_accuracy(seq)[2])),
))


def _runs(subject_id):
    """Analysable, non-degenerate localization runs of one subject, chronological."""
    localization = hr.Subject(subject_id).localization
    items = sorted(localization.items(), key=lambda kv: parse_loc_key(kv[0]))
    return [(key, seq) for key, seq in items
            if _is_analysable(seq) and not is_degenerate(seq)]


def _learning_runs(runs):
    """The subset on a modified HRIR — what the modal tests below are taken over."""
    return [(key, seq) for key, seq in runs
            if condition_of_meta(extract_seq_meta(seq, key)) == 'learning']


def _curve_hrir(runs):
    """The HRIR name the learning curve is measured on = the modal one.

    Taking the mode rather than "any modified HRIR" is what excludes a day-1
    ladder over other-ear treatment or n_keep: those are one block each, the
    trained condition is every other day.
    """
    names = [extract_seq_meta(seq, key).get('hrir')
             for key, seq in _learning_runs(runs)]
    names = [n for n in names if n]
    return Counter(names).most_common(1)[0][0] if names else None


def _curve_stim(runs):
    """The stimulus the curve is measured on = the modal one, per participant.

    Not a constant, because the participants are not on the same stimulus: FS
    and IR ran on gapped pinknoise throughout, AS on the rippled stimulus from
    her first afternoon block onward. Comparing days WITHIN a participant needs
    one stimulus held constant, and taking each one's own mode is what does
    that — it is also strictly better than the old "anything but USO" rule,
    which let AS's single day-1 noise block onto a curve that is otherwise all
    ripple, and which cost her 0.06 gain and 1.6 deg of polar error on day 1
    for reasons that had nothing to do with learning.
    """
    stims = [(extract_seq_meta(seq, key).get('stim') or '').lower()
             for key, seq in _learning_runs(runs)]
    stims = [s for s in stims if s]
    return Counter(stims).most_common(1)[0][0] if stims else None


def _curve_n_trials(runs):
    """The modal trial count of the trained condition — the full-length test.

    FS's day 1 contains four 10-trial rungs of the externalization ladder
    (`run_ext_check`), one of which is on the curve's own HRIR and stimulus and
    passes every other test. Ten trials is a probe, not a measurement: that
    rung alone pulls his day-1 elevation gain from 0.42 to 0.32. Runs shorter
    than MIN_TRIALS_FRACTION of the modal length are dropped.
    """
    counts = [len(seq.data) for _key, seq in _learning_runs(runs)]
    return Counter(counts).most_common(1)[0][0] if counts else 0


def subject_series(subject_id, aggregate=AGGREGATE, off_stimulus=OFF_STIMULUS):
    """One participant's curve.

    Returns
    -------
    dict with 'days' (1-based day index), one array per key of METRICS, the
    day-1 own-HRTF AR reference ('own'), the free-field dome reference
    ('dome'), and 'excluded' — runs that were on the curve's HRIR but dropped,
    with the reason, so a gap in the curve can always be explained.
    """
    runs = _runs(subject_id)
    metas = [extract_seq_meta(seq, key) for key, seq in runs]
    field = trained_hemifield(metas)
    hrir = _curve_hrir(runs)
    stim = _curve_stim(runs)
    min_trials = MIN_TRIALS_FRACTION * _curve_n_trials(runs)

    last_day = extract_day(runs[-1][0]) if runs else None

    by_day, uso_by_day, excluded = OrderedDict(), OrderedDict(), []
    own_reference, dome_reference, post_reference = None, None, None
    for (key, seq), meta in zip(runs, metas):
        day = extract_day(key)
        by_day.setdefault(day, [])
        run_stim = (meta.get('stim') or '').lower()

        if meta.get('hrir') is None:                       # free-field dome
            if dome_reference is None and run_stim != 'uso':
                dome_reference = seq
            continue
        if condition_of_meta(meta) == 'baseline':          # own HRTF through AR
            # Midline-only blocks (azimuth_range (-1, 1)) are a different task:
            # elevation gain on one azimuth column is not comparable to the
            # sector blocks the curve is made of, and FS's reads 1.01 where his
            # sector reference reads 0.75. Take the first non-USO SECTOR block.
            if (own_reference is None and run_stim != 'uso'
                    and _azimuth_span(seq) > 2):
                own_reference = seq
            # The same measurement repeated on the LAST day is the post-training
            # own-HRTF test — the aftereffect probe, "has the native map moved".
            # `last` wins, so a day holding several keeps the final one.
            elif (day == last_day and run_stim != 'uso'
                  and _azimuth_span(seq) > 2):
                post_reference = seq
            continue
        if meta.get('hrir') != hrir:
            excluded.append((key, f'other HRIR ({meta.get("hrir")})'))
            continue
        # transfer_block returns None for any USO run, so the block has to be
        # resolved on a meta with the stimulus masked out — otherwise a USO run
        # of the trained condition is indistinguishable from a mirrored one.
        if transfer_block({**meta, 'stim': None}, field) != 'A':
            excluded.append((key, 'not the trained condition'))
            continue
        if len(seq.data) < min_trials:
            excluded.append((key, f'{len(seq.data)} trials — a probe, not a test'))
            continue
        if run_stim != stim:
            # The trained condition, but not on the stimulus the rest of the
            # series is on. Kept aside rather than dropped — what happens to it
            # is decided below by `off_stimulus`.
            uso_by_day.setdefault(day, []).append(seq)
            excluded.append((key, f'{run_stim} stimulus (curve is {stim})'))
            continue
        by_day[day].append(seq)

    def _aggregate(seqs, extract):
        scores = [extract(seq) for seq in seqs]
        scores = [v for v in scores if v is not None and numpy.isfinite(v)]
        if not scores:
            return numpy.nan
        if aggregate == 'first':
            return scores[0]
        if aggregate == 'last':
            return scores[-1]
        return float(numpy.mean(scores))

    # --- decide what the off-stimulus blocks are allowed to do ---------------
    # 'fill_gaps' promotes one onto the curve ONLY on a day that has no
    # on-stimulus test at all. That is the completeness case and nothing more:
    # it recovers IR's day 1, which is USO because his whole first day was, and
    # leaves FS's day-4 USO probe and AS's day-1 noise block off the curve,
    # since those days already have the real measurement and mixing a second
    # stimulus in would only add variance the day does not have.
    filled = set()
    if off_stimulus in ('fill_gaps', 'always'):
        for day, seqs in uso_by_day.items():
            if off_stimulus == 'always' or not by_day.get(day):
                by_day.setdefault(day, []).extend(seqs)
                filled.add(day)

    days = [day for day, seqs in by_day.items() if seqs]
    uso_days = [day for day in by_day if uso_by_day.get(day) and day not in filled]
    series = {'subject_id': subject_id, 'hrir': hrir, 'trained_field': field,
              'stim': stim, 'min_trials': min_trials, 'off_stimulus': off_stimulus,
              'dates': days, 'excluded': excluded, 'uso_dates': uso_days,
              'n_tests': [len(by_day[d]) for d in days]}
    # Day index counts EVERY session the subject attended, so a day whose
    # trained-condition blocks were all excluded leaves a visible gap in x
    # rather than silently closing up.
    all_days = list(by_day)
    series['days'] = numpy.array([all_days.index(d) + 1 for d in days], dtype=float)
    series['uso_days'] = numpy.array([all_days.index(d) + 1 for d in uso_days],
                                     dtype=float)
    #: mask over 'days' — True where the point is an off-stimulus stand-in.
    series['filled'] = numpy.array([day in filled for day in days], dtype=bool)
    #: which stimulus those stand-ins were run on (they are all the same one
    #: on this cohort; the first is representative).
    stand_in = [seq for day in filled for seq in uso_by_day[day]]
    series['uso_stim'] = ((getattr(stand_in[0], 'stim', None) or '').lower()
                          if stand_in else None)

    for name, (_label, extract) in METRICS.items():
        series[name] = numpy.array(
            [_aggregate(by_day[day], extract) for day in days], dtype=float)
        series[f'uso_{name}'] = numpy.array(
            [_aggregate(uso_by_day[day], extract) for day in uso_days], dtype=float)
        series[f'own_{name}'] = (extract(own_reference) if own_reference is not None
                                 else numpy.nan)
        series[f'dome_{name}'] = (extract(dome_reference) if dome_reference is not None
                                  else numpy.nan)
        series[f'post_{name}'] = (extract(post_reference) if post_reference is not None
                                  else numpy.nan)
    return series


def group_mean(series_list, key):
    """Per-day mean ±SEM across participants, over the days anyone has.

    Days with a single participant get an SEM of 0, which would draw as a point
    with no bar and read as a precise estimate; those are returned as nan so the
    caller can draw the marker without one.
    """
    max_day = int(max(s['days'].max() for s in series_list))
    days = numpy.arange(1, max_day + 1, dtype=float)
    means, sems, counts = [], [], []
    for day in days:
        values = [float(s[key][s['days'] == day][0]) for s in series_list
                  if (s['days'] == day).any()]
        values = [v for v in values if numpy.isfinite(v)]
        counts.append(len(values))
        means.append(numpy.mean(values) if values else numpy.nan)
        sems.append(numpy.std(values, ddof=1) / numpy.sqrt(len(values))
                    if len(values) > 1 else numpy.nan)
    return days, numpy.array(means), numpy.array(sems), numpy.array(counts)


def figure(subject_ids=SUBJECT_IDS, aggregate=AGGREGATE,
           off_stimulus=OFF_STIMULUS, show=SHOW):
    series_list = [subject_series(sid, aggregate, off_stimulus)
                   for sid in subject_ids]

    figsize_cm = (17.5, 7.5)
    fs = 8
    plt.rcParams.update({
        'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
        'axes.titlesize': fs, 'lines.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.linewidth': 0.7,
        'axes.spines.right': False, 'axes.spines.top': False,
    })
    fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54),
                     constrained_layout=True, dpi=264)
    ax_gain = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    axes = {'gain': ax_gain,
            'polar': plt.subplot2grid((2, 3), (0, 2)),
            'sd': plt.subplot2grid((2, 3), (1, 2))}

    dome_x, own_x = 0.0, 0.45      # references, left of day 1
    max_day = int(max(s['days'].max() for s in series_list))
    post_x = max_day + 0.55        # post-training own HRTF, right of the last day
    has_post = any(numpy.isfinite(s[f'post_{k}']) for s in series_list
                   for k in METRICS)

    for name, axis in axes.items():
        # --- references: real ear on the dome, own HRTF through the AR chain -
        references = [(dome_x, 'dome', 'o'), (own_x, 'own', 'D')]
        if has_post:
            references.append((post_x, 'post', 'D'))
        for x, prefix, marker in references:
            values = [s[f'{prefix}_{name}'] for s in series_list]
            for series, value in zip(series_list, values):
                if numpy.isfinite(value):
                    axis.plot([x], [value], marker=marker, ms=2.2,
                              markerfacecolor='none',
                              markeredgecolor=COLOURS[series['subject_id']],
                              alpha=0.55, linestyle='None', zorder=1)
            finite = [v for v in values if numpy.isfinite(v)]
            if finite:
                axis.plot([x], [numpy.mean(finite)], marker=marker, ms=3.4,
                          markerfacecolor='none', markeredgecolor='0.3',
                          linestyle='None', zorder=2)

        # --- individual traces -----------------------------------------------
        for series in series_list:
            colour = COLOURS[series['subject_id']]
            days_i, values, filled = series['days'], series[name], series['filled']
            # A segment that touches a filled point is dashed: the two ends are
            # not the same measurement, so the slope across it is not a clean
            # day-to-day change.
            for j in range(len(days_i) - 1):
                dashed = bool(filled[j] or filled[j + 1])
                axis.plot(days_i[j:j + 2], values[j:j + 2],
                          linestyle='--' if dashed else '-', color=colour,
                          alpha=0.55, lw=0.8, zorder=2)
            axis.plot(days_i[~filled], values[~filled], 'o', ms=2.2, color=colour,
                      alpha=0.55, linestyle='None', zorder=2)
            axis.plot(days_i[filled], values[filled], 'o', ms=2.2,
                      markerfacecolor='0.75', markeredgecolor=colour,
                      alpha=0.9, linestyle='None', zorder=2)
            # trained condition measured on a DIFFERENT stimulus: same marker,
            # grey face, never joined to the curve (the repo's convention, see
            # elevation_learning.is_uso). Not the same measurement, so it must
            # not be read as one.
            if series['uso_days'].size:
                axis.plot(series['uso_days'], series[f'uso_{name}'], 'o', ms=2.2,
                          markerfacecolor='0.75',
                          markeredgecolor=COLOURS[series['subject_id']],
                          linestyle='None', alpha=0.9, zorder=2)

        # --- group mean +- SEM ------------------------------------------------
        days, means, sems, counts = group_mean(series_list, name)
        complete = counts == len(series_list)
        # a day whose group value rests on an off-stimulus stand-in is marked,
        # so the mean never claims more comparability than its inputs have
        impure = numpy.array([
            any(bool(s['filled'][s['days'] == d][0]) for s in series_list
                if (s['days'] == d).any())
            for d in days])
        axis.plot(days, means, '-', color='0', lw=1.3, zorder=3)
        solid = complete & ~impure
        axis.errorbar(days[solid], means[solid], yerr=sems[solid],
                      fmt='o', ms=3.2, color='0', ecolor='0', elinewidth=0.8,
                      capsize=1.6, zorder=4)
        mixed = complete & impure
        if mixed.any():
            axis.errorbar(days[mixed], means[mixed], yerr=sems[mixed], fmt='o',
                          ms=3.2, markerfacecolor='0.75', markeredgecolor='0',
                          ecolor='0', elinewidth=0.8, capsize=1.6, zorder=4)
        # Days not every participant reached carry no SEM. Drawn open and
        # without a bar so an n=1 or n=2 day is never read as a group estimate.
        if (~complete).any():
            axis.plot(days[~complete], means[~complete], 'o', ms=3.2,
                      markerfacecolor='white', markeredgecolor='0',
                      linestyle='None', zorder=4)

        ticks = [dome_x, own_x] + list(range(1, max_day + 1))
        # The reference ticks are only labelled on the wide panel; in B and C
        # 'ear' and 'own' collide at this width.
        labels = (['ear', 'own'] if name == 'gain' else ['', ''])
        labels += [str(d) for d in range(1, max_day + 1)]
        if has_post:
            ticks.append(post_x)
            labels.append('post' if name == 'gain' else '')
            axis.axvline(max_day + 0.28, color='0.88', lw=0.6, zorder=-1)
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.set_xlim(dome_x - 0.35, (post_x if has_post else max_day) + 0.35)
        axis.set_ylabel(METRICS[name][0])
        for y in axis.get_yticks():
            axis.axhline(y, color='0.92', lw=0.5, zorder=-1)

    ax_gain.set_ylim(0, 1.02)
    ax_gain.set_yticks(numpy.arange(0, 1.2, 0.2))
    ax_gain.set_xlabel('Days')
    axes['polar'].set_xticklabels([])
    axes['sd'].set_xlabel('Days')
    for name in ('polar', 'sd'):
        axes[name].set_ylim(0, None)

    for letter, axis, offset in (('A', ax_gain, -0.10), ('B', axes['polar'], -0.30),
                                 ('C', axes['sd'], -0.30)):
        axis.annotate(letter, xy=(offset, 1.005), xycoords='axes fraction',
                      fontsize=fs, weight='bold')

    handles = [Line2D([0], [0], color=COLOURS[s['subject_id']], marker='o',
                      ms=2.2, lw=0.8, alpha=0.7, label=s['subject_id'])
               for s in series_list]
    # Only the series belong in the legend. The marker conventions are spelled
    # out in the footnote instead: as entries they made the box wide enough to
    # sit on IR's day-1 off-curve point.
    handles += [Line2D([0], [0], color='0', marker='o', ms=3.2, lw=1.3,
                       label=f'mean ± SEM (n={len(series_list)})')]
    ax_gain.legend(handles=handles, frameon=False, fontsize=6, loc='lower right',
                   handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3, ncol=2,
                   columnspacing=1.0)

    fig.suptitle('Relearning with own envelope + donor fine detail on one ear',
                 fontsize=fs + 1)
    # Say how much of each participant's series is actually there. A curve that
    # stops early because the participant is mid-experiment looks identical to
    # one that stops because the effect plateaued.
    parts = []
    for series in series_list:
        sessions = int(series['days'].max()) if series['days'].size else 0
        missing = sorted({int(d) for d in range(1, sessions + 1)}
                         - {int(d) for d in series['days']})
        # name the days that are only there because an off-stimulus test filled
        # them — "IR 4/4 d" would otherwise read as four comparable days
        stood_in = [int(d) for d, f in zip(series['days'], series['filled']) if f]
        if missing:
            gap = f", no day {', '.join(map(str, missing))} point"
        elif stood_in:
            gap = (f", day {', '.join(map(str, stood_in))} on "
                   f"{series['uso_stim'] or 'another stimulus'}")
        else:
            gap = ''
        parts.append(f"{series['subject_id']} {len(series['days'])}/{sessions} d "
                     f"on {series['stim']}{gap}")
    fig.text(0.5, -0.02,
             'open circle = free field, own ears (midline)   ·   open diamond = '
             'own HRTF through AR (\'own\' before training, \'post\' after)   ·   '
             'grey face + dashed segment = only test that day was on another '
             'stimulus',
             ha='center', va='top', fontsize=5.5, color='0.35')
    fig.text(0.5, -0.055, ' · '.join(parts), ha='center', va='top', fontsize=5.5,
             color='0.35')
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return fig, series_list


if __name__ == '__main__':
    fig, series_list = figure()
    for series in series_list:
        print(f"\n{series['subject_id']}  hrir={series['hrir']}  "
              f"stim={series['stim']}  trained field={series['trained_field']}  "
              f"min trials={series['min_trials']:.0f}")
        print(f"  {'day':>4s} {'date':>6s} {'n':>2s} {'gain':>6s} "
              f"{'polar':>7s} {'sd':>6s}")
        for i, day in enumerate(series['days']):
            print(f"  {int(day):4d} {series['dates'][i]:>6s} "
                  f"{series['n_tests'][i]:2d} {series['gain'][i]:6.2f} "
                  f"{series['polar'][i]:7.2f} {series['sd'][i]:6.2f}")
        for key, reason in series['excluded']:
            print(f"    excluded {key}: {reason}")

    out_dir = paths.ANALYSIS_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'group_learning_' + '_'.join(s['subject_id'] for s in series_list)
    for suffix in ('png', 'svg'):
        out = out_dir / f'{stem}.{suffix}'
        fig.savefig(out, bbox_inches='tight')
        print(f'\nwrote {out}')
