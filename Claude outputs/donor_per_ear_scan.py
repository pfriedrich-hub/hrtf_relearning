"""
RUN ON THE RIG (needs slab). Three things per-ear selection still owes.

Per-ear selection is now IN the package -- donor_selection.PER_EAR_SELECTION is
True, a candidate is a donor EAR, and DonorModification threads donor_ear
through build/stage/swap. Nothing here duplicates that. What is left is the
part that needs SOFAs, which cannot be read anywhere but this machine (PyPI is
blocked from both the cloud container and the desktop VM, so no slab there).

  STEP 1  RECOMPUTE TARGET_R_MATCH.  It is 0.58, computed on a SEVEN-member,
          person-level pool. The pool is now 12 recordings and candidates are
          ears, so the distribution it is the median of has changed twice over.
          The band tier is defined as +-0.05 around this number; leave it stale
          and the band is centred on a cohort that no longer exists. THIS IS
          THE ONE BLOCKING ITEM -- do it before the next participant.

  STEP 2  Sanity-check the per-ear table for the next subject, and see how far
          it moves the pick against the old person-level rule.

  STEP 3  Does any acoustic metric predict acute retained EG? If nothing does,
          the 0.30-0.55 retained band stays a SCREEN criterion only and the
          shortlist can order candidates but cannot promise the band.

    python "Claude outputs/donor_per_ear_scan.py"
"""
import numpy
from hrtf_relearning.hrtf.analysis import donor_selection as sel


# ---------------------------------------------------------------- step 1
def recompute_target(pool=sel.DONOR_POOL):
    """The median between-candidate r_match -- what TARGET_R_MATCH is defined as."""
    print('=' * 78)
    print(f'STEP 1  TARGET_R_MATCH is {sel.TARGET_R_MATCH:.2f} '
          f'(set on the 7-member person-level pool)')
    print('=' * 78)
    cands = sel.load_candidates('__none__', pool=pool)
    print(f'  pool: {len(cands)} recordings -> '
          f'{2 * len(cands) - len(sel.EXCLUDED_DONOR_EARS)} donor ears\n')
    out = {}
    for per_ear in (False, True):
        vals, _ = sel.pairwise_r_match(cands, per_ear=per_ear)
        q = numpy.percentile(vals, [25, 50, 75])
        label = 'per-EAR   ' if per_ear else 'per-person'
        print(f'  {label}  n={len(vals):5d} pairs   min {vals.min():.3f} | '
              f'Q1 {q[0]:.3f} | median {q[1]:.3f} | Q3 {q[2]:.3f} | '
              f'max {vals.max():.3f}')
        out[per_ear] = float(q[1])
    print(f"\n  -> set TARGET_R_MATCH = {out[True]:.3f}  "
          f"(per-ear median; PER_EAR_SELECTION is True)")
    print(f"     for reference, the person-level median is {out[False]:.3f}")
    if abs(out[True] - sel.TARGET_R_MATCH) > sel.TOLERANCE:
        print(f"  !! the target moves by more than one tolerance band "
              f"({abs(out[True] - sel.TARGET_R_MATCH):.3f} > {sel.TOLERANCE}), "
              f"so the current constant is selecting on the wrong centre.")
    return out[True]


# ---------------------------------------------------------------- step 2
def compare(listener, trained_ear, pool=sel.DONOR_POOL, top=12):
    """Per-ear table for one listener, beside the old person-level ranking."""
    print('\n' + '=' * 78)
    print(f'STEP 2  {listener}, trained ear {trained_ear}')
    print('=' * 78)
    own = sel.load_candidates('__none__', pool=(listener,)).get(listener)
    if own is None:
        print(f'  !! no recording for {listener}')
        return
    cands = sel.load_candidates(listener, pool=pool)

    new = sel.shortlist(own, cands, trained_ear=trained_ear, per_ear=True)
    ref, _ = sel.pairwise_r_match(cands, per_ear=True)
    sel.report(new[:top], ref)

    print('\n  the OLD person-level rule, for contrast:')
    old = sel.shortlist(own, cands, trained_ear=trained_ear, per_ear=False)
    for r in old[:5]:
        print(f"      {r['rank']}. {r['donor']:12} r_match {r['r_match']:5.2f}  "
              f"ridge {r['ridge_slope']:+5.2f}  strength "
              f"{r['donor_strength']:4.1f}  [{r['tier']}]")
    a, b = new[0], old[0]
    same = a['donor'] == b['donor']
    print(f"\n  rank 0: per-ear {a['donor']}({a['donor_ear']}) vs "
          f"person-level {b['donor']}  --> "
          f"{'SAME recording' if same else 'DIFFERENT recording'}")
    return new, old


# ---------------------------------------------------------------- step 3
# (listener, donor, trained_ear, own_EG, acute retained EG). Day-1 mean over
# trained-ear unmirrored blocks, n>=50, |el| <= 30, from the subject JSONs.
# NB these are the MONAURAL training blocks; donor_screening's eg_retained is
# measured on the BINAURAL screen composite and is a different number.
PAIRINGS = [
    ('AS', 'GS',       'left',  0.75,  0.35), ('FP', 'pilot/AH', 'left',  0.84,  0.35),
    ('FP', 'FS',       'left',  0.84, -0.00), ('FS', 'AS',       'left',  0.86,  0.34),
    ('GM', 'FS',       'right', 0.82,  0.33), ('IR', 'AS',       'left',  0.51,  0.23),
    ('LS', 'pilot/AH', 'right', 1.30,  0.25), ('NR', 'pilot/AH', 'right', 0.70,  0.01),
    ('NR', 'pilot/SW', 'right', 0.70,  0.24), ('TS', 'pilot/AH', 'left',  0.54,  0.23),
]
METRICS = ('r_match', 'ridge_slope', 'donor_strength',
           'cue_gradient', 'cue_monotonicity', 'cue_decodability')
BAND = (0.30, 0.55)


def predicts_retained_eg():
    print('\n' + '=' * 78)
    print(f'STEP 3  acoustic metrics vs acute retained EG (n={len(PAIRINGS)})')
    print('        scored on the DELIVERED ear pair, not the ear average')
    print('=' * 78)
    splits, rows = {}, []
    for listener, donor, trained, own_eg, retained in PAIRINGS:
        for who in (listener, donor):
            if who not in splits:
                got = sel.load_candidates('__none__', pool=(who,)).get(who)
                splits[who] = sel.median_plane_split(got) if got is not None else None
                if got is None:
                    print(f'  !! could not load {who}')
        if splits[listener] is None or splits[donor] is None:
            continue
        # the composite was built same-side, so donor_ear == trained ear
        pair = sel.pair_metrics(splits[listener], splits[donor], donor_ear=trained)
        cue = sel.cue_gradient(splits[donor], ear=trained)
        rows.append(dict(listener=listener, donor=donor, ear=trained,
                         own_eg=own_eg, retained=retained,
                         r_match=pair['per_ear'][trained]['r_match'],
                         ridge_slope=pair['per_ear'][trained]['ridge_slope'],
                         donor_strength=sel.detail_strength(splits[donor], ear=trained),
                         cue_gradient=cue['gradient'],
                         cue_monotonicity=cue['monotonicity'],
                         cue_decodability=cue['decodability']))

    print(f"{'listener':9}{'donor':11}{'ear':6}{'ownEG':>7}{'retEG':>7}"
          + ''.join(f'{m[:9]:>10}' for m in METRICS))
    for r in rows:
        print(f"{r['listener']:9}{r['donor']:11}{r['ear']:6}{r['own_eg']:7.2f}"
              f"{r['retained']:7.2f}" + ''.join(f"{r[m]:10.2f}" for m in METRICS))

    print('\nSpearman rho vs retained EG, and in-band vs out-of-band means:')
    ret = [r['retained'] for r in rows]
    for m in METRICS:
        v = [r[m] for r in rows]
        inb = [r[m] for r in rows if BAND[0] <= r['retained'] <= BAND[1]]
        out = [r[m] for r in rows if not BAND[0] <= r['retained'] <= BAND[1]]
        gap = (numpy.mean(inb) - numpy.mean(out)) if inb and out else float('nan')
        print(f'  {m:20} rho {sel._spearman(v, ret):+.2f}   '
              f'in-band {numpy.mean(inb):6.2f}   out {numpy.mean(out):6.2f}   '
              f'gap {gap:+.2f}')
    print('\n  |rho| below ~0.65 at n=10 is not a predictor. If none clears it,')
    print('  the retained-EG band cannot be targeted acoustically and the day-1')
    print('  screen is the only thing that can enforce it.')
    return rows


if __name__ == '__main__':
    recompute_target()
    predicts_retained_eg()
    # next participant -- id and the pre-registered trained ear:
    # compare('XX', 'right')
