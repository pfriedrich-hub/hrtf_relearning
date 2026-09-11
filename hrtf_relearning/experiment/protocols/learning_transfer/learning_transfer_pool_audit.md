# Participant pool audit — learning_transfer (donor-detail)

Audit date 2026-09-08. Companion to `learning_transfer_block_order.csv`, which
carries the same verdicts in machine-readable form and is still read unchanged
by `learning_transfer.py::_load_subject_params` (the extra columns are ignored
by `csv.DictReader`). Previous sheet kept as
`learning_transfer_block_order.csv.bak_20260908`.

All numbers below are recomputed from `data/results/<id>/<id>.json` with the
canonical metrics in `experiment/analysis/localization/localization_analysis.py`
(`polar_error`, `localization_accuracy`). Blocks are classified into design
cells A/B/C/D from `mirrored` x `settings['azimuth_range']` x trained ear.

## 1. The cohort as run

| id | ear | days | games | final 2x2 | stimulus | day-1 impairment | verdict |
|---|---|---|---|---|---|---|---|
| FS | L | 4 | 38 | complete | noise | +13.6 | noise control subgroup |
| FP | L | 4 | 38 | **missing** | ripple | +9.3 | excluded (age 41; aborted) |
| AS | L | 4 | 48 | complete | ripple | **+2.5** | excluded (manipulation check) |
| IR | L | 4 | 48 | complete | uso -> noise | +9.6 | noise control subgroup |
| LS | R | 4 | 56 | complete | ripple | +13.9 | on hold (render anomaly) |
| NR | R | 3 | **17** | **missing** | ripple | +13.4 | excluded (non-completion) |
| GM | R | 2 | 18 | pending | noise + ripple | +18.7 prov. | in progress |
| PF | R | 1 | 2 | none | — | — | Paul's self-test, not a participant |

Day-1 impairment = donor block mean local polar error minus own-HRTF block,
same day, matched stimulus. The manipulation-check band is +8 to +14 deg.

### Exclusions, each on an outcome-independent ground
- **FP** — age 41, outside the 18-40 inclusion criterion. Pre-registrable and
  decided without reference to her data. Independently, she aborted: day 4 was
  one PRE block and 5 games.
- **AS** — day-1 impairment +2.5 deg, below the manipulation-check band. Her
  composite barely bit, so her 0.41 -> 0.51 gain change is recovery from a
  2.5 deg perturbation, not relearning. Keep her as the worked example of a
  failed manipulation check.
- **NR** — protocol non-completion: 3 of 4 days, 17 training games against
  38-56 for everyone else, no final 2x2. **The dome EG 0.74 gate is not a
  valid reason** and should not be cited: her three 21-trial dome blocks in one
  sitting were 0.18 / 0.45 / 0.74, a spread wider than the noise band for that
  block length.
- **PF** — self-test, never ran donor blocks. Remove from the design entirely.

### The two stimulus groups are not one pool
`learning_transfer.py` already says subjects run before the switch were tested
on noise and are pilots. FS and IR are noise-era; AS, FP, LS, NR are ripple-era.
FS additionally reported a timbre-to-elevation lookup on the frozen noise token,
which is *why* the stimulus was changed — so the noise group is not a neutral
control, it is the condition the ripple was introduced to prevent. Report it as
such: "a pilot subgroup (n=2) tested with a frozen noise token, in which one
listener reported an explicit timbre-based strategy".

IR is weaker still: his baselines are uso and his adaptation and final blocks
are noise, so his pre-post change scores are stimulus-confounded.

**GM is currently a third case** — dome and baselines A/D on noise, training and
PRE/POST on ripple. That mixes the eras *within one subject* and makes her
baseline-to-final change scores non-comparable. The own-HRTF full-field NOISE
block on day 2 is still owed; run it before day 3.

## 2. The laser / head-pose offset is NOT yet corrected in any data

Estimated as the day-1 own-HRTF AR elevation bias minus the day-1 dome bias
(the per-session constant `c`, the only correction the existing data supports):

| FS | FP | AS | IR | LS | NR | **GM** |
|---|---|---|---|---|---|---|
| +3.3 | +0.2 | -2.8 | +2.2 | +0.1 | +0.6 | **+9.0** |

Only GM is materially affected; everyone else is within +-3.3 deg. Her day-2
offset is ~+3.0, so the alignment on day 2 was much better and her day-2 donor
blocks (bias +24.1 / +19.2) are mostly genuine donor displacement, not laser.

Two corrections to the assumption that this is fixed by now: **no sequence in
any subject's file carries `calibration_poses`**, including GM's session run
today — the logging added 2026-09-07 has not yet been exercised on the rig; and
the hardware fix (gluing the laser to the sensor) was only decided today, so it
is in front of every subject still to be run, not behind them. Until then the
only retrospective correction is the per-session constant, which needs a
same-session dome block and therefore exists on day 1 only.

Per-block mean removal is not an alternative: it deletes 20-30% of the real
donor displacement along with the offset.

## 3. Effect size, and why the answer to "8 or 16" is "neither yet"

Noise floor on a 75-trial donor block is EG 0.18 / PE 4.2 deg; a change score
is a difference of two such blocks, so the band a change must clear is
**EG 0.25 / PE 5.9 deg**.

**Ripple-era trained-ear learning, first to last donor block (n=4):**

| | AS | FP | LS | NR | mean | SD | dz |
|---|---|---|---|---|---|---|---|
| PE improvement (deg) | +0.4 | +0.9 | +4.3 | -1.8 | +0.95 | 2.52 | **0.38** |
| EG change | +0.10 | +0.04 | +0.18 | -0.11 | +0.05 | 0.12 | **0.43** |

Three of four are inside the noise band. At dz 0.38-0.43, 80% power needs
**n = 45-58**. Neither 8 nor 16 is in the running for the effect *as currently
observed in the ripple era*.

**Transfer contrast (change in A minus change in D), the four with a complete
final 2x2:**

| | dPE_A | dPE_D | contrast | dEG_A | dEG_D | contrast |
|---|---|---|---|---|---|---|
| FS (noise) | +10.4 | -2.1 | **+12.5** | +0.38 | +0.00 | +0.38 |
| IR (uso/noise) | +6.0 | -1.2 | **+7.2** | +0.35 | +0.09 | +0.26 |
| AS (ripple) | +1.0 | +0.7 | +0.3 | +0.08 | +0.13 | -0.05 |
| LS (ripple) | +4.3 | +5.9 | -1.6 | +0.18 | +0.44 | -0.26 |

Contrast: mean +4.6 deg, SD 6.5, **dz 0.71 -> n = 18 for 80% power**. That is
the number that argues for 16 rather than 8. But it is carried entirely by the
two noise-era subjects; among the ripple-era pair it is ~0 with the sign
reversed, and the whole contrast is only interpretable in subjects who learned
with the trained ear in the first place.

**What each N buys** (paired, alpha .05 two-tailed):

| N | 80% power detects | 90% power detects |
|---|---|---|
| 8 | dz >= 1.16 | dz >= 1.34 |
| 16 | dz >= 0.75 | dz >= 0.87 |
| 24 | dz >= 0.60 | dz >= 0.69 |

For scale, the earmold benchmark (dEG +0.50, 15/15 subjects up, ~50x the
exposure) is dz ~ 2.0-2.5 and would be detected at n = 4-6. A real adaptation
effect is trivially detectable at 8. The AR effect as it currently stands is
not detectable at 16 either.

### Recommendation

**Plan for 16, commit to 8, and put a decision point between them.** The sheet
is laid out that way: cohort `ripple_n8` is the six open cells that complete a
first balanced ripple-era cycle (LS and GM hold R/1 and R/3), and cohort
`ripple_n16_conditional` is the second replicate, to be run only if the interim
check passes.

**Interim check at N=8** — on trained-ear learning only, before spending the
second replicate: median dPE_A >= 5.9 deg or median dEG_A >= 0.25, i.e. outside
the noise band. If it passes, the transfer contrast at dz ~0.7 justifies 16.
If it fails, 8 more subjects will not rescue it and the constraint is the
manipulation or the exposure, not N — the earmold study got its effect with
about 50x the training exposure, which is the more likely lever.

Committing to 8 is also cheap insurance: the design needs multiples of 8 for the
Williams x ear crossing anyway, so stopping at 8 leaves a balanced, reportable
dataset rather than a half-filled square.

### The cost, stated plainly
Analysable ripple-era subjects today: **0** (LS on hold, GM incomplete). At
~6 h per participant, reaching a clean N=8 is 6 more subjects (~36 h of
testing); N=16 is 14 more (~84 h).

## 4. Before the next participant sits down
1. Glue the laser to the sensor and measure `k` (procedure already written up);
   confirm `calibration_poses` actually appears in the saved sequence.
2. Fix one stimulus for the whole experiment and do not switch mid-subject.
3. Run GM's owed own-HRTF full-field NOISE block on day 3.
4. Pool the three dome blocks (3 x 21 = 63 trials) before applying any dome
   inclusion criterion; do not gate on a single 21-trial block.
5. Put LS's right-trained render on KEMAR to resolve the hold.
