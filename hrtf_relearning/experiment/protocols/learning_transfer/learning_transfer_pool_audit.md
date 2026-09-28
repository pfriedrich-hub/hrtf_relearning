# Participant pool audit — learning_transfer (donor-detail)

Updated 2026-09-28. Companion to `learning_transfer_block_order.csv`, which
carries the same verdicts in the `analysis_set` column and is still read
unchanged by `learning_transfer.py::_load_subject_params`. Backups:
`...csv.bak_20260908`, `...csv.bak_20260928`.

## The inclusion rule (Paul, 2026-09-28)
**Completed the final 2×2 AND day-1 impairment ≥ +8° after the per-session laser
correction.** Both conditions are measurable before any training, so **no
exclusion depends on whether the subject learned.** Adaptation is reported as a
continuous covariate, never as a gate.

Ripple continues as the stimulus. FS and IR are kept although neither was tested
on ripple — to be declared in the paper.

Rows are retained, never deleted: the protocol resolves `trained_ear` and
`block_order` from this file, so a deleted row makes that subject un-runnable and
un-re-analysable. Subject pkl/json data is untouched.

## Verdicts

| id | cell | stimulus | impairment raw → corrected | adaptation ΔPE / ΔEG | analysis_set |
|---|---|---|---|---|---|
| FS | L/1 | noise | +13.6 → **+17.2** | +10.4 / +0.38 | **include** |
| IR | L/4 | uso → noise | +9.6 → **+11.5** | +6.1 / +0.36 | **include** |
| LS | R/1 | ripple | +13.9 → **+13.9** | +4.4 / +0.18 | **include** |
| GM | R/3 | ripple trained, noise tested | +5.8 → **+8.2** | +5.9 / +0.11 | **include** |
| AS | L/3 | ripple | +4.1 → **+2.4** | +0.5 / +0.10 | exclude — manipulation check |
| FP | L/2 | ripple | +9.3 | +0.9 / +0.04 | exclude — protocol (age 41, aborted day 4) |
| NR | R/2 | ripple | +9.3 | −1.9 / −0.11 | exclude — protocol (3 days, 17 games) |
| PF | R/4 | — | — | — | not a participant (self-test) |

FP and NR both pass the manipulation check; their ground is completion, not the
manipulation. AS is the only subject excluded by the check itself.

## The laser correction, applied to everyone
`c` = own-HRTF AR intercept − dome intercept on the reference day; subtracted from
both the own-HRTF and the donor block before recomputing polar error. Applying it
to only the subject who looked wrong would be the forking path `donor_screening`
already warns about, so it is applied to the whole cohort:

| | FS | IR | AS | LS | NR | GM | FP |
|---|---|---|---|---|---|---|---|
| c (°) | +3.3 | +2.8 | −3.4 | +0.1 | +0.5 | **+8.8** | +0.3 |
| impairment shift | +3.6 | +1.9 | −1.7 | 0.0 | −0.1 | **+2.4** | +0.1 |

Only GM's verdict changes — she is the only subject whose offset is large enough
to matter, and she crosses the threshold by **0.2°**. State that margin in the
paper rather than presenting +8.2 as comfortable. The same correction moves AS
*further* out (+4.1 → +2.4), so the rule does not bend toward a convenient answer.

## Caveats on the included four
- **LS** — the only one with no caveat. Ripple throughout, deepest perturbation,
  render exonerated (removing her azimuth expansion moves polar error by under 1°).
- **FS** — noise era, and he reported the timbre→elevation lookup on the frozen
  token that the ripple exists to prevent. Not a neutral control.
- **IR** — uso baselines against noise adaptation and finals; his change scores
  carry that stimulus confound.
- **GM** — marginal on the check, trained on ripple but tested on noise, and her
  ΔPE is confounded with a 9.8° drop in elevation bias while her bias-immune
  metric moved only +0.11. No own-HRTF block on her final day to separate them.

## Where the design stands
Analysis set **n = 4** (FS, IR, LS, GM), two of them ripple-exposed. LS and GM
hold R/1 and R/3, so **six cells remain open** for a balanced ripple N = 8:
L1 L2 L3 L4 · R2 R4 — about 36 h of testing. `ripple_n16_conditional` holds the
second replicate, to be run only if the interim check passes.

Interim check at N = 8, on trained-ear adaptation: median ΔPE_A ≥ 5.9° or
ΔEG_A ≥ 0.25.

Group figures for this set: adaptation ΔPE +6.68, t(3) = 5.18, **p = .014**;
ΔEG +0.26, t(3) = 3.90, **p = .030**. Transfer ΔPE +0.74, p = .71; ΔEG +0.10,
p = .47. Adaptation is established; transfer is not distinguishable from zero.

## Still open
- The head-pose offset is **not corrected in any collected data** — no sequence in
  any subject's file carries `calibration_poses`. The correction above is the
  retrospective per-session constant, which needs a same-session dome block and
  therefore exists on day 1 only.
- Whether the ripple's source-spectrum variation masks adaptation for modified
  cues is unresolved; the within-subject probe ({own, donor} × {noise, ripple},
  order counterbalanced) has not been run.
- **Add a short dome block on the FINAL day.** GM's case is exactly what its
  absence costs: her adaptation cannot be separated from her pose.
