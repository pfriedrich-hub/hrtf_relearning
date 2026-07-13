# Expectation Transfer: Does Real-Speaker Exposure Recalibrate Virtual Localization?

Design background for `protocols/expectation_transfer.py`.

## 1. Observation motivating this experiment

Naive listeners tested on individual-HRIR localization over headphones (AR, via
pybinsim) typically fail to externalize the sound and localize poorly — first
exposure to the virtual condition. Informally, if listeners first localize real
loudspeakers in the dome and are *then* given the virtual test, externalization
and accuracy both look normal — close to indistinguishable from the real
speakers. This suggests real-speaker exposure updates some prior/expectation
about how spectral cues map to source position (or convinces the listener the
space is "real"), rather than the AR signal itself being unlocalizable.

## 2. The confound a naive pre/post design can't rule out

AR → Dome → AR within one group looks like a clean before/after, but a second
AR test is *also* the listener's second time doing the task at all — second
time calibrating the head tracker, second time trusting the response
procedure, general warm-up. Any of that could produce a pre→post improvement
that has nothing to do with the dome. A within-subject-only design can't tell
these apart.

## 3. Design: two-arm, single session, matched-effort control

| Group | Block 1 | Block 2 | Block 3 |
|---|---|---|---|
| **dome** (experimental) | AR_pre | **Dome** (real loudspeakers) | AR_post |
| **control** | AR_pre | **AR_filler** (more virtual trials, same locations/count as dome) | AR_post |

Both groups get *identical* AR_pre and AR_post blocks (same HRIR, same
locations, same trial count). Only block 2 differs: real acoustic exposure vs.
more of the same virtual task. If the dome group improves pre→post more than
the control group, that isolates the real-speaker-specific effect from mere
repetition/practice — the control group absorbs the practice effect and
nothing else changes for them.

An AR_filler control was chosen over a passive wait so that time-on-task,
number of head-orienting responses, and fatigue/arousal are matched between
groups; the only thing that differs is whether the exposure block was real or
virtual.

## 4. Location/trial matching (uses existing modules as-is)

`Localization_dome.LocalizationDome` already plays from the 7 vertical-midline
dome speakers (az ≈ 0). `Localization_AR.Localization` with
`kind='standard', azimuth_range=(-1, 1)` (as already used in
`HRIR_Recording.py` step 5) samples the *same* physical directions virtually.
No new location logic is needed — AR_pre, AR_post, dome, and AR_filler all
draw from this shared vertical-midline set, so the three (or four) blocks are
directly comparable. Recommended `targets_per_speaker = 3` (21 trials/block)
for all four block types.

## 5. Measures

- **Localization error** — already computed by the existing pipeline
  (`target_p`, `plot_localization`, `plot_elevation_response` over the
  sequence). Primary DV: elevation error pre vs. post, compared between
  groups.
- **Externalization** — not currently instrumented anywhere in the codebase.
  Added a short console-collected rating after AR_pre and AR_post only
  (0–10 "how far outside your head did the sound feel" + yes/no "could you
  tell these weren't real loudspeakers"), stored as
  `sequence.externalization_rating` / `sequence.plausibility_response`. This
  is new and lightweight — reword/replace freely, it's a first pass.

## 6. Session structure (single day, one visit)

Assumes the subject already has a recorded individual HRIR and calibrated
headphone filter (`HRIR_Recording.py`). Then, in order:

1. AR_pre (~10 min, 21 trials) + externalization rating
2. Exposure block — dome or AR_filler depending on assigned group (~10 min)
3. AR_post (~10 min, 21 trials) + externalization rating

~35–40 min including instructions/breaks.

## 7. Group assignment

Between-subjects, alternating dome/control by recruitment order, tracked in
`data/documentation/expectation_transfer_block_order.csv` (same convention as
`exp1_transfer_block_order.csv`). Don't tell subjects which arm they're in or
what's predicted to change, to avoid biasing the externalization rating.

## 8. Planned analysis (sketch, not a power analysis)

2 (group: dome/control) × 2 (time: pre/post) mixed design on localization
error and on the externalization rating; the interaction is the effect of
interest — dome group improving pre→post while control does not. Treat any n
below as a placeholder: run a small pilot batch (6–8/arm) first to get an
effect-size estimate before committing to a full sample size.

## 9. Alternatives considered — not implemented now, natural follow-ups

- **Order crossover** (dome-first vs. AR-first, both groups get all three
  blocks): tests whether starting with real speakers changes the whole
  trajectory, not just a mid-session boost. Worth doing once the basic effect
  above is confirmed.
- **Day-separated version** (dome exposure day 1, AR test day 2, same room vs.
  a different room as control): dissociates "auditory expectation update"
  from "being in the same physical room as the speakers just heard" (a pure
  context effect). Needs a second AR-capable room; a good second study once
  the single-session effect size is known.
- **Within-subject A-B-A, no control arm**: cheapest to pilot informally, but
  per §2, any improvement is uninterpretable without the AR-filler control.

## 10. What this protocol does not resolve

Whether the effect is acoustic/perceptual recalibration vs. purely cognitive
("I now believe I'm in a real room") — that's what the day-separated
follow-up (§9) targets. This is also a single-session snapshot; it says
nothing about how long the effect lasts.
