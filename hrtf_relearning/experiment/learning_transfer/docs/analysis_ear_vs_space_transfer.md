# Ear-specific vs space-specific learning — analysis plan for the 2x2 (Ear x Side)

Noted 2026-09-29. Basis: Van Wanrooij & Van Opstal (2005), J Neurosci 25:5413-5424
("Relearning sound localization with a new ear").

## Model
Two stages: (1) each ear maps its own spectrum to an elevation estimate E_L, E_R;
(2) binaural weighting, perceived elevation = w_L(az)*E_L + w_R(az)*E_R, near ear
dominant, ~equal weights at the midline.

VW&VO 2005: unilateral-mold adaptation was ear-specific, at the mapping stage (MM
improved like FM, MF did not). Yet the improvement GREW toward the mold side
(Fig. 14, dGain ~+0.1 contra -> +0.3 ipsi) because the learned E_L is scaled by
w_L(az). Acute FM (Fig. 4): the far ear still carries a lot of elevation at
+-20-35 deg (local gain ~0.4-0.6), so far-ear cells are not hopeless in our range.
They state (p. 5422) their design cannot dissociate the order of the two stages.

**Key correction to intuition:** ear-specific learning is ALSO azimuth-dependent —
the weights are the channel it reaches behaviour through. Only learning located
AFTER binaural combination (tied to a side of space) is flat across azimuth.

## Predictions (left ear trained; untrained ear carries an elevation-flat envelope)
| hypothesis | locus | A | D | B | C |
|---|---|---|---|---|---|
| ear-specific (VW&VO result) | E_L only | up, grows laterally | none | up* near midline, falls laterally | none |
| shape learning shared by both ears | template read by E_L and E_R | up, grows laterally | up, mirror of A | up* near midline, falls | up* near midline, falls |
| tied to side of space | after combination | up, FLAT over az | none | none | up, FLAT over az |

*B and C present the trained ear's DTFs for CONTRALATERAL directions; if training
only sampled the trained hemifield these were never trained, weakening B and C
under every hypothesis.

## Discriminators
1. dD with an azimuth gradient mirroring A -> spectral shapes learned independently
   of the ear. Novel vs VW&VO: their MF put DIFFERENT new DTFs in the other ear;
   our D delivers the SAME learned DTFs. FS/IR/GM dD ~ 0 -> consistent with ear-specific.
2. dB vs dC near the midline (far-ear handicap matched and smallest):
   B > C -> ear;  C > B -> side of space;  both ~0 -> far-ear cue unused (uninformative).
3. Shape of d vs azimuth: gradient following the cue-carrying ear -> pre-combination;
   flat within the trained hemifield -> post-combination.

## Corollary: ceilings
The cue-less ear still contributes its weighted, flat estimate, so even perfect
learning caps gain at ~the weight of the cue-carrying ear: high for A/D laterally,
low for B/C (day-1 B/C gains 0.07-0.28). Compare B with C, not with A.

## To do
- Analyse every cell as LOCAL elevation gain (and PE) in azimuth bins
  (VW&VO: 30 deg windows stepped 5 deg; here e.g. 0-15 / 10-25 / 20-35 deg).
- Needs day-1 B/C blocks (SH onward); FS-AS have B/C on the final day only.
- 75 trials/block is thin per bin -> more final-day B/C trials (2 blocks or 150).
- Far-ear usability check: own-HRTF detail in one ear only, tested on the far
  side, day 1 (+ final day). Decides per subject whether a null C is interpretable.
- Check which directions the training game samples (trained hemifield only?).

## Weighting model fitted to the 2x2 (2026-11-02/03)
Cell gain = weight of the cue-carrying ear x that ear's mapping; the own-envelope
ear is assumed to add no elevation slope. Near-ear weight w, extra trained-ear
weight d (reweighting), mappings gT/gU, trained-side factor s:

| cell | cue in | gain |
|---|---|---|
| A | trained ear, near | s (w + d) gT |
| B | trained ear, far | (1 - w + d) gT |
| C | untrained ear, far | s (1 - w - d) gU |
| D | untrained ear, near | (w - d) gU |

Day 1: gT = gU (same donor detail, mirrored), so the day-1 2x2 measures w.
Fitted per azimuth bin (|az| < 18 / > 18 deg, own w per bin), day-1 side
asymmetry s1 free, 16 observations per participant (SH, SK, DA, LP, PA).

Fingerprints: trained-ear mapping -> dA:dB = w:(1-w), C/D flat; untrained-ear
mapping -> C/D up, D more; reweighting -> dA = dB > 0, dC = dD < 0;
side of space -> A and C up together.

Results:
- w_med < w_lat in all five (SH .56/.76, LP .61/.72, PA .72/.97, DA .83/.92,
  SK .49/.54) -- the expected azimuth dependence, read from day 1 alone.
- Reweighting never improves the fit (all p >= .2). PA = ear-specific mapping
  (gT .24 -> .64, gU flat); her medial-D drop is within noise.
- D transfer follows gU: roughly doubles in SH/SK/DA, small in LP, flat in PA.
- Trained-side advantage grows in SH (x1.5, p=.02) and LP (x1.7, p=.06),
  shrinks in SK (day-1 x1.75 -> x1.1); DA fit degenerate (day-1 B/C at 0).
- Day-1 side asymmetries x0.6-1.75 exist before training -- unexplained,
  check laser / tracker zero.
- Fit within noise: SK, SH, PA; misfit: DA, LP (chi2 17-20 on 7 dof).

Limits: a constrained description, not a mechanism. Shared cue learning and a
general change in response scaling both raise gT and gU and cannot be
separated here. Gain only; polar error not modelled. Explanatory (discussion),
not pre-registered. Fit scripts were scratch-only, not yet in the repo.
