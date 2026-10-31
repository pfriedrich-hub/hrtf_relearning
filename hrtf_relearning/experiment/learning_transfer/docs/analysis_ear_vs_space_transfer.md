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
