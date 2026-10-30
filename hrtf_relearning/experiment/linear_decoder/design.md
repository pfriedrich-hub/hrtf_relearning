# Exp 2 — linear decoding vs template matching (`linear_decoder`)

Status: **concept, 2026-10-26; under section-by-section review before
implementation (2026-10-30).** Nothing implemented yet. Decided points are
marked **[decided]**, everything else is open and listed at the end.

## Question

Grant Exp. 2. Two accounts of how spectral cues are mapped to elevation:

- **Template correlation** (Middlebrooks 1992; Langendijk & Bronkhorst 2002):
  the incoming spectrum is compared with stored per-direction templates.
- **Linear decoding** (Blauert 1969, "combination of simple features"):
  elevation is a weighted sum of spectral feature-detector activity.

They make different predictions for spectra from elevations that were never
trained. We train listeners on a synthetic, linearly varying cue with a gap in
elevation and test inside the gap.

Reference models for the predictions: Baumgartner et al. (2014) as the main
template model, Barumerli et al. (2023, Bayesian) as a robustness check, and
the McDermott lab's neural-network localization model (Francl & McDermott
2022; implemented in our lab), to be trained on the linear-notch set with the
gap and tested inside it.

## Predictions for gap targets

Updated 2026-10-30 after the model simulations (concept figures 5-7,
native-weights learning model):

| account | gap prediction | status |
|---|---|---|
| template matching | step near the gap edges; responses only at learned directions | robust across models (2014, 2023 with/without template interpolation, MAP or sampling) |
| linear readout | graded if the detectors are broad relative to the gap (band levels); with gradient (DCN-like) detectors step-like or undershooting | open: depends on representation and learning (Things to figure out #4; two-notch idea) |
| elimination ("unfamiliar sound -> the place I never heard") | inside the gap, flat vs target elevation | unchanged |

Within-gap gain alone no longer separates the accounts. Measure: the response
distribution per gap target, compared through model likelihoods (each model
fitted on trained-region trials, then used to predict the gap trials); see
Measures and Things to figure out.

## Cohort and rendering

- New cohort. **Binaural**, frontal field **[decided]**.
- **Same feature at both ears [decided]**; ITD/ILD from the spherical-head
  expansion.
- **Stimuli: ripple for localization AND training [decided]**, as in Exp 1
  (STIM='ripple', rms_tilt=3, TRAINING_STIM=STIM), same reason: the source
  spectrum varies every trial, so the task cannot be solved as a fixed
  timbre->elevation lookup.
- **Days, block structure and trial counts as in Exp 1 [decided]**: day 1
  own-HRTF reference + screen + baseline; days 2-3 pre-test -> training ->
  post-test; day 4 pre-test -> training -> final test [decided 2026-10-27].
- Shared engines (Training_AR, Localization_AR) with `EAR=None`; Exp 2
  behaviour comes in through hooks that are off by default **[decided]**.
- Results under `data/results/linear_decoder/<ID>/` **[decided]**.
- Pilot evidence is monaural: the synthetic-notch pilots (March 2026) trained
  `<ID>_notch_left` (left-ear notch, left hemifield, noise stimulus, own fine
  structure kept). JP learned, SK did not; why is open. So the pilots show that
  a notch like this can be learned, not that this binaural set is learnable --
  that is what the day-1 screen checks (see Day 1 screen).

## Cue (per participant)

**Cue synthesis remains the critical part.** Before implementing
`synth_cue.py`: look into the existing Exp 1 and pilot data for which cues were
learned (JP learned, SK did not -- why?) and use that to choose the cue.

- **Base:** the participant's own n_keep=4 cepstral envelope, **averaged over
  all elevations**, so the band below the feature region is identical at every
  direction and carries no elevation information.
- **Binaural cues:** ITD/ILD from the spherical-head expansion at the fitted
  head radius, as in the current pipeline.
- **Feature:** one Gaussian notch, parameters from the March 2026 pilots
  (`*_notch_left`; JP learned, SK did not) **[decided as starting point]**.
  These values were chosen arbitrarily at the time and worked in some cases;
  they are a starting point, not a derived optimum:
  - depth ~15 dB
  - FWHM ~0.11 oct
  - centre frequency log-linear in elevation, ~0.008 oct/deg
    (pilot: 7.2 -> 10.9 kHz over +-37.5 deg); one FWHM ~13 deg of elevation
  - Measured from the pilot SOFAs (2026-10-29, concept fig 12): 7.2 -> 11.0
    kHz over +-37.5 deg (0.0082 oct/deg), ~15 dB, the same notch for JP and
    SK, added on top of the listener's own HRTF and rendered on the left ear
    only. For JP it ran parallel to his own main notch, ~0.3 oct above it.
- **Second notch -- under consideration, not decided.** Would turn the gap
  test into a test of linear combination: each notch in the gap lands on
  detectors the other notch drove at trained elevations (concept fig 8).
  - Own HRTFs (98 ears, concept fig 9): the second notch sits ~0.5 oct above
    the main one and is mostly flat (median tilt 0.8 vs 6.8 milli-oct/deg for
    the main notch); it falls in 8 of 50 current-pipeline ears. A
    non-parallel second notch is therefore realistic.
  - Open: spacing (>= gap width x sweep, ~0.18 oct for the current gap), its
    tilt (parallel / flat / falling), and how a linear readout splits its
    weights between the two notches. Decide after the cue-data review and a
    simulation.
- **Placement ("similar enough"):** frequency range inside the band where the
  participant's own notches lie. In the pilots, learning ordered with
  similarity to the own DTF (JP r=0.85 learned, SK 0.61 did not learn, others <=0.43
  flat; n=2 learners, both with poor own-HRTF AR baselines).
- **Cue type [decided 2026-10-26]: option (a), novel cue.** The set
  contains only the n=4 envelope (for externalization) plus the synthetic
  notch -- no native cue. Paul tests it; if it is not learnable, fall back to
  option (b), a gain remap (same frequency range and notch shape as the
  listener's own, elevation mapping stretched so the native map reads it at
  reduced gain; question then = does recalibration of the trained region carry
  into the gap). Risk of (b): a different kind of remapping, may produce a
  bistable percept (cf. Van Opstal monaural-learning work).
- The pilot sets above came from an earlier pipeline stage in which HRIR
  recording and externalization did not yet work well (hence the poor native
  AR baselines of those subjects); their acute numbers do not transfer to the
  current pipeline.
- **No native cue in the set:** the pilot sets kept the own fine detail
  (native ridge slope 0.9-1.0); Exp 2 removes it (averaged envelope). How the
  native map reads the synthetic notch, in and outside the gap, is measured
  on day 1, not required.
- **Grid:** synthetic, so not tied to the recorded +-37.5 deg arc; can be
  built to +-50 deg if the gap needs it.

### Toy-model check (linear readout of ERB-channel excitation, ridge)

Pilot-like notch (7.2 -> 10.9 kHz over +-37.5 deg, 15 dB), within-gap gain / RMSE:

| notch FWHM | gap 10 deg | 15 deg | 20 deg | 25 deg |
|---|---|---|---|---|
| 0.11 oct (pilot) | 0.95 / 2.5 | 0.96 / 2.6 | 1.03 / 3.6 | 1.01 / 4.6 |
| 0.20 oct | 1.00 / 1.5 | 0.96 / 1.5 | 1.00 / 2.0 | 0.99 / 2.4 |

Caveat (2026-10-30): these numbers come from a linear readout of broad
ERB-channel levels. With gradient (DCN-like) detectors the readout
interpolates far less (concept figs 6-7, native-weights model), so this table
alone does not support a single-notch design.

What matters is channel tuning in degrees = (auditory filter width + feature
width) / sweep rate. The grant's original design (semitone notch, 5 -> 18 kHz
over 115 deg, 28 deg gap) fails this for the linear model too (gain 0.46,
RMSE 18 deg).

## Gap

- A band in **head-relative elevation**, all azimuths.
- Must stay clear of 0 (every training trial converges on 0; overshoot).
  Second reason: responses to unknown DTFs land near 0 deg (Paul's
  observation; the Bayesian model's prior reproduces it), so a gap containing
  0 could not separate gap responses from default responses.
- Needs trained cue on both sides (interpolation, not extrapolation). Above
  the current gap there are only 15 deg of trained range (+35...+50).
- Width trade-off (updated 2026-10-30): wider gives more power to tell the
  response distributions apart (30 deg clearly better than 23 deg in the
  power simulation, 2026-10-28); a linear readout only interpolates if
  detector tuning exceeds ~half the gap; a wider gap costs more gated
  training time; very wide gaps may match neither edge template.
- If a second notch is used: full coverage needs at least as much trained
  range above the gap as the gap is wide -> a gap of ~15 deg, a lower gap, or
  a set extended upward (pointing comfort at +-50 deg still untested).
- **Starting point [decided, field-test]: gap +12...+35 deg, synthetic set
  over -50...+50 deg** (head-pointing comfort to +-50 deg still to check).
  Alternative was -35...-12 deg. Upper field costs less exposure (crossed by ~32-36 % of logged
  training trials vs ~42-46 % for a lower gap) and avoids the stronger
  compression of low targets. Width (15 / 23 / 30 deg) to be fixed after the
  cue-data review and simulation.

## Training (Training_AR + gap hook)

**The hurdle.** The cue depends on head-RELATIVE elevation (target minus head
pitch), so the gap is defined in that frame -- and every training trial
sweeps the relative elevation continuously from its onset value to 0. A
target 45 deg above the head passes through +35...+12 on the way; with Exp 1's
target draw, 35 % of logged training trials would even have started inside a
+12...+35 gap. Without countermeasures gap spectra are heard all the time.

**The two fixes (both essential):**

1. **Gating.** While the relative elevation is in the gap, the target is
   rendered through a gated filter (averaged envelope, low-passed, nothing
   above a cutoff below the lowest notch in the map). Why not a flat
   broadband sound or masking noise: a "no-notch" spectrum heard at gap
   directions becomes a template for the gap, and a novel gap notch is closer
   to it than to any trained spectrum -- template matching would then also
   put gap sounds into the gap. Why keep the low band: ITD/ILD stay there
   (azimuth guidance keeps working), it is the same sound (no extra event),
   and it stays externalised.
2. **Target draw rule.** Onset offset outside the gap; crossing allowed and
   gated (the only way relative elevations above the gap get trained, so the
   gap keeps trained cue on both sides -- interpolation, not extrapolation);
   absolute gap elevations remain pointing targets, approached from small
   offsets, so the response prior is not biased against the gap.

- Closed-loop game unchanged otherwise.
- **Gap gating [decided]:** while the target's head-relative elevation is in
  the gap, the target is rendered through the **gated filter** instead of the
  cue set: the averaged envelope (as above), low-passed with **no energy above
  the cutoff**. The cutoff lies below the lowest frequency any feature reaches
  anywhere in the map, including skirts, and the stopband residual must be
  below threshold at playback level. Azimuth guidance (ITD/ILD in the low
  band) stays spatial.
- Head-relative elevations outside the synthetic set's range are gated the
  same way, so feature positions beyond the range are never heard in
  training (needed for the probes).
- Gated filter parameters [decided as starting point, field-test]: cutoff
  0.25 oct below the lowest notch centre in the map (notch sigma ~0.047 oct,
  so >5 sigma clear of the skirt); minimum-phase FIR, >=60 dB stopband;
  residual above cutoff checked against threshold at the rig's playback
  level. Applied to the averaged envelope at each azimuth, ITD/ILD kept.
- Ramp at gate transitions; hysteresis at the boundary; nearest-neighbour grid
  snapping moves the effective edges to grid points **[decided]**.
- **Gap masker [decided 2026-10-30, supervisor's recommendation]:** while the
  head-relative elevation is in the gap, a flat masker noise is added. The
  target keeps playing at its fixed position and is still rendered through
  the gated filter underneath (silently), so the listener hears the target
  being masked rather than losing information. Purpose: the drop of the cue
  band is no longer an audible event of its own; ramps and hysteresis become
  less critical.
  - Risk: if the masker fuses with the target, the percept at gap directions
    is a flat, notch-free spectrum -- the "no-notch template for the gap" the
    gating was chosen to avoid (a novel gap notch is closer to it than to any
    trained spectrum, so template matching would also put gap sounds into the
    gap). The masker must be heard as a separate source.
  - Defaults (to confirm in the field): spectrally flat **above the cutoff
    only** (low band of the target stays audible with ITD/ILD, azimuth
    guidance intact); **diotic**, not spatialised or head-tracked (in-head /
    diffuse, apart from the externalised target); own on/off ramps, and a
    temporal structure that does not share the target's envelope (e.g.
    continuous vs pulsed target, if the training sound is pulsed); level just
    high enough that the low-passing of the target is inaudible.
  - The masker exists only in training. At test, gap targets are presented
    with their notch and no masker.
- **Target draw [decided; corrected 2026-10-26]:** the onset head-relative
  elevation must lie outside the gap. Crossing the gap on the way to 0 is
  allowed -- that is what the gating is for, and it is the only way relative
  elevations above the gap (+35...+50) are ever presented. (An earlier draft
  also forbade crossing; that would have left the gap without trained cue
  above it.) Absolute target directions may lie in the gap: they are drawn
  with a head pose that puts the onset offset outside the gap, so gap
  elevations are trained as pointing destinations and the response prior is
  not biased against them.
- To test in the field: residual audibility of the gated band, whether the
  gated sound itself has an elevation (without a cue ~half of listeners point
  to a non-zero elevation), transition artefacts, exposure lost to gating
  (from pose traces). Masker: one sound or two during a crossing (ask, or
  press-on-change), masker level that hides the low-passing at the rig's
  playback level, comfort/annoyance, effect on pointing during crossings.

## Localization (Localization_AR + hooks)

- Head starts at 0, so head-relative = absolute target elevation.
- Trial types:
  - trained-region targets
  - **gap targets**, dense enough for a within-gap slope
  - **probe trials [decided]:** spectra that never occurred in training.
    Pilot: one type only, the notch at positions beyond the set's range
    (never heard, see gating), ~10 % of test trials. Non-monotonic
    combinations deferred. Elimination sends them into the gap, template matching to
    the nearest trained elevation, a linear decoder to its weighted-sum
    estimate. Needs extra filters in the pyBinSim database and a per-trial
    switch (separate hook).
- Trial budget: within-gap slope SE ~0.2 at 65 gap trials, ~0.1 needs ~250
  (target spread scales the SE; a 28 deg gap gives ~2.5x the SE of the
  +-35 deg field).

## Measures

- Primary: within-gap elevation gain, change from the day-1 baseline (the
  pre-training gap interpretation is measured, not required to be zero).
- Secondary: within-gap polar error vs the same elevations without a gap;
  probe response locations; elevation gain AND polar error in the trained
  region (reported together).
- Later: model-based. Fit a linear and a template readout to each listener's
  trained-region responses and predict their gap and probe trials.

## Day 1 screen [decided 2026-10-26: same rules as learning_transfer]

Applied to the trained region of the synthetic set, against the day-1 own-HRTF
reference, with donor_screening.py's gates:

- elevation gain retained 30-55 % of own-HRTF gain (MIN/MAX_EG_RETAINED),
  absolute backstop EG >= 0.15 (MIN_EG)
- polar-error perturbation band (synthetic minus own, same day and stimulus)
- azimuth check for a broken render (ITD/ILD)

**Candidates [decided for the pilot: option A].** Three variants of the
synthetic set that differ only in frequency placement relative to the
listener's own main notch path (detected with the edge_shift detector):
(1) on that range, (2) 1/3 oct below, (3) 1/3 oct above. **Everything but
position is fixed for everyone [decided 2026-10-27]:** depth, width and sweep
rate stay at the pilot values; depth is NOT matched to the listener's own
notch. Sweep rate is deliberately not matched either: a synthetic notch with
the listener's own range AND own tilt is just a shifted copy of their own
cue, which the native map reads as a bias (Exp 1 dropped the ERB shift for
this reason; Van Wanrooij & Van Opstal saw no adaptation to pure shifts).

**Shift check (report only, never blocks).** A fixed tilt can still happen to
match a listener's own tilt (example listener: own ~0.005 oct/deg vs fixed
0.008). Rather than fitting the own tilt (fragile: own notch paths are not
straight lines), each variant is checked acoustically before the session
with the same ridge-slope measure the donor selection uses: if the native map
would read a variant as a shifted copy of the own cue (ridge slope > 0.5),
that variant is flagged in the screen table. What to do with a flagged
variant (skip it, or change its tilt) is decided during the pilot.

Frequency limit [decided 2026-10-27]: the top of every variant stays below
~15 kHz. If a listener's own notch range is so high that a variant would
exceed it, the placement is moved down (all three variants keep their 1/3 oct
spacing), rather than dropping a variant. Expected gradient: on-range -> native map reads it -> high retained gain
(likely > 55 %); shifted -> lower. The screen picks the in-band variant, as
the donor shortlist does in Exp 1. ~35 trials per variant, shuffled order,
retained gain over the trained region only. This is the crucial part of the
experiment; the strategy may be adjusted during the pilot (other knobs held
in reserve: sweep direction, notch+peak).

**Screen sampling.** EG slope SE = sigma / (SD(target el) * sqrt(n)), so
moving trials from azimuth columns to more elevation levels does not by
itself improve precision; what helps is (i) more trials, (ii) a wider
elevation range (the trained region here spans more than +-35 deg), and
(iii) lower response scatter, i.e. targets nearer the median plane, where
elevation responses are least noisy. Proposed: few azimuths close to the
midline, elevations across the full trained range. Applies to the Exp 1
screen as well (for Exp 1 after data collection).

**Decided (2026-10-26): (iii)** -- screen azimuths within +-20 deg of the
midline, elevations spread across the whole trained region. Proposed grid:
3 azimuths (-20, 0, +20) x 12 elevations = 36 trials per variant (to confirm
with the localization sequence, step 4).

The gap is NOT a screen criterion. Listeners will interpret gap spectra
somehow before training; that is measured as the day-1 baseline, and the
primary outcome is the change from it.

## Code layout [decided]

```
experiment/linear_decoder/
  design.md          this file
  linear_decoder.py  protocol (# %% cells)
  participants.csv
  synth_cue.py       per-participant cue set + gated filter + QC figure
  gap.py             gap definition, gating, target draw
  screen.py          day-1 screen
  analysis/          later
```

- Exp-specific signal processing lives here, not in `hrtf/`.
- Engine hooks in Training_AR / Localization_AR, off by default; Exp 1 path
  verified unchanged by dry run.
- Developed on branch `exp2-linear-decoder`; nothing reaches the lab PCs
  during Exp 1 (until 13.11.2026). `learning_transfer/` moves out of
  `protocols/` after Exp 1.

## Things to figure out

Issues raised in the concept review (2026-10-28). Not decisions -- questions
the design has to answer before the predictions can be stated firmly.

1. **The template prediction depends on the readout.**
   - Maximum (winner-take-all) or *sampling* readout (one response drawn per
     trial from the similarity-weighted template directions, as in
     Baumgartner et al. 2014): responses only at template directions, so no
     mass inside the gap beyond response scatter; for a gap target the
     responses split between the two edges (bimodal).
   - *Averaging* readout (similarity-weighted mean of template directions):
     interpolates, graded responses inside the gap -- indistinguishable from
     linear decoding in this design (it is effectively a linear readout over
     template-tuned units).
   - So the paradigm can reject template matching with a max/sampling
     readout, not template matching in general. State this explicitly.
2. **Similarity metric and gap width.** "Snap to the nearer edge" needs the
   gap spectra to still overlap the edge templates. If the gap is so wide
   that a gap spectrum overlaps neither, the nearest template is arbitrary
   and responses scatter over trained or native directions instead.
3. **Native templates.** Gap spectra can match templates from the
   listener's own ears; responses then follow the native map (seen in the
   toy model for the on-range placement variant). Check per variant
   acoustically before the session.
4. **Detector width (linear prediction).** Grounded at the periphery: a notch
   affects auditory filters over ~1 ERB (0.13-0.16 oct at 8-12 kHz,
   Glasberg & Moore 1990); plus notch FWHM 0.11 oct, divided by the sweep
   rate 0.008 oct/deg -> each channel sees the notch over ~30 deg of
   elevation. Assumption: the detectors feeding the readout are at least
   that broad (could be narrower, e.g. DCN edge units, Reiss & Young 2005;
   or broader, cortical elevation tuning, Trapeau & Schoenwiesner 2018).
   Interpolation is only predicted if effective width > ~half the gap;
   choose gap width and sweep rate so this holds comfortably at the
   peripheral width, otherwise linear decoding also predicts no
   interpolation and the experiment cannot decide.
5. **Default response to an unknown DTF (~0 deg, Paul's observation).** Gap
   responses from a listener who cannot use the cue there would be flat at
   ~0 deg, outside the +12...+35 gap -- distinguishable from graded and from
   edge clusters. Not diagnostic of mechanism on its own (a linear readout's
   intercept or a template model's prior at 0 deg give the same). Another
   reason the gap must not contain 0 deg. Learned elimination (flat at the
   gap centre) remains a separate possible strategy.
6. **Native weights in the linear account.** With native weights as the
   starting point (toy model, 2026-10-28) the post-training linear curve in
   the gap is smooth but not the identity and depends on input coding and
   native rehearsal; within-gap gain alone does not separate the accounts.
   The discriminating feature is the response *distribution* per target
   (bimodal vs unimodal), not the mean.
7. **Statistical power** of the bimodal-vs-unimodal test at realistic
   response scatter (simulation run 2026-10-28; to be discussed).

## Open decisions

1. Head-pointing comfort at +-50 deg (field test).
1a. Gap width, 15 vs 23 vs 30 deg: 30 deg clearly better for the two-cluster
    test (power simulation 2026-10-28), but the linear prediction needs
    detector width > ~half the gap (Things to figure out #4), a wider gap
    costs more gated training time, and a second notch needs trained range
    above the gap >= gap width (favours ~15 deg or a lower gap).
1b. Test-block composition: gap trials vs trained-region trials per block.
    Simulation: ~80-160 gap trials per listener pooled over the post-training
    tests of days 2-4 (response scatter 8-12 deg; Exp 1 data: own-HRTF median
    6.7 deg, modified HRTF median 10.2 deg).
1c. Analysis criterion fixed in advance: two-cluster vs smooth model per
    listener, with a minimum cluster separation / BIC threshold calibrated by
    simulation to ~5 % false alarms (compressed graded responses otherwise
    produce up to 30-40 % false "two clusters").
2. Exact screen grid (3 x 12 proposed).
3. Split of test trials between trained region, gap and probes.
4. Gated-filter cutoff and level; masker band, level and ramps (field test).
