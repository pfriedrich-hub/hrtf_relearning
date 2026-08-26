# Protocol: source spectrum × interaural spectral difference

Operational protocol and pre-specified analysis plan. The *rationale* lives in
`source_spectrum_isd_design.md` and is not repeated here; the *code* is
`source_spectrum_isd.py`. Written 2026-08-18.

---

## 1. Question and predictions

Does the auditory system use the interaural spectral difference to cope with an
unknown source spectrum? The source cancels exactly in `P_L/P_R = H_L/H_R`, at
every frequency and every ripple density. Removing that channel should
therefore hurt **only** when the source varies inside the cue band, where no
filter can separate source from filter.

The design crosses ISD (natural / diotic) with source band (below / in), plus a
noise anchor per ISD level. **The test is the interaction:**

```
I = (in − below)_diotic  −  (in − below)_natural        [degrees of polar error]
```

| outcome | reading |
|---|---|
| I > 0, CI excludes 0 | the ISD is what normally rescues in-band source variation |
| I ≈ 0, both in-band costs > 0 | in-band variation hurts, but not because of the binaural channel — the monaural prior question becomes the whole story |
| I ≈ 0, both in-band costs ≈ 0 | uninformative: `RMS_CUE` too low, or both cells at ceiling. Re-run the ladder, do not interpret |
| below-band ≠ noise in either ISD level | manipulation check failed — the below-band stimulus was supposed to be filterable |

The third row is the one that kills the experiment, and it is why the pilot
ladder in §4 is not optional.

## 2. Participants

Normal-hearing adults with an individually recorded HRIR and a calibrated
headphone filter already on file (`HRIR_Recording.py` completed).

**Pre-specified exclusions**, applied before looking at the interaction:

1. **No binaural channel to remove.** `isd_depth` on the native SOFA below
   1.0 dB rms (cue band, az 0). Such a listener is already effectively diotic
   at the midline and cannot test the hypothesis. Report them; exclude from
   the interaction.
2. **Floor.** Mean polar error in the *natural / in-band* cell above 30°, or
   quadrant-error rate above 20% in any cell. Nothing can be inferred from a
   cell at floor.
3. **Aborted or partial block** — any cell with fewer than 40 analysable
   midline trials.
4. **Diotic build QC failed** (§4, step 4).

Exclusions are recorded with the reason in the session log; no subject is
dropped for a reason invented after seeing their interaction.

## 3. Sample size

Trial-level SD of local polar error, measured from AS's native-HRTF AR blocks
(2026-08-18): **7.5°**. Donor-HRTF blocks run 10–12°, so a study on modified
HRTFs would need more subjects than this.

Within-subject SE of the interaction, `7.5 · √(4 / n_trials)`:

| midline trials per cell | within-subject SE of I |
|---|---|
| 34 | 2.57° |
| 51 (the protocol) | 2.10° |
| 68 | 1.82° |

Minimum detectable interaction, one-sample *t* across subjects on the
per-subject estimate, 80% power, two-sided α = .05, at 51 trials per cell:

| between-subject SD | 6 | 8 | 12 | 16 | 20 subjects |
|---|---|---|---|---|---|
| 0° | 3.0 | 2.4 | 1.9 | 1.6 | 1.4 |
| 2° | 4.1 | 3.3 | 2.6 | 2.2 | 1.9 |
| 3° | 5.2 | 4.2 | 3.3 | 2.7 | 2.4 |
| 4° | 6.4 | 5.2 | 4.0 | 3.4 | 3.0 |

The between-subject SD of the interaction is unknown — there is no pilot for
it. **Plan for 12, look after 4.** After the fourth subject, estimate the
between-subject SD from the four per-subject interactions and re-read this
table; that look is on the *variance only* and is pre-specified, so it costs
nothing in error rate. For scale: the whole ripple-vs-noise cost on AS's own
HRTF was +3.65°, so an interaction much above 3° would be surprisingly large.

Note where the leverage is. Once the between-subject SD exceeds ~2°, doubling
the trials per cell (51 → 102, an extra 50 min per session) moves the total SE
by less than adding two subjects does. **Recruit rather than lengthen.**

## 4. Preparation, before the subject arrives

Days ahead, once per subject:

1. **Native SOFA is current.** Built after the 2026-08-19 ILD fix and with
   `itd_method='phase'`. Check with
   `fit_from_sofa(native_path)` — a residual above a few µs means the legacy
   `'onset'` expansion; rebuild before running, because the diotic set is
   expanded with `'phase'` and the two conditions would then differ in ITD.
2. **Head radius** from that same fit. Cell 1 does this automatically and
   passes it to the diotic build — never let `expand_from_midline` default to
   0.0875.
3. **Run cell 1** (build the diotic SOFA). QC, all three must hold:
   - the duplicated ear's monaural cue depth is **unchanged**;
   - the ISD depth falls to ≈0 **at every azimuth**, not just at 0 — a level
     offset and a time shift restore no source information, and this is where
     that is verified rather than asserted;
   - native vs diotic head radius agree to better than 1 mm.
4. **Run cell 2** and read the printed cue:source ratios. Below-band must leave
   the cue standing clear inside the cue band; in-band must bring it to roughly
   the cue's own level.
5. **Loudness match** with `match_ar_dome_loudness.py`, at OS volume 50%.
   Record `GAIN`. Both SOFAs are level-normalised per build, but check the
   diotic render by ear as well — both ears now carry one ear's DTF.
6. **Pre-build the diotic binsim database** by constructing a `Localization`
   against it once. The first build takes minutes and should not eat session
   time.

Once per study, not per subject: **the `RMS_CUE` ladder** (cell 0). Sweep
`rms_cue` on the shallowest-cue subject in the pool and pick the value that
costs a measurable but sub-floor amount of polar error. The current 2.0 dB is a
1:1 cue:source guess and has not been laddered. Until it has, treat the first
subject as a pilot.

## 5. Session run sheet

Total ≈ 90 min including breaks. Six blocks of 63 trials (51 midline + 12
off-midline fillers), ~9 min each.

| # | step | ~min |
|---|---|---|
| 1 | Consent, demographics (`collect_demographics`), headphones and tracker fitted | 10 |
| 2 | Cell 3 preflight: OS volume pinned to 50%, tracker charged, print the block order | 3 |
| 3 | Familiarisation: one short natural/noise block, discarded | 5 |
| 4 | Blocks 1–3 (first ISD half, in the printed order), externalization rating after each | 30 |
| 5 | Break, headphones off | 10 |
| 6 | Blocks 4–6 (second ISD half), externalization rating after each | 30 |
| 7 | Debrief: ask what the two halves sounded like, log it verbatim | 5 |

**Run the block cells in the order cell 3 prints.** That order *is* the
counterbalance: the ISD level is blocked because switching it rebuilds the
binsim database, its order alternates with `SUBJECT_INDEX`, and the two ripple
conditions rotate within each half so neither is systematically last.

At the start of every block the subject recentres on the LED and presses enter;
the sensor calibrates on that. Between blocks the headphones stay on unless
the subject asks — a re-seat changes the headphone transfer function.

**Externalization is rated after every block** (`collect_externalization_rating`,
the standard prompt). The diotic render puts one ear's DTF on both sides and
will sound more in-the-head; a difference in externalization is a live
alternative explanation for a difference in polar error, so it is measured
rather than assumed away.

## 6. What is recorded, and where

Everything lands in `subject.localization` under keys tagged `_ssISD-<isd>-<band>`,
written after every trial. Per sequence: `isd`, `source_band`, `diotic_ear`,
`subject_index`, `target_set` (per-trial 'midline'/'filler'), `stim_params`
(per-trial DCT coefficients — the reverse correlation depends on these),
`stim_settings`, `hrir_params` (the diotic SOFA's embedded
`GLOBAL_ModificationParams`), and `externalization_rating`.

Also log by hand, in the session notes: OS volume, `GAIN`, the cell-1 QC
printout, the cell-2 ratios, anything unusual, and the debrief.

## 7. Analysis plan (pre-specified)

**Outcome.** Mean absolute local polar error over **midline trials only** —
fillers are dropped, they are a response-strategy control and carry no
independent spectral information in AR. Quadrant errors (|error| > 90°) split
off and reported as a rate.

**Primary.** Per subject, compute `I` as in §1. One-sample two-sided *t* across
subjects against 0, α = .05. Report the mean interaction with its 95% CI and
every per-subject value — with a dozen subjects the individual estimates matter
more than the *p*.

**Sensitivity.** Linear mixed model on trial-level |polar error|: fixed effects
ISD × band, random intercept and random ISD × band slope per subject. Reported
alongside; the *t* test is what the conclusion rests on, because it is the one
whose assumptions can be checked by eye.

**Secondary, pre-specified.**

1. *Externalization.* Same interaction on the ratings. If it patterns with the
   polar-error interaction, the binaural interpretation is confounded and must
   be qualified.
2. *Within-block learning.* Slope of |polar error| against trial number per
   cell. Prediction: the diotic in-band cell improves across trials while the
   natural one does not — a fall back on priors over the source ensemble.
3. *Manipulation check.* below-band vs noise within each ISD level; expected to
   be indistinguishable.

**Exploratory, and honestly labelled as such.** The source-bias slope: regress
the per-trial elevation residual on the trial's imposed envelope, using the
logged coefficients. Two predictors — broad tilt (HF−LF in dB) and, now that
`rms_cue > 0`, the projection of the cue-band component onto the local DTF
elevation gradient. Prediction: a steeper slope in the diotic condition. This
needs no design change and can be run on data already collected, which is
exactly why it is exploratory here rather than primary — it will be
pre-specified in whatever study follows.

**Not analysed:** filler trials, the familiarisation block, azimuth gain.

## 8. Troubleshooting

| symptom | do this |
|---|---|
| cell 1 ISD depth not ≈0 off the midline | stop; the arc duplication or the expansion is wrong. Do not run the subject |
| native fit residual > few µs | rebuild the native SOFA with `itd_method='phase'` first |
| diotic blocks sound in-the-head | expected. Record the rating, carry on; that is what §7 secondary 1 is for |
| subject reports the two halves sound like different experiments | log it verbatim, carry on. It is a real observation about the manipulation, not a failure |
| a block aborts mid-way | it is already written trial-by-trial. Re-run the same cell; the new run gets its own timestamped key. Note which one to use |
| binsim rebuild on every construction is eating the session | pre-build both databases the day before (§4 step 6) |

## 9. Checklist

Before the subject:

- [ ] native SOFA current, phase ITD, radius fitted
- [ ] cell 1 run, all three QC criteria pass
- [ ] cell 2 ratios read and recorded
- [ ] `RMS_CUE` from the study ladder (or subject flagged as pilot)
- [ ] loudness matched, `GAIN` recorded, OS volume 50%
- [ ] both binsim databases pre-built
- [ ] `SUBJECT_INDEX` set to recruitment order

After the session:

- [ ] six blocks on file, all with `isd` / `source_band` set
- [ ] six externalization ratings
- [ ] session notes written, including the debrief
- [ ] cell 10 run, output pasted into the notes
