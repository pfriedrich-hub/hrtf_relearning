# Source spectrum × interaural spectral difference

Design note for `source_spectrum_isd.py`. Written 2026-08-18.

Three files, three jobs: **this one is the rationale** — why the experiment is
shaped the way it is, and what was decided against.
`source_spectrum_isd_protocol.md` is the run sheet, the power calculation and
the pre-specified analysis plan. `source_spectrum_isd.py` is the code.

## 1. The question

The spectrum at the eardrum is the product of an unknown source spectrum and
the direction-dependent filter:

    P(f) = S(f) · H(f, dir)

One equation, two unknowns. Every account of spectral-cue localization is a
proposal for how the auditory system closes that gap, and there are only four:

1. **Assume `S` is flat** and template-match on the absolute spectrum.
2. **Cancel `S` by differentiating** along frequency — derivative operators are
   blind to smooth multiplicative colouration (Zakarauskas & Cynader 1993;
   Baumgartner's positive spectral gradient extraction).
3. **Cancel `S` binaurally** — `P_L/P_R = H_L/H_R`, exact, at every frequency
   and every ripple density, with no assumption about `S` at all.
4. **Infer `S` and direction jointly** under a prior over source spectra
   (Reijniers et al. 2014; Barumerli et al. 2023; Baumgartner et al. 2026).

This experiment isolates (3). It does not decide between (1), (2) and (4) —
that needs the ensemble-entropy and burst-coherence manipulations described in
§7.

## 2. Why the current stimulus cannot answer it

The validated ripple stimulus (`rms_tilt=3, rms_cue=0`) confines source
variation to **below 0.5 ripples/oct**, while the elevation cue lives in
**0.5–2 ripples/oct**. Source and cue therefore occupy disjoint bands, and any
band-selective read-out separates them with no inference whatsoever. A null
result with that stimulus is uninformative: mechanisms (1)–(4) all predict it.

Only **in-band** source variation forces the issue. Inside the cue band a
source feature and a directional feature are formally indistinguishable in a
single monaural spectrum — the information is not there — so performance must
then rest on a prior, on across-trial statistics, or on the interaural
comparison.

Note this is an identifiability limit, not a processing failure. An in-band
effect on its own does not show that the system "cannot reconstruct the source
spectrum"; it shows that the monaural single-snapshot channel cannot. Which is
exactly why it has to be crossed with the availability of another channel.

## 3. The manipulation

`hrtf/processing/diotic.py` duplicates one ear's measured az=0 arc onto both
ears and re-expands across azimuth, so the interaural spectral difference is
frequency-flat while the spherical-head ILD and ITD are imposed as usual. The
sound still lateralizes; only the *difference* between the ears' spectra stops
carrying elevation information.

How much is removed is measured, not assumed. From
`localization_helpers/spectral_metrics.py` — 0.5–2 ripples/oct, az 0, dB rms
per DCT coefficient on the 3.5–16 kHz axis this codebase measures cue:source
ratios on:

| set | monaural L | monaural R | ISD | ISD/mon |
|---|---|---|---|---|
| AS | 7.84 | 6.25 | 5.92 | 0.76 |
| GS | 13.79 | 12.87 | 5.60 | 0.41 |
| FS | 13.59 | 12.89 | 12.32 | 0.91 |
| KU100 | 12.50 | 12.01 | 6.48 | 0.52 |
| KEMAR (symmetric dummy) | 12.57 | 12.57 | **0.00** | 0.00 |

KEMAR is the anchor: a symmetric head has *no* midline ISD cue at all, so the
channel exists only because real ears are asymmetric. It is worth checking per
subject — a listener whose ISD depth is already near zero cannot test the
hypothesis.

The absolute numbers depend on the analysis axis, so quote the axis whenever
you quote them. On a 0.5–16 kHz axis with the rms taken over the reconstructed
shape rather than per DCT coefficient, the same measurement gives AS
2.02/1.86/1.21 (0.60), GS 2.47/2.35/1.37 (0.55), FS 2.62/2.53/2.05 (0.78),
KEMAR 0.00. What survives both choices is the part the design rests on: real
ears retain roughly half to nine tenths of the monaural cue depth through the
L−R subtraction, and a symmetric head retains none.

**The sphere radius must be read off the native SOFA, not defaulted.** The two
conditions have to differ *only* in the interaural spectral difference.
`expand_from_midline` defaults to `head_radius=0.0875` while the recording
protocol measures ~0.0725 per subject, so a defaulted rebuild would change the
diotic set's ITD and ILD as well and confound the manipulation. `diotic_hrtf`
therefore refuses to run without an explicit radius, and cell 1 recovers it
with `fit_head_radius.fit_from_sofa(native_path)` — which returns the imposed
radius with ~0 µs residual for anything this pipeline built — then re-fits the
written diotic file and prints the difference. A large residual on the *native*
fit means that SOFA was expanded with the legacy `itd_method='onset'`; rebuild
it before running, because re-expanding here uses `'phase'`.

**Unavoidable side effect.** Making the two ears equal necessarily puts one
ear's pattern on the other. The duplicated ear's own monaural cue is untouched,
so use the better/trained ear and counterbalance `DIOTIC_EAR` across subjects
if the overwritten ear needs ruling out.

## 4. Design

2 × 2 plus an anchor per ISD level, all midline, all on the native HRTF:

| ISD | stimulus | expectation |
|---|---|---|
| natural | noise | anchor; also a render sanity check |
| natural | below-band ripple | ≈ noise — filtering suffices |
| natural | in-band ripple | degraded |
| diotic | noise | anchor; checks the diotic render still localizes |
| diotic | below-band ripple | ≈ noise — filtering still suffices |
| diotic | in-band ripple | degraded **more** |

**The result is the interaction**, not any main effect:

    (diotic_in − diotic_below) − (natural_in − natural_below)

Positive and clear of zero → the interaural difference is what normally
rescues in-band source variation. Indistinguishable from zero → the listener is
not using that channel, and the monaural-prior question becomes the whole
story. Both outcomes are informative, which is the point.

**`RMS_CUE` is a constant, the same for every participant** (decided
2026-08-18). The alternative was to calibrate it per subject to a fixed
cue:source ratio, which would equalise difficulty across listeners. That was
rejected: every comparison this experiment rests on is within subject, so a
listener who localizes a given ripple better than another changes nothing, and
a constant stimulus is one line of method rather than a per-subject procedure
on test day. Cue depth still differs enough between subjects to matter (AS
7.84 dB, GS 13.79, FS 13.59), so the constant is chosen **once, against the
shallowest cue in the pool** — the same worst-case rule `stimulus_check.py`
cell 6 already uses to pick `rms_tilt` from the weaker ear. What a constant
cannot absorb is a listener at floor; report each subject's cue depth
alongside their data, and exclude from the interaction anyone whose natural
in-band cell is already at floor.

The 1:1 starting point is a guess and should be replaced by a ladder — sweep
`rms_cue` in a pilot the way the dome ladder swept `rms_tilt`, and take the
value that costs a measurable but sub-floor amount of polar error. A null
interaction is uninterpretable if both cells sat on the floor.

## 5. Why midline only

Azimuth in these SOFAs is **synthesised**. Only the az=0 arc is measured; every
other direction is that arc times a spherical-head model
(`expand_azimuths_with_binaural_cues`). Verified in the files: the left-ear
elevation pattern at az 0 correlates r = 1.00 with the pattern at ±21° and
±42° for AS, GS and FS, while KEMAR and KU100 fall to r = 0.64–0.79 at 40°.

Three consequences:

- az 0 is the only azimuth where the AR render is acoustically faithful to the
  measurement, so a midline-only test is the fair comparison, not merely a
  convenient one.
- Off-midline targets would add no independent spectral information, so they
  cannot strengthen the ISD contrast.
- A midline-vs-lateral version of this experiment **cannot be run in AR** with
  these SOFAs at all. It needs the dome, or a real 2-D measurement.

Cost of the choice: with every target on one arc the response space is
effectively 1-D and subjects may restrict their responses accordingly. The fix
is a minority of off-midline **filler trials** — `kind='midline_filler'` in
`make_sequence.py`, 12 fillers against 51 midline trials. They keep the task
2-D, they are tagged `'filler'` on `sequence.target_set`, and they are dropped
before every analysis. They are emphatically not a condition: in AR an
off-midline filler is the same measured arc with a level offset and a time
shift, so it carries no independent spectral information at all.

Do not compare gains from these blocks against 2-D sector blocks either way.

## 5a. Learning is an outcome here, not only a confound

If removing the binaural channel forces a fall back on priors over the source
ensemble, the diotic in-band block should **improve across trials** while the
natural one does not. That is a direct prediction of the statistical-learning
account (Baumgartner et al. 2026) and it costs nothing to measure — cell 11
reports the within-block slope per cell.

It does constrain the design:

- The ISD level is blocked, because switching it rebuilds the pybinsim
  database. Its order therefore alternates across subjects (`SUBJECT_INDEX`),
  and the stimulus order rotates within each half so neither ripple condition
  is systematically last.
- Carryover cuts the other way: if the diotic half teaches something it may
  persist into the natural half. Reporting the first half of each subject
  separately is the check.
- True trial-by-trial interleaving would remove both problems and is not
  available: pyBinSim selects filters by head orientation, so two conditions
  cannot be hidden in one database at fake azimuths.

## 5b. The reverse correlation needs no design change

Because `sequence.stim_params` logs the exact DCT coefficients of every trial's
envelope, the source-bias slope — does the imposed spectrum leak into the
responses, and does it leak *more* when the ISD is gone? — can be computed
after the fact from data collected under this protocol. It is arguably the
sharper dependent measure, since it is mechanistic rather than a difference of
difficulties, and with `rms_cue > 0` it can be measured **inside** the cue band
for the first time. Deliberately left out of the protocol: it is an analysis,
not a procedure, and nothing about the session changes if it becomes the
primary outcome later.

## 6. Outcome measure

Mean absolute **local polar error**, with quadrant errors (|error| > 90°) split
off and reported as a rate (`localization_analysis.polar_error`). Elevation
gain is secondary: it is a slope, so it can look healthy while absolute
accuracy is poor, and the fitted intercept absorbs any constant bias. On the
midline polar error reduces to plain elevation error.

Trial counts: 17 midline sources × 3 = 51 trials per block, 6 blocks. That
gives roughly ±3° of bootstrap CI per cell, which is the same order as the
expected effect — so this is a within-subject design that needs several
subjects, not a single-subject result.

## 7. What this deliberately does not test

The other three ways of coping with an unknown source spectrum need their own
manipulations, all cheap on top of this machinery:

- **Ensemble entropy** — fresh envelope every trial vs a fixed set of 4 vs one
  per block, via the existing `seed=` argument. Identical instantaneous cue
  SNR, different learnability. Tests the statistical-learning account.
- **Burst coherence** — the envelope is currently shared across all five 25 ms
  bursts (r = 0.95). Randomising it per burst lets within-trial averaging
  converge on `H` while leaving the instantaneous SNR untouched. The cleanest
  separation of a snapshot read-out from an integrating one.
- **Feature type** — smooth in-band ripple vs a spurious notch of matched rms.
  Tests whether the prior is band-shaped or feature-shaped.

Run this one first: it is the only one of the four that removes a channel
rather than changing the stimulus, so it is the one whose null is cleanest.
