# Encoding of Spectral Cues for Elevation: Models, Predictions, and a Diagnostic Manipulation Matrix

Scope: vertical/sagittal-plane localization from monaural pinna spectral cues (≈4–16 kHz), framed for an experiment in which a listener localizes with **their own (individual) HRTFs** and we apply **precise, targeted DTF manipulations** to dissociate competing accounts — including manipulations designed to *collapse* localization by removing a single posited cue.

---

## 1. Model families

| Model family | Core cue / computation | Frequency emphasis | Key references | Computational implementation |
|---|---|---|---|---|
| **Wideband spectral correlation / template match** | Cross-correlate the *whole* incoming DTF magnitude spectrum against stored, learned, direction-specific templates; pick best match. | Whole 4–16 kHz pattern | Blauert 1969; Hebrank & Wright 1974; Middlebrooks 1992; Hofman & Van Opstal 1998; Langendijk & Bronkhorst 2002 | Langendijk2002 (AMT); comparison stage of Baumgartner2014 |
| **Positive spectral gradient / rising edge** | Half-wave-rectified spectral derivative along tonotopic axis; keep only *rising* (positive) edges; match to template. Physiologically grounded in cat DCN type IV edge sensitivity. | Rising flank of N1 region | Reiss & Young 2005 (DCN); Davis et al. 2003 (ICC type O); Baumgartner, Majdak & Laback 2014 | **Baumgartner2014 (AMT), `do=1`** = positive-gradient front end; `do=0` disables it |
| **Spectral notch center frequency (two-notch)** | Elevation read from center frequencies of first/second pinna notches (N1, N2); ≥2 notches needed because notch-CF↔elevation is non-monotonic; P1 (~4 kHz, elevation-independent) acts as a reference. | N1 ≈ 6–10 kHz, N2 higher | Bloom 1977; Iida et al. 2007; Iida & Ishii 2018; Rajendran & Gamper 2019 | Parametric notch–peak (PNP) HRTF model (Iida toolkit) |
| **Covert peak area / spectral peak** | A (narrow-band) sound is heard at the elevation whose DTF has a gain *peak* at that frequency; broadband percept displaced away from attenuated regions. | Peaks (~8 kHz → "up") | Butler & Belendiuk 1977; Musicant & Butler 1984; Rogers & Butler 1992 | — (feature extraction; partial overlap with notch/peak models) |
| **Spectral derivative (two-sided)** | Match 1st/2nd spectral derivatives to templates; assumes source spectrum locally flat or locally constant-slope. Precursor to the rising-edge model, but does not rectify to positive only. | Slopes across band | Zakarauskas & Cynader 1993 | Conceptually subsumed by Baumgartner front end |
| **Bayesian / weighted ideal-observer** | Reliability-weighted spectral likelihood across bands, combined with spatial priors → MAP estimate. Wideband but with non-uniform band weights. | Weighted, peak weight ~8 kHz | Reijniers et al. 2014; Ege et al. 2018; Zonooz/Van Opstal 2019; ideal-observer (Sci Rep 2025) | Van Opstal weighted cross-correlation + Bayesian decision |

Note: families overlap. Baumgartner2014 is effectively a **feature (rising-edge) front end feeding a template-correlation back end**, which is why it is the most useful single tool here — toggling `do` moves it along the feature↔wideband axis.

---

## 2. What each model predicts under targeted DTF manipulations

Outcomes: **shift** = systematic elevation bias; **collapse** = stimulus–response slope (gain) → ~0; **degrade** = reduced gain/precision but residual mapping; **~none** = little change.

| Manipulation (per-direction, individual DTF) | Rising-edge (DCN / Baumgartner do=1) | Notch-CF (Iida two-notch) | Wideband template (Langendijk / Middlebrooks) | Bayesian-weighted (Van Opstal) | What it isolates |
|---|---|---|---|---|---|
| **A. Shift whole notch up** (edges + minimum together) | shift up | shift up | shift up | shift up | Dose / sanity check — all agree |
| **B. Shift rising edge up, notch minimum pinned** | **shift up** | **~none** | partial shift | partial shift | **Rising-edge vs notch-CF (the key contrast)** |
| **C. Shift falling edge down, notch minimum pinned** | ~none | ~none | partial shift | small | Falling-edge contribution; control for B (derivative model predicts an effect here) |
| **D. Fill the notch** (raise minimum to flat; removes edges + min) | collapse | collapse | degrade | degrade + upward bias | Necessity of the notch complex |
| **E. Reduce notch depth** (shallower; edges + min preserved) | gain ↓ (weaker edge) | robust-ish (CF intact) | graded degrade | gain ↓ | Depth matters more to edge/gradient than to notch-CF |
| **F. Flatten the whole notch band** (≈5.7–11.3 kHz → band-average) | collapse | collapse | strong degrade | strong degrade | Establishes the critical band & a collapse baseline (Langendijk) |
| **G. Flatten an *out-of-notch* band** (e.g. 4–5.7 kHz or >11.3 kHz; notch intact) | ~none | ~none | measurable degrade | small degrade | **Wideband/weighted vs pure single-feature** (Zonooz's argument) |
| **H. Remove positive gradients only** (flatten rising flanks, keep falling) | collapse | degrade/collapse | degrade | degrade | Direct lesion of the edge cue; pair with falling-gradient-removal control |
| **I. Spectral smoothing** (cepstral truncation; coarse shape kept) | gain ↓ with smoothing | notches merge → CF degraded | robust to moderate smoothing | robust-ish | How much fine detail each model needs (Kulkarni & Colburn) |
| **J. Raise in-notch power / contrast, no edge move** | ~none | ~none | ~none | **upward bias** | Confound control — isolates the Zonooz power effect to subtract from B/E |
| **K. Shift N2 independently of N1** | effect if N2 has in-range rising edge | systematic, dissociable from N1 | shift | shift | Tests the two-notch model specifically |

---

## 3. Design logic: "shift" vs "collapse"

- A **shift** manipulation (A–C, K) identifies the cue that *drives* the percept — which feature the system follows when features disagree.
- A **collapse / ablation** manipulation (D, F, G, H, I) identifies which cue is *necessary* — removing it should flatten the stimulus–response relation if and only if that cue carries the elevation information.
- The strongest tests are **minimal lesions**: remove exactly one model's posited cue while preserving the others (e.g., H removes rising edges; G removes wideband context but leaves the notch; B moves the edge but pins the minimum). Crossing a "shift" and a "collapse" version of the same feature is the most diagnostic (e.g., B + H both target the rising edge from opposite directions).

Why the **individual-HRTF baseline** matters: each listener already localizes near-optimally with their own DTFs, giving a high, stable ceiling against which the *selective* damage from a targeted lesion is measurable. Non-individualized HRTFs depress the baseline and confound "the manipulation broke it" with "the ears were never right."

---

## 4. Confounds to control (carry into the manipulation code)

1. **Power / contrast (Zonooz).** Edge/notch warps incidentally change in-band level and notch depth; added in-notch power alone biases elevation upward (~+30°) and can cause up-down confusions. → Match mean in-band level and notch depth across conditions; include manipulation J as an explicit control.
2. **Sound level (Macpherson & Sabin — "negative level effect").** Brief, high-level sounds degrade peripheral spectral coding. → Moderate level, duration ≥ ~80–150 ms.
3. **Within-session relearning.** Listeners adapt to altered spectra (Hofman 1998; Van Wanrooij & Van Opstal 2005; Carlile 2014; recent slab-based adaptation work). → Interleave/counterbalance conditions; keep blocks short; treat baseline drift as a covariate.
4. **Binaural integrity.** Off-median directions carry notches in both ears; manipulation is magnitude-only and must preserve ITD/onset so lateral position is unchanged.
5. **Coordinates.** Analyze in lateral/polar (double-pole) coordinates, not raw az/el, so sagittal planes pool and results compare directly to Baumgartner output (polar response angle, PE/QE).

---

## 5. Predict-then-test loop

1. Build baseline + manipulated DTF sets (A–K as needed).
2. Run each through the **Baumgartner2014** model with `do=1` (rising-edge on) and `do=0` (off), using the listener's own baseline DTFs as the template. Read predicted polar-response distributions and PE/QE.
3. For each manipulation, also generate the **wideband template** prediction (Langendijk2002 / Hofman–Van Opstal cross-correlation) as the competing hypothesis.
4. Pre-register the divergent predictions (esp. rows B, G, H where models split), power the study to that effect size, then test.
5. Post hoc, compare human gain/bias/precision to both model predictions per condition.

---

## References

- Baumgartner R, Majdak P, Laback B (2014). Modeling sound-source localization in sagittal planes for human listeners. *JASA* 136(2):791–802.
- Bloom PJ (1977). Creating source elevation illusions by spectral manipulation. *JAES*.
- Butler RA, Belendiuk K (1977). Spectral cues in the localization of sound in the median sagittal plane. *JASA*.
- Davis KA, Ramachandran R, May BJ (2003). Auditory processing of spectral cues for sound localization in the inferior colliculus. *JARO* 4:148–163.
- Hebrank J, Wright D (1974). Spectral cues used in localization on the median plane. *JASA* 56.
- Hofman PM, Van Riswick JG, Van Opstal AJ (1998). Relearning sound localization with new ears. *Nat Neurosci* 1:417–421.
- Iida K, Itoh M, Itagaki A, Morimoto M (2007). Median plane localization using a parametric model of the HRTF based on spectral cues. *Appl Acoust* 68:835–850.
- Iida K, Ishii Y (2018). Effects of adding a spectral peak (P2) to a parametric HRTF model on upper-median-plane localization. *Appl Acoust* 129:239–247.
- Kulkarni A, Colburn HS (1998). Role of spectral detail in sound-source localization. *Nature* 396:747–749.
- Langendijk EHA, Bronkhorst AW (2002). Contribution of spectral cues to human sound localization. *JASA* 112:1583–1596.
- Macpherson EA, Middlebrooks JC (2003). Vertical-plane sound localization probed with ripple-spectrum noise. *JASA* 114:430–445.
- Macpherson EA, Sabin AT (2013). Vertical-plane sound localization with distorted spectral cues. *Hear Res*.
- Middlebrooks JC (1992). Narrow-band sound localization related to external ear acoustics. *JASA* 92:2607–2624.
- Musicant AD, Butler RA (1984). The influence of pinnae-based spectral cues on sound localization. *JASA* 75:1195–1200.
- Rajendran VG, Gamper H (2019). Spectral manipulation improves elevation perception with non-individualized HRTFs. *JASA-EL* 145(3):EL222.
- Reijniers J, Vanderelst D, Jin C, Carlile S, Peremans H (2014). An ideal-observer model of human sound localization. *Biol Cybern* 108:169–181.
- Reiss LAJ, Young ED (2005). Spectral edge sensitivity in neural circuits of the dorsal cochlear nucleus. *J Neurosci* 25(14):3680–3691.
- Van Wanrooij MM, Van Opstal AJ (2005). Relearning sound localization with a new ear. *J Neurosci* 25:5413–5424.
- Zakarauskas P, Cynader MS (1993). A computational theory of spectral cue localization. *JASA* 94:1323–1331.
- Zonooz B, Arani E, Körding KP, Aalbers PATR, Celikel T, Van Opstal AJ (2019). Spectral weighting underlies perceived sound elevation. *Sci Rep* 9:1642.
