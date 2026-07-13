# Learning-transfer experiment — methods

This folder holds the protocol runner (`learning_transfer.py`), the per-subject
counterbalance sheet (`learning_transfer_block_order.csv`), and this methods
note. Session-1 acquisition (HRIR recording, headphone calibration, dome
externalization check, midline AR localization) runs first from
`../HRIR_Recording.py`; everything from the full-field baseline onward runs from
`learning_transfer.py`.

## Goal of the manipulation

Modify an individually measured HRTF enough to **disrupt** elevation perception,
but not so much that it cannot be **relearned** through the training paradigm.
The manipulation must therefore displace the elevation cue to a new, consistent
mapping (an ear-mould / non-individual-HRTF situation, demonstrably relearnable)
rather than destroy it or create a self-contradictory cue that listeners simply
down-weight.

## The modification: coherent ERB translation of the fine spectral structure

Implemented in `hrtf.processing.modify.shift_detail`. Per HRIR, per ear:

1. **Log-magnitude spectrum.** `L(f) = log|H(f)|` from the rFFT of the HRIR.
2. **Coarse/fine split (Kulkarni & Colburn 1998, *Nature* 396:747).** Fit a
   truncated cosine (Fourier) series to `L(f)` and keep the first `M`
   coefficients → the coarse envelope `E(f)`. The residual `D(f) = L(f) − E(f)`
   is the fine structure: the pinna peaks and notches that carry the vertical
   (elevation) cue. We use `M = 4` — few enough that listeners cannot localise
   from the envelope alone, but the sound still externalises (the K&C result).
3. **ERB translation of the fine structure.** Each output frequency `f` samples
   the detail at the frequency whose ERB-number is `ERB(f) − Δ`, i.e.
   `D'(f) = D(erb_to_hz(hz_to_erb(f) − Δ))`. This is a constant step of `Δ` ERB
   (Glasberg & Moore 1990), so notch **spacing is preserved on the auditory
   scale**. `Δ > 0` shifts cues up, `Δ < 0` down. The envelope is held fixed, so
   every direction keeps a unique pattern — just displaced by `Δ`. This is a
   bijective remap, not a destroyed or conflicting cue.
4. **Band selection (Trapeau peak-VSI octave); pattern translates up in place.**
   The full detail is shifted first — each output frequency `f` takes the detail
   from `ERB(f) − Δ` — and the band window (**5.7–11.3 kHz**, the peak-VSI band
   where the elevation cue is strongest, Trapeau et al. 2016; 0.25-octave skirt)
   then selects it. So *inside* the band the pattern slides up and replaces the
   higher-frequency content there — because content is sampled from below, the
   band top is never cut off — while *outside* the band the native detail is
   kept. `band=None` shifts the whole spectrum.
5. **In-band energy equalisation.** The energy of the shifted detail is matched
   to the source, per direction and ear, so relocating a non-stationary residual
   does not change in-band spectral contrast — removing the overall level /
   in-notch-power cue (cf. Zonooz et al. 2019) as a confound. Only the cue
   *position* differs between native and modified.
6. **Magnitude-only, original phase.** The new magnitude is recombined with the
   **original phase** (`H' = |H'|·e^{i∠H}`). No minimum-phase step, no ITD
   restoration here: interaural time/level cues are imposed upstream, when the
   measured frontal arc is expanded across azimuth
   (`hrtf.record.processing.expand_azimuths_with_binaural_cues`), so the SOFA read
   here already carries the imposed ITD in its phase and we must preserve it.

### Δ selection

`SHIFT_ERB` is set per pilot. A constant ERB shift is ≈ a constant frequency-scale
factor above ~1 kHz: factor 1.3 ≈ 2.4 ERB, factor 1.4 ≈ 3.0 ERB (Δ = 1.5 ERB ≈
+17 %, Δ = 2 ERB ≈ +24 %) — on the order of natural between-listener HRTF scaling,
i.e. a shift the system plausibly can relearn. Default `SHIFT_ERB = 2.5`; tune from
the pilot, then rebuild the modified SOFA.

### Configuration (in `learning_transfer.py`)

| param | default | meaning |
|---|---|---|
| `SHIFT_BAND` | (5700, 11300) | peak-VSI octave selected, then shifted (`None` = whole spectrum) |
| `SHIFT_ERB` | 2.5 | ERB displacement of the fine detail |
| `SHIFT_ENV_NKEEP` | 4 | Fourier coeffs kept for the envelope (`M`) |
| `SHIFT_SKIRT` | 0.25 | raised-cosine taper on the selection window [octaves] |
| `SHIFT_EQ_RMS` | True | match in-band detail energy per direction/ear |

The same `SHIFT_BAND`, `Δ` and `M` are used for **every direction and both ears** —
a single map to learn; a direction- or ear-dependent shift would collapse the design
or read as a lateral cue.

## Quality control

`build_modified_sofa()` writes `<subject>_shift.sofa` and saves a **split-QC
panel** (`<subject>_shift_split_qc.png`): full log-magnitude vs. the `M = 4`
envelope, stacked by elevation. The envelope must be smooth **and roughly
elevation-invariant**; if the envelope still tracks elevation, `M` is freezing a
cue that also carries elevation (a cue conflict) — lower `M`. A baseline-vs-
modified transfer-function image is also available via `modify.plot`.

## Session flow

- **Session 1 (`../HRIR_Recording.py`):** record individual HRIR + reference,
  deconvolve/equalise, (optional) midline ITD/ILD alignment, azimuth expansion
  with imposed binaural cues, write `<subject>.sofa`; calibrate headphones; dome
  localization (real speakers, vertical midline) to establish externalization;
  midline AR localization.
- **Day 1 (`learning_transfer.py`):** `native` — original HRIR, binaural, full
  field (baseline); **build the modified HRTF (ERB shift)**; `baseline_A` and
  `baseline_D` — naive modified monaural references matched to final A and D.
- **Adaptation days:** `daily` monaural trained-ear test (training game runs from
  `Training.py`).
- **Final day:** 2×2 (Ear × Side) — A/B/C/D, in the subject's counterbalanced
  `block_order`. D (untrained ear, mirrored) is the main transfer test.

## References

- Kulkarni & Colburn (1998). Role of spectral detail in sound-source
  localization. *Nature* 396, 747–749.
- Glasberg & Moore (1990). Derivation of auditory filter shapes. *Hear. Res.*
- Trapeau, Aubrais & Schönwiesner (2016) — VSI band; fast/persistent adaptation.
- Zonooz et al. (2019) — spectral in-notch power weighting.
