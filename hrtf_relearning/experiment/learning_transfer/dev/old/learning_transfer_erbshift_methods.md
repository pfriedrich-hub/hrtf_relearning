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

Implemented in `hrtf.processing.shift_spectral_detail.shift_spectral_detail`
(`modify.shift_detail` is a thin wrapper around it). Per HRIR, per ear:

1. **Log-magnitude spectrum.** `L(f) = log|H(f)|` from the rFFT of the HRIR.
2. **Coarse/fine split (Kulkarni & Colburn 1998, *Nature* 396:747).** Fit a
   truncated cosine (Fourier) series to `L(f)` and keep the first `M`
   coefficients → the coarse envelope `E(f)`. The residual `D(f) = L(f) − E(f)`
   is the fine structure: the pinna peaks and notches that carry the vertical
   (elevation) cue. We use `M = 4` — few enough that listeners cannot localise
   from the envelope alone, but the sound still externalises (the K&C result).
3. **Selection (Trapeau peak-VSI octave).** A window `w(f)` picks the detail to
   move: **5.7–11.3 kHz**, the peak-VSI band where the elevation cue is
   strongest (Trapeau et al. 2016). `D_sel = w·D` is transported; the residual
   `D_rest = D − D_sel` stays exactly where it is. Edges are **hard**
   (`SHIFT_SKIRT = 0`): under transport a tapered edge would leave a partial copy
   of a feature at its origin *and* deposit a partial copy at the target — the
   same notch at reduced depth in two places. `band=None` transports the whole
   detail.
4. **ERB transport of the selected structure.** `D_sel` is carried by a constant
   step of `Δ` ERB (Glasberg & Moore 1990), implemented by sampling at
   `ERB(f) − Δ` with zero extrapolation outside the source support, so a feature
   at `f₀` reappears at `erb_to_hz(hz_to_erb(f₀) + Δ)`. Constant on the ERB axis
   means notch **spacing is preserved on the auditory scale**. `Δ > 0` moves cues
   up, `Δ < 0` down. The envelope is held fixed, so every direction keeps a unique
   pattern — just displaced by `Δ`: a bijective remap, not a destroyed or
   conflicting cue. Final spectrum: `L' = E + D_rest + D_moved`.

   **The modified region moves with the content.** The window says *which*
   features move, not *where* the result is allowed to live; for `Δ > 0` they land
   in `target_band(band, Δ)`, above 11.3 kHz. Nothing is dropped at the band edge
   and nothing is duplicated: every selected feature reappears at its new
   frequency with its depth intact.
5. **Per-ERB energy equalisation.** The transported detail is rescaled so its
   per-ERB RMS matches the selected detail, per direction and ear, so relocating a
   non-stationary residual does not change spectral contrast — removing the overall
   level / in-notch-power cue (cf. Zonooz et al. 2019) as a confound. Because the
   transport is a pure translation on the ERB axis this is already true up to
   interpolation loss, so it is a small correction. Only the cue *position*
   differs between native and modified.
6. **Magnitude-only, original phase.** The new magnitude is recombined with the
   **original phase** (`H' = |H'|·e^{i∠H}`). No minimum-phase step, no ITD
   restoration: the SOFA read here already carries its binaural cues, and keeping
   the original phase passes them through intact. ITD is untouched; broadband ILD
   is preserved because both ears receive the same ERB transport, with its fine
   structure travelling along with the cue.

### Δ selection

`SHIFT_ERB` is set per pilot. A constant ERB shift is ≈ a constant frequency-scale
factor above ~1 kHz: factor 1.3 ≈ 2.4 ERB, factor 1.4 ≈ 3.0 ERB (Δ = 1.5 ERB ≈
+17 %, Δ = 2 ERB ≈ +24 %) — on the order of natural between-listener HRTF scaling,
i.e. a shift the system plausibly can relearn. Default `SHIFT_ERB = 2.5`; tune from
the pilot, then rebuild the modified SOFA.

### Configuration (in `learning_transfer.py`)

| param | default | meaning |
|---|---|---|
| `SHIFT_BAND` | (5700, 11300) | peak-VSI octave: which features are transported (`None` = whole spectrum) |
| `SHIFT_ERB` | 2.5 | ERB displacement of the fine detail; features land in `target_band(SHIFT_BAND, Δ)` |
| `SHIFT_ENV_NKEEP` | 4 | Fourier coeffs kept for the envelope (`M`) |
| `SHIFT_SKIRT` | 0.0 | taper on the selection window [octaves]; 0 = hard edges, no ghosting |
| `SHIFT_EQ_RMS` | True | match per-ERB detail RMS between source and target |

`describe(SHIFT_BAND, SHIFT_ERB)` prints the selection window, where the features
land, and the shift in Hz at each edge (the same ERB step spans far more Hz at
11 kHz than at 5.7 kHz). Check that the target band stays below Nyquist —
`shift_spectral_detail` warns if it does not, since features above Nyquist really
are lost.

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
