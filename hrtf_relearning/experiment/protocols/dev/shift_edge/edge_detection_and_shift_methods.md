# Detection and Manipulation of Pinna Spectral Notch Edges via Cepstral Smoothing

Scope: the notch/saddle detection and edge-shift procedure used to generate the rising-edge, falling-edge, and whole-notch manipulation conditions (rows A–C of the manipulation matrix in `elevation_spectral_cue_models.md`), as implemented in `edge_shift.py`.

> **Note (current implementation).** §2–3 below document the original cepstral (`n_keep`) identity + raw-derivative edge-extent method. The shipping detector now uses a light **Iida et al. (2007) Gaussian** (σ≈122 Hz) for notch *identity* — it de-ripples without displacing the extrema, so notch centre frequencies read true and closely-spaced pinna notches stay resolved — and reads edge *extent* off the derivative of that **same smoothed** curve rather than the raw spectrum (the raw derivative is dominated by single hyper-narrow ripples). The cepstral path is retained only as an explicit legacy fallback (`detect_notches(n_keep=…)`). The module docstring in `edge_shift.py` is authoritative; §6 below covers the elevation-continuity tracking layer added on top. See also `cue_perception_synthesis.md`.

---

## 1. Rationale

Individual DTFs contain fine spectral ripple superimposed on the broader notch/peak structure created by pinna reflections. Two related problems follow from this:

1. **Notch identity is not obvious in raw resolution.** Two closely-spaced fine ripples can be the auditory system's single, unresolved notch rather than two independent features (a specific case: a subject's DTF where two adjacent local minima ~1 kHz apart are separated by a shallow secondary peak; treating these as two independently manipulable notches vs. one notch with a superimposed ripple materially changes what a rising/falling-edge manipulation means). A detector must therefore decide notch *count and identity* from something coarser than the raw magnitude spectrum.
2. **A rising or falling edge is not well described by a single boundary point.** An edge manipulation needs a *span* — an onset and an offset in frequency — not just a center frequency, and that span must leave enough untouched spectrum on either side that shifting one notch's edge does not silently distort a neighboring notch.

Both problems are solved with one tool, applied at two different resolutions: truncated cosine-series ("cepstral") reconstruction of the log-magnitude spectrum (Kulkarni & Colburn, 1998), i.e.

```
log|H(f)| ≈ Σ_{k=0}^{n_keep-1} a_k cos(2π k f / N)
```

with only the first `n_keep` coefficients retained. This is the same smoothing operator already used for the envelope/detail split in the cue-shift ("shift_band") manipulation; here it is reused for detection rather than manipulation.

An alternative, time-domain method — locating notches via the group delay of the LP-residual autocorrelation of the pinna-isolated HRIR (Raykar, Duraiswami & Yegnanarayana, 2005) — was implemented and validated first, and gives equivalent notch-identity/merging behavior. It was set aside in favor of the cepstral approach below because the latter achieves the same correctness with substantially less machinery, and operates in the same representation (the log-magnitude spectrum) already used for the manipulation itself.

---

## 2. Notch and saddle detection

For a single HRIR, the one-sided magnitude spectrum |H(f)| is obtained by FFT and converted to log-magnitude, `L(f) = 20·log10|H(f)|`. `L(f)` is smoothed with `n_keep = 60` (the *coarse* curve, `L_c`). This value was chosen empirically: it is high enough that individually distinct notches survive as separate local minima, and low enough that fine ripple that is genuinely part of one auditory feature is absorbed into a single trough rather than resolved into separate ones. (At substantially higher `n_keep`, e.g. 150, previously-merged notches can re-split into two, reproducing the identity ambiguity described in §1; at substantially lower `n_keep`, e.g. ≤ 30, shallow individual notches can be smoothed away entirely.)

Notch minima and their bracketing saddle peaks are then identified as prominence-thresholded local extrema of `L_c`:

```
notches = find_peaks(-L_c, prominence = 3 dB)
saddles = find_peaks( L_c, prominence = 3 dB)
```

For an outermost notch with no saddle on one side within the analysis band (3–17.5 kHz here), that side's bound is the edge of the band rather than a detected saddle.

---

## 3. Edge definition and extent

No separate "edge point" is detected. Instead, for a given notch, the **rising edge** is a frequency span lying between that notch's minimum and its nearest saddle above it; the **falling edge** lies between the nearest saddle below it and the minimum. The two bounding features (minimum, saddle) are shared with the neighboring notch on that side — this is structural, since exactly one saddle separates two consecutive minima — so the edge's extent, not just its existence, has to be determined without assuming the neighbor's territory is free to use.

The extent is found from the derivative of the **unsmoothed** (raw) log-magnitude spectrum, `dL/df`, evaluated only within the window bounded by the minimum and its nearest saddle:

1. Take the maximum of `dL/df` within that window (the window's own steepest point).
2. Set a local threshold `ε = 0.15 × that maximum` — 15% of this specific edge's own peak steepness, not a fixed or global value.
3. Mark every sample in the window where `dL/df ≥ ε` (rising) or `−dL/df ≥ ε` (falling).
4. The edge span is the frequency range between the outermost (nearest- and farthest-from-minimum) marked samples — not the first contiguous run from the peak.

Step 4 is the operative difference from a conventional relative-height peak-width measure (e.g. `scipy.signal.peak_widths`): the latter stops at the first point the signal drops below threshold when walking outward from the peak, so a secondary ripple or shoulder on the same flank truncates the detected span early. Taking the outermost threshold-crossing across the whole window instead bridges such ripples, since points do not need to be contiguous to count — only extremal.

Because a notch's flank typically becomes shallow well before reaching the bracketing saddle (a broad, low-slope shoulder near the peak, rather than a steep approach all the way to it), the detected edge span, in practice, terminates before the shared saddle, leaving an untouched plateau between one notch's edge and the next notch's. This plateau is what a shift manipulation (§4) uses as headroom, without needing any special-case handling of shared boundaries.

Detection parameters and results were checked visually and quantitatively (plateau width between adjacent edges, and edge coverage against the visible slope in the raw DTF) across two individually-measured subjects at three median-plane elevations, and cross-checked for generality across five HRTF datasets (two additional individual subjects, KEMAR and FABIAN dummy heads), both ears, three elevations each (88 notches total): every edge was successfully detected (no failures), with a median inter-notch plateau of ≈500 Hz — comfortably larger than a typical 1 ERB shift target (≈900–1500 Hz in absolute frequency at these center frequencies is the shift itself, not the required headroom, which is smaller) — and a minority of cases (closely spaced notches, particularly at frontal, 0° elevation, or ears with more than four notches in-band) with under 100 Hz of headroom, which the shift procedure reports and clamps rather than silently exceeding.

---

## 4. Edge-shift manipulation

Given a notch's minimum, edge span, and available headroom, a target shift Δ (in ERB) is applied as follows:

- The notch **minimum is pinned** (unmodified in frequency) — it is the notch's identity/characteristic-frequency anchor, and stays fixed under both the rising- and falling-edge conditions, which is what dissociates them from the whole-notch-shift condition (below).
- The edge span (onset → offset, §3) is **rigidly translated** by Δ: onset and offset are both shifted by the identical amount, so the manipulated flank is an exact, undistorted copy of the measured one, only moved — not a stretched or re-warped version of it.
- The gap this opens between the pinned minimum and the translated span's new onset is filled with a **flat hold at the minimum's own log-magnitude value** (a zero-order hold), rather than interpolated — i.e. the notch floor widens by Δ, the slope is not shallowed.
- Any compensating stretch needed so the manipulated curve still lands correctly on the next truly fixed feature (a neighboring notch's own pinned minimum, or the edge of the analysis band) is confined to the plateau beyond the translated edge span — in practice this rarely encroaches on the neighboring notch's own edge, for the reason given at the end of §3.
- `mode = 'whole'` is a distinct manipulation: the notch **minimum** itself is translated by Δ with **both** bracketing saddles pinned, reshaping both flanks around a recentered notch. This is not an edge-only manipulation and is used as an all-models-agree reference condition (row A), not to dissociate rising- from falling-edge accounts.

Additional invariants enforced after the warp: notch **depth is preserved** (verified at the shifted, not the original, saddle location, since the saddle itself has moved for the shifted flank); in-band **RMS power is matched** pre/post to remove the level/power confound described under manipulation J in `elevation_spectral_cue_models.md` (cf. Zonooz et al., 2019); the original **phase spectrum is unchanged** (ITD/onset structure is untouched by an otherwise magnitude-only manipulation).

---

## 5. Parameters

| Parameter | Value | Role |
|---|---|---|
| Analysis band | 3–17.5 kHz | frequency range searched for notches/saddles |
| `n_keep` (coarse smoothing) | 60 | cosine-series coefficients retained for notch/saddle **identity** |
| Prominence | 3 dB | minimum depth/height for a local extremum of the coarse curve to count as a notch or saddle |
| Smoothing for edge extent | none (raw spectrum) | derivative computed directly on the unsmoothed log-magnitude |
| `eps_frac` | 0.15 | fraction of a given edge window's own peak `|dL/df|` used as the extent threshold |

---

## 6. Elevation-continuity tracking (robust notch identity across a cone)

Detection and gating are computed independently per direction, which flickers: a real notch whose depth momentarily dips below the gate is shifted at some elevations but not adjacent ones, and a "deepest = primary" label swaps between unrelated notches when two are close in depth. Measured across 18 subject-ears (real subjects + KEMAR/FABIAN/KU100 dummies), the per-direction gate produces **21 interior gate-dropouts** (a notch shifted, then not, then shifted again along its own trajectory) and worst-case **primary-CF jumps of ~1 octave** between directions only 4° apart. Median-plane notches, however, are smooth continuous trajectories in elevation (visible as unbroken troughs in the DTF magnitude image), so these are labelling/gating artifacts, not real cue movement.

The fix (`stabilized_valid_cfs`, opt-in via `edge_shift_set`/`manipulate_hrtf` `use_tracking=True`):

1. **Group** directions into constant-azimuth elevation arcs (cones of confusion); a midline-only SOFA is a single arc.
2. **Link** each direction's detected minima into tracks by nearest centre frequency in log-frequency (`tol_oct` = 0.22; a track may bridge up to `max_gap` = 3 directions where the minimum was momentarily undetected). Greedy nearest-CF association is adequate because tracks are well separated (>~0.5 oct) and move slowly (typically <~0.06 oct/direction, with occasional steeper local segments up to ~0.19 oct). `tol_oct` and `max_gap` are the knobs against **over-segmentation** (one physical trough split into several tracks): they were raised from an initial 0.14/1 after the database overview showed single notches broken where the CF steps steeply between two elevations (>0.14 oct → `tol_oct`) or the notch fades below detection for a couple of directions (→ `max_gap`). `tol_oct` stays well below the ~0.5 oct inter-notch spacing, so distinct notches are not merged (no over-merge across the measured database). Splits that remain where a notch genuinely fades for more than `max_gap` directions are correct and are deliberately not bridged. `sigma_hz` is **not** the lever here — the splits are a linking issue, not detection resolving one trough into two minima.
3. **Decide validity once per track:** a track is a real elevation cue if it passes the per-direction depth/width gate at ≥ `min_valid_frac` (default 0.5) of the directions where a minimum was actually detected, and spans ≥ `min_len` directions. That single decision is applied to **every** detected member of the track.

Consequences: a notch that momentarily dips below the depth gate is still shifted at that elevation (the manipulation becomes elevation-consistent), a track that is only briefly deep or too short is excluded everywhere (transient/noise minima rejected), and tracks are labelled N1, N2, … by ascending mean CF — stable across the whole arc, replacing the per-direction depth rank. On the same 18 subject-ears this reduces interior gate-dropouts from **21 to 0**. `detect_notches` itself is unchanged; when `use_tracking=False` (default) the per-direction `select_features` path is byte-identical to before.

Validated on real DTFs (h5py-read SOFA, note the raw `Data.IR` axis order is `(dir, ear, tap)` whereas slab's `hrtf_to_array` yields the module's `(dir, tap, ear)`): 21→0 interior dropouts; `valid_cfs=None` reproduces the legacy `select_features` output exactly across all four modes and both ears.

---

## References

- Kulkarni A, Colburn HS (1998). Role of spectral detail in sound-source localization. *Nature* 396:747–749.
- Raykar VC, Duraiswami R, Yegnanarayana B (2005). Extracting the frequencies of the pinna spectral notches in measured head related impulse responses. *JASA* 118(1):364–374.
- Zonooz B, Arani E, Körding KP, Aalbers PATR, Celikel T, Van Opstal AJ (2019). Spectral weighting underlies perceived sound elevation. *Sci Rep* 9:1642.

See also `elevation_spectral_cue_models.md` for the modeling framework (rows A–C) this manipulation is designed to dissociate.
