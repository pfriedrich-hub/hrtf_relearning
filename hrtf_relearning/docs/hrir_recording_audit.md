# HRIR recording pipeline — audit

Scope: `hrtf_relearning/hrtf/record/` (dome / freefield pipeline) and its entry
point `experiment/protocols/HRIR_Recording.py`. `hrtf/record_mesm/` is out of
scope for this pass.

Nothing has been deleted or changed. This is the state of the pipeline as of
commit `14a1af9` plus the findings, ranked.

> **Status, 2026-08-19/20.** Findings A, B and D are closed in the code.
> A: `record_dome`'s parameter is now `equalize_dome` (was `equalize`, which
> silently defeated the first fix attempt and mis-recorded `ref_19.08`), the
> default is `False`, and `record_reference()` — step 0b in
> `HRIR_Recording.py` — records the reference with the same flag as the
> subject, so the mismatch is closed by construction from `ref_20.08` on.
> Data already on disk keeps the old reference and its residual **by decision
> (Paul, 2026-08-19)**: subjects localized well, a rebuild is not warranted.
> Do not re-raise A as blocking, and do not try to settle an EQ question
> acoustically across sessions — read `params.txt` and trust the code path.
> B: fixed (`overwrite=True` passed explicitly). D: fixed, along with a third
> name-resolution bug of the same class (`center_key` never matched for
> subjects — magnitude effect 0.016 dB rms, no rebuild owed).
> The rest of the findings below stand as written.

---

## 1. What the pipeline actually does

```
HRIR_Recording.py  (# %% cells, first session)
  │
  ├─ step 1 ── record_hrir()                        hrtf/record/record_hrir.py
  │              │
  │              ├─ Recordings.record_dome(subject) recordings.py
  │              │     for each of n_directions head positions:
  │              │       LED cue → wait for Enter → sweep every frontal speaker
  │              │       (az ∈ [-1,1], el ∈ [-37.5, 37.5]), n_recordings each,
  │              │       highpass at hp_freq, store as {idx_az_el: [Binaural]}
  │              │     → rec/<id>/recordings.npz + params.txt
  │              │
  │              ├─ Recordings.record_dome(reference)   (once, n_directions=1)
  │              │     → rec/reference/<ref_id>/
  │              │
  │              ├─ compute_ir()  ×2                 processing.py
  │              │     align repeats (±10 samples, xcorr) → average
  │              │     → regularised inversion of the chirp → deconvolve
  │              │     → time_align_irs: global shift so speaker 23 onset = 1 ms
  │              │       (optionally zero ITD+ILD on frontal directions)
  │              │
  │              ├─ equalize()                       divide subject IR by the
  │              │     same-speaker reference IR, onset-align, Hann window
  │              │     (2.5 ms total), crop to n_samples_out
  │              │
  │              ├─ lowfreq_extrapolate()            below 800 Hz, replace
  │              │     magnitude with spherical-head anchors, keep phase
  │              │
  │              ├─ expand_azimuths_with_binaural_cues()   copy the vertical arc
  │              │     across az ∈ [-50,50], apply spherical-head magnitude
  │              │     *relative to frontal at the same elevation*, then match
  │              │     spherical-head ITD by shifting the right ear
  │              │
  │              └─ to_slab_hrtf() → sofa/<id>/<id>.sofa  + midline TF plot
  │
  ├─ step 2 ── calibrate_headphones()   record/calibration/calibrate_headphones.py
  │              n_rec re-seatings of the headphone, average, regularised
  │              minimum-phase inverse → rec/<id>/<hp>_equalization.npz
  │              + repeatability QC figure + before/after result figure
  │
  ├─ step 3 ── acoustic_test()   (optional, commented out in the protocol)
  ├─ step 4 ── LocalizationDome  (real speakers)
  └─ step 5 ── Localization (AR)  via pybinsim
```

Every step in that chain is reached and does something distinct. **No step is
redundant.** The problems are elsewhere: two of them silently change what the
data mean, and a third throws away the record of what was done.

---

## 2. Findings

### A. Blocking — subject and reference are recorded under different speaker equalisation

Confirmed across every recording on disk:

| | `equalize_dome` |
|---|---|
| all 16 subjects (AH … TS, Jun–Aug 2026) | `False` |
| `ref_03.04`, `kemar_reference` | `True` |

`record_dome` passes this straight to `freefield.play_and_record(equalize=...)`,
so the reference sweeps went out through the dome EQ filter and the subject
sweeps did not. `equalize()` then divides subject by reference per speaker. If
`E(f)` is the dome EQ filter, the division leaves `HRTF(f) / E(f)` rather than
`HRTF(f)` — i.e. the raw speaker response is imprinted on every HRIR instead of
being cancelled, and because each elevation is a different speaker, the residual
is **elevation-dependent**. That is exactly the axis the project measures.

How this happened: `record_hrir()` declares `equalize_dome: bool = False` while
the module-level config block above it says `equalize_dome = True`. The protocol
never passes the argument, so the default wins and the config line is decorative.
The reference predates that default.

**Before anything else, verify** what `equalize=True` does in the freefield fork
when a dome calibration is loaded, then either re-record a reference with
`equalize=False` (cheap — one sweep set) or confirm the residual is negligible.
Everything downstream — VSI/RMSE, notch tracking, cue edits — sits on top of this.

### B. Blocking — `overwrite` in `record_hrir` resolves to a module global, not the argument

`record_hrir.py:108` and `:127`:

```python
subject_rec.to_npz(subj_dir, overwrite=overwrite)   # ← module global, always False
```

`overwrite` is not a parameter of `record_hrir` (params are `overwrite_rec` and
`overwrite_hrir`). The protocol calls it with `overwrite_rec=True`. Result: when
`rec/<id>/recordings.npz` already exists, the subject is re-recorded, the fresh
sweeps are used in memory to build the SOFA, and `to_npz` silently skips the
write — so the SOFA and the raw data on disk come from **different sessions**,
with no warning. `params.txt` is rewritten unconditionally, so even the timestamp
does not flag it.

Worth checking whether any subject was ever re-recorded; if so, their `.npz` is
not the source of their `.sofa`.

### C. Serious — processing provenance is dropped twice

- `equalize()` (processing.py:384) builds a fresh `params` dict containing only
  `fs`, `signal`, `equalize`. Everything accumulated upstream — `id`,
  `n_recordings`, `highpass_frequency`, `equalize_dome`, the recording
  `datetime`, the `compute_ir` block, the `time_alignment` block — is discarded.
- `to_slab_hrtf()` ignores `self.params` entirely, so the written SOFA carries no
  processing metadata at all.

`rec/<id>/params.txt` preserves the *recording* parameters, so nothing is lost
forever, but a SOFA cannot be traced to the settings that produced it. This is
the same gap that `GLOBAL_ModificationParams` closed for the cue-modification
side; the recording side never got it.

### D. Serious — `key` is both a flag and a loop variable in `record_dome`

`recordings.py:102` takes `key=True` (whether to prompt for Enter);
`recordings.py:142` reassigns `key` to the dict key string inside the speaker
loop. After the first speaker, `key` is a truthy string forever.

Currently harmless — the reference is recorded with `key=False` *and*
`n_directions=1`, so the prompt is only evaluated before the reassignment. Any
reference with `n_directions > 1` would start prompting from direction 2. Rename
the parameter (`wait_for_key`) and the loop variable.

### E. Moderate — hardcoded / inconsistent magic values

| Where | Value | Issue |
|---|---|---|
| `compute_ir` | `center_key='23_0.0_0.0'` | Speaker 23 hardcoded as the time-alignment anchor. Falls back to `keys[0]` with a warning if absent — that changes the global time reference silently. |
| `record_hrir` | `(hp_freq, 20e3)` for `compute_ir`, `(hp_freq, 18e3)` for `equalize` | Two different inversion ranges, undocumented. |
| `equalize` | fade 0.25 ms in / 1.5 ms plateau / 1.0 ms out | Was 4.8/5.8 ms (still there commented out at processing.py:359). A 2.5 ms window is short — worth a line saying which reflection it excludes. |
| `record_dome` | `res = abs(speakers[0].elevation - speakers[1].elevation)/n_directions` | Assumes speaker-table order and uniform elevation spacing. |
| `calibrate_headphones` | `beta=0.01` at the call site | Module constant `BETA = 0.1` is unused; `N_OUT = 256` is unused (`n_samp_out` defaults to 1024). |

### F. Moderate — module-level config blocks shadowing function defaults

`record_hrir.py` (lines 15–30) and `calibrate_headphones.py` (lines 61–79)
define script-style config at import time. Some of it is genuinely dead
(`subject_id = 'kemar_pir'`, `n_directions = 1`, `show`, `align_interaural`);
some leaks into signatures (`head_radius: float = head_radius`,
`subject_id=SUB_ID`, `hp_id=HP_ID`); one contradicts the real default
(`equalize_dome`, finding A). A reader cannot tell which values are live. These
belong in the protocol cell, not the library module.

### G. Moderate — azimuth convention is mixed within one HRTF

Measured keys keep the dome's signed azimuth (`..._0.00_...`), while
`expand_azimuths_with_binaural_cues` wraps expanded azimuths into `[0, 360)`
(`-50°` → `310.0`). Both go into `get_sources()` → `slab.HRTF` unchanged, so one
SOFA mixes the two conventions. Probably fine as long as everything downstream
wraps consistently, but it is worth a line in the docstring and a check against
what `binsim` / `Localization_AR` assume.

### H. Cleanup — dead code (nothing deleted, listed for approval)

| Item | Lines | Notes |
|---|---|---|
| `processing.py:987–1203` | ~217 | Commented-out previous version of `expand_azimuths_with_binaural_cues` (absolute-ILD variant). Superseded by the relative-to-frontal version. Recoverable from git. |
| `record/test_hrir_recording.py` | 165 | Calls `record_hrir(..., overwrite=False)` — **TypeError**, that parameter does not exist. Its `acoustic_test` is duplicated near-verbatim in `HRIR_Recording.py`. `behavioral_test` is a stub of TODOs. Not a test (no assertions, name collides with pytest discovery). |
| `record/deprecated/record_hrir_old.py` | 553 | Superseded; only reference to `to_wav`/`from_wav`. |
| `calibrate_dome_pyfar.py` | 384 | `main()` raises `NameError: speaker_idx` at line 332; `OUTPUT_FILE = 'C:/projects/music_space/calibration.pkl'` points outside the project. Duplicates `align_recordings` from `processing.py` in monaural form. |
| `wav_to_npz`, `Recordings.to_wav/from_wav`, `pyfar2wav` | ~90 | Wav path referenced only by `deprecated/`. `Recordings.load()` still falls back to `from_wav` — check no subject folder relies on it before removing (none currently do; all 18 have `.npz`). |
| `wait_for_button` + `pynput` import | record_hrir.py:206–216 | Unused; the import runs on every `record_hrir` import. |
| `ImpulseResponses.waterfall`, `.time_freq`, `Recordings.plot` | ~230 | Never called anywhere in the repo. Keep if used interactively — but say so in the docstring. |
| Unused imports | — | `matplotlib` in `record_hrir.py`, `recordings.py`, `calibrate_headphones.py`, `calibrate_dome_pyfar.py`, `test_hrir_recording.py`; `hrtf_relearning` in `record_hrir.py`; duplicate `import logging` at processing.py:23. |
| `record/calibration/*.pkl` | 5 files | `calibration_dome_19.02`, `_frontal`, `_full`, `_auditory_attention`, `data/calibration_dome.pkl`. None is read by any code path — the dome EQ is loaded from `freefield.DIR/data/`. |
| `record/calibration/*_equalization.wav` | 2 files | `DT770`, `MYSPHERE` — legacy format, no DT770 anywhere else in the codebase. |

### I. Cleanup — structural

- `processing.py` is 1210 lines holding three unrelated things: the
  `ImpulseResponses` container, the DSP chain, and ~230 lines of plotting.
  Splitting plotting out (or moving it next to the other QC helpers) would make
  the DSP readable.
- `calibrate_headphones.py` imports `_agg_figure` from `hrtf.modify.edge_shift`
  (twice, at call time). `modify/` is meant to hold only experimental cue
  manipulations — a save-only figure helper belongs in `utils/`.
- Docstring drift: `ImpulseResponses` says "Values are `slab.Filter` (FIR)" and
  `to_slab_hrtf` repeats it; the values are `pyfar.Signal`.
- `record_hrir.py`'s docstring is two bullets. The seven-step docstring inside
  `record_hrir()` is good and should be what the module says.
- `HRIR_Recording.py` lists five steps but defines step 3 as a helper function
  in a cell of its own and leaves it commented out; the cells run 1, 2, 4, 5b.
  The leading `' TODO check led, make run'` string literal is a stray.
- Five subjects have raw recordings but no SOFA: `AH`, `CA`, `JS`, `LS`, `SZ`.
  Intentional (dropouts?) or unprocessed — worth recording either way.

---

## 3. Proposed order of work

1. **Resolve finding A.** Verify the freefield `equalize` semantics, decide
   whether a reference re-recording is needed. Blocks any reprocessing.
2. **Fix B** (one-line: use `overwrite_rec`), then check whether any subject's
   `.npz` and `.sofa` disagree.
3. **Fix C**: thread `params` through `equalize`, add a `lowfreq_extrapolate`
   block, and write the dict into the SOFA as a global attribute (mirroring
   `GLOBAL_ModificationParams`).
4. **Fix D and F**: rename `key`, delete the dead config globals, move live ones
   into explicit defaults.
5. **Document**: module docstring for `record/`, a short `docs/hrir_recording.md`
   describing the chain and the parameter choices in E.
6. **Delete H** in one commit, after approval.
7. **Split processing.py** (I) last — the largest diff, the least risk.

Steps 1–3 change results and should be followed by reprocessing all subjects
from `rec/` and diffing the resulting SOFAs against the current ones.
