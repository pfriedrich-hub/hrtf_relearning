# Per-ear donor selection + pool extension — 2026-09-17

A donor candidate is now a donor **EAR**, and `DONOR_POOL` gained GM.
**12 recordings x 2 ears - 1 excluded = 23 candidates per listener, was 11.**

## Why

`r_match`, `ridge_slope` and `donor_strength` were ear AVERAGES while the
monaural composite delivers exactly ONE donor ear (trained ear = own envelope +
donor's same-side detail; untrained ear = own envelope only). Half of every
ranking quantity was measured on an ear whose composite is never built, and the
two ears of one recording differ a lot (pilot/VD cue gradient: left 0.45, right
1.09). Splitting also doubles the candidate set for free — the pool is limited
by qualification blocks, not by recordings.

Acoustically legitimate: on the az=0 arc both ears see the same source, and
ILD/ITD are imposed downstream by `expand_from_midline`'s spherical-head step,
not taken from the donor. A donor's LEFT-ear detail on a listener's RIGHT ear is
another person's elevation cue, not a mirrored one.

## Changed

### `hrtf/analysis/donor_selection.py`
- `PER_EAR_SELECTION = True`. Set False to reproduce any pre-2026-09-17 ranking
  exactly — that is the only way to get the old numbers back.
- `shortlist(..., per_ear=)` enumerates both donor ears and scores each on the
  delivered pairing: `pair_metrics(..., donor_ear=side)['per_ear'][trained_ear]`.
  Requires `trained_ear`; mutually exclusive with `donor_ear`. Rows carry
  `donor_ear`.
- `detail_strength(..., ear=)` — the last ranking quantity with no per-ear path.
- `pairwise_r_match(..., per_ear=)` — per-ear yardstick; a person's own two ears
  are not counted as a pair.
- `BAND_RANK_BY = 'distance'` — NEW SWITCH. The band tier's docstring has always
  said "ranked by `donor_strength` DESCENDING" while the code sorted by
  `distance`. Default left on `'distance'` so no subject already run moves;
  `'strength'` gives the documented rule. **This now decides something:** with
  11 candidates the band usually held one row, with 23 it will not.
- `EXCLUDED_DONOR_EARS = {('GM', 'left')}` — per-ear exclusion for a recording
  with a known one-sided defect. Person-level qualification cannot express it:
  `MIN_DONOR_ELEVATION_GAIN` comes from a BINAURAL block, so "this person's cue
  works" is evidence about the pair of ears, not each one. GM's left ear lost
  17% of its cue depth to a badly seated mic.
- `DONOR_POOL` += `'GM'`.

### `protocols/learning_transfer/donor_modification.py`
`donor_ear` threaded through `__init__` / `shortlist` / `_pick` / `build` /
`prepare_shortlist` / `use_donor` / `screen_name` / `screen_settings` /
`load_existing` / `discard_unused`; stored on
`subject.active_donor["donor_ear"]`, read back by `_recorded_donor_ear()`.
`donor_detail_dtf` and `modification_params` already **had** a `donor_ear`
parameter — it had simply never been passed.

### `learning_transfer.py`
The screen keyed `measured[row["donor"]]`, which with two ears per donor made
the second screen block silently overwrite the first — now keyed
`(donor, donor_ear)`. `use_donor` / `build_donor_sofa` wrappers and a new
`DONOR_EAR` global take the ear.

### `donor_screening.py`
Ear column in the report; `chosen` matched on (donor, ear).

## File-naming compatibility rule — do not break this

`_donor_tag()`: the ear tag appears **only when `donor_ear != trained_ear`**.

- same-side build -> `<SID>_donor_FS_env4_right`, byte-identical to every
  composite, binsim database and stored `seq.hrir` already on disk
- cross-ear build -> `<SID>_donor_FS-L_env4_right`, a name that could not have
  existed before and so collides with nothing

`is_training_sofa` still resolves (the tag sits before the `_env<k>_<ear>` tail
and contains no `_n<digits>`). `group_transfer` / `elevation_learning` match
these strings exactly, which is why it matters. `active_donor["donor_ear"] =
None` likewise means "same side" and must NOT be normalised to the trained ear,
or an old record starts claiming an ear it never recorded.

`discard_unused` now spares EVERY ear variant of a named donor: over-sparing
leaves a stray file, under-sparing deletes the composite a participant is
mid-study on.

## Verified

`~/an/test_perear.py` on the desktop VM — 12 checks, all pass. `slab`/`scipy`
stubbed and `median_plane_split` monkeypatched onto synthetic splits (the
scoring path is pure numpy; slab is only SOFA I/O). Covers row count, the
exclusion, equality with `per_ear[trained_ear]`, that per-ear differs from the
ear average, per-ear `detail_strength`, `per_ear=False` backwards compatibility,
both guards, and the report rendering.

## BLOCKING before the next participant

**`TARGET_R_MATCH = 0.58` is now doubly stale** — computed on a seven-member,
person-level pool; the pool is 12 recordings and candidates are ears. The band
tier is +-0.05 around it. Run `donor_per_ear_scan.py` step 1 on the rig; it
prints both medians.

## Pool extension — GM is the only addition available

Re-derived from the subject JSONs with `baseline_run`'s own rule (first finished
non-dome own-HRTF run of >=50 trials) at |el| <= 30:

| candidate | own EG | n |
|---|---|---|
| **GM  <-- NEW** | **0.82** | 131 |
| LS | 1.30 | 131 |
| CO | 0.98 | 129 |
| pilot/AGV | 0.96 | 64 |
| pilot/AH | 0.86 | 62 |
| FS | 0.86 | 129 |
| FP | 0.84 | 130 |
| FD | 0.82 | 134 |
| NR | 0.70 | 130 |
| AS | 0.63 | 130 |
| *--- floor 0.6 ---* | | |
| TS | 0.54 | 131 |
| IR | 0.51 | 126 |
| pilot/AS | 0.51 | 45 |
| SS | 0.32 | 128 |
| PF | 0.28 | 130 |
| JF | 0.16 | 130 |

GM is the only recording that newly clears the floor. 25 recordings still have
no own-HRTF block at all. Relaxing the floor to 0.5 would add TS, IR, pilot/AS
(-> 15 recordings, 29 donor ears).

Existing members all reproduce within +0.10 of their quoted numbers, the small
positive offsets being the elevation-window convention (this table uses
|el| <= 30; `vsi_rmse.EL_LIMIT=None` takes the narrowest span present).

**Provenance narrowed from four unverifiable members to two.** `pilot/AGV.json`
and `pilot/backup/AH.json` both exist and reproduce exactly (0.96 n=64,
0.86 n=62); the earlier audit looked only for `pilot/<id>/<id>.json`. Only
**pilot/SW (0.93) and pilot/VD (0.89)** remain unverifiable.

## Two traps found on the way

1. **`results/` and `sofa/` are independent namespaces.** CO, FD, FP, TS, SS and
   JF have their SOFA at `sofa/<N>/` but their results under
   `results/pilot/<N>/`. Only AS, PF and LS genuinely exist twice as different
   recordings (the pool comment documents AS and PF; LS is undocumented). A scan
   keyed on the bare id silently merges participant AS with pilot AS and reports
   AS at 0.51 instead of 0.63.
2. `build()`'s docstring still promises it "saves the candidate ranking as a CSV
   for the supplement". It never writes one. Not fixed here.

---

# Update 2026-09-28 — run for real, TARGET_R_MATCH set

## The VM CAN run the pipeline (PyPI is reachable again)

On 2026-09-17 pip was blocked from the desktop VM and the cloud container, so
the per-ear code was only verified against synthetic splits. **On 2026-09-28
PyPI is reachable from the desktop VM again** and the whole thing was rebuilt
and run on real SOFAs. Recipe (venv OUTSIDE the mount, as in
reference_vm_python_env):

    python3 -m venv $HOME/venv
    $HOME/venv/bin/pip install numpy scipy matplotlib slab h5py h5netcdf pyfar
    mkdir -p $HOME/shim
    printf "raise ImportError('no PortAudio in this VM; DSP only')\n" \
        > $HOME/shim/sounddevice.py
    export PYTHONPATH=$HOME/shim      # MUST come before site-packages

`h5py` + `h5netcdf` are required for SOFA reading and are NOT slab dependencies.
The sounddevice shim must raise **ImportError**, not OSError — slab guards its
import with `except ImportError`, and the real package raises
`OSError('PortAudio library not found')`, which is not caught. Do not build
PortAudio. `freefield` was NOT needed for donor selection or the composite build.
Background jobs do not survive a `device_bash` call, so keep runs under ~170 s
(a full per-ear shortlist for one listener is ~25 s, ~50 s with the yardstick).

## TARGET_R_MATCH = 0.626   (was 0.58)

Measured with `pairwise_r_match` on the 12-recording pool:

| | n pairs | min | Q1 | median | Q3 | max |
|---|---|---|---|---|---|---|
| per-person | 66 | 0.103 | 0.490 | **0.601** | 0.702 | 0.820 |
| per-EAR | 264 | **-0.189** | 0.487 | **0.626** | 0.719 | 0.861 |

0.58 was 0.046 low — just inside one tolerance band, so the band it defined
[0.53, 0.63] overlapped the right one [0.576, 0.676] rather than missing it.

**The per-ear minimum is NEGATIVE** (-0.189) where the person-level minimum was
+0.103: some donor-EAR pairings are anti-correlated with the listener's own
detail at the same elevation. Ear-averaging had been hiding the most strongly
perturbing pairings in the pool. They are not selected (the target is
mid-range), but the distribution now has a real tail.

## Verified on real SOFAs

**Selection**, 2 listeners:

- **GM**, trained right, 11 recordings -> 22 donor ears: tier **BAND**, 7 band
  members, 20/22 ridge-eligible. Per-ear rank 0 = pilot/VD(right); the old
  person-level rank 0 was FS.
- **PF**, trained right, 12 recordings -> 23 donor ears: tier **WIDENED**,
  0 band members, 13/23 ridge-eligible. Still the r_match outlier he always was
  (best available 0.47 against a 0.626 target) — not a target problem. But
  per-ear helps him: rank 0 moves from pilot/VD at r_match 0.30 to NR(left) at
  0.47, much closer to target.

**Naming + build path**, GM x FD, writing nothing:

- `GM_donor_FD_env4_right` for same-side, byte-identical to the old spelling
- `donor_ear='right'` (== trained ear) yields the IDENTICAL name
- `GM_donor_FD-L_env4_right` for cross-ear; `is_training_sofa` accepts both;
  the binaural QC name `GM_donor_FD-L` is still not mistaken for a training sofa;
  `binsim_names` follows to `GM_donor_FD-L_env4_right_mirrored`
- midline QC **PASS** for both donor ears, ILD broadband 0.000, ITD 0.000 us,
  processed-ear elevation SD 3.398 -> 0.812 dB
- the two donor ears really do produce different composites (max |diff| 0.34)

## Bug found and fixed in `report()`

The per-ear header took `rows[0]['cue_ear']` and announced it for the whole
table — so a right-trained listener whose rank 0 was a LEFT donor ear got
"cue columns measured on the left ear" printed over a right-ear protocol. Now
it says the columns are measured on each row's own donor ear.

## OPEN DECISION: BAND_RANK_BY should probably become 'strength'

GM's band tier, sorted by distance to target:

| donor | ear | distance | strength |
|---|---|---|---|
| pilot/VD | right | 0.0221 | 3.5 |
| FD | right | 0.0224 | 4.0 |
| FP | left | 0.0323 | 3.1 |
| pilot/SW | right | 0.0341 | 2.6 |
| NR | right | 0.0365 | 2.0 |
| FS | right | 0.0400 | 2.7 |
| FS | left | 0.0440 | 2.6 |

**pilot/VD wins rank 0 over FD by 0.0003 of r_match** while FD carries 0.5 dB
more cue depth and a much better gradient (1.24 vs 1.10). That is selecting on
noise, which is the exact failure the tolerance band was introduced to prevent —
the module's own comment says so ("min |dissimilarity - target| is decided by
differences of 0.01-0.02, which is well inside the measurement noise ... the
band says close enough to target and then picks on a quantity that is actually
meaningful"). With 11 candidates the band held one row and it never mattered;
with 22 it decides. `band_rank_by='strength'` makes rank 0 FD(right).

Not flipped — it moves the rank-0 pick for every subject and is Paul's call.
