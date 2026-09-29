# Day 1 — why each cell does what it does

Moved out of `protocols/learning_transfer/learning_transfer.py` 2026-09-28 so
the cells are runnable without reading. Thresholds and mechanics live in
`donor_screening.py` and `donor_selection.py` docstrings; this is the protocol
reasoning only.

## Native reference
The first anchor. This is the best the chain can sound, so its externalization
rating defines the top of the 0–10 scale for everything after it. Also the
own-HRTF block every screen is measured against, so it must share geometry and
stimulus with them.

## Staging — before the session
`prepare_donor_shortlist(n, screen=True)` builds the top-n composites, their
pyBinSim databases (mirrored and un-mirrored) and the binaural database each
screen block plays. Run it with nobody in the rig: `write_filters` passes over
all 475 directions, so it is minutes per donor. Rank 0 is left active.

n=3 is the minimum. n=5–6 costs only wall-clock and means `screen_more()` never
has to build filters with the participant waiting. Unused builds cost disk
only; `discard_unused()` clears them.

## Screening
One 35-trial binaural full-field block per candidate, ~3 min each, scored
against the native block. It MEASURES — it does not gate and does not choose
(simplified 2026-09-28; see `donor_screening.measure` for why). Read elevation
gain and polar error off the table and pick.

The screen is itself donor exposure. Take the day-1 impairment for the chosen
donor from ITS screen block, and report baseline A as post-screen.

Retro-test on every screen run before the gates were removed, with elevation
gain leading — kept because it shows what the numbers look like in practice:

| subject | donor | reading |
|---|---|---|
| NR | pilot/AH | EG 0.00 — cue abolished, not degraded |
| NR | pilot/SW | EG 0.25, 39% of her own gain kept |
| FP | FS | EG 0.00 |
| FP | pilot/AH | EG 0.30, 37% kept |
| LS | pilot/AH | EG 0.09, 7% kept — she ran the whole study on it |
| AS | GS | EG 0.41 looks fine absolutely, but that is 65% of her own gain and impairment was only +2.5°: the manipulation never bit |
| SH | GM right | EG 0.41, 60% kept, impairment +5.0° |

AS/GS is the case an absolute EG threshold cannot catch and the RATIO can.

### Screening more
`screen_more(subject, native, n=2)` measures the next unscreened candidates and
MERGES them into the record — earlier blocks are kept, keyed on (donor, ear).
Each extra block is 35 more trials of exposure before the naive baselines. Two
or three is cheap; past that the "naive" baseline is drifting and that belongs
on the record.

## Choosing
`use_donor(donor_id=..., donor_ear=..., reason=...)` is the whole step. The
composite already exists from staging — `use_donor` repoints, it does not
build, and it raises if the SOFA is missing.

The ear is REQUIRED: a donor id alone matches two shortlist rows since per-ear
selection, and `use_donor` refuses to guess.

`reason` is the only record of why this participant got this composite. Write
what you saw.

**Do not call `build_donor_sofa()` after `use_donor()`.** It defaults to
`rank=0` and would re-point `active_donor` back to the rank-0 donor, silently
undoing the choice. It is only needed when the donor was never staged — and
then it needs the same `donor_id=`/`donor_ear=` you chose.

`require_screen` refuses to commit a donor that does not appear in the screen
record. `override_reason=` proceeds anyway and lands in the subject file.

## `load_existing_donor()`
Not required. The config cell already resolved the donor from the subject file.
It re-reads the SOFA's embedded params and prints them, so you can confirm what
is on disk is what was built before running a block on it. Useful on later
sessions, redundant on day 1.

## Day-1 baselines A/B/C/D
All four since 2026-09-28. Transfer is read from within-cell pre-post change
scores, and the cells start from different naive levels: even the mirror pair
A/D differed by up to ±0.25 EG on day 1 in the pilots, about the size of the
change-score noise band. Final-day B and C ranged 0.05–0.73 and 0.07–0.62 EG,
so without a day-1 value they cannot be read.

What the four cells distinguish (+ = improves day 1 → final day):

| where learning lives | A | B | C | D |
|---|---|---|---|---|
| trained ear's monaural pathway | + | + | − | − |
| binaural stage organised by side | + | − | + | − |
| shared stage (ear- and side-general) | + | + | + | + |
| only trained ear × side | + | − | − | − |

Without C, the side-specific and combination-only accounts both predict
(+, −, −) and cannot be told apart. B and C are far-ear cells — the donor
detail reaches only the head-shadowed far ear — which is why they need their
own baselines rather than being dropped.

Run in the subject's FINAL_ORDER (Williams square from the block-order CSV) so
each cell holds the same position on day 1 and the final day.

Subjects run before this change (FS, IR, AS, LS, GM, NR, FP) have day-1 A and D
only; their B and C are post-only and drop out of the B/C change scores.

Externalization ratings: native anchor plus baselines A and D
(`RATED_BASELINES`). A is the trained-ear condition and D the main transfer
cell; B and C are far-ear variants of the same two composites, so rating all
four measures boredom rather than externalization.
