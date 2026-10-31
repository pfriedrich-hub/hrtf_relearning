# Verification precursor: does the freefield→VR failure still exist?

Design note for `expectation_transfer_verification.py`. Read alongside
`expectation_transfer_design.md` — this is the go/no-go you run *before*
committing subjects to that mechanism study.

## 1. Why this comes first

The whole motivation for the expectation-transfer work is the observation that
naive listeners fail to externalize/localize individual-HRIR audio in the VR
room, but localize near real-ear level once they've been in front of the real
speakers. That observation predates the current HRIR pipeline. The HP
equalization now works much better, so the old failure might have been the old
signal chain failing to faithfully reproduce spectral cues — not a psychological
context effect at all. If so, immediate freefield→VR transfer now just works,
the VR baseline can be taken directly, and there is no mechanism to study.

You cannot tell these apart from the historical data because room and signal-era
are confounded there. You *can* tell them apart by holding the new rendered
signal constant and varying only room/presence/priming. That is this script.

## 2. The three blocks and the two comparisons

| Block | What | Where | State |
|---|---|---|---|
| 1 `AR_VR_immediate` | virtual AR | VR room | naive, unprimed, no speakers — record → straight here, nothing between |
| 2 `dome_ref` | real speakers | freefield | real-ear reference / rendering ceiling |
| 3 `AR_freefield_primed` | virtual AR | freefield | best case: in-room presence + freshly primed by Block 2 |

Two orthogonal reads, both with the new signal held constant:

- **Fidelity = Block 3 vs Block 2.** Best-case AR against real-ear. If primed,
  in-room AR still can't approach the dome, the rendering itself is the limit.
- **Context = Block 1 vs Block 3.** Same rendered signal, differing only in
  room/presence/priming. A gap here is the contextual effect — the thing the
  mechanism study is about.

Headline: **Block 1 vs Block 2** — is immediate transfer already good enough to
just take the baseline?

## 3. Order is one-way

Block 1 must be the participant's first localization of the day, in the VR room,
with nothing between it and the HRIR recording. Any localization run first —
even a warm-up — primes or task-familiarizes them and destroys the "immediate"
reading (the second-time-at-task confound from `expectation_transfer_design.md`
§2). You cannot un-prime, so Blocks 2 and 3 necessarily come after and are
"primed" by construction; that is intended. If a participant is accidentally
primed before Block 1, that subject's Block 1 is spent — reschedule rather than
run a contaminated one.

This is a within-subject go/no-go, not a controlled effect-size estimate: Block
1 is also the first time at task, so the Block-1↔Block-3 gap is confounded with
practice. That is acceptable here because the question is only *which regime
we're in* (A/B/C below), and practice can only *shrink* a real context gap —
so a gap that survives is conservative. The clean effect size is what the
matched-control `expectation_transfer.py` design exists to measure.

## 4. Pass/fail criteria (placeholders — tune after first pilots)

Per subject, relative to their own dome (real-ear) block:

- **elevation gain** ≥ `ELE_GAIN_ADEQUATE_FRAC` × dome gain (default 0.70) —
  primary, this is the cue the work is about;
- **elevation RMSE** ≤ dome RMSE + `ELE_RMSE_MARGIN`° (default 7.5);
- **externalization** ≥ `EXT_ADEQUATE` on the 0–10 post-block rating
  (default 6), applied to the AR blocks.

All computed from the existing `localization_accuracy(sequence)` (returns
`elevation_gain, ele_rmse, …`). Set the thresholds from the first few dome
references before trusting the verdict — the defaults are guesses.

## 5. The three verdicts and where each routes

- **A — immediate transfer adequate** (Block 1 meets criteria vs dome). The new
  pipeline fixed it. Take the VR baseline directly; the mechanism study is
  probably unnecessary. Cheapest outcome.
- **B — immediate poor, primed good** (Block 1 fails, Block 3 meets criteria).
  The phenomenon persists and is contextual, not a signal problem. Proceed to
  `expectation_transfer.py`, and treat the VR baseline as state-dependent
  (see below).
- **C — even best-case AR falls short** (Block 3 also fails vs dome). Rendering
  fidelity is the bottleneck (HP eq / spectral-cue capture). Debug the signal
  chain before any psychology — a mechanism study on an unfaithful signal
  measures nothing.

## 6. What still isn't answered (and is deliberately out of scope)

- **Presence vs priming.** Block 1 carries neither; Block 3 carries both. This
  script does not separate belief-in-speakers from acoustic priming — that's the
  belief × priming design, run as the distant 1–2 day side study, kept out of
  the 4-day learning experiment.
- **Persistence / decay.** How long the primed state survives the walk to the VR
  room. If Verdict B, add a repeat VR AR block ~20–30 min after Block 1's
  freefield induction to estimate the decay slope — that number decides whether
  the transfer baselines are trustworthy or need a portable warm-up on each
  training day.
- **Baseline↔training state match.** If the baseline is freshly primed but
  training days start cold, "improvement" partly reflects state, not learning.
  A short standardized virtual warm-up at the start of each session addresses
  this; verify it holds a state only real speakers can induce.
