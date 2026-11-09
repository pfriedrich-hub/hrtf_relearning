# What predicts learning -- donor selection vs early learning (2026-11-09)

Question (Paul): what can the donor selection tell us about who learns, to raise
the learner rate? Sample: 18 donor-detail participants -- 8 learners (FS, IR, SH,
PA, SK, DA, LP, JO) vs 10 non-learners (LGL, AS, GM, LS, IK, EM, MM, KH, SHE, AB);
NR aborted, left out. Post-hoc, 23 variables, n = 18: treat as hypotheses.

## 1. Donor and day-1 variables do NOT separate (all AUC 0.32-0.61, p > 0.3)

| variable | learners (median) | non-learners | AUC |
|---|---|---|---|
| r_match (trained ear) | 0.56 | 0.55 | 0.55 |
| ridge slope | 0.28 | 0.37 | 0.55 |
| donor cue gradient / monotonicity | 0.62 / 0.83 | 0.77 / 0.82 | 0.43 / 0.52 |
| donor / own gradient, strength | 0.84, 0.96 | 1.07, 1.00 | 0.47, 0.41 |
| cross-ear donor | 7/8 | 8/10 | 0.51 |
| day-1 A retained gain | 39% | 40% | 0.44 |
| day-1 PE impairment | +6.9 deg | +7.5 deg | 0.43 |
| own-HRTF gain | 0.75 | 0.83 | 0.36 |

- Donor identity: AH -> LP (L), LS, KH (N); SH -> SK (L), AB (N); AS -> FS, IR (L);
  LS as donor 0/2 (LGL, MM).
- r_match / ridge only vary inside the window the selection rule allows
  (r_match 0.51-0.74): they do not matter *within* it; says nothing about the window.
- Screen band: of screened donors retaining > 55%, 3 of 4 failed (EM 61, SHE 69,
  KH 77; SH 60 learned) vs 2 of 6 inside 30-55% (p = .52). Keep the band.

## 2. Early learning predicts it: day-2 post-training gain minus day-1 A

| | day-2 post - day-1 A (gain) |
|---|---|
| learners | +0.18 .. +0.39 (all 8) |
| non-learners | -0.09 .. +0.11 (9 of 10); IK +0.22 |

AUC 0.97 (p < .001); polar error the same direction (AUC 0.88). A +0.15 rule after
day 2 would have stopped 9/10 non-learners and kept every learner -> two days per
failure instead of four. Threshold chosen on these data.

## 3. Technical problems cluster in non-learners (6/10 vs 1/8, p = .07)

Non-learners: GM (laser offset +13 deg, stimulus switch), EM (frontal-ITD fault,
rebuilt after day 1), MM (-8 deg azimuth bias, tracker/chair suspected), KH (laser
offset +15..+27 deg throughout), SHE (head-radius fallback, rebuilt after day 1),
IK (mixed stimuli). Learners: JO (rebuilt after day 1). The rest of the
non-learners are participant-side: AS/LGL little impairment or high start, LS
extreme impairment (7% retained), AB rerun.

## Consequences (console reminders in session_checks.py, print only)

| step | reminder | rule |
|---|---|---|
| HRIR_Recording step 0 | head_radius_check | fallback radius -> re-seat mics, redo fit |
| HRIR_Recording step 4 | dome_check | best free-field EG < 0.5 -> do not train (GM .30, KH .44, IK .48; lowest learner FS .52) |
| HRIR_Recording 5b / learning_transfer native | baseline_check | own-HRTF EG < 0.5 -> do not train (KH .41; lowest learner LP .64) |
| every learning_transfer block (run_phase) | block_check | calibration pitch outside -17..0 deg, or elevation bias > 10 deg -> check laser / tracker zero |
| adaptation-day post test | progress_check | day >= 2 and gain step < +0.15 over day-1 A -> consider stopping |
