"""Day-1 behavioural donor screening — a REJECT filter, NOT a ranking.

The acoustic properties of a donor pairing do not predict who relearns. Over
the 27 mold sets of the earmold experiment (15 subjects, a study where
relearning demonstrably happened), r_match, the rms spectral difference, detail
strength, cue gradient and cue monotonicity all correlate with day-1
impairment, day-1 performance and eventual recovery at |rho| <= 0.28. So a
donor cannot be chosen from its spectrum, and this module chooses behaviourally
instead — but only to REJECT, for a reason that is arithmetic rather than
philosophical.

WHY IT CANNOT RANK. From the test-retest floor measured on the earmold
ears-free repeats (polar error SD 1.61 deg, elevation gain SD 0.07 at n=132),
the difference two donors must show before you can tell them apart is:

    n=10   16.2 deg / 0.70 EG          n=60    6.6 deg / 0.29 EG
    n=21   11.2 deg / 0.49 EG          n=75    5.9 deg / 0.26 EG
    n=30    9.4 deg / 0.41 EG          n=100   5.1 deg / 0.22 EG
    n=50    7.3 deg / 0.32 EG          n=132   4.5 deg / 0.19 EG

The observed spread between the best and worst donor pairing across subjects is
about 10 deg of polar error. A ~50-trial screen therefore separates a BROKEN
pairing from a workable one and nothing finer; ranking two workable donors
needs n >= 100 each, which is a whole session for three candidates and still
only resolves 5 deg. Every screen this protocol has actually run bears that
out: FP's 24.08 pick (donor FS 21.1 deg vs donor AH 16.0 deg at n=75) rested on
a 5.1 deg difference against a 5.9 deg detection limit, i.e. on noise; FS's
28.07 screens were n=10 each, where the limit is 16.2 deg.

So: gate, then fall back on the pre-registered `donor_selection.shortlist()`
order for the choice among survivors. Picking the best-scoring survivor is
selecting on noise and turns a fixed rule into a garden of forking paths.

WHAT THE GATES ARE, AND WHERE THE NUMBERS COME FROM.

1. Azimuth sanity (the one criterion with a demonstrated effect size).
   Across all 238 earmold blocks -- ears free, molds week 1 AND week 2 --
   azimuth gain sits at 1.01-1.04 (IQR 0.98-1.06) and azimuth RMSE at
   5.3-5.5 deg. Azimuth is INVARIANT to a spectral manipulation. A donor that
   moves it has a broken render (ITD/ILD), not a hard spectrum, and no amount
   of training will fix it. NR's 25.08 screen: donor pilot/AH gave azimuth gain
   1.60 and RMSE 15.7 deg against donor pilot/SW's 1.04 and 4.8 deg. LS was
   never screened and ran her whole experiment at azimuth gain 1.25-1.38 and
   RMSE 10.7 deg, against her own-HRTF 1.04 and 7.8 deg. Both would have been
   caught here on day 1.

2. Perturbation band. donor polar error minus own-HRTF polar error, same day,
   same stimulus, same geometry. The earmold molds displaced listeners by
   +8.3 to +23.8 deg (median +12.2). Below about +6 deg the manipulation has
   not bitten and there is nothing to relearn -- AS was displaced +2.5 deg,
   improved 0.41 -> 0.51 in gain over three days, and that "recovery" is inside
   the n=75 noise band. Above about +20 deg there is a floor risk.

3. Elevation gain floor, deliberately loose. A very low day-1 gain WITH a large
   polar error suggests an unreadable map, but do not set this tight: FS opened
   at 0.37 and IR at 0.20 and both were among the better outcomes on record.

The bands are set from the distributions above and validated post hoc against
NR and LS. They have not been applied prospectively to anyone. Treat them as
screening thresholds, not estimates, and re-fit them once more subjects have
been through.

See also `localization_analysis.manipulation_check` (the same day-1 comparison
for a single already-chosen donor) and `donor_selection.cue_gradient` (reported,
never gating, for the same reason as above).
"""

import numpy

# --- measurement floor (earmold ears-free repeats, 15 subjects, n=132) -------
NOISE_PE_SD = 1.61       # deg, within-subject SD across sessions
NOISE_EG_SD = 0.07
NOISE_N = 132

# --- gates ------------------------------------------------------------------
# ELEVATION GAIN IS THE PRIMARY GATE (Paul, 2026-08-31). It is asked twice,
# because one number cannot answer both questions:
#   is there a cue at all?      -> absolute EG      (MIN_EG)
#   was the listener perturbed? -> EG as a FRACTION of their own-HRTF EG
# The second is necessary and the first cannot replace it. AS ran her donor
# block at EG 0.41 -- comfortably above any usable absolute floor -- but her own
# EG was 0.63, so she had kept 65% of her cue and had nothing to relearn. FS
# opened at 0.37 against an own EG of 0.84 (44% kept) and was the one
# unambiguous learner on record. 0.41 and 0.37 are indistinguishable in
# absolute terms and opposite in outcome; the ratio separates them.
MIN_EG = 0.15             # below this the composite has abolished the cue
                          # (LS 0.09, NR/FP 0.00 -- a block at chance teaches
                          # nothing and measures nothing)
EG_TARGET = 0.30          # where a good pairing lands, per Paul. Reported.
EG_TARGET_BAND = (0.20, 0.45)
MAX_EG_RETAINED = 0.55    # donor EG / own EG. Above this the manipulation did
                          # not bite. AS 0.65 (rejected); FS 0.44, IR 0.41,
                          # FP 0.38, NR 0.38 (all pass); LS 0.075 (floor).

# Polar-error impairment is KEPT as a co-gate, not demoted, because for the one
# question the EG ratio answers weakly it answers strongly: against an AS-like
# donor whose true impairment is +2.5 deg, a 35-trial block catches it 84% of
# the time on impairment and 67% on the EG ratio. Requiring either to fire puts
# the combined catch rate near 95%.
IMPAIRMENT_REJECT = (6.0, 20.0)   # deg of polar error vs the native block
IMPAIRMENT_TARGET = (8.0, 14.0)   # deg; reported, not enforced

# AZIMUTH IS A LOOSE SANITY CHECK, NOT A GATE -- demoted 2026-08-31.
# It looked like the strongest criterion until the composites were measured:
# every one of them carries ITD 287 us at +-35 deg, the same number for every
# subject, against natives at 256-263 us, because `build()` re-expanded the
# azimuths at the pipeline default head radius instead of the subject's fitted
# one (fixed the same day; see DonorModification.head_radius). The azimuth
# offsets the screen would have flagged were reading that fault, which is
# systematic and identical across donors -- so it cannot discriminate between
# them, and Paul was right that azimuth should be reconstructed correctly by
# the spherical model in any case. What is left is a coarse check that a
# particular block was not a write-off: NR's pilot/AH block came in at azimuth
# gain 1.60 with EG 0.00 and polar error 24.8 -- not a broken render but a
# participant not doing the task.
AZ_GAIN_REJECT = (0.70, 1.40)   # egregious only
AZ_GAIN_FLAG = (0.85, 1.15)     # reported, never rejects
AZ_RMSE_FACTOR = 2.0            # x the subject's own-HRTF azimuth RMSE


def resolution(n):
    """Smallest difference between two blocks of ``n`` trials that is not noise.

    1.96 * sqrt(2) * SD(n), with SD scaled from the n=132 floor as 1/sqrt(n).
    Returns ``{'pe': deg, 'eg': gain}``. Print it next to any screen so the
    reader can see whether the block was long enough to support the call.
    """
    f = numpy.sqrt(NOISE_N / float(n))
    k = 1.96 * numpy.sqrt(2.0)
    return {"pe": float(k * NOISE_PE_SD * f), "eg": float(k * NOISE_EG_SD * f)}


def impairment_se(n_screen, n_reference):
    """Standard error of (screen polar error - reference polar error), in deg.

    The impairment gate is a difference of two noisy means, so it is the WEAK
    gate and no feasible screen length fixes that. Against a +6 deg floor, the
    chance of missing an AS-like donor whose true impairment is +2.5 deg runs
    17% at n=30, 12% at n=50, 9% at n=75 -- and still 2% at n=400. Lengthening
    the screen buys almost nothing, which is why the block is short and why
    anything near a boundary comes back MARGINAL rather than PASS.
    """
    f = lambda n: NOISE_PE_SD * numpy.sqrt(NOISE_N / float(n))
    return float(numpy.hypot(f(n_screen), f(n_reference)))


def evaluate(reference, candidates,
             min_eg=MIN_EG, max_eg_retained=MAX_EG_RETAINED,
             impairment_reject=IMPAIRMENT_REJECT,
             az_gain_reject=AZ_GAIN_REJECT, az_gain_flag=AZ_GAIN_FLAG,
             az_rmse_factor=AZ_RMSE_FACTOR):
    """Apply the gates. Returns ``(rows, chosen)``.

    Parameters
    ----------
    reference : dict
        The own-HRTF block, SAME geometry and stimulus as the screens --
        normally the `native` phase (binaural, full field). Needs ``pe``,
        ``eg`` and ``az_rmse``; ``n`` is used for the resolution note.
    candidates : list of dict
        One per screened donor, in SHORTLIST RANK ORDER, each with ``donor``,
        ``rank``, ``pe``, ``eg``, ``az_gain``, ``az_rmse``, ``n``.

    Returns
    -------
    rows : list of dict
        Input rows plus ``impairment``, ``eg_retained``, ``passed``,
        ``reasons`` (empty when it passed) and ``marginal``.
    chosen : dict or None
        The FIRST SURVIVOR IN RANK ORDER -- not the best-scoring one. None if
        every candidate was rejected, which means stage more donors rather than
        pick the least-bad.
    """
    own_eg = reference.get("eg")
    rows = []
    for c in candidates:
        r = dict(c)
        r["impairment"] = float(c["pe"] - reference["pe"])
        r["eg_retained"] = (float(c["eg"] / own_eg)
                            if own_eg and own_eg > 0.1 else float("nan"))
        reasons, marginal = [], []

        # -- primary: elevation gain, asked twice ---------------------------
        if c["eg"] < min_eg:
            reasons.append(
                f"elevation gain {c['eg']:.2f} < {min_eg:.2f} — the composite "
                f"has abolished the cue, not degraded it")
        if numpy.isfinite(r["eg_retained"]) and r["eg_retained"] > max_eg_retained:
            reasons.append(
                f"kept {r['eg_retained']*100:.0f}% of their own elevation gain "
                f"({c['eg']:.2f} of {own_eg:.2f}) — above "
                f"{max_eg_retained*100:.0f}%, the manipulation did not bite")

        # -- co-gate: did the perturbation land at all ----------------------
        if r["impairment"] < impairment_reject[0]:
            reasons.append(
                f"impairment only +{r['impairment']:.1f} deg — nothing to relearn")
        elif r["impairment"] > impairment_reject[1]:
            reasons.append(f"impairment +{r['impairment']:.1f} deg — floor risk")

        # -- sanity: azimuth. Egregious rejects only; see the header. -------
        if not (az_gain_reject[0] <= c["az_gain"] <= az_gain_reject[1]):
            reasons.append(
                f"azimuth gain {c['az_gain']:.2f} outside "
                f"{az_gain_reject[0]:.2f}-{az_gain_reject[1]:.2f} — this block "
                f"is a write-off (check the participant was doing the task), "
                f"not evidence about the donor")
        elif not (az_gain_flag[0] <= c["az_gain"] <= az_gain_flag[1]):
            marginal.append(f"azimuth gain {c['az_gain']:.2f} outside "
                            f"{az_gain_flag[0]:.2f}-{az_gain_flag[1]:.2f} (noted, "
                            f"does not reject)")
        limit = az_rmse_factor * reference["az_rmse"]
        if c["az_rmse"] > limit:
            marginal.append(f"azimuth RMSE {c['az_rmse']:.1f} deg > {limit:.1f} "
                            f"({az_rmse_factor:g}x own) (noted, does not reject)")

        # -- MARGINAL: passed, but a gated quantity is within one SE of an
        #    edge, so the verdict would flip on a re-run.
        eg_se = NOISE_EG_SD * numpy.sqrt(NOISE_N / float(c["n"]))
        if abs(c["eg"] - min_eg) < eg_se:
            marginal.append(f"elevation gain {c['eg']:.2f} is within one SE "
                            f"({eg_se:.2f}) of the {min_eg:.2f} floor")
        if numpy.isfinite(r["eg_retained"]) and own_eg:
            if abs(c["eg"] - max_eg_retained * own_eg) < eg_se:
                marginal.append(
                    f"kept {r['eg_retained']*100:.0f}% — within one SE of the "
                    f"{max_eg_retained*100:.0f}% did-not-bite edge")
        se = impairment_se(c["n"], reference.get("n", c["n"]))
        for edge, side in ((impairment_reject[0], "floor"),
                           (impairment_reject[1], "ceiling")):
            if abs(r["impairment"] - edge) < se:
                marginal.append(
                    f"impairment {r['impairment']:+.1f} deg is within one SE "
                    f"({se:.1f}) of the {side} at {edge:+.0f}")

        r["reasons"] = reasons
        r["marginal"] = [] if reasons else marginal
        r["passed"] = not reasons
        rows.append(r)

    survivors = [r for r in rows if r["passed"]]
    survivors.sort(key=lambda r: r["rank"])
    return rows, (survivors[0] if survivors else None)


def report(rows, chosen, reference):
    """Print the screen. Everything a supplement needs is on these lines."""
    ns = sorted({int(r["n"]) for r in rows})
    res = resolution(min(ns)) if ns else {"pe": float("nan"), "eg": float("nan")}
    print("\n" + "=" * 78)
    print("DAY-1 DONOR SCREEN — reject filter, not a ranking")
    print("=" * 78)
    print(f"reference (own HRTF, same geometry): elevation gain "
          f"{reference.get('eg', float('nan')):.2f}, polar error "
          f"{reference['pe']:.1f} deg, azimuth RMSE {reference['az_rmse']:.1f} deg"
          + (f", n={reference['n']}" if reference.get("n") else ""))
    print(f"screen blocks n={'/'.join(str(x) for x in ns)}  ->  two donors are "
          f"distinguishable only beyond {res['pe']:.1f} deg polar error / "
          f"{res['eg']:.2f} gain.")
    print("Do NOT read the ordering below as a ranking; it is rank order from "
          "donor_selection.shortlist().\n")
    print(f"primary gate is ELEVATION GAIN: >= {MIN_EG:.2f} absolute (a cue "
          f"exists) AND <= {MAX_EG_RETAINED*100:.0f}% of the listener's own "
          f"(it was perturbed). Target ~{EG_TARGET:.2f}.\n")
    print(f"{'rank':>4} {'donor':>14} {'EG':>6} {'kept':>6} {'PE':>6} "
          f"{'impair':>7} {'azGain':>7} {'azRMSE':>7}  verdict")
    for r in sorted(rows, key=lambda r: r["rank"]):
        mark = ("MARGINAL" if r["passed"] and r.get("marginal")
                else "PASS" if r["passed"] else "REJECT")
        star = " <--" if chosen is not None and r["donor"] == chosen["donor"] else ""
        kept = (f"{r['eg_retained']*100:5.0f}%"
                if numpy.isfinite(r.get("eg_retained", float("nan"))) else "    —")
        print(f"{r['rank']:>4} {r['donor']:>14} {r['eg']:6.2f} {kept:>6} "
              f"{r['pe']:6.1f} {r['impairment']:+7.1f} {r['az_gain']:7.2f} "
              f"{r['az_rmse']:7.1f}  {mark}{star}")
        for reason in r["reasons"] + list(r.get("marginal", [])):
            print(f"{'':>46}   - {reason}")
    print()
    if chosen is None:
        print("!! EVERY candidate was rejected. Stage more donors "
              "(prepare_shortlist(n=5, screen=True)) rather than run the "
              "least-bad one — a rejected donor is rejected for a reason that "
              "training cannot fix.")
    else:
        print(f"chosen: {chosen['donor']} (rank {chosen['rank']}) — the first "
              f"survivor in the pre-registered order, NOT the best-scoring one.")
        if not (EG_TARGET_BAND[0] <= chosen["eg"] <= EG_TARGET_BAND[1]):
            print(f"  note: elevation gain {chosen['eg']:.2f} is outside the "
                  f"target band {EG_TARGET_BAND[0]:.2f}-{EG_TARGET_BAND[1]:.2f} "
                  f"(centre {EG_TARGET:.2f}) though inside the reject band. "
                  f"Report it.")
        if not (IMPAIRMENT_TARGET[0] <= chosen["impairment"] <= IMPAIRMENT_TARGET[1]):
            print(f"  note: impairment {chosen['impairment']:+.1f} deg is outside "
                  f"the target band {IMPAIRMENT_TARGET[0]:g}-"
                  f"{IMPAIRMENT_TARGET[1]:g} deg though inside the reject band.")
        if chosen.get("marginal"):
            print("  ** MARGINAL — this donor passed, but on a quantity close "
                  "enough to a boundary that a re-run could flip it. Run it, "
                  "then re-check against baseline A before committing the "
                  "participant to four days; use_donor() still swaps in "
                  "seconds at that point.")
        print("  REMINDER: the screen is itself donor exposure. THIS block is "
              "the naive measurement of the chosen donor — take the day-1 "
              "impairment from here, not from baseline A. Baseline A now runs "
              "post-screen for every participant; that is uniform and "
              "interpretable, but it has to be reported.")
    print("=" * 78)
    return chosen
