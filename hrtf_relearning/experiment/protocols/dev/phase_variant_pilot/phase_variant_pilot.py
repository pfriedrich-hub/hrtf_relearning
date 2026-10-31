"""
phase_variant_pilot.py

Cell-by-cell protocol for the PHASE-VARIANT pilot: does elevation localization
need the monaural TIME structure of the HRIR, or only its magnitude spectrum?

Batteau (1967) proposed that the pinna encodes direction as short reflection
delays that the nervous system inverts. Because an echo of delay tau and a comb
with notches at (2k+1)/2tau are the same thing, no manipulation of the HRIR
MAGNITUDE can separate the two accounts — including everything else in
hrtf.modify. What separates them is a manipulation that leaves |H(f)| identical
and rearranges the impulse response in time. See hrtf/modify/phase_variants.py
for the derivation and the conditions.

All localization tests run in AR (HRTF over headphones, Localization_AR) on the
VERTICAL MIDLINE only, matched to the dome speaker layout, with head tracking
live.

WHAT IS AND IS NOT ALREADY KNOWN (verify both methods sections before citing)
  Kulkarni, Isabelle & Colburn (1999, JASA 105:2821) built these same
    conditions — min-phase-plus-delay, linear-phase, reversed-phase-plus-delay
    — but measured DISCRIMINATION, not localization. Listeners could not hear a
    difference PROVIDED the low-frequency ITD was appropriate; where they could,
    ITD was the cue. Hence the ITD gate in phase 0.
  Kistler & Wightman (1992, JASA 91:1637) measured LOCALIZATION with min-phase
    -plus-delay reconstructions: judgments nearly identical to measured HRTFs.
    So `minphase` here is a REPLICATION CONTROL, not an open question.
  Nobody appears to have run localization with REVERSED or MAXIMUM phase. That
    is the experiment. Discrimination and localization are different tests: a
    listener can hear a difference that carries no directional information, or
    point differently without being able to say anything changed.

Individual HRIRs and live head tracking are the second novelty — statically,
a running-echo readout and a magnitude-pattern readout are much harder to pull
apart than they are under self-motion.

CONDITIONS (all 1024 taps, all built from this subject's own SOFA)
  baseline        native HRIR, zero-padded to the same length as the variants.
                  THIS, not the native SOFA, is the control — otherwise the
                  conditions differ in filter length as well as in phase.
  reversed        h[::-1] with the arrival put back. Magnitude identical.
                  *** the primary contrast: baseline vs reversed ***
  minphase        all-pass component removed. REPLICATION CONTROL — Kistler &
                  Wightman already showed this localizes like the original; if
                  it does not here, something is wrong with the setup, not with
                  Batteau.
  maxphase        all-pass component sign-flipped. (minphase, maxphase) is an
                  exactly matched pair differing ONLY in the sign of the phase,
                  and the pair contrast is cleaner than either against baseline
                  because it does not go through the reconstruction twice.
  allpass_<d>ms   dispersive all-pass, same filter on both ears, so ITD/ILD/IPD
                  are untouched by construction. The parametric condition: sweep
                  the dose and find where elevation breaks. Compare the
                  breakpoint against auditory-filter time constants (~0.25 ms at
                  16 kHz) and against pinna echo delays (60-160 us).

PREDICTIONS
  Batteau-style temporal readout : reversed and maxphase collapse; allpass
      degrades from the smallest dose upward.
  magnitude-spectrum readout     : reversed, minphase and maxphase are all
      equivalent to baseline; allpass survives until the dispersion exceeds the
      cochlear integration time, i.e. milliseconds, not microseconds.

DESIGN NOTES
  * Stimulus is continuous NOISE, not clicks. Reversal turns the HRIR's decay
    into a pre-echo, which with transients is an audible attack change and a
    non-spatial cue the listener can latch onto. Noise removes it.
  * Run baseline FIRST and AGAIN LAST (phase 6) as a drift control.
  * Order of the middle blocks should be counterbalanced across subjects; with
    one participant per session just record the order you used.
  * Before interpreting any null result, check phase 2b: if the listener cannot
    DISCRIMINATE baseline from reversed at all, the localization result was
    never in question and the interesting number is the discrimination one.

Run cell by cell (# %%). Build a condition's SOFA (phase 0) before running its
AR block. Rerun any cell as needed.
"""

# %% imports and config ------------------------------------------------------
import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.hrtf.modify import phase_variants as pv
from hrtf_relearning.utils import paths

SUBJECT_ID = "GS"                # edit per participant
HP = "DT990"                     # headphone EQ profile
PROTOCOL_TAG = "phaseV"          # tags every run's filename/sequence.name

# --- manipulation config
ITD_MODE = "phase"               # match the low-frequency IPD (see phase_variants)
ALIGN = "fractional"             # 'integer' for a bit-exact magnitude instead
N_OUT = 1024                     # filter length for every condition, baseline included
DISPERSION_BAND = (3000.0, 16000.0)
DISPERSIONS = (0.5, 2.0, 5.0)    # allpass dose series [ms], run ascending

# --- QC gate: refuse to run a condition whose magnitude moved
MAX_DB_TOLERANCE = 0.5           # worst single bin, top 40 dB of each direction
MAX_ITD_TOLERANCE_US = 20.0      # phase-derived ITD change, ~the ITD JND

# --- AR block config
TARGETS_PER_SPEAKER = 3          # -> 21 trials/block on the 7 midline positions
MIN_DISTANCE = 15
GAIN = 0.07                      # matched to dome at Windows master volume 50%
QC_DIRECTION = (0.0, 0.0)

SOFA_DIR = paths.SOFA_DIR / SUBJECT_ID
BASE_SOFA_PATH = SOFA_DIR / f"{SUBJECT_ID}.sofa"

AR_MIDLINE_SETTINGS = {
    "kind": "standard",
    "azimuth_range": (-1, 1),
    "elevation_range": (-35, 35),
    "targets_per_speaker": TARGETS_PER_SPEAKER,
    "min_distance": MIN_DISTANCE,
    "gain": GAIN,
    "stim": "noise",             # NOT clicks -- see DESIGN NOTES
}


def label_for(condition, dispersion_ms=None):
    """'phase_reversed', 'phase_allpass_2ms', ... -> <subj>_<label>.sofa"""
    if dispersion_ms is None:
        return f"phase_{condition}"
    return f"phase_{condition}_{pv.dispersion_tag(dispersion_ms)}"


def hrir_settings(label):
    """binaural, vertical-midline; `name` resolves
    data/hrtf/sofa/<subj>/<subj>_<label>.sofa."""
    return {
        "name": f"{SUBJECT_ID}_{label}",
        "subject_id": SUBJECT_ID,
        "ear": None, "mirror": False,
        "reverb": True, "drr": 20,
        "hp_filter": True, "hp": HP,
        "convolution": "cpu", "storage": "cpu",
    }


def _tag(loc_test, stage):
    """Append '_<PROTOCOL_TAG>-<stage>' before .run(), so these runs can be
    picked out of subject.localization later. Same convention as
    expectation_transfer.py."""
    loc_test.filename = f"{loc_test.filename}_{PROTOCOL_TAG}-{stage}"
    loc_test.sequence.name = loc_test.filename
    return loc_test


def assert_clean(result, label):
    """Gate: a condition only reaches a participant if the magnitude really did
    not move and the ITD really was preserved.

    This is the whole experiment in one check. If the magnitude drifted, the
    manipulation is no longer 'phase only' and a localization difference has an
    ordinary spectral explanation; if the ITD drifted, an ITD explanation. Both
    thresholds are deliberately generous relative to what phase_variants
    actually achieves (~0.2 dB, <1 us) — tripping one means something is wrong,
    not merely tight."""
    max_db = result["max_db"]
    itd = result["itd_phase_delta_us_max"]
    if max_db > MAX_DB_TOLERANCE:
        raise RuntimeError(
            f"{label}: magnitude moved by {max_db:.3f} dB (> {MAX_DB_TOLERANCE}). "
            f"This is no longer a phase-only manipulation. Increase N_OUT, or "
            f"use align='integer'.")
    if itd > MAX_ITD_TOLERANCE_US:
        raise RuntimeError(
            f"{label}: ITD moved by {itd:.1f} us (> {MAX_ITD_TOLERANCE_US}). "
            f"Use itd_mode='phase', or align='fractional' if you switched to "
            f"integer alignment.")
    print(f"  OK  {label}: {max_db:.4f} dB, {itd:.2f} us")


def run_ar(subject, label, stage=None):
    """Build the binsim files for <subj>_<label>.sofa and run one AR midline
    block. Opens a QC preview of the HRIR being run (baseline vs condition)."""
    preview(label)
    loc = _tag(Localization(subject, hrir_settings(label), AR_MIDLINE_SETTINGS),
               stage or label)
    loc.run()
    return subject.localization[loc.filename]


def preview(label, block=False):
    """Non-blocking window with baseline vs <label>: impulse responses,
    cochleagrams, and the two panels that must superimpose (magnitude spectrum
    and time-integrated excitation). If those two separate, stop."""
    import matplotlib.pyplot as plt
    variant = slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa"))
    fig = pv.qc_figure(baseline_hrtf, {label: variant}, direction=QC_DIRECTION,
                       title=f"{SUBJECT_ID} — running: {label}")
    plt.show(block=block)
    plt.pause(0.1)
    return fig


# %% load the native HRTF and build the length-matched baseline ---------------
subject = hr.Subject(SUBJECT_ID)
base_hrtf = slab.HRTF(str(BASE_SOFA_PATH))
base_hrtf.name = SUBJECT_ID
print(f"{SUBJECT_ID}: {base_hrtf.n_sources} directions, "
      f"{base_hrtf[0].n_samples} taps, {base_hrtf[0].samplerate:.0f} Hz")

baseline_hrtf = pv.pad_hrtf(base_hrtf, N_OUT)
baseline_label = label_for("baseline")
baseline_hrtf.write_sofa(str(SOFA_DIR / f"{SUBJECT_ID}_{baseline_label}.sofa"))
print(f"baseline -> {SUBJECT_ID}_{baseline_label}.sofa ({N_OUT} taps)")

# %% PHASE 0 -- build and verify every condition ------------------------------
# Nothing runs on a participant until this cell prints OK for the conditions you
# intend to use. The printed report is the methods section.
results = {}
for condition in ("reversed", "minphase", "maxphase"):
    label = label_for(condition)
    _, result = pv.save_condition_sofa(
        base_hrtf, condition, SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa",
        plot_dir=paths.subject_acoustic_dir(SUBJECT_ID),
        direction=QC_DIRECTION, itd_mode=ITD_MODE, align=ALIGN, n_out=N_OUT)
    results[label] = result
    print(pv.format_report(result))
    assert_clean(result, label)
    print()

for dispersion in DISPERSIONS:
    label = label_for("allpass", dispersion)
    _, result = pv.save_condition_sofa(
        base_hrtf, "allpass", SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa",
        plot_dir=paths.subject_acoustic_dir(SUBJECT_ID),
        direction=QC_DIRECTION, dispersion_ms=dispersion,
        band=DISPERSION_BAND, n_out=N_OUT)
    results[label] = result
    print(pv.format_report(result))
    assert_clean(result, label)
    print()

# %% PHASE 0b -- the matched-pair check ---------------------------------------
# minphase and maxphase should differ ONLY in the sign of the phase, so their
# magnitude spectra agree to machine precision. This is a stronger check than
# either against baseline, because it does not go through the minimum-phase
# reconstruction twice.
pair = pv.verify(slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}_{label_for('minphase')}.sofa")),
                 slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}_{label_for('maxphase')}.sofa")))
print(f"minphase vs maxphase: magnitude deviation {pair['max_db']:.3e} dB")

# %% PHASE 0c -- overview figure for the lab book -----------------------------
overview = {lab: slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}_{lab}.sofa"))
            for lab in (label_for("reversed"), label_for("maxphase"),
                        label_for("allpass", DISPERSIONS[-1]))}
fig = pv.qc_figure(baseline_hrtf, overview, direction=QC_DIRECTION,
                   title=f"{SUBJECT_ID} — phase variants")
plot_dir = paths.subject_acoustic_dir(SUBJECT_ID)
plot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_dir / f"{SUBJECT_ID}_phase_overview.png", bbox_inches="tight", dpi=140)
print(f"wrote {plot_dir / f'{SUBJECT_ID}_phase_overview.png'}")

# %% status check (rerun anytime) ---------------------------------------------
done = list(getattr(subject, "localization", {}).keys())
own = [k for k in done if f"_{PROTOCOL_TAG}-" in k]
print(f"phase_variant runs already on file ({len(own)}):")
for k in own:
    print(f"   - {k}")

# %% PHASE 1 -- baseline block (control) --------------------------------------
run_ar(subject, baseline_label, stage="baseline_pre")

# %% PHASE 2 -- reversed block (THE primary contrast) -------------------------
run_ar(subject, label_for("reversed"), stage="reversed")

# %% PHASE 2b -- discrimination check  *** NOT IMPLEMENTED — read this ***
# This is the direct replication of Kulkarni, Isabelle & Colburn (1999), and it
# is worth running BEFORE interpreting any localization result, for two reasons.
#
#  1. Their null was conditional on ITD. If your listeners CAN discriminate
#     baseline from reversed, the first hypothesis is not "Batteau was right",
#     it is "the ITD restoration leaked" -- check results[...] from phase 0
#     against the ITD JND before concluding anything.
#  2. Discrimination and localization dissociate in both directions. Hearing a
#     difference does not mean the difference carries directional information;
#     pointing differently does not require being able to report a change.
#
# What is needed: 2I-2AFC same/different over the same midline directions,
# baseline vs reversed, FIXED head position (no tracking -- this is the static
# replication), ~40 trials. There is no discrimination task in the package yet;
# build it against localization_helpers/stimulus.make_gapped_pinknoise and the
# binsim stream, or run it offline on rendered wavs.

# %% PHASE 3 -- the matched pair: minphase then maxphase ----------------------
run_ar(subject, label_for("minphase"), stage="minphase")

# %%
run_ar(subject, label_for("maxphase"), stage="maxphase")

# %% PHASE 4 -- allpass dose series, ASCENDING --------------------------------
# Set DISPERSION_TO_RUN to each value of DISPERSIONS in turn and rerun this cell.
# Stop when elevation performance clearly departs from baseline; that dose is
# the breakpoint. Read it against the filter-ringing curve in the QC figure --
# a magnitude readout should hold well past the pinna echo delays (60-160 us)
# and start to fail only around the cochlear integration time.
DISPERSION_TO_RUN = 0.5
run_ar(subject, label_for("allpass", DISPERSION_TO_RUN),
       stage=f"allpass_{pv.dispersion_tag(DISPERSION_TO_RUN)}")

# %% PHASE 5 -- baseline again (drift control) --------------------------------
# Same condition as phase 1. If baseline_post differs from baseline_pre, the
# between-condition differences are contaminated by fatigue/practice and the
# session needs re-running with a counterbalanced order.
run_ar(subject, baseline_label, stage="baseline_post")

# %% preview any condition on demand (edit `label`) ---------------------------
label = label_for("reversed")
preview(label, block=True)
