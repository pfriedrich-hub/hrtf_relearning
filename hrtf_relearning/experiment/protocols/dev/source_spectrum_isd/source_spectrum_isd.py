"""
source_spectrum_isd.py

Protocol for the source-spectrum x interaural-spectral-difference experiment
(see source_spectrum_isd_design.md for the full rationale). Asks whether the
binaural channel is what lets listeners cope with an unknown source spectrum.

    P(f) = S(f) * H(f, dir)              one equation, two unknowns
    P_L(f) / P_R(f) = H_L(f) / H_R(f)    the source cancels exactly

Source variation BELOW 0.5 ripples/oct can be filtered out without solving
anything, because it lives in a different ripple-density band from the cue.
Source variation INSIDE the 0.5-2 ripples/oct cue band cannot: there the two
are formally indistinguishable in a single monaural spectrum, and only a
prior, an interaural comparison, or across-trial statistics can help. So the
in-band condition is where inference is actually required, and the diotic
condition removes the one channel that solves it without any inference.

2 x 2 plus an anchor, all on the vertical midline, all on the NATIVE HRTF:

    ISD          stimulus        expectation
    natural      noise           anchor; also checks the render is sane
    natural      below-band      ~ noise (filtering suffices)
    natural      in-band         degraded
    diotic       noise           anchor; checks the diotic SOFA still localizes
    diotic       below-band      ~ noise (filtering still suffices)
    diotic       in-band         degraded MORE, if the ISD is what rescues it

The result is the INTERACTION, not any main effect. No interaction means the
listener is not using the interaural spectral difference -- which is itself
the answer, and is what AS's donor blocks already hint at (he tracked the
source spectrum at +0.80 deg/dB in a condition whose ISD cue was, if anything,
unusually deep).

The diotic condition is diotic only ON the midline. Off it, the spherical-head
ILD and ITD are imposed exactly as for the native set, so the sound still
lateralizes -- and a level offset plus a time shift restores no
source-spectrum information, which is the point. Cell 1 verifies that: the
cue-band ISD depth must be ~0 at every azimuth, not just at 0.

Why midline only: azimuth in these SOFAs is synthesised. Only the az=0 arc is
measured; every other direction is that arc times a spherical-head model
(processing/midline.py), so off-midline targets carry no independent spectral
detail and az 0 is the only azimuth where the AR render is acoustically
faithful to the measurement. It is also where polar error reduces to plain
elevation error. Filler trials off the midline keep the response space 2-D so
a listener cannot collapse to a 1-D pointing strategy; they are NOT a
condition and are dropped before analysis.

Every run is tagged '_ssISD-<stage>' on both filename and sequence.name, so
these blocks can be pulled back out of subject.localization with a key filter.

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as
a plain script. Cell 0 is run ONCE for the whole study, not per subject.
Cell 1 is per subject, cells 4-9 are the blocks, in the order cell 3 prints.

Time: 6 blocks x 63 trials ~ 70-85 min with breaks. If that is too long, drop
TARGETS_PER_SPEAKER to 2 (46 trials, ~55 min) before dropping any block -- the
two noise anchors are the only check that the diotic render is not simply
broken.
"""

# %% imports and config ------------------------------------------------------
import numpy
import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.localization_helpers import spectral_metrics as sm
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.experiment.protocols.protocol_helpers import (
    collect_demographics, collect_externalization_rating)
from hrtf_relearning.hrtf.processing.diotic import save_diotic_sofa
from hrtf_relearning.hrtf.record.fit_head_radius import fit_from_sofa
from hrtf_relearning.utils import paths
# private, but it is the one coordinate transform in the codebase -- importing
# it beats keeping a second copy that can drift out of step
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy, _interaural_polar)

SUBJECT_ID = "AS"                 # edit per participant
SUBJECT_INDEX = 1                 # 1-based recruitment order; sets the counterbalance
PROTOCOL_TAG = "ssISD"

DIOTIC_EAR = "left"               # ear whose DTF is duplicated onto both sides.
                                  # Its own monaural cue is untouched, so pick the
                                  # better/trained ear and counterbalance across
                                  # subjects if the overwritten ear matters.
TARGETS_PER_SPEAKER = 3           # x 17 midline sources = 51 midline trials
N_FILLER = 12                     # + off-midline fillers = 63 trials per block
MIN_DISTANCE = 15                 # deg between consecutive targets
GAIN = 0.2                        # pybinsim gain. Re-derive per subject with
                                  # match_ar_dome_loudness.py; only valid at the
                                  # OS volume that match was made at (50%).
HP = "DT990"

RMS_TILT = 3.0                    # below-band source variation, held CONSTANT in
                                  # every ripple block. 3 dB is the value AS's
                                  # dome ladder settled on (tilt 4 and 8 cost
                                  # precision, 3 does not).
RMS_CUE = 2.0                     # in-band source variation, dB. FIXED across
                                  # participants -- the comparison is within
                                  # subject, so the stimulus stays constant and
                                  # some listeners simply localize this ripple
                                  # better than others. Chosen ONCE in cell 0
                                  # against the SHALLOWEST cue in the pool, so
                                  # nobody is floored. Re-run cell 0 only if the
                                  # subject pool changes; never per subject.

BELOW_BAND = {"rms_tilt": RMS_TILT, "rms_cue": 0.0}    # source outside the cue band
IN_BAND = {"rms_tilt": RMS_TILT, "rms_cue": RMS_CUE}   # source inside it as well

NATIVE_SOFA = SUBJECT_ID
DIOTIC_SOFA = f"{SUBJECT_ID}_diotic_{DIOTIC_EAR}"


def hrir_settings(sofa_name):
    """Binaural render of one SOFA. `ear=None` means no ear reduction, so
    `other_ear` never applies -- both ears keep whatever the SOFA holds."""
    return {"name": sofa_name, "subject_id": SUBJECT_ID,
            "ear": None, "mirror": False,
            "reverb": True, "drr": 20,
            "hp_filter": True, "hp": HP,
            "convolution": "cpu", "storage": "cpu"}


def loc_settings(stim, stim_settings=None):
    """Midline arc plus off-midline fillers, one stimulus condition. Fillers are
    tagged 'filler' on sequence.target_set and dropped in cell 10."""
    return {"kind": "midline_filler",
            "elevation_range": (-35, 35),
            "targets_per_speaker": TARGETS_PER_SPEAKER,
            "n_filler": N_FILLER,
            "filler_azimuth_range": (-35, 35),
            "min_distance": MIN_DISTANCE,
            "gain": GAIN,
            "stim": stim,
            "stim_settings": stim_settings or {}}


def session_order(index=None):
    """Block order for this subject, from recruitment order.

    The ISD level is BLOCKED -- switching it rebuilds the pybinsim database --
    so its order alternates across subjects, and the noise anchor always opens
    each half so a broken render shows up before the ripple blocks. Within a
    half the two ripple conditions rotate, and the second half runs them in the
    opposite order, so neither is systematically last.
    """
    index = SUBJECT_INDEX if index is None else index
    isd = ("natural", "diotic") if index % 2 else ("diotic", "natural")
    ripples = ("below", "in") if (index // 2) % 2 == 0 else ("in", "below")
    return ([(isd[0], "noise"), (isd[0], ripples[0]), (isd[0], ripples[1])]
            + [(isd[1], "noise"), (isd[1], ripples[1]), (isd[1], ripples[0])])


def _tag(loc_test, stage):
    """Label a run so it can be filtered back out of subject.localization.
    Must be called after construction and before .run() -- write() keys on
    filename."""
    loc_test.filename = f"{loc_test.filename}_{PROTOCOL_TAG}-{stage}"
    loc_test.sequence.name = loc_test.filename
    return loc_test


def run_block(subject, isd, band):
    """One cell of the design. isd: 'natural' | 'diotic'. band: 'noise' |
    'below' | 'in'."""
    sofa = NATIVE_SOFA if isd == "natural" else DIOTIC_SOFA
    stim_settings = {"noise": None, "below": BELOW_BAND, "in": IN_BAND}[band]
    loc_test = Localization(subject, hrir_settings(sofa),
                            loc_settings("noise" if band == "noise" else "ripple",
                                         stim_settings))
    _tag(loc_test, f"{isd}-{band}")
    loc_test.run()
    # the diotic render puts one ear's DTF on both sides and will sound more
    # in-the-head. A difference in externalization is a live alternative
    # explanation for a difference in polar error, so measure it rather than
    # assuming it away -- same prompt wording as every other protocol.
    collect_externalization_rating(loc_test)
    sequence = subject.localization[loc_test.filename]
    sequence.isd = isd                      # 'natural' | 'diotic'
    sequence.source_band = band             # 'noise' | 'below' | 'in'
    sequence.diotic_ear = DIOTIC_EAR if isd == "diotic" else None
    sequence.subject_index = SUBJECT_INDEX
    subject.write()
    return loc_test


def midline_errors(sequence):
    """(local polar errors, quadrant-error rate) over MIDLINE trials only.

    Fillers exist to keep the task 2-D and are not part of any condition, so
    they are dropped here. Older sequences without target_set are treated as
    all-midline.
    """
    data = numpy.asarray(sequence.data).reshape(len(sequence.data), 2, 2)
    tags = numpy.asarray(getattr(sequence, "target_set", ["midline"] * len(data)))
    data = data[tags == "midline"]
    _, target = _interaural_polar(data[:, 1, 0], data[:, 1, 1])
    _, response = _interaural_polar(data[:, 0, 0], data[:, 0, 1])
    error = (target - response + 180.0) % 360.0 - 180.0
    quadrant = numpy.abs(error) > 90.0
    return numpy.abs(error[~quadrant]), float(quadrant.mean())


# %% 0. ONCE PER STUDY: choose RMS_CUE ---------------------------------------
# Not a per-subject cell. The in-band source variation is held constant across
# participants -- the comparison is within subject, so a listener who localizes
# this ripple better than another is not a problem. What IS a problem is a
# listener at floor, so the constant is chosen against the SHALLOWEST cue depth
# in the pool: run this against the weakest subject's native SOFA, not a
# typical one, and take a ratio near 1:1. Then paste the number into RMS_CUE
# above and leave it alone.
#
# Cue depths measured so far (0.5-2 rip/oct, az 0, left ear): AS 7.84, GS 13.79,
# FS 13.59 -- AS is the binding case by some margin.
slab.set_default_samplerate(48828)
_weakest = sm.load_hrtf("AS", "AS")          # edit to the weakest SOFA in the pool
_rms_cue, _report = sm.calibrate_rms_cue(_weakest, ear="left", target_ratio=1.0,
                                         rms_tilt=RMS_TILT)
print(f"\npaste into RMS_CUE: {_rms_cue}")


# %% 1. per subject: build the diotic SOFA -----------------------------------
# Duplicates DIOTIC_EAR's measured az=0 arc onto both ears and re-expands, so
# the interaural spectral difference is frequency-flat while the spherical-head
# ILD and ITD are imposed as usual. Two manipulation checks in the printout:
#   - the duplicated ear's monaural depth must be UNCHANGED
#   - the ISD depth must fall to ~0 at EVERY azimuth, not just at 0. A level
#     offset and a time shift carry no source-spectrum information, and this is
#     where that assumption is verified rather than asserted.
# If the native ISD depth is already near zero this subject has no binaural
# channel to remove and cannot test the hypothesis -- note it and continue, the
# monaural cells are still valid.
#
# The sphere radius is READ BACK OFF THE NATIVE SOFA rather than defaulted:
# the two conditions must differ only in the interaural SPECTRAL difference, so
# the diotic set has to be expanded with the same radius, and expand_from_midline
# would otherwise silently use 0.0875 while the protocol records ~0.0725. A
# residual above a few microseconds means the native set was built with the
# legacy itd_method='onset' -- rebuild it before running the experiment, or the
# diotic set will differ in ITD too.
native_path = paths.SOFA_DIR / SUBJECT_ID / f"{NATIVE_SOFA}.sofa"
native_fit = fit_from_sofa(native_path)
print(f"native head radius {native_fit['head_radius']:.4f} m, "
      f"residual {native_fit['residual_us']:.1f} us")

native = sm.load_hrtf(NATIVE_SOFA, SUBJECT_ID)
diotic_path = paths.SOFA_DIR / SUBJECT_ID / f"{DIOTIC_SOFA}.sofa"
diotic = save_diotic_sofa(native, diotic_path, ear=DIOTIC_EAR,
                          head_radius=native_fit['head_radius'])
diotic.name = DIOTIC_SOFA

diotic_fit = fit_from_sofa(diotic_path)
print(f"diotic head radius {diotic_fit['head_radius']:.4f} m, "
      f"residual {diotic_fit['residual_us']:.1f} us  "
      f"(delta {1000 * abs(diotic_fit['head_radius'] - native_fit['head_radius']):.2f} mm)")

print(f"cue depth (dB rms, 0.5-2 rip/oct, az 0)")
print(f"{'':10s} {'mon L':>7s} {'mon R':>7s}")
for label, hrtf in (("native", native), ("diotic", diotic)):
    print(f"{label:10s} {sm.cue_depth(hrtf, 'left'):7.2f} {sm.cue_depth(hrtf, 'right'):7.2f}")
print(f"\nISD depth by azimuth")
print(f"{'':10s}" + "".join(f"{a:>8.0f}" for a in (-35, -20, 0, 20, 35)))
for label, hrtf in (("native", native), ("diotic", diotic)):
    print(f"{label:10s}" + "".join(
        f"{sm.isd_depth(hrtf, azimuth=a):8.2f}" for a in (-35, -20, 0, 20, 35)))
print(f"\n-> {diotic_path}")


# %% 2. check the stimulus conditions ----------------------------------------
# Both ripple conditions share rms_tilt; only the in-band component differs.
# The printed ratios are the manipulation check: below-band should leave the
# cue standing clear inside the cue band, in-band should bring the source down
# to roughly the cue's own level there.
_depth = sm.cue_depth(native, DIOTIC_EAR)
for label, settings in (("below-band", BELOW_BAND), ("in-band", IN_BAND)):
    _source = sm.source_rms(n=120, **settings)
    print(f"{label:11s} {settings} -> source {_source:5.2f} dB, "
          f"cue:source {_depth / _source:5.2f}:1")


# %% 3. preflight (rerun anytime) --------------------------------------------
# Checklist before the first block: headphones on and seated, tracker charged,
# OS volume pinned to the level GAIN was matched at, and no block from this
# protocol already on file that you are about to overwrite. Run the block cells
# in the order printed here -- that IS the counterbalance.
set_windows_volume(50)
subject = hr.Subject(SUBJECT_ID)
collect_demographics(subject)
done = [k for k in subject.localization if f"_{PROTOCOL_TAG}-" in k]
print(f"{SUBJECT_ID} (subject {SUBJECT_INDEX}): {len(done)} block(s) on file")
for key in done:
    print("   ", key)
print("\nrun the block cells in this order:")
for position, (isd, band) in enumerate(session_order(), start=1):
    print(f"  {position}. {isd:8s} {band}")


# %% 4. natural / noise ------------------------------------------------------
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "natural", "noise")


# %% 5. natural / below-band ripple ------------------------------------------
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "natural", "below")


# %% 6. natural / in-band ripple ---------------------------------------------
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "natural", "in")


# %% 7. diotic / noise -------------------------------------------------------
# The first diotic block rebuilds the binsim database, which takes a few
# minutes. Expect the sound to be noticeably more in-the-head than the natural
# blocks: both ears now carry one ear's DTF.
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "diotic", "noise")


# %% 8. diotic / below-band ripple -------------------------------------------
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "diotic", "below")


# %% 9. diotic / in-band ripple ----------------------------------------------
subject = hr.Subject(SUBJECT_ID)
run_block(subject, "diotic", "in")


# %% 10. results (rerun anytime) ---------------------------------------------
# Polar error over midline trials only; gain is secondary. The number that
# tests the hypothesis is the INTERACTION on the bottom line: how much more the
# in-band stimulus costs when the interaural spectral difference is gone.
subject = hr.Subject(SUBJECT_ID)
blocks = {}
for key, sequence in subject.localization.items():
    if f"_{PROTOCOL_TAG}-" not in key:
        continue
    blocks[(getattr(sequence, "isd", None), getattr(sequence, "source_band", None))] = sequence

print(f"{'ISD':10s} {'stimulus':11s} {'n':>4s} {'|pol err|':>10s} {'QE %':>6s} "
      f"{'EG':>6s} {'extern':>7s}")
cells = {}
for isd in ("natural", "diotic"):
    for band in ("noise", "below", "in"):
        sequence = blocks.get((isd, band))
        if sequence is None:
            continue
        errors, quadrant_rate = midline_errors(sequence)
        cells[(isd, band)] = errors
        rating = getattr(sequence, "externalization_rating", None)
        print(f"{isd:10s} {band:11s} {len(errors):4d} {errors.mean():10.2f} "
              f"{100 * quadrant_rate:6.1f} {localization_accuracy(sequence)[0]:6.2f} "
              f"{'--' if rating is None else rating:>7}")

_needed = [("natural", "in"), ("natural", "below"), ("diotic", "in"), ("diotic", "below")]
if all(key in cells for key in _needed):
    def in_band_cost(isd):
        return cells[(isd, "in")].mean() - cells[(isd, "below")].mean()

    rng = numpy.random.default_rng(0)
    draws = []
    for _ in range(10000):
        resampled = {key: value[rng.integers(0, len(value), len(value))]
                     for key, value in cells.items()}
        draws.append((resampled[("diotic", "in")].mean() - resampled[("diotic", "below")].mean())
                     - (resampled[("natural", "in")].mean() - resampled[("natural", "below")].mean()))
    low, high = numpy.percentile(draws, [2.5, 97.5])
    print(f"\nin-band cost, natural ISD     : {in_band_cost('natural'):+6.2f} deg")
    print(f"in-band cost, diotic  ISD     : {in_band_cost('diotic'):+6.2f} deg")
    print(f"interaction (diotic - natural): "
          f"{in_band_cost('diotic') - in_band_cost('natural'):+6.2f} deg "
          f"[{low:+.2f}, {high:+.2f}]")
    print("positive and clear of zero -> the interaural spectral difference is "
          "what normally rescues in-band source variation")


# %% 11. within-block learning (rerun anytime) -------------------------------
# Secondary, and a real prediction rather than a nuisance check: if removing the
# binaural channel forces a fall back on priors over the source ensemble, the
# diotic in-band block should IMPROVE across trials while the natural one does
# not. A negative slope is improvement.
print(f"{'ISD':10s} {'stimulus':11s} {'slope':>16s}")
for (isd, band), errors in sorted(cells.items()):
    trial = numpy.arange(len(errors))
    slope = numpy.polyfit(trial, errors, 1)[0] * len(errors)
    print(f"{isd:10s} {band:11s} {slope:+10.2f} deg/block")
