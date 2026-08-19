"""
learning_transfer.py

Adaptation-transfer experiment protocol — DONOR-DETAIL cue manipulation.

This is the current protocol. The previous version, which translated the
participant's own spectral detail along the ERB axis, is kept for reference at
protocols/dev/old/learning_transfer_erbshift.py; design, grid, counterbalancing
and the final-day 2x2 are unchanged from it, only the cue manipulation differs.
Instead of moving the participant's own detail, it is REPLACED with a donor's:


    log|H_modified(direction)| = envelope_4( log|H_own| ) + detail( log|H_donor| )

with the participant's own phase and own per-direction broadband level, so ITD
and broadband ILD are exactly as measured and only the spectral shape above the
envelope scale changes. See hrtf.modify.donor_detail.

WHY NOT THE ERB SHIFT. A constant translation keeps the participant's own
pattern intact, so the existing spectral-to-spatial map can still match each
modified DTF to one of its own templates and read out a coherent — merely
displaced — elevation. Van Wanrooij & Van Opstal (2005) saw exactly that: the
listeners whose molds only shifted the main notch showed a bias with preserved
gain and never adapted over the whole study, while those whose spectral cues
were decorrelated did adapt. Measured here, a 1-ERB shift leaves the
own-vs-modified correlation ridge at slope ~1.0, i.e. fully absorbable as a
bias; a donor's detail collapses it.

DONOR SELECTION is the only thing that varies between participants, and it is
made by a fixed rule in hrtf.analysis.donor_selection — see
DONOR_POOL / TARGET_DISSIMILARITY / TOLERANCE / MAX_RIDGE_SLOPE there, and
docs/methods_donor_detail.md for the paragraph this becomes in a paper.
Everything else (n_keep, band, filter bank, target) is identical for everyone.

DONOR SWAPS. The rule produces a ranked shortlist, not a single name, so a
participant who is at floor with the first donor can be moved to the second
without inventing a criterion on the spot. Stage the alternates before the
session with prepare_donor_shortlist(), swap with use_donor(rank=1, reason=...).
The active donor is recorded in the subject pickle (subject.active_donor), so a
later session reloads the donor the participant was actually trained on and not
whatever the rule ranks first once the pool has grown. Only the current donor is
kept there; the candidate ranking behind the choice is embedded in the composite
SOFA as GLOBAL_ModificationParams.

The monaural ear treatment is orthogonal and selectable via OTHER_EAR
('flat' | 'envelope' | 'native'); which one to use is still an open question,
tested per participant by the ladder in protocols/dev/ladder.py.

Counterbalancing is read from learning_transfer_block_order.csv IN THIS FOLDER.
Write each subject's id into the 'subject' column before running.

RUN ORDER. Cells top to bottom are the protocol proper, in the order they are
performed:
    day 1            status -> native reference -> build donor -> baseline A/D
    adaptation days  PRE test -> train -> POST test   (three cells, in order)
    final day        the counterbalanced 2x2
Everything under MISC at the bottom is diagnostic and is NOT run as a matter of
course -- the externalization ladder, the cepstral-split QC, the n_keep=8 build
and the other-ear probe live there.

The OS master volume is forced to OS_VOLUME (50%) at the start of every
localization block and every training run, because the pybinsim gain was
matched to the dome at that setting. Off Windows this is a logged no-op.

Run cell by cell (# %%) in an IDE/console -- do NOT run top-to-bottom.

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT.
------------------------------------------------------------------------------
"""

SUBJECT_ID = ("AS")

# %% imports and config #------------------------------------------------------
import csv
import os
import subprocess
import sys
import slab
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.hrtf.analysis.vsi import vsi as vsi_of
from hrtf_relearning.hrtf.modify.donor_detail import donor_detail_dtf, modification_params
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf, ENVELOPE_BAND
from hrtf_relearning.hrtf.processing.midline import (midline_arc, expand_from_midline,
                                                     qc_midline, format_qc)
from hrtf_relearning.hrtf.modify.edge_shift import (embed_modification_params,
                                                    read_modification_params)
from hrtf_relearning.experiment.protocols.protocol_helpers import (
    collect_demographics, collect_externalization_rating, externalization_check,
    externalization_ladder)
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.hrtf.modify.plot_compare import plot_ears, plot_split_qc
from hrtf_relearning.utils import paths

CSV_PATH = (hr.PATH / "experiment" / "protocols" / "learning_transfer"
            / "learning_transfer_block_order.csv")


def _load_subject_params(subject_id, csv_path=CSV_PATH):
    """Look up trained_ear and final block order for this subject."""
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject", "").strip() == subject_id:
                ear   = row["trained_ear"].strip()
                order = [c.strip() for c in row["block_order"].split("-")]
                return ear, order
    raise ValueError(
        f"Subject '{subject_id}' not found in the 'subject' column of\n  {csv_path}\n"
        f"Add its id to a row there (replacing an '(assign)' cell) before running.")


TRAINED_EAR, FINAL_ORDER = _load_subject_params(SUBJECT_ID)

NATIVE_SOFA = f"{SUBJECT_ID}"      # individual measured HRTF
DONOR_ID = None                    # filled in by the build cell (or set by hand
                                   # to re-use a previously chosen donor)
MODIFIED_SOFA = None               # <SUBJECT_ID>_donor_<DONOR_ID>, set below

HP = "DT990"

# --- what the non-listening ear gets in monaural blocks ----------------------
# 'flat' | 'envelope' | 'native' -- hrtf.processing.{flatten,envelope,native}
OTHER_EAR = "envelope"
ENV_NKEEP = selection.N_KEEP
PROBE_OTHER_EAR = "envelope"

# --- which build this subject's filters were made with -----------------------
# 'v1'  Donor detail applied to the finished 475-direction SOFA; the monaural
#       reduction applied again at RENDER time by hrtf2binsim. Every build
#       before 2026-08.
# 'v2'  Donor detail AND the monaural reduction applied to the 19 MEASURED
#       az=0 DTFs, then re-expanded through the spherical head model
#       (hrtf.processing.midline). Three things follow: the ITD comes from the
#       model's interaural phase so native and modified sets share it exactly;
#       the envelope is fitted to the pinna response rather than to the pinna
#       response times the head shadow; and it is averaged over elevation, so
#       the untrained ear carries no elevation-dependent spectral structure at
#       all -- measured, v1 left about half of it there.
#
# Subjects already run stay on v1. The change alters the untrained ear, and
# swapping builds mid-cohort would put a discontinuity in the middle of their
# own pre/post comparison, which is worse than the bias itself.
LEGACY_V1_SUBJECTS = ("FS", "GS", "IR", "TS", "PF")
PIPELINE = "v1" if SUBJECT_ID in LEGACY_V1_SUBJECTS else "v2"

# --- shared localization sampling grid (do not change) -----------------------
SECTOR_SIZE        = (7, 14)
ELEVATION_RANGE    = (-35, 35)
TARGETS_PER_SECTOR = 3
MIN_DISTANCE       = 20
GAIN               = 0.2
# Every block in this file inherits STIM, and that is the point: baselines,
# daily tests and the final 2x2 must all be measured with the SAME stimulus or
# the change scores are meaningless. What must never happen is a mixture within
# a subject.
#   !! TS (10.08) and IR (11.08) had their day-1 blocks run with STIM='uso'
#      because this was left set to 'uso'. See docs/stimulus_spectral_variation.md.
#
# 'ripple' since 2026-08-17. Plain noise has essentially no across-trial source
# variation (0.4 dB SD at 1/6 oct against a ~3.3 dB elevation cue), so the sound
# at the eardrum carries a fixed map from absolute spectrum to elevation and the
# task can be solved by template matching, without ever separating source from
# filter. That cannot distinguish a spectral-to-spatial recalibration from a
# learned timbre->elevation lookup, which is what FS reported doing. The test
# stimulus therefore varies its source spectrum on every trial, in EVERY block,
# so the measure is source-invariant throughout rather than only on the last day.
# Training stays on noise (SOUND_FILE=None -> pink noise).
#
# Subjects run before this date were tested on noise. They are pilots and are
# not pooled with what follows.
STIM               = "ripple"      # -> "ripple" once the depth is settled, below
# Envelope parameters for STIM='ripple'. Empty dict = inherit the defaults in
# localization_helpers.stimulus (the single source of truth); set rms_tilt here
# only to override for a specific cohort, and it is recorded per block in
# sequence.stim_settings either way.
#   !! Depth is NOT yet settled: the current default (8 dB rms, 27 dB median
#      peak-to-trough) sits above the ~20 dB depth at which Macpherson &
#      Middlebrooks report localization degradation. Run the free-field check
#     nm and the AR self-test
#      (cells 7-9) BEFORE flipping STIM to 'ripple' for a participant.
STIM_SETTINGS      = {'rms_tilt': 3}
MIDLINE_TOL        = 1.0
FULL_FIELD = (-35, 35)

if TRAINED_EAR == "left":
    UNTRAINED_EAR, TRAINED_HEMI, MIRRORED_HEMI = "right", (-35, 0), (0, 35)
elif TRAINED_EAR == "right":
    UNTRAINED_EAR, TRAINED_HEMI, MIRRORED_HEMI = "left", (0, 35), (-35, 0)
else:
    raise ValueError("TRAINED_EAR must be 'left' or 'right'.")


# The ladder compares two composite strengths, so more than one modified SOFA
# can exist per subject. n_keep=N_KEEP (4) keeps the plain name; anything else
# gets an _n<k> suffix, so the training/testing SOFA is never ambiguous.
LADDER_N_KEEP = (4, 8)


def _modified_name(donor_id, n_keep=None):
    """The SOFA the protocol's blocks load.

    On v2 the monaural reduction is already inside the file, so it is part of
    the name -- there is no longer a render-time switch that could be set
    differently between two runs of the same block.
    """
    n_keep = selection.N_KEEP if n_keep is None else n_keep
    stem = f"{SUBJECT_ID}_donor_{donor_id.split('/')[-1]}"
    if n_keep != selection.N_KEEP:
        stem = f"{stem}_n{n_keep}"
    if PIPELINE == "v2":
        stem = f"{stem}_env{ENV_NKEEP}_{TRAINED_EAR}"
    return stem


def _binaural_name(donor_id, n_keep=None):
    """v2 only: the composite BEFORE the monaural reduction. QC reference."""
    n_keep = selection.N_KEEP if n_keep is None else n_keep
    stem = f"{SUBJECT_ID}_donor_{donor_id.split('/')[-1]}"
    return stem if n_keep == selection.N_KEEP else f"{stem}_n{n_keep}"


def hrir_settings(sofa_name, ear=None, mirror=False, other_ear=None):
    # v2 bakes the monaural reduction into the SOFA, so hrtf2binsim must NOT
    # apply it a second time -- ear=None means "take the file as it is".
    # `mirror` still happens at render time: it is a channel/source swap and
    # commutes with everything above it, so blocks C and D come off this file.
    baked = PIPELINE == "v2" and f"_env{ENV_NKEEP}_" in sofa_name
    return {
        "name": sofa_name,
        "subject_id": SUBJECT_ID,
        "ear": None if baked else ear,
        "other_ear": OTHER_EAR if other_ear is None else other_ear,
        # pins the v1 render-time envelope to its historical behaviour; unused
        # when `baked`. See hrtf2binsim and hrtf.processing.envelope.
        "env_elevation_average": False,
        "env_n_keep": ENV_NKEEP,
        "native_sofa": NATIVE_SOFA,
        "mirror": mirror,
        "reverb": True,
        "drr": 20,
        "hp_filter": True,
        "hp": HP,
        "convolution": "cpu",
        "storage": "cpu",
    }


def loc_settings(azimuth_range, exclude_midline=False):
    return {
        "kind": "sectors",
        "azimuth_range": azimuth_range,
        "elevation_range": ELEVATION_RANGE,
        "targets_per_speaker": 3,
        "targets_per_sector": TARGETS_PER_SECTOR,
        "min_distance": MIN_DISTANCE,
        "gain": GAIN,
        "stim": STIM,
        "stim_settings": STIM_SETTINGS,
        "sector_size": SECTOR_SIZE,
        "replace": False,
        "exclude_midline": exclude_midline,
        "midline_tol": MIDLINE_TOL,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

_SHORTLIST = None      # cached rows from selection.shortlist(), rule order


def donor_shortlist(refresh=False, quiet=False):
    """This subject's candidate donors in rule order — cached.

    ``[0]`` is the protocol pick, ``[1]`` and ``[2]`` are the alternates
    ``use_donor(rank=...)`` swaps to. Scoring the pool takes ~20 s, so it is
    computed once and reused; pass ``refresh=True`` after editing
    selection.DONOR_POOL.
    """
    global _SHORTLIST
    if _SHORTLIST is not None and not refresh:
        return _SHORTLIST

    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    own = slab.HRTF(str(sofa_dir / f"{NATIVE_SOFA}.sofa"))
    own.name = SUBJECT_ID
    candidates = selection.load_candidates(SUBJECT_ID)
    if not candidates:
        raise RuntimeError("empty donor pool — check selection.DONOR_POOL")
    if not quiet:
        print(f"donor pool ({len(candidates)}): {', '.join(candidates)}")

    _SHORTLIST = selection.shortlist(own, candidates)
    if not quiet:
        reference, _ = selection.pairwise_reference({SUBJECT_ID: own, **candidates})
        selection.report(_SHORTLIST, reference)
    return _SHORTLIST


def _set_active_donor(donor_id, rank, reason=""):
    """Record the donor this participant is currently on, in their pickle.

    Stored as subject.active_donor so a later session can reload the donor the
    participant was actually trained on rather than whatever the selection rule
    ranks first today (the pool grows as subjects are added, so rank 0 is not
    stable over time). Overwritten on each change — only the current donor is
    kept; the candidate ranking behind the choice lives in the composite SOFA
    as GLOBAL_ModificationParams.
    """
    import datetime
    subject = hr.Subject(SUBJECT_ID)
    subject.active_donor = {
        "donor": donor_id,
        "rank": rank,
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    subject.write()
    return subject.active_donor


def _save_qc_figure(own, modified, name, chosen, n_keep, donor_id,
                    show=True, quiet=False, overwrite=False):
    """Write plots/acoustic/<name>.png — the before/after 2x2.

    ALWAYS saved, `show` only decides whether it is displayed. It is provenance
    in the same sense as the ranking CSV: the picture of what this participant
    was actually given. Tying the save to a display flag is what left AS with
    six donor SOFAs and no figures — staging builds pass show_qc=False, and by
    the time the day-1 cell ran the files existed, so the build returned early
    and never reached the plot.

    ``modified`` may be a path, so this also works on the already-built path
    where nothing is in memory.
    """
    plot_dir = paths.subject_acoustic_dir(SUBJECT_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / f"{name}.png"
    if out_png.exists() and not overwrite and not show:
        return out_png
    if not isinstance(modified, slab.HRTF):
        modified = slab.HRTF(str(modified))
        modified.name = name
    fig = plot_ears(own, modified, vsi_dis=chosen["vsi_dissimilarity"],
                    vsi_bw=selection.DEFAULT_BAND, band=selection.DEFAULT_BAND,
                    suptitle=f"{SUBJECT_ID}  own envelope (n_keep={n_keep}) "
                             f"+ {donor_id} detail", show=show)
    fig.savefig(out_png, bbox_inches="tight")
    if not show:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    if not quiet:
        print(f"QC figure: {out_png}")
    return out_png


def build_donor_sofa(overwrite=False, show_qc=True, n_keep=None, rank=0,
                     donor_id=None, set_active=True, quiet=False):
    """Build one composite, write it, return (path, donor_id).

    Writes <SUBJECT_ID>_donor_<DONOR>.sofa next to the native one, embeds the
    full selection record as GLOBAL_ModificationParams, and saves the candidate
    ranking as a CSV for the supplement.

    ``rank`` picks which entry of :func:`donor_shortlist` to build — 0 is the
    protocol choice, 1 and 2 are the alternates. ``donor_id`` overrides the rank
    and names a donor directly (use only to rebuild something already run).
    ``set_active=False`` builds without repointing the protocol, which is what
    :func:`prepare_donor_shortlist` uses to stage the alternates.
    """
    global DONOR_ID, MODIFIED_SOFA

    n_keep = selection.N_KEEP if n_keep is None else n_keep
    is_default = n_keep == selection.N_KEEP
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    own = slab.HRTF(str(sofa_dir / f"{NATIVE_SOFA}.sofa"))
    own.name = SUBJECT_ID

    candidates = selection.load_candidates(SUBJECT_ID)
    if not candidates:
        raise RuntimeError("empty donor pool — check selection.DONOR_POOL")

    rows = donor_shortlist(quiet=quiet)
    if donor_id is not None:
        matches = [row for row in rows if row["donor"] == donor_id]
        if not matches:
            raise ValueError(f"{donor_id} is not in this subject's pool: "
                             f"{', '.join(r['donor'] for r in rows)}")
        chosen = matches[0]
    else:
        if rank >= len(rows):
            raise IndexError(f"requested rank {rank} but only {len(rows)} "
                             f"candidates were scored")
        chosen = rows[rank]

    if not quiet:
        print(f"\ndonor rank {chosen['rank']}: {chosen['donor']}   VSI dissimilarity "
              f"{chosen['vsi_dissimilarity']:.3f} (target "
              f"{selection.TARGET_DISSIMILARITY:.2f} ± {selection.TOLERANCE:.2f})"
              f"   ridge slope {chosen['ridge_slope']:+.2f}"
              f"   cue strength {chosen['donor_strength']:.1f} dB"
              f"   [{chosen['tier']}]")
    if chosen["fallback"]:
        print("  !! FALLBACK: no candidate met the ridge criterion; lowest slope "
              "used. This must be reported.")
    elif chosen["tier"] == "widened" and not quiet:
        print(f"  !  no candidate landed within {selection.TOLERANCE:.2f} of the "
              f"target; nearest-to-target used. Report the tier.")

    # The donor is always selected at the protocol n_keep, so the two ladder
    # strengths differ ONLY in how much of the cue is handed over -- not in
    # whose cue it is.
    donor_id = chosen["donor"]
    name = _modified_name(donor_id, n_keep)
    if is_default and set_active:
        DONOR_ID, MODIFIED_SOFA = donor_id, name
        _set_active_donor(donor_id, chosen["rank"], chosen["tier"])
    out_path = sofa_dir / f"{name}.sofa"
    if out_path.exists() and not overwrite:
        if not quiet:
            print(f"{out_path.name} already exists (overwrite=False) -- skipping build")
        # self-healing: the SOFA may have been written by a staging build that
        # never drew the figure. Re-read it rather than leave the record short.
        _save_qc_figure(own, out_path, name, chosen, n_keep, donor_id,
                        show=show_qc, quiet=quiet)
        return out_path, donor_id

    # extra provenance for whatever this build did beyond the donor step. The
    # file name says _env4_<ear>; the embedded record has to say the same, or
    # it is another SOFA nobody can trace (see FD 12:13).
    pipeline_params = {}
    binaural_path = None

    if PIPELINE == "v1":
        modified = donor_detail_dtf(own, candidates[donor_id], n_keep=n_keep)
    else:
        # --- v2: modify the 19 MEASURED directions, then re-expand -----------
        # The az=0 arc in the finished SOFA is magnitude-identical to what went
        # into the expansion (step 2 skips az=0), so it is read back from there
        # rather than replayed from the npz -- no reference recording needed.
        own_arc = midline_arc(own)
        donor_arc = midline_arc(candidates[donor_id])
        arc = donor_detail_dtf(own_arc, donor_arc, n_keep=n_keep)
        # binaural composite, kept as the QC reference and for the ladder
        binaural = expand_from_midline(arc)
        binaural.name = _binaural_name(donor_id, n_keep)
        binaural_path = sofa_dir / f"{binaural.name}.sofa"
        binaural.write_sofa(str(binaural_path))

        arc = envelope_dtf(arc, ear=TRAINED_EAR, n_keep=ENV_NKEEP)
        report = qc_midline(own_arc, arc, processed_ear=UNTRAINED_EAR,
                            raise_on_fail=True)
        print(format_qc(report))
        modified = expand_from_midline(arc)

        pipeline_params = {
            "pipeline": "v2",
            "chain": ("midline_arc -> donor_detail_dtf -> envelope_dtf -> "
                      "qc_midline -> expand_azimuths_with_binaural_cues"),
            "expansion": {"itd_method": "phase", "az_range": [-50, 50]},
            "monaural": {"other_ear": "envelope", "ear_kept": TRAINED_EAR,
                         "ear_processed": UNTRAINED_EAR, "env_n_keep": ENV_NKEEP,
                         "env_band_hz": list(ENVELOPE_BAND),
                         "elevation_average": True},
            "midline_qc": {k: v for k, v in report.items()},
        }

    modified.name = name
    own_vsi, modified_vsi = vsi_of(own), vsi_of(modified)
    print(f"VSI  own={own_vsi:.3f}  modified={modified_vsi:.3f}  "
          f"(diagnostic only -- not diffuse-field normalised, see vsi.py)")

    def _params(**extra):
        return modification_params(
            SUBJECT_ID, donor_id, n_keep=n_keep,
            target_dissimilarity=selection.TARGET_DISSIMILARITY,
            band=selection.DEFAULT_BAND, resolution=selection.DEFAULT_RESOLUTION,
            max_ridge_slope=selection.MAX_RIDGE_SLOPE, pool=list(candidates),
            fallback=chosen["fallback"],
            scores={k: chosen[k] for k in ('vsi_dissimilarity', 'vsi', 'own_vsi',
                                           'i_sim', 'peak_r', 'ridge_slope')},
            ranking=[{k: row[k] for k in ('donor', 'rank', 'tier', 'donor_strength',
                                          'vsi_dissimilarity', 'vsi', 'ridge_slope',
                                          'eligible', 'in_band')} for row in rows],
            **extra)

    modified.write_sofa(str(out_path))
    embed_modification_params(out_path, _params(**pipeline_params))
    print(f"wrote {out_path}")

    # the binaural composite is a MODIFIED file too -- label it, or it is an
    # unattributable SOFA sitting next to the native one
    if binaural_path is not None:
        embed_modification_params(binaural_path, _params(
            **{**pipeline_params, "monaural": None,
               "note": "binaural composite, before the monaural reduction; "
                       "QC reference for the _env file next to it"}))
        print(f"wrote {binaural_path}")

    _save_qc_figure(own, modified, name, chosen, n_keep, donor_id,
                    show=show_qc, quiet=quiet)
    return out_path, donor_id


def _binsim_names(sofa_name, mirror):
    """The binsim database directory name hrtf2binsim would produce."""
    name = sofa_name
    # v2 carries the reduction in the SOFA name already, so hrtf2binsim adds
    # nothing here -- see hrir_settings, which passes ear=None for those files.
    baked = PIPELINE == "v2" and f"_env{ENV_NKEEP}_" in sofa_name
    if TRAINED_EAR and not baked:
        if OTHER_EAR == "flat":
            name += f"_{TRAINED_EAR}"
        elif OTHER_EAR == "envelope":
            name += f"_{TRAINED_EAR}_env{ENV_NKEEP}"
        elif OTHER_EAR == "native":
            name += f"_{TRAINED_EAR}_nat"
    if mirror:
        name += "_mirrored"
    return name


def prepare_donor_shortlist(n=3, mirrored=True, overwrite=False):
    """Stage the top ``n`` donors so a mid-session swap costs seconds.

    Builds, for each of the first ``n`` entries of :func:`donor_shortlist`, the
    composite SOFA and the pyBinSim filter databases the protocol actually
    plays: trained ear + OTHER_EAR, un-mirrored (day-1 baseline A, the daily
    test, training) and mirrored (baseline D and the final C/D blocks).

    RUN THIS BEFORE THE PARTICIPANT ARRIVES. write_filters passes over all 475
    directions, so a database is minutes, not seconds; doing it with someone in
    the chair is what makes a donor swap expensive. Once staged,
    :func:`use_donor` is instant.

    Only the rank-0 donor is left active — the alternates are built but not
    selected, so nothing about the default path changes.
    """
    rows = donor_shortlist()
    n = min(n, len(rows))
    print(f"\nstaging {n} donors for {SUBJECT_ID}: "
          f"{', '.join(r['donor'] for r in rows[:n])}")

    from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim

    staged = []
    for row in rows[:n]:
        donor = row["donor"]
        print(f"\n--- rank {row['rank']}: {donor} "
              f"(VSI-dis {row['vsi_dissimilarity']:.3f}, ridge "
              f"{row['ridge_slope']:+.2f}, strength {row['donor_strength']:.1f} dB, "
              f"{row['tier']}) ---")
        # set_active=False: staging must not silently repoint the protocol
        build_donor_sofa(overwrite=overwrite, show_qc=False,
                         donor_id=donor, set_active=False, quiet=True)
        name = _modified_name(donor)
        for mirror in ((False, True) if mirrored else (False,)):
            db = paths.BINSIM_DIR / _binsim_names(name, mirror)
            if db.exists() and not overwrite:
                print(f"    binsim {db.name} exists — skipping")
                continue
            print(f"    building binsim {db.name} ...")
            hrtf2binsim(hrir_settings(name, ear=TRAINED_EAR, mirror=mirror),
                        overwrite=overwrite, build=True)
        staged.append(donor)

    # rank 0 is the protocol donor; make it the active one
    build_donor_sofa(overwrite=False, show_qc=False, rank=0, quiet=True)
    print(f"\nstaged: {', '.join(staged)}")
    print(f"active: {DONOR_ID}  ({MODIFIED_SOFA})")
    print(f"swap with  use_donor(rank=1, reason='...')")
    print(f"once the donor is settled: discard_unused_donors()")
    return staged


def discard_unused_donors(dry_run=True, keep=None):
    """Delete every staged donor for this subject except the active one.

    Staging costs ~4 MB of SOFA plus two pyBinSim databases per candidate, and
    after the day-1 baselines the choice is made — the alternates are dead
    weight that also makes it ambiguous, months later, which composite the
    participant actually heard. Run this once the donor is settled.

    Keeps the active donor's SOFA and databases, and never touches the native
    <SUBJECT_ID>.sofa. The discarded ones are reproducible at any time from
    build_donor_sofa(donor_id=...), so nothing unrecoverable is lost — the
    selection record stays in the surviving SOFA's GLOBAL_ModificationParams
    and in subject.active_donor.

    Defaults to ``dry_run=True``: it prints what it would remove and removes
    nothing. Call again with ``dry_run=False`` to actually delete. Pass
    ``keep=['XX']`` to spare extra donors (e.g. one you still mean to compare).
    """
    import shutil

    active = DONOR_ID or _last_active_donor()
    if active is None:
        raise RuntimeError(
            "no active donor for this subject — run build_donor_sofa() or "
            "load_existing_donor() first, or this would delete everything.")

    # Match on the exact composite name, not on the donor id appearing in the
    # stem: '<SID>_donor_AS' is a PREFIX of '<SID>_donor_AS_n8', so a substring
    # or startswith test would silently spare the n_keep=8 ladder build too.
    # That build is a diagnostic, not what the participant was trained on, so
    # it goes with the rest unless named in `keep`.
    # The binsim directory name depends on OTHER_EAR, which may have been a
    # different setting when a database was built than it is in the config
    # block today. Spare every variant the active composite could have
    # produced, so a since-changed setting can never make this delete the
    # database the participant is actually being tested on.
    global OTHER_EAR
    spared_sofa, spared_binsim = set(), set()
    for donor in {active, *(keep or [])}:
        name = _modified_name(donor)
        # On v2 a donor has TWO files: the reduced one the blocks load
        # (_modified_name, '<SID>_donor_<D>_env4_<ear>') and the binaural
        # composite kept as its QC reference (_binaural_name, '<SID>_donor_<D>').
        # Sparing only the first would delete the QC reference for the donor
        # that was actually used. On v1 the two names coincide.
        spared_sofa.update((name, _binaural_name(donor)))
        for other_ear in ("flat", "envelope", "native"):
            OTHER_EAR, current = other_ear, OTHER_EAR
            try:
                spared_binsim.update(_binsim_names(name, m) for m in (False, True))
            finally:
                OTHER_EAR = current

    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    victims = [p for p in sorted(sofa_dir.glob(f"{SUBJECT_ID}_donor_*.sofa"))
               if p.stem not in spared_sofa]
    victims += [p for p in sorted(paths.BINSIM_DIR.glob(f"{SUBJECT_ID}_donor_*"))
                if p.is_dir() and p.name not in spared_binsim]

    if not victims:
        print(f"{SUBJECT_ID}: nothing to discard (active donor {active})")
        return []

    total = sum(f.stat().st_size for v in victims
                for f in ([v] if v.is_file() else v.rglob("*")) if f.is_file())
    print(f"{SUBJECT_ID}: keeping {', '.join(sorted(spared))}; "
          f"{'would remove' if dry_run else 'removing'} {len(victims)} item(s), "
          f"{total / 1e6:.1f} MB")
    for v in victims:
        print(f"    {'[dry-run] ' if dry_run else ''}{v.relative_to(paths.DATA_DIR)}")
        if not dry_run:
            shutil.rmtree(v) if v.is_dir() else v.unlink()
    if dry_run:
        print("  nothing deleted — re-run with dry_run=False to apply")
    return victims


def use_donor(rank=None, donor_id=None, reason=""):
    """Switch the active donor mid-session. Instant if pre-staged.

    Use when a participant cannot localize at all with the current composite —
    a floor-level block is uninformative, and the design needs an acute
    degradation, not an abolished cue. ``rank=1`` moves to the next donor on
    the rule's list.

    ``reason`` is stored on subject.active_donor and should say what was observed
    ('EG 0.02 on baseline A, at floor'), because a swap made after seeing the
    data has to be reportable as such. It is NOT free: the discarded block was
    still run, and both the discarded and the replacement donor belong in the
    participant's record.

    Refuses to swap if the replacement's binsim database has not been built,
    rather than silently starting a multi-minute build with someone in the rig
    — pass ``build=True`` to override that by hand if you really want to wait.
    """
    global DONOR_ID, MODIFIED_SOFA

    rows = donor_shortlist(quiet=True)
    if donor_id is not None:
        matches = [row for row in rows if row["donor"] == donor_id]
        if not matches:
            raise ValueError(f"{donor_id} is not in this subject's pool")
        row = matches[0]
    elif rank is not None:
        if rank >= len(rows):
            raise IndexError(f"rank {rank} but only {len(rows)} candidates")
        row = rows[rank]
    else:
        raise ValueError("pass rank= or donor_id=")

    previous = DONOR_ID
    name = _modified_name(row["donor"])
    sofa_path = paths.SOFA_DIR / SUBJECT_ID / f"{name}.sofa"
    if not sofa_path.exists():
        raise FileNotFoundError(
            f"{sofa_path.name} has not been built — run "
            f"prepare_donor_shortlist() before the session, or "
            f"build_donor_sofa(rank={row['rank']}) now (slow).")

    missing = [_binsim_names(name, m) for m in (False, True)
               if not (paths.BINSIM_DIR / _binsim_names(name, m)).exists()]
    if missing:
        print(f"  [!] binsim database(s) not staged: {', '.join(missing)}")
        print(f"      building now — this takes minutes. Ctrl-C and run "
              f"prepare_donor_shortlist() before the next session.")
        from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
        for mirror in (False, True):
            if _binsim_names(name, mirror) in missing:
                hrtf2binsim(hrir_settings(name, ear=TRAINED_EAR, mirror=mirror),
                            overwrite=False, build=True)

    DONOR_ID, MODIFIED_SOFA = row["donor"], name
    _set_active_donor(row["donor"], row["rank"],
                      reason or f"(no reason given; from {previous})")
    print(f"\nactive donor: {previous} -> {row['donor']}   (rank {row['rank']}, "
          f"{row['tier']}, VSI-dis {row['vsi_dissimilarity']:.3f}, ridge "
          f"{row['ridge_slope']:+.2f}, strength {row['donor_strength']:.1f} dB)")
    print(f"MODIFIED_SOFA = {MODIFIED_SOFA}")
    print(f"  !! set DONOR_ID = '{row['donor']}' in the config block at the top "
          f"so later sessions reload THIS donor, not the rank-0 one.")
    if not reason:
        print("  !! no reason recorded — re-run use_donor(..., reason=...) "
              "to say what was observed.")
    return row


def _last_active_donor():
    """The donor this participant is currently on, from their pickle."""
    return (hr.Subject(SUBJECT_ID).active_donor or {}).get("donor")


def show_donor_log():
    """The donor this participant is on. Read before analysis."""
    record = hr.Subject(SUBJECT_ID).active_donor or {}
    if not record:
        print(f"no donor recorded for {SUBJECT_ID}")
        return
    print(f"{SUBJECT_ID}: donor {record.get('donor')} (rank {record.get('rank')}) "
          f"set {record.get('timestamp')}")
    if record.get("reason"):
        print(f"  reason: {record['reason']}")


def load_existing_donor():
    """Point the protocol at this subject's already-built donor SOFA.

    Recovers the donor from disk rather than from a variable that resets every
    time the config cell is re-run, which is how a subject could silently end up
    trained on one stimulus and tested on another. The donor id is read back
    from the SOFA's embedded modification params, so what is loaded is what was
    built. Refuses to guess if there is more than one candidate file.
    """
    global DONOR_ID, MODIFIED_SOFA

    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    if DONOR_ID is not None:
        matches = [sofa_dir / f"{_modified_name(DONOR_ID)}.sofa"]
    else:
        # The ladder writes extra strengths as <name>_n<k>.sofa; those are
        # diagnostics, never the training/testing stimulus, so they must not
        # make this look ambiguous.
        import re
        matches = [path for path in sorted(sofa_dir.glob(f"{SUBJECT_ID}_donor_*.sofa"))
                   if not re.search(r"_n\d+$", path.stem)]

    if not matches:
        raise FileNotFoundError(
            f"no {SUBJECT_ID}_donor_*.sofa in {sofa_dir} — run build_donor_sofa() first")
    if len(matches) > 1:
        # prepare_donor_shortlist() stages alternates, so several composites
        # existing is now NORMAL and must not be read as ambiguity. The donor
        # log says which one was last made active; that is the authority.
        active = _last_active_donor()
        if active is None:
            raise RuntimeError(
                f"several donor SOFAs for {SUBJECT_ID}: "
                f"{', '.join(p.name for p in matches)}, and no donor log to say "
                f"which is active. Set DONOR_ID in the config block to the one "
                f"this subject was trained on.")
        wanted = sofa_dir / f"{_modified_name(active)}.sofa"
        if wanted not in matches:
            raise RuntimeError(
                f"donor log says the active donor is {active} but "
                f"{wanted.name} is not on disk")
        print(f"  {len(matches)} composites on disk (staged alternates); donor "
              f"log says {active} is active")
        matches = [wanted]

    path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_donor_sofa() first")

    params = read_modification_params(path) or {}
    embedded = params.get("donor_id")
    if embedded is None:
        print(f"  [warn] {path.name} carries no modification params — donor id "
              f"taken from the filename, which is not authoritative")
        embedded = path.stem.replace(f"{SUBJECT_ID}_donor_", "")
    DONOR_ID = embedded
    MODIFIED_SOFA = path.stem

    scores = params.get("scores", {})
    print(f"using {path.name}   donor={DONOR_ID}"
          + (f"   VSI-dis {scores['vsi_dissimilarity']:.3f}, ridge "
             f"{scores['ridge_slope']:+.2f}" if scores else "")
          + ("   [built with FALLBACK donor]" if params.get("fallback") else ""))
    return path


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phases():
    """Built on demand so MODIFIED_SOFA is picked up after the build cell."""
    return {
        "native":     ("Native reference",         "Day 1", NATIVE_SOFA,   None,        False, FULL_FIELD,    "binaural, native HRTF, full field"),
        "baseline_A": ("Baseline A: trained/same", "Day 1", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "naive trained ear, modified filter (matches final A)"),
        "baseline_D": ("Baseline D: untrnd/mirr",  "Day 1", MODIFIED_SOFA, TRAINED_EAR, True,  MIRRORED_HEMI, "naive untrained ear, MIRRORED modified filter (matches final D)"),
        "daily":      ("Daily training test",      "Adaptation days", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "monaural trained ear, trained hemifield"),
        "A":          ("Final A: trained/same",    "Final day", MODIFIED_SOFA, TRAINED_EAR, False, TRAINED_HEMI,  "trained ear, same locations (baseline retest)"),
        "B":          ("Final B: trained/mirr",    "Final day", MODIFIED_SOFA, TRAINED_EAR, False, MIRRORED_HEMI, "trained ear, mirrored locations"),
        "C":          ("Final C: untrnd/same",     "Final day", MODIFIED_SOFA, TRAINED_EAR, True,  TRAINED_HEMI,  "untrained ear (mirrored modified HRIR), same locations"),
        "D":          ("Final D: untrnd/mirr",     "Final day", MODIFIED_SOFA, TRAINED_EAR, True,  MIRRORED_HEMI, "untrained ear (mirrored modified HRIR), mirrored locations  [MAIN]"),
    }


def _n_sectors(rng, size):
    import numpy
    return len(numpy.arange(rng[0] + size / 2, rng[1], size))


def _est_trials(az_range):
    return (_n_sectors(az_range, SECTOR_SIZE[0])
            * _n_sectors(ELEVATION_RANGE, SECTOR_SIZE[1]) * TARGETS_PER_SECTOR)


def _describe(key):
    label, when, sofa, ear, mirror, az, desc = phases()[key]
    midline = "excluded" if tuple(az) != tuple(FULL_FIELD) else "kept"
    other = f"{OTHER_EAR}" if ear else "-"
    return (f"{label}  [{when}]\n      {desc}\n"
            f"      SOFA={sofa}  ear={ear or 'binaural'}  other ear={other}  mirror={mirror}\n"
            f"      azimuth={az}  elevation={ELEVATION_RANGE}  sector={SECTOR_SIZE}  "
            f"tps={TARGETS_PER_SECTOR}  midline az=0 {midline}\n"
            f"      ~{_est_trials(az)} trials")


OS_VOLUME = 50   # Windows master slider, %. The pybinsim gain was matched to the
                 # dome at this setting (match_ar_dome_loudness.py), so every
                 # localization test and training run forces it before starting;
                 # a moved slider silently invalidates the presentation level.


def _fix_output_level():
    """Force the OS master volume to OS_VOLUME. No-op off Windows."""
    if not set_windows_volume(OS_VOLUME):
        print(f"  [!] OS volume NOT set programmatically — check the slider is "
              f"at {OS_VOLUME}% before continuing")


def run_phase(key, subject, other_ear=None):
    label, when, sofa, ear, mirror, az, desc = phases()[key]
    if sofa is None:
        raise RuntimeError("MODIFIED_SOFA is not set — run build_donor_sofa() "
                           "or load_existing_donor() first")
    print("\n" + "=" * 70)
    print(f"RUNNING:  {_describe(key)}"
          + (f"\n      OTHER EAR OVERRIDE: {other_ear}" if other_ear else ""))
    print("=" * 70)
    _fix_output_level()
    one_sided = tuple(az) != tuple(FULL_FIELD)
    test = Localization(subject,
                        hrir_settings(sofa, ear=ear, mirror=mirror, other_ear=other_ear),
                        loc_settings=loc_settings(az, exclude_midline=one_sided))
    test.run()
    print(f"Done: {test.filename}")
    return test


def run_anchor(subject):
    """Short native-HRTF block + rating -- the top of the 0-10 scale.

    'A real external loudspeaker' is not a sound the participant has heard over
    these headphones, so an unanchored rating is a number about their
    imagination. This is the participant's OWN unmodified HRTF played
    binaurally: the ceiling of the whole delivery chain, and the closest thing
    to a 10 the setup can produce. ~10 trials, about a minute.

    Run it once per day, before that day's first test, so ratings are comparable
    within a day AND across days -- a rating of 6 on day 1 and 6 on day 4 only
    means the same thing if the anchor also came out the same. If the anchor
    itself drifts, that is a delivery-chain problem (headphone seat, HP filter,
    OS volume), not adaptation, and it is worth catching before the day's data.
    """
    settings = loc_settings(FULL_FIELD)
    settings.update(sector_size=(14, 14), targets_per_sector=1)   # ~10 trials
    return externalization_check(subject, hrir_settings(NATIVE_SOFA, ear=None),
                                 settings, label=f"{SUBJECT_ID} anchor (own HRTF, binaural)")


def ladder_settings(rung):
    """(hrir_settings, loc_settings) for one rung of the externalization ladder.

    Coarse grid on purpose (~10 trials): these blocks are for the rating, not
    for elevation-gain statistics. 'anchor' is the participant's own unmodified
    HRTF played binaurally — the ceiling of the whole delivery chain, which is
    what gives the 0-10 scale a top.
    """
    settings = loc_settings(TRAINED_HEMI, exclude_midline=True)
    settings.update(sector_size=(14, 14), targets_per_sector=1)   # ~10 trials
    if rung == "anchor":
        return hrir_settings(NATIVE_SOFA, ear=None), settings
    if rung.startswith("donor_n"):
        # composite STRENGTH: same donor, same other-ear treatment, only n_keep
        # differs. Lower n_keep hands over more of the cue.
        n_keep = int(rung.split("_n")[1])
        return (hrir_settings(_modified_name(DONOR_ID, n_keep), ear=TRAINED_EAR),
                settings)
    # ear TREATMENT: n_keep=4 composite on the trained ear, other ear varies
    return hrir_settings(MODIFIED_SOFA, ear=TRAINED_EAR, other_ear=rung), settings


TRAINING_SCRIPT = hr.PATH / "experiment" / "training" / "Training_AR.py"


def run_training(hrir_name=None, ear=None, az_range=None):
    """Launch Training_AR.py with this subject's modified HRIR and ear settings."""
    hrir_name = MODIFIED_SOFA if hrir_name is None else hrir_name
    if hrir_name is None:
        raise RuntimeError("MODIFIED_SOFA is not set — build or load the donor SOFA first")
    ear = TRAINED_EAR if ear is None else ear
    az_range = TRAINED_HEMI if az_range is None else az_range

    sofa_path = paths.SOFA_DIR / SUBJECT_ID / f"{hrir_name}.sofa"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Modified HRIR not found:\n  {sofa_path}")

    print("-" * 64)
    print(f"TRAINING   subject={SUBJECT_ID}   ear={ear}   az_range={az_range}")
    print(f"           HRIR={hrir_name}.sofa   HP={HP}")
    print(f"           other ear={OTHER_EAR} (n_keep={ENV_NKEEP})")
    print("-" * 64)
    _fix_output_level()

    env = dict(os.environ,
               TRAINING_SUBJECT_ID=SUBJECT_ID,
               TRAINING_HRIR_NAME=hrir_name,
               TRAINING_EAR=ear,
               TRAINING_OTHER_EAR=OTHER_EAR,
               TRAINING_ENV_NKEEP=str(ENV_NKEEP),
               TRAINING_NATIVE_SOFA=NATIVE_SOFA,
               TRAINING_AZ_RANGE=f"{az_range[0]},{az_range[1]}",
               TRAINING_HP=HP)
    subprocess.run(
        [sys.executable, "-m", "hrtf_relearning.experiment.training.Training_AR"],
        env=env, cwd=str(hr.PATH.parent), check=False)


def show_status(subject):
    print("\n" + "-" * 70)
    print(f"SUBJECT: {SUBJECT_ID}    TRAINED EAR: {TRAINED_EAR}    UNTRAINED: {UNTRAINED_EAR}")
    print(f"manipulation: donor detail (n_keep={selection.N_KEEP}), donor="
          f"{DONOR_ID or '(not selected yet)'}   other ear={OTHER_EAR}")
    print(f"hemifields -> trained {TRAINED_HEMI}, mirrored {MIRRORED_HEMI}")
    print(f"modified SOFA: {MODIFIED_SOFA}    final-day order: {'-'.join(FINAL_ORDER)}")
    done = list(getattr(subject, "localization", {}).keys())
    if done:
        print(f"\nLocalization runs on file for {SUBJECT_ID} ({len(done)}):")
        for k in done:
            print(f"   - {k}")
    else:
        print(f"\nNo localization runs on file yet for {SUBJECT_ID}.")
    print("-" * 70)







# %% status check (rerun anytime) --------------------------------------------
subject = hr.Subject(SUBJECT_ID)
collect_demographics(subject)      # once per participant; skipped if on file
show_status(subject)

# %% day 1: native reference (original HRIR, full field) ----------------------
# Doubles as the first anchor: this is the best the chain can sound, so its
# rating defines the top of the 0-10 scale for everything that follows.
native = run_phase("native", subject)
collect_externalization_rating(native)

# %% BEFORE THE SESSION: stage the top 3 donors -------------------------------
# Run this with nobody in the rig. Builds the rank 0/1/2 composites AND their
# pyBinSim databases (mirrored and un-mirrored), so that if the participant
# turns out to be at floor with the rank-0 donor you can swap in seconds
# instead of rebuilding filters while they wait. Leaves rank 0 active.
prepare_donor_shortlist(n=3)

# %% day 1: select the donor and build the modified HRTF -- run ONCE ----------
# Prints the full candidate ranking, the chosen donor and why, writes
# <SUBJECT_ID>_donor_<DONOR>.sofa with the selection embedded, plus a ranking
# CSV and before/after figures. Note the donor id -- put it in DONOR_ID at the
# top so later sessions can skip straight to load_existing_donor().
# Redundant if prepare_donor_shortlist() was run; harmless to run anyway.
build_donor_sofa(overwrite=False)
subject = hr.Subject(SUBJECT_ID)

# %% later sessions: reload the modified HRTF without rebuilding --------------
load_existing_donor()

# %% IN SESSION: participant is at floor -- swap to the next donor ------------
# Only when the composite has abolished the cue rather than degraded it (a
# block at chance tells you nothing). Moves down the rule's ranked list, which
# was fixed before the participant heard anything -- see selection.shortlist.
# WRITE WHAT YOU SAW in reason=: the swap is a data-dependent decision and has
# to be reportable as one. Then set DONOR_ID at the top of this file so later
# sessions reload the donor you switched TO.
#   use_donor(rank=1, reason
#   ="EG 0.03 on baseline A, responses at chance")
# use_donor(rank=1, reason="")

# %% which donors this participant has been on ---------------------------------
show_donor_log()  # todo move this out of the way

# %% day 1: baseline A -- trained ear, same loc (matches final A) -------------
baseline_A = run_phase("baseline_A", subject)
collect_externalization_rating(baseline_A)

# %% day 1: baseline D -- untrained ear, mirrored loc (matches final D) -------
baseline_D = run_phase("baseline_D", subject)
collect_externalization_rating(baseline_D)

# ---------------------------------------------------------------------------
# ADAPTATION DAYS
# anchor -> PRE test -> train -> POST test, so within-session change is
# separable from overnight consolidation and every rating has a same-day top.
# Run the four cells in order.
# ---------------------------------------------------------------------------

# %% adaptation day: 0. ANCHOR (~10 trials, own HRTF) -------------------------
# Re-tops the 0-10 scale for today and checks the delivery chain before any
# data is collected. If this rating is well below yesterday's, stop and check
# headphone seating / HP filter / OS volume rather than logging the day.
subject = hr.Subject(SUBJECT_ID)
run_anchor(subject)
# todo im not going to run this every day, too expensive. if externalization is stable for modified ears the second day
#  i trust it will stay so until the end of the experiment
#  also, i dont need to collect externalization ratings beyond the baseline tests


# %% adaptation day: 1. PRE-training test -------------------------------------
subject = hr.Subject(SUBJECT_ID)
daily_pre = run_phase("daily", subject)
collect_externalization_rating(daily_pre)

# %% adaptation day: 2. TRAIN --------------------------------------------------
run_training()

# %% adaptation day: 3. POST-training test ------------------------------------
subject = hr.Subject(SUBJECT_ID)
daily_post = run_phase("daily", subject)
collect_externalization_rating(daily_post)

# %% final day: 0. ANCHOR (run before the 2x2) --------------------------------
# Same-day top of the scale. The four final ratings are compared against each
# other and against the day-1 baselines, so both need an anchor from their own
# day; four days of headphone re-seating is enough to move the scale on its own.
subject = hr.Subject(SUBJECT_ID)
run_anchor(subject)

# %% final day: all 4 conditions in this subject's counterbalanced order -------
subject = hr.Subject(SUBJECT_ID)
print(f"Running final tests in order: {FINAL_ORDER}")
for key in FINAL_ORDER:
    collect_externalization_rating(run_phase(key, subject))

# %% final day: A -- trained ear, same locations (redo individually) -----------
collect_externalization_rating(run_phase("A", subject))

# %% final day: B -- trained ear, mirrored locations (redo individually) -------
collect_externalization_rating(run_phase("B", subject))

# %% final day: C -- untrained ear, same locations (redo individually) ---------
collect_externalization_rating(run_phase("C", subject))

# %% final day: D -- untrained ear, mirrored locations [MAIN] -----------------
collect_externalization_rating(run_phase("D", subject))


# ===========================================================================
# MISC — diagnostics, not part of the per-participant protocol.
# Nothing below runs as a matter of course. Reach for it when something looks
# wrong, or on the odd participant where the extra measurement is worth the
# time. Each cell stands alone; the config cell at the top must have been run.
# ===========================================================================

# %% misc: QC the cepstral split the manipulation depends on ------------------
# Envelope (red) should be smooth and roughly elevation-invariant; if it tracks
# elevation, the split is freezing part of the cue instead of separating it.
# Worth a look on the first few participants and whenever a donor composite
# looks odd in the before/after figure.
plot_split_qc(slab.HRTF(str(paths.SOFA_DIR / SUBJECT_ID / f"{NATIVE_SOFA}.sofa")),
              envelope_n_keep=selection.N_KEEP, ear=TRAINED_EAR,
              band=selection.DEFAULT_BAND)

# %% misc: build the second composite strength (n_keep=8) ---------------------
# Half as much of the cue handed over. Only needed as a rung of the
# externalization ladder below; the training/testing SOFA stays the n_keep=4 one.
build_donor_sofa(overwrite=False, show_qc=False, n_keep=8)

# %% misc: externalization + acute-degradation ladder -------------------------
# ~50 trials plus ratings, so it is NOT run on every participant. Use it when
# externalization is in doubt, when picking OTHER_EAR for a new cohort, or to
# check that a donor composite lands in the intended acute-degradation range.
# Requires the n_keep=8 SOFA from the cell above.
#
# Blocks of ~10 trials in a per-subject RANDOM order, each followed by the 0-10
# rating; elevation gain is reported alongside.
#   anchor     own unmodified HRTF, binaural      <- ceiling of the whole chain
#   flat       other ear = delta impulse          }
#   native     other ear = own full DTF           } ear treatment
#   donor_n4   composite n_keep=4, other ear = OTHER_EAR   }
#   donor_n8   composite n_keep=8, other ear = OTHER_EAR   } composite strength
# donor_n4 is the condition the experiment actually runs. Read the EG column for
# the acute degradation (target 0.3-0.5) and the rating column for
# externalization -- but see the caveat the summary prints about 10-trial EG.
subject = hr.Subject(SUBJECT_ID)
externalization_ladder(
    subject, ladder_settings, seed=SUBJECT_ID,
    rungs=("anchor", "flat", "native", "donor_n4", "donor_n8"))

# %% misc: daily PROBE (only meaningful when OTHER_EAR = 'native') ------------
# Repeats the daily test with the other ear reduced, so relearning of the
# modified cue can be told apart from reweighting toward the intact ear.
subject = hr.Subject(SUBJECT_ID)
run_phase("daily", subject, other_ear=PROBE_OTHER_EAR)

# %% misc: externalization ratings so far, in order ---------------------------
# Every rating on file for this participant, with its block, so drift in the
# anchor can be told apart from drift in the conditions. Anchors are the
# ~10-trial native binaural blocks; they should stay roughly flat across days.
subject = hr.Subject(SUBJECT_ID)
print(f"{'run':46s} {'n':>4s} {'ear':6s} {'mir':5s} {'rating':>6s}")
for _name, _seq in subject.localization.items():
    _rating = getattr(_seq, "externalization_rating", None)
    if _rating is None:
        continue
    _n = len(getattr(_seq, "data", []) or [])
    _tag = " <- anchor" if (_n <= 12 and getattr(_seq, "hrir", "") == NATIVE_SOFA) else ""
    print(f"{_name:46s} {_n:4d} {str(getattr(_seq, 'ear', None)):6s} "
          f"{str(getattr(_seq, 'mirrored', None)):5s} {_rating:6.1f}{_tag}")

# %% misc: force the OS output level on its own -------------------------------
# run_phase() and run_training() already do this. Use it when checking levels
# by ear outside a block, or after someone has touched the volume slider.
_fix_output_level()
