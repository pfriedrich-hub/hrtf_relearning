"""
learning_transfer_donor.py

Adaptation-transfer protocol with the DONOR-DETAIL manipulation.

Same design, grid, counterbalancing and final-day 2x2 as
protocols/learning_transfer/learning_transfer.py — only the cue manipulation
differs. Instead of translating the participant's own spectral detail along the
ERB axis, their detail is REPLACED with a donor's:


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
DONOR_POOL / TARGET_DISSIMILARITY / MAX_RIDGE_SLOPE there, and
docs/methods_donor_detail.md for the paragraph this becomes in a paper.
Everything else (n_keep, band, filter bank, target) is identical for everyone.

The monaural ear treatment is orthogonal and still selectable via OTHER_EAR
('flat' | 'envelope' | 'native'), as in learning_transfer_env.py.

Counterbalancing is read from learning_transfer_donor_block_order.csv IN THIS
FOLDER. Write each subject's id into the 'subject' column before running.

Run cell by cell (# %%) in an IDE/console -- do NOT run top-to-bottom.

------------------------------------------------------------------------------
EDIT THE CONFIG BLOCK BELOW PER PARTICIPANT.
------------------------------------------------------------------------------
"""

SUBJECT_ID = ("FS")

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
from hrtf_relearning.hrtf.modify.edge_shift import (embed_modification_params,
                                                    read_modification_params)
from hrtf_relearning.experiment.protocols.protocol_helpers import (
    collect_externalization_rating, externalization_ladder)
from hrtf_relearning.hrtf.modify.plot_compare import plot, plot_split_qc
from hrtf_relearning.utils import paths

CSV_PATH = (hr.PATH / "experiment" / "protocols" / "dev"
            / "learning_transfer_donor_block_order.csv")


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
# 'flat' (parent protocol) | 'envelope' | 'native' -- see learning_transfer_env.py
OTHER_EAR = "envelope"
ENV_NKEEP = selection.N_KEEP
PROBE_OTHER_EAR = "envelope"

# --- shared localization sampling grid (do not change) -----------------------
SECTOR_SIZE        = (7, 14)
ELEVATION_RANGE    = (-35, 35)
TARGETS_PER_SECTOR = 3
MIN_DISTANCE       = 20
GAIN               = 0.2
STIM               = "noise"
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
    n_keep = selection.N_KEEP if n_keep is None else n_keep
    stem = f"{SUBJECT_ID}_donor_{donor_id.split('/')[-1]}"
    return stem if n_keep == selection.N_KEEP else f"{stem}_n{n_keep}"


def hrir_settings(sofa_name, ear=None, mirror=False, other_ear=None):
    return {
        "name": sofa_name,
        "subject_id": SUBJECT_ID,
        "ear": ear,
        "other_ear": OTHER_EAR if other_ear is None else other_ear,
        "env_n_keep": ENV_NKEEP,
        "native_sofa": NATIVE_SOFA,
        "mirror": mirror,
        "reverb": True,
        "drr": 20,
        "hp_filter": True,
        "hp": HP,
        "convolution": "cuda",
        "storage": "cuda",
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
        "sector_size": SECTOR_SIZE,
        "replace": False,
        "exclude_midline": exclude_midline,
        "midline_tol": MIDLINE_TOL,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_donor_sofa(overwrite=False, show_qc=True, n_keep=None):
    """Select the donor, build the composite, write it, return (path, donor_id).

    Writes <SUBJECT_ID>_donor_<DONOR>.sofa next to the native one, embeds the
    full selection record as GLOBAL_ModificationParams, and saves the candidate
    ranking as a CSV for the supplement. Run ONCE per participant; every
    subsequent cell loads the result.
    """
    global DONOR_ID, MODIFIED_SOFA

    n_keep = selection.N_KEEP if n_keep is None else n_keep
    is_default = n_keep == selection.N_KEEP
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    own = slab.HRTF(str(sofa_dir / f"{NATIVE_SOFA}.sofa"))
    own.name = SUBJECT_ID

    candidates = selection.load_candidates(SUBJECT_ID)
    print(f"donor pool ({len(candidates)}): {', '.join(candidates)}")
    if not candidates:
        raise RuntimeError("empty donor pool — check selection.DONOR_POOL")

    chosen, rows = selection.select_donor(own, candidates)
    reference, _ = selection.pairwise_reference({SUBJECT_ID: own, **candidates})
    selection.report(rows, reference)
    print(f"\nchosen donor: {chosen['donor']}   VSI dissimilarity "
          f"{chosen['vsi_dissimilarity']:.3f} (target {selection.TARGET_DISSIMILARITY:.2f})"
          f"   ridge slope {chosen['ridge_slope']:+.2f}")
    if chosen["fallback"]:
        print("  !! FALLBACK: no candidate met the ridge criterion; lowest slope "
              "used. This must be reported.")

    # The donor is always selected at the protocol n_keep, so the two ladder
    # strengths differ ONLY in how much of the cue is handed over -- not in
    # whose cue it is.
    donor_id = chosen["donor"]
    name = _modified_name(donor_id, n_keep)
    if is_default:
        DONOR_ID, MODIFIED_SOFA = donor_id, name
    out_path = sofa_dir / f"{name}.sofa"
    if out_path.exists() and not overwrite:
        print(f"{out_path.name} already exists (overwrite=False) -- skipping build")
        return out_path, donor_id

    modified = donor_detail_dtf(own, candidates[donor_id], n_keep=n_keep)
    modified.name = name
    own_vsi, modified_vsi = vsi_of(own), vsi_of(modified)
    print(f"VSI  own={own_vsi:.3f}  modified={modified_vsi:.3f}  "
          f"(diagnostic only -- not diffuse-field normalised, see vsi.py)")

    modified.write_sofa(str(out_path))
    embed_modification_params(out_path, modification_params(
        SUBJECT_ID, donor_id, n_keep=n_keep,
        target_dissimilarity=selection.TARGET_DISSIMILARITY,
        band=selection.DEFAULT_BAND, resolution=selection.DEFAULT_RESOLUTION,
        max_ridge_slope=selection.MAX_RIDGE_SLOPE, pool=list(candidates),
        fallback=chosen["fallback"],
        scores={k: chosen[k] for k in ('vsi_dissimilarity', 'vsi', 'own_vsi',
                                       'i_sim', 'peak_r', 'ridge_slope')},
        ranking=[{k: row[k] for k in ('donor', 'vsi_dissimilarity', 'vsi',
                                      'ridge_slope', 'eligible')} for row in rows]))
    print(f"wrote {out_path}")

    plot_dir = paths.subject_plot_dir(SUBJECT_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    with open(plot_dir / f"{name}_donor_ranking.csv", "w", newline="") as handle:
        fields = ['donor', 'vsi_dissimilarity', 'vsi', 'own_vsi', 'i_sim',
                  'peak_r', 'ridge_slope', 'ridge_bias', 'distance', 'eligible']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    if show_qc:
        for kind in ("image", "waterfall"):
            fig = plot(own, modified, kind, ear=UNTRAINED_EAR,
                       vsi_orig=own_vsi, vsi_mod=modified_vsi,
                       vsi_dis=chosen["vsi_dissimilarity"],
                       vsi_bw=selection.DEFAULT_BAND)
            fig.savefig(plot_dir / f"{name}_{kind}.png", bbox_inches="tight")
        print(f"QC figures in {plot_dir}")
    return out_path, donor_id


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
        raise RuntimeError(
            f"several donor SOFAs for {SUBJECT_ID}: "
            f"{', '.join(p.name for p in matches)}. Set DONOR_ID in the config "
            f"block to say which one this subject was trained on.")

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


def run_phase(key, subject, other_ear=None):
    label, when, sofa, ear, mirror, az, desc = phases()[key]
    if sofa is None:
        raise RuntimeError("MODIFIED_SOFA is not set — run build_donor_sofa() "
                           "or load_existing_donor() first")
    print("\n" + "=" * 70)
    print(f"RUNNING:  {_describe(key)}"
          + (f"\n      OTHER EAR OVERRIDE: {other_ear}" if other_ear else ""))
    print("=" * 70)
    one_sided = tuple(az) != tuple(FULL_FIELD)
    test = Localization(subject,
                        hrir_settings(sofa, ear=ear, mirror=mirror, other_ear=other_ear),
                        loc_settings=loc_settings(az, exclude_midline=one_sided))
    test.run()
    print(f"Done: {test.filename}")
    return test


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
show_status(subject)

# %% day 1: native reference (original HRIR, full field) ----------------------
run_phase("native", subject)


# %% day 1: select the donor and build the modified HRTF -- run ONCE ----------
# Prints the full candidate ranking, the chosen donor and why, writes
# <SUBJECT_ID>_donor_<DONOR>.sofa with the selection embedded, plus a ranking
# CSV and before/after figures. Note the donor id -- put it in DONOR_ID at the
# top so later sessions can skip straight to load_existing_donor().
build_donor_sofa(overwrite=False)
subject = hr.Subject(SUBJECT_ID)

# %% later sessions: reload the modified HRTF without rebuilding --------------
load_existing_donor()

# %% day 1: QC the split the manipulation depends on --------------------------
# Envelope (red) should be smooth and roughly elevation-invariant; if it tracks
# elevation, the split is freezing part of the cue instead of separating it.
plot_split_qc(slab.HRTF(str(paths.SOFA_DIR / SUBJECT_ID / f"{NATIVE_SOFA}.sofa")),
              envelope_n_keep=selection.N_KEEP, ear=TRAINED_EAR,
              band=selection.DEFAULT_BAND)

# %% day 1: build the second composite strength for the ladder ---------------
# Same donor, n_keep=8: half as much of the cue handed over. Only needed for the
# ladder; the training/testing SOFA stays the n_keep=4 one.
build_donor_sofa(overwrite=False, show_qc=False, n_keep=8)

# %% day 1: externalization + acute-degradation ladder ------------------------
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
externalization_ladder(
    subject, ladder_settings, seed=SUBJECT_ID,
    rungs=("anchor", "flat", "native", "donor_n4", "donor_n8"))

# %% day 1: baseline A -- trained ear, same loc (matches final A) -------------
baseline_A = run_phase("baseline_A", subject)
collect_externalization_rating(baseline_A)

# %% day 1: baseline D -- untrained ear, mirrored loc (matches final D) -------
baseline_D = run_phase("baseline_D", subject)

collect_externalization_rating(baseline_D)

# %% adaptation days: TRAIN ----------------------------------------------------
run_training()

# %% adaptation days: daily TEST -----------------------------------------------
subject = hr.Subject(SUBJECT_ID)
daily = run_phase("daily", subject)
collect_externalization_rating(daily)

# %% adaptation days: daily PROBE (only when OTHER_EAR = 'native') -------------
# Repeats the daily test with the other ear reduced, so relearning of the
# modified cue can be told apart from reweighting toward the intact ear.
run_phase("daily", subject, other_ear=PROBE_OTHER_EAR)

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
