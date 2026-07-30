"""
ladder.py — what breaks externalization, and what it costs in localization.

Run in the FREE-FIELD lab, straight after the dome localization block and before
moving to the VR lab, so the participant's real-source performance from minutes
earlier is the reference for everything here.

THE QUESTION
------------
Symmetric detail removal does NOT break externalization: with both ears smoothed
the image stays outside the head even when elevation localization is gone
(n_keep=8 leaves some localization, n_keep=4 none; both well externalized —
measured here, and consistent with Kulkarni & Colburn 1998). Flat DTFs, on the
other hand, are not externalized. So the culprit is one of three things:

  1. BINAURAL MISMATCH   the two ears carry incongruent spectra
  2. THE FLAT DTF        a delta is spectrally implausible in itself
  3. THE UNKNOWN EAR     non-individual detail

Each is isolated by making the two ears SYMMETRIC in the suspect property —
mismatch only exists when the ears differ, the other two exist even when they
do not:

    rung             L / R                        isolates
    anchor           native / native              ceiling of the whole chain
    sym_envelope     envelope / envelope          no detail, no mismatch, own
    sym_donor        donor / donor                unknown ear, NO mismatch
    mono_native_env  native / envelope            mismatch, both ears plausible
    mono_native_flat native / flat                the classic monaural condition
    mono_donor_env   donor / envelope             training candidate
    mono_donor_flat  donor / flat                 training candidate

Predictions, so the result is interpretable before it is collected:

                        sym_donor    mono_native_env
    mismatch is it        ext          INTERNAL
    unknown ear is it   INTERNAL         ext

(The symmetric-flat cell that would separate "flat in itself" from "mismatch"
is deliberately omitted — flat/flat is not expected to externalize, so it buys
little. That leaves those two explanations partly confounded: every flat rung
here is also a mismatch rung.)

The last two rungs are not diagnostic, they are the per-participant choice of
OTHER_EAR for the learning_transfer protocol.

ALL rungs sample the same full frontal field, midline included. A degraded
contralateral ear matters most near the midline, where both ears are weighted
about equally, and least for lateral targets, where the listening ear dominates
— so flat/flat failing to externalize says nothing about whether native/flat
fails off to the side. `split_by_laterality` reads that out.

Each rung is 3 azimuth columns (-30, 0, +30) x 10 elevations = 30 trials, so
10 per laterality cell. Seven rungs is 210 trials — budget roughly an hour, and
drop rungs from LADDER_RUNGS if the session is tighter than that. Order is
randomised per participant.
"""

SUBJECT_ID = 'FS'
HP = 'DT990'
ENV_N_KEEP = 4
AZIMUTH_COLUMNS = (-30, 0, 30)   # laterality is the variable, so fix the columns
TARGETS_PER_COLUMN = 10          # 3 x 10 = 30 trials per rung

LADDER_RUNGS = (
    'anchor',
    'sym_envelope',
    'sym_donor',
    'mono_native_env',
    'mono_native_flat',
    'mono_donor_env',
    'mono_donor_flat',
)

# %% imports and config -------------------------------------------------------
import csv
import re

import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.protocols.protocol_helpers import (
    collect_demographics, externalization_ladder)
from hrtf_relearning.hrtf.modify.edge_shift import (embed_modification_params,
                                                    read_modification_params)
from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf
from hrtf_relearning.utils import paths

NATIVE_SOFA = SUBJECT_ID
ELEVATION_RANGE = (-35, 35)
CSV_PATH = (hr.PATH / "experiment" / "protocols" / "learning_transfer"
            / "learning_transfer_block_order.csv")


def trained_ear(subject_id=SUBJECT_ID, csv_path=CSV_PATH):
    """The ear the monaural rungs listen with — same sheet the protocol uses."""
    with open(csv_path, newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('subject', '').strip() == subject_id:
                return row['trained_ear'].strip()
    raise ValueError(f"{subject_id} not in the 'subject' column of {csv_path}")


TRAINED_EAR = trained_ear()
TRAINED_HEMI = (-35, 0) if TRAINED_EAR == 'left' else (0, 35)


def smoothed_name(n_keep=ENV_N_KEEP):
    return f'{SUBJECT_ID}_smooth_n{n_keep}'


def donor_sofa_name():
    """This participant's donor composite, as built by the learning_transfer
    protocol. Ladder-only strength variants (<name>_n<k>) are excluded."""
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    matches = [p for p in sorted(sofa_dir.glob(f'{SUBJECT_ID}_donor_*.sofa'))
               if not re.search(r'_n\d+$', p.stem)]
    if not matches:
        raise FileNotFoundError(
            f'no {SUBJECT_ID}_donor_*.sofa — run build_donor_sofa() in '
            f'protocols/learning_transfer/learning_transfer.py first')
    if len(matches) > 1:
        raise RuntimeError(f'several donor SOFAs: {[p.name for p in matches]}')
    params = read_modification_params(matches[0]) or {}
    print(f'donor SOFA: {matches[0].name}   donor={params.get("donor_id", "?")}')
    return matches[0].stem


def hrir_settings(sofa_name, ear=None, other_ear=None):
    return {
        'name': sofa_name, 'subject_id': SUBJECT_ID,
        'ear': ear, 'other_ear': other_ear, 'env_n_keep': ENV_N_KEEP,
        'native_sofa': NATIVE_SOFA, 'mirror': False,
        'reverb': True, 'drr': 20,
        'hp_filter': True, 'hp': HP,
        'convolution': 'cuda', 'storage': 'cuda',
    }


def loc_settings():
    """Three fixed azimuth columns, 10 elevations each.

    Laterality is the variable of interest here, not a nuisance, so the columns
    are fixed rather than sampled from sectors: a degraded contralateral ear
    should cost most at az 0, where both ears are weighted about equally, and
    least at +-30, where the listening ear dominates.
    """
    return {
        'kind': 'columns',
        'azimuths': AZIMUTH_COLUMNS, 'targets_per_column': TARGETS_PER_COLUMN,
        'elevation_range': ELEVATION_RANGE, 'azimuth_tol': 5,
        'min_distance': 20, 'gain': 0.2, 'stim': 'noise', 'replace': False,
    }


def build_smoothed_sofa(n_keep=ENV_N_KEEP, overwrite=False, show_qc=True):
    """<SUBJECT_ID>_smooth_n<k>.sofa — own HRTF, BOTH ears reduced to the
    envelope. Symmetric: no detail anywhere, no mismatch, still the own ear."""
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    out_path = sofa_dir / f'{smoothed_name(n_keep)}.sofa'
    if out_path.exists() and not overwrite:
        print(f'{out_path.name} exists — skipping')
        return out_path
    own = slab.HRTF(str(sofa_dir / f'{NATIVE_SOFA}.sofa'))
    own.name = SUBJECT_ID
    smoothed = envelope_dtf(own, ear='both', n_keep=n_keep)
    smoothed.name = smoothed_name(n_keep)
    smoothed.write_sofa(str(out_path))
    embed_modification_params(out_path, {
        'condition': 'binaural_smoothing', 'subject_id': SUBJECT_ID,
        'n_keep': n_keep, 'ears': 'both', 'phase': 'original',
        'level': 'per-direction energy matched',
        'reference': 'Kulkarni & Colburn 1998'})
    print(f'wrote {out_path.name}')
    if show_qc:
        plot_dir = paths.subject_acoustic_dir(SUBJECT_ID)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_ears(own, smoothed,
                        suptitle=f'{SUBJECT_ID}  both ears -> envelope n_keep={n_keep}')
        fig.savefig(plot_dir / f'{smoothed.name}.png', bbox_inches='tight')
    return out_path


# rung -> (sofa, ear, other_ear). ear=None is binaural.
#
# EVERY rung uses the SAME full frontal field, including the midline. Two
# reasons. Restricting the monaural rungs to the trained hemifield (as the
# training protocol does) would make them differ from the symmetric rungs in
# azimuth as well as in ear treatment, which is not a comparison worth having.
# And laterality is the interesting variable here rather than a nuisance: a
# degraded contralateral ear should matter most near the midline, where both
# ears are weighted about equally, and least for lateral targets, where the
# intact ipsilateral ear dominates. So a flat ear can perfectly well internalize
# flat/flat and still leave native/flat externalized off to the side. Sampling
# the whole field lets that show up — see split_by_laterality below.
def ladder_settings(rung):
    donor = donor_sofa_name()
    table = {
        'anchor':           (NATIVE_SOFA,     None,        None),
        'sym_envelope':     (smoothed_name(), None,        None),
        'sym_donor':        (donor,           None,        None),
        'mono_native_env':  (NATIVE_SOFA,     TRAINED_EAR, 'envelope'),
        'mono_native_flat': (NATIVE_SOFA,     TRAINED_EAR, 'flat'),
        'mono_donor_env':   (donor,           TRAINED_EAR, 'envelope'),
        'mono_donor_flat':  (donor,           TRAINED_EAR, 'flat'),
    }
    sofa, ear, other_ear = table[rung]
    return hrir_settings(sofa, ear=ear, other_ear=other_ear), loc_settings()


def split_by_laterality(subject, boundary=15):
    """Elevation error near the midline vs lateral, per rung.

    The test of the weighting account: a degraded contralateral ear costs most
    where both ears are weighted about equally (near the midline) and least
    where the listening ear dominates (lateral). If a rung is uniformly bad the
    cause is not binaural weighting. Ipsilateral / contralateral is relative to
    the LISTENING ear, so for the monaural rungs 'lateral' means targets on the
    listening side.
    """
    import numpy
    rows = []
    for name, sequence in getattr(subject, 'localization', {}).items():
        rung = getattr(sequence, 'ladder_rung', None)
        data = getattr(sequence, 'data', None)
        if rung is None or not data:
            continue
        trials = numpy.asarray([t for t in data if t is not None],
                               dtype=float).reshape(-1, 2, 2)
        target, response = trials[:, 1], trials[:, 0]
        error = numpy.abs(response[:, 1] - target[:, 1])
        near = numpy.abs(target[:, 0]) <= boundary
        rows.append((rung,
                     float(error[near].mean()) if near.any() else float('nan'),
                     int(near.sum()),
                     float(error[~near].mean()) if (~near).any() else float('nan'),
                     int((~near).sum())))

    print(f'\nelevation error by target laterality (boundary |az| = {boundary} deg)')
    print(f'  {"rung":>17} {"near midline":>14} {"lateral":>14}')
    for rung, near_error, near_n, far_error, far_n in rows:
        print(f'  {rung:>17} {near_error:9.1f} (n={near_n:2d}) '
              f'{far_error:9.1f} (n={far_n:2d})')
    print('  near >> lateral  -> the degraded ear is being weighted; consistent\n'
          '                      with a binaural-weighting account\n'
          '  near ~= lateral  -> uniform degradation; weighting is not the story')
    return rows


def dome_elevation_gain(subject):
    """Elevation gain of this subject's most recent free-field dome block.

    The ladder runs minutes after it, in the same room, so this is the
    real-source ceiling every virtual rung should be read against. Picked up
    automatically rather than typed in, so it cannot drift.
    """
    from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
        localization_accuracy)
    runs = [(name, seq) for name, seq in getattr(subject, 'localization', {}).items()
            if name.endswith('_dome')]
    if not runs:
        print('  [warn] no *_dome block on file for this subject')
        return None
    name, sequence = runs[-1]
    try:
        gain = localization_accuracy(sequence)[0]
    except Exception as exc:
        print(f'  [warn] could not score {name}: {exc}')
        return None
    print(f'dome reference: {name}   EG {gain:.2f}')
    return float(gain)


def save_summary(results, dome_reference=None):
    """One row per rung in results/<id>/ladder_<id>.csv, for cross-subject work."""
    out_path = paths.RESULTS_DIR / SUBJECT_ID / f'ladder_{SUBJECT_ID}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['subject', 'rung', 'run_position', 'rating', 'elevation_gain',
              'ele_rmse', 'ele_sd', 'n_trials', 'dome_elevation_gain']
    with open(out_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rung in LADDER_RUNGS:
            row = results.get(rung)
            if row:
                writer.writerow({'subject': SUBJECT_ID, 'rung': rung,
                                 'dome_elevation_gain': dome_reference,
                                 **{k: row.get(k) for k in fields[2:-1]}})
    print(f'wrote {out_path}')
    return out_path


# %% build what the ladder needs -- run once ----------------------------------
# The donor SOFA must already exist (learning_transfer.build_donor_sofa).
build_smoothed_sofa()
donor_sofa_name()

# %% run the ladder -----------------------------------------------------------
# Randomised order per participant, ~10 trials per rung, rating after each.
subject = hr.Subject(SUBJECT_ID)
collect_demographics(subject)      # once per participant; skipped if on file
results = externalization_ladder(subject, ladder_settings, rungs=LADDER_RUNGS,
                                 seed=SUBJECT_ID)

# %% laterality split ---------------------------------------------------------
# A degraded contralateral ear should cost most near the midline and least for
# lateral targets. flat/flat failing to externalize does not imply native/flat
# fails off to the side — this is where that shows up.
split_by_laterality(subject)

# %% summary ------------------------------------------------------------------
# The dome block run minutes ago is picked up automatically and stored beside
# the virtual rungs, so each condition has this participant's real-source
# ceiling next to it.
save_summary(results, dome_reference=dome_elevation_gain(subject))

print("""
READ-OUT
  sym_envelope externalized, anchor externalized
      -> removing detail does not internalize (expected; replicates)
  sym_donor externalized
      -> a foreign ear is not what internalizes; the donor paradigm is safe
  sym_donor internal
      -> non-individual detail internalizes even without mismatch, which
         constrains the whole donor design
  mono_native_env internal, sym_envelope externalized
      -> MISMATCH is the cause: the same envelope ear is fine when both ears
         have it and bad when only one does
  mono_native_env externalized
      -> mismatch alone is not enough; the flat ear does something extra
  mono_donor_env vs mono_donor_flat
      -> not diagnostic; this is the OTHER_EAR choice for this participant
""")
