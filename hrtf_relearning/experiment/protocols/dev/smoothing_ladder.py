"""
smoothing_ladder.py — how much spectral detail does this listener actually need?

Kulkarni & Colburn (1998, Nature 396:747) replicated in this setup. The
participant's OWN HRTF is presented binaurally, with the log-magnitude spectrum
of every direction reduced to its first n_keep cosine coefficients — the same
cepstral smoother the cue manipulations use. Original phase is kept and each
direction is rescaled to its original broadband energy, so ITD and broadband ILD
are exactly as measured and only the magnitude fine structure is removed. No
monaural reduction, no donor: this is purely "how coarse can the spectrum get
before it stops working".

WHAT THEY FOUND, and what it predicts here
------------------------------------------
Their M is the same index as n_keep (their sum runs n=0..M inclusive, this code
keeps n < n_keep, so their M=16 is n_keep=17 — off by one, same scale).

  n_keep 16   discrimination from a real source at chance -> should be
              indistinguishable from the native rung
  n_keep  8   just discriminable; the reported cue was that the virtual image
              sat HIGHER than the real one (smoothed spectra resemble
              high-elevation HRTFs) -> expect an upward elevation BIAS, not
              necessarily a gain loss
  n_keep  4   below anything they tested; the whole perceptually relevant range
              (M ~ 4-16) is gone

  AND, the part that matters most here: at their most extreme smoothing every
  listener still reported COMPLETE EXTERNALIZATION. They concluded spectral
  detail is not what externalizes, and credited their open-canal tube-phones.

So if externalization falls off across these rungs in this setup, that is a
finding about the DELIVERY (headphone coupling, HP equalisation, the midline ILD
offset) rather than about spectral detail — and it would explain why removing
detail at one ear hurt FS so badly while Kulkarni & Colburn saw no such thing.

Run cell by cell (# %%). Four blocks of ~10 trials, randomised order, ~15 min.
"""

SUBJECT_ID = 'FS'
N_KEEP_RUNGS = (16, 8, 4)     # plus 'native' as the unsmoothed reference
HP = 'DT990'

# %% imports and config -------------------------------------------------------
import slab

import hrtf_relearning as hr
from hrtf_relearning.experiment.protocols.protocol_helpers import externalization_ladder
from hrtf_relearning.hrtf.modify.edge_shift import embed_modification_params
from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf
from hrtf_relearning.utils import paths

NATIVE_SOFA = SUBJECT_ID
ELEVATION_RANGE = (-35, 35)
AZIMUTH_RANGE = (-35, 35)     # full frontal field: binaural, nothing to restrict


def smoothed_name(n_keep):
    return f'{SUBJECT_ID}_smooth_n{n_keep}'


def hrir_settings(sofa_name):
    """Binaural, no monaural reduction — ear=None."""
    return {
        'name': sofa_name, 'subject_id': SUBJECT_ID,
        'ear': None, 'mirror': False,
        'reverb': True, 'drr': 20,
        'hp_filter': True, 'hp': HP,
        'convolution': 'cuda', 'storage': 'cuda',
    }


def loc_settings():
    """Coarse grid, ~10 trials — for the rating and a gross gain estimate."""
    return {
        'kind': 'sectors',
        'azimuth_range': AZIMUTH_RANGE, 'elevation_range': ELEVATION_RANGE,
        'sector_size': (14, 14), 'targets_per_sector': 1,
        'targets_per_speaker': 3, 'min_distance': 20,
        'gain': 0.2, 'stim': 'noise', 'replace': False,
    }


def build_smoothed_sofas(n_keeps=N_KEEP_RUNGS, overwrite=False, show_qc=True):
    """Write <SUBJECT_ID>_smooth_n<k>.sofa for each k. Idempotent."""
    sofa_dir = paths.SOFA_DIR / SUBJECT_ID
    own = slab.HRTF(str(sofa_dir / f'{NATIVE_SOFA}.sofa'))
    own.name = SUBJECT_ID
    plot_dir = paths.subject_acoustic_dir(SUBJECT_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for n_keep in n_keeps:
        out_path = sofa_dir / f'{smoothed_name(n_keep)}.sofa'
        if out_path.exists() and not overwrite:
            print(f'{out_path.name} exists — skipping')
            continue
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
            fig = plot_ears(own, smoothed,
                            suptitle=f'{SUBJECT_ID}  binaural smoothing, n_keep={n_keep}')
            fig.savefig(plot_dir / f'{smoothed.name}.png', bbox_inches='tight')
    print(f'QC figures in {plot_dir}')


def ladder_settings(rung):
    sofa = NATIVE_SOFA if rung == 'native' else smoothed_name(int(rung[1:]))
    return hrir_settings(sofa), loc_settings()


# %% build the smoothed SOFAs -- run once -------------------------------------
build_smoothed_sofas()

# %% run the ladder -----------------------------------------------------------
# native / n16 / n8 / n4, ~10 trials each, per-subject randomised order.
subject = hr.Subject(SUBJECT_ID)
rungs = ('native',) + tuple(f'n{k}' for k in N_KEEP_RUNGS)
results = externalization_ladder(subject, ladder_settings, rungs=rungs,
                                 seed=f'{SUBJECT_ID}_smoothing')

# %% read-out -----------------------------------------------------------------
# What to look for, against Kulkarni & Colburn:
#   n16 ~ native on both rating and EG        -> replicates them
#   n8  shows an UPWARD elevation bias        -> replicates their reported cue
#   ratings stay high while EG falls          -> detail carries elevation but not
#                                                externalization (their claim)
#   ratings fall with n_keep                  -> does NOT replicate; in this setup
#                                                externalization depends on detail,
#                                                which points at the delivery chain
print('\nbias column is the key one for n8 — Kulkarni & Colburn report the\n'
      'smoothed image sitting HIGHER than the real source.')
for rung in rungs:
    row = results.get(rung)
    if row:
        print(f'  {rung:>7}  rating {str(row["rating"]):>5}   '
              f'EG {row["elevation_gain"]:5.2f}   RMSE {row["ele_rmse"]:5.1f}   '
              f'SD {row["ele_sd"]:5.1f}   n {row["n_trials"]}')
