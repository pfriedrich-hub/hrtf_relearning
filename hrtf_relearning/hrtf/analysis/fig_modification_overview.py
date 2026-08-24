"""
fig_modification_overview.py — one figure for what the donor-detail
manipulation does to a participant's ears.

Three stages, both ears, one colour scale (median-plane arc, which is the only
arc that is MEASURED — see donor_detail.py):

    1  measured        the participant's own recorded DTF
    2  envelope n=4    the cepstral envelope both ears share once the fine
                       structure is removed; per-direction, NOT averaged, so
                       what is gone is exactly the elevation cue
    3  delivered       the SOFA the experiment actually loads:
                       trained ear   = own envelope + DONOR fine detail
                       untrained ear = own envelope, averaged over elevation

Stage 3 is read from disk rather than rebuilt, so the figure is a picture of
the stimulus that was presented, not of a reconstruction of it. Stage 2 is the
one synthetic panel: it is the common base both ears of stage 3 are built on
(`envelope_dtf(ear='both')` is exactly the split
`hrtf.modify.donor_detail.split_log_magnitude` performs), and it is drawn so the
step from 2 to 3 reads as "the donor's detail is written back on the left, and
the right keeps the envelope with its elevation dependence averaged out".

The dB label in each panel is that ear's spread across elevation inside the
scoring band — the cue the ear still carries. It should stay high on the
trained ear (the donor's cue replaces the participant's own, it is not removed)
and fall to ~0 on the untrained one.

Run this file directly; it saves to ``results/<id>/plots/acoustic/``.
"""

from pathlib import Path

import slab

from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.hrtf.modify.plot_compare import plot_stages
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf
from hrtf_relearning.hrtf.processing.midline import midline_arc
from hrtf_relearning.utils import paths

# --- configuration -----------------------------------------------------------
SUBJECT_ID = 'AS'
DELIVERED_SOFA = 'AS_donor_GS_env4_left'   # the file the protocol's blocks load
DONOR_ID = 'GS'
TRAINED_EAR = 'left'
N_KEEP = selection.N_KEEP                  # 4 — the same split at every stage
SHOW = False


def _load(subject_id, stem):
    path = paths.SOFA_DIR / subject_id / f'{stem}.sofa'
    if not path.exists():
        raise FileNotFoundError(f'no SOFA at {path}')
    hrtf = slab.HRTF(str(path))
    hrtf.name = stem
    return hrtf


def modification_stages(subject_id=SUBJECT_ID, delivered_sofa=DELIVERED_SOFA,
                        donor_id=DONOR_ID, trained_ear=TRAINED_EAR,
                        n_keep=N_KEEP):
    """The three HRTFs the figure draws, as ``[(label, slab.HRTF), ...]``.

    Everything is reduced to the measured median-plane arc first: the off-median
    azimuths in every SOFA are generated from that arc by
    ``expand_azimuths_with_binaural_cues``, so plotting them would be plotting
    the expansion, not the manipulation.
    """
    own = midline_arc(_load(subject_id, subject_id))
    delivered = midline_arc(_load(subject_id, delivered_sofa))
    envelope = envelope_dtf(own, ear='both', n_keep=n_keep,
                            elevation_average=False)
    untrained = 'right' if trained_ear == 'left' else 'left'
    return [
        (f'{subject_id} measured', own),
        (f'envelope only (n={n_keep})', envelope),
        (f'delivered\n{trained_ear} + {donor_id} detail / {untrained} averaged',
         delivered),
    ]


def figure(subject_id=SUBJECT_ID, delivered_sofa=DELIVERED_SOFA,
           donor_id=DONOR_ID, trained_ear=TRAINED_EAR, n_keep=N_KEEP,
           band=None, show=SHOW):
    band = selection.DEFAULT_BAND if band is None else band
    stages = modification_stages(subject_id, delivered_sofa, donor_id,
                                 trained_ear, n_keep)
    untrained = 'right' if trained_ear == 'left' else 'left'
    return plot_stages(
        stages, band=band, show=show,
        ear_labels={trained_ear: f'{trained_ear} ear (trained)',
                    untrained: f'{untrained} ear (untrained)'},
        suptitle=(f'{subject_id}: own envelope (n={n_keep}) + {donor_id} fine '
                  f'detail on the {trained_ear} ear   '
                  f'[{band[0] / 1000:.1f}–{band[1] / 1000:.1f} kHz scoring band]'))


if __name__ == '__main__':
    fig = figure()
    out_dir = paths.subject_acoustic_dir(SUBJECT_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{SUBJECT_ID}_modification_overview'
    for suffix in ('png', 'svg'):
        out = Path(out_dir) / f'{stem}.{suffix}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'wrote {out}')
