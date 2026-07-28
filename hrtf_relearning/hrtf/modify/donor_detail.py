"""
donor_detail.py — keep your own coarse envelope, wear someone else's fine detail.

    log|H_new(direction)| = envelope_n_keep( log|H_own| )  +  detail( log|H_donor| )

The cepstral split (Kulkarni & Colburn 1998) is the same one
:mod:`hrtf_relearning.hrtf.modify.shift_spectral_detail` uses, at the same
``n_keep``. Magnitude-only: the listener's own phase is kept, so ITD is exact,
and each direction is rescaled to the listener's own broadband energy, so ILD is
exact too.

WHY SPLIT IT THIS WAY. Measured on this cohort, the n_keep=4 split separates the
elevation map from everything else almost perfectly:

    own envelope + donor detail   ->  I_sim 0.256, peak r 0.68, ridge slope +0.23
    donor DTF, whole              ->  I_sim 0.251, peak r 0.62, ridge slope +0.10
    donor envelope + OWN detail   ->  I_sim 0.199, peak r 0.99, ridge slope +1.00

Swapping only the detail buys the entire decorrelation of a full donor swap;
keeping your own detail keeps your own map completely intact no matter whose
envelope carries it. So the detail is the cue and the envelope is the timbre —
which means you can hand over the cue that has to be relearned while keeping
level, ILD, head shadow and the individual colouration that make the sound
externalize. The composite also inherits normal ear statistics (donor detail RMS
and notch depths, across-direction SD between the two individuals').

CHOOSING THE DONOR — see :mod:`hrtf_relearning.hrtf.analysis.donor_selection`.
Trapeau et al. (2016) validated their earmolds as physiologically plausible by
showing that the free-vs-mold VSI dissimilarity fell inside the distribution of
VSI dissimilarities between PAIRS OF PARTICIPANTS. A donor swap is that
distribution, so any donor is plausible by construction — but do not simply take
the most dissimilar one. In their data larger VSI dissimilarity meant a larger
initial loss of elevation performance AND less or slower improvement over six
days (it still explained 26% of the variance in accuracy at the end). Van
Wanrooij & Van Opstal (2005) push the other way: too similar and the old map
absorbs the change as a bias and nothing is relearned. The two bracket an
optimum; pick a target dissimilarity in the middle of the between-subject
distribution rather than the maximum, and treat it as a parameter to calibrate
on your own pilot data.

Trapeau's molds also did not reduce VSI (free ears 0.76, with molds 0.81) — they
changed the cue without removing information — so it is tempting to require the
same of a candidate. Do not: VSI is defined on diffuse-field-normalised DTFs and
these recordings sample only the az=0 arc, so the quantity is not available here
and the available proxies are too unstable to gate on (0.19-1.01 across seven
listeners). VSI is reported; the ridge slope decides. See the constants block in
:mod:`hrtf_relearning.hrtf.analysis.donor_selection`.
"""

import copy
import logging

import numpy

from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude
from hrtf_relearning.hrtf.processing.native import source_index_map

logger = logging.getLogger(__name__)

DEFAULT_N_KEEP = 4
_EPS = numpy.finfo(float).tiny


def split_log_magnitude(mag, n_keep=DEFAULT_N_KEEP):
    """Split a linear magnitude spectrum into (envelope_db, detail_db).

    ``mag`` is ``(n_bins, n_channels)``. Pure numpy, no HRTF plumbing, so the
    manipulation can be unit-tested on synthetic spectra.
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError('mag must have shape (n_bins, n_channels)')
    log_mag_db = 20.0 * numpy.log10(numpy.maximum(mag, _EPS))
    envelope_db = 20.0 * numpy.log10(
        numpy.maximum(smooth_magnitude(mag, n_keep=int(n_keep)), _EPS))
    return envelope_db, log_mag_db - envelope_db


def compose_magnitude(own_mag, donor_mag, n_keep=DEFAULT_N_KEEP):
    """Own envelope + donor detail, returned as a linear magnitude spectrum.

    Both inputs ``(n_bins, n_channels)``, linear. This is the whole
    manipulation; :func:`donor_detail_dtf` is the HRTF wrapper.
    """
    envelope_db, _ = split_log_magnitude(own_mag, n_keep)
    _, detail_db = split_log_magnitude(donor_mag, n_keep)
    return 10.0 ** ((envelope_db + detail_db) / 20.0)


def donor_detail_dtf(hrtf, donor, n_keep=DEFAULT_N_KEEP, target_ear=None,
                     donor_ear=None, match_level=True):
    """Give one or both ears the donor's fine spectral detail.

    Parameters
    ----------
    hrtf : slab.HRTF
        The listener's own measured HRIR. Not modified in place.
    donor : slab.HRTF
        The donor's HRIR, same source grid and tap count.
    n_keep : int, default 4
        Cosine coefficients kept for the envelope. Everything above this is
        taken from the donor, everything at or below it stays the listener's.
    target_ear : {'left', 'right', None}, default None
        Which of the listener's ears receives the donor detail. ``None`` does
        both — use that when building a SOFA that the monaural ``other_ear``
        machinery will reduce afterwards, and a single ear for a genuinely
        one-sided perturbation.
    donor_ear : {'left', 'right', None}, default None
        Which of the donor's ears to take detail from. ``None`` = same side as
        ``target_ear`` (and same side per ear when doing both).
    match_level : bool, default True
        Rescale each direction to the listener's own broadband energy, making
        the per-direction ILD exactly what it was.

    Returns
    -------
    slab.HRTF
        Deep copy with the composite spectra. The listener's phase is kept
        throughout, so ITD is untouched.
    """
    if target_ear not in ('left', 'right', None):
        raise ValueError(f"target_ear must be 'left', 'right' or None, got {target_ear!r}")
    if donor_ear not in ('left', 'right', None):
        raise ValueError(f"donor_ear must be 'left', 'right' or None, got {donor_ear!r}")

    out = copy.deepcopy(hrtf)
    index = source_index_map(hrtf, donor)

    n_taps = out[0].data.shape[0]
    if donor[0].data.shape[0] != n_taps:
        raise ValueError(f'tap count differs: own {n_taps} vs donor '
                         f'{donor[0].data.shape[0]} — cannot compose')
    if not numpy.isclose(float(out[0].samplerate), float(donor[0].samplerate)):
        raise ValueError('samplerate differs between the listener and the donor')

    ears = ('left', 'right') if target_ear is None else (target_ear,)
    pairs = [(0 if e == 'left' else 1,
              (0 if e == 'left' else 1) if donor_ear is None
              else (0 if donor_ear == 'left' else 1))
             for e in ears]
    logger.info('Donor detail (n_keep=%d) onto %s ear(s)', n_keep,
                '+'.join(ears))

    for source_idx in range(out.n_sources):
        own_ir = numpy.asarray(out[source_idx].data, dtype=float)
        donor_ir = numpy.asarray(donor[index[source_idx]].data, dtype=float)
        own_spectrum = numpy.fft.rfft(own_ir, axis=0)
        donor_spectrum = numpy.fft.rfft(donor_ir, axis=0)

        for own_ch, donor_ch in pairs:
            mag = compose_magnitude(numpy.abs(own_spectrum[:, [own_ch]]),
                                    numpy.abs(donor_spectrum[:, [donor_ch]]),
                                    n_keep=n_keep)[:, 0]
            # listener's own phase -> onset and ITD exactly as measured
            new_ir = numpy.fft.irfft(
                mag * numpy.exp(1j * numpy.angle(own_spectrum[:, own_ch])),
                n=n_taps)
            if match_level:
                own_energy = float(numpy.linalg.norm(own_ir[:, own_ch]))
                new_energy = float(numpy.linalg.norm(new_ir))
                if new_energy > _EPS:
                    new_ir = new_ir * (own_energy / new_energy)
            out[source_idx].data[:, own_ch] = new_ir
    return out


def modification_params(subject_id, donor_id, n_keep=DEFAULT_N_KEEP,
                        target_ear=None, donor_ear=None, match_level=True,
                        **extra):
    """Parameter record describing a composite, for embedding in the SOFA.

    Same role as :func:`hrtf_relearning.hrtf.modify.edge_shift.modification_params`
    — the ground truth of what was done, so a run can be traced back to its
    stimulus. Pass the result to ``edge_shift._embed_modification_params`` after
    writing the SOFA.
    """
    import datetime
    import subprocess
    from pathlib import Path
    try:
        git_hash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).resolve().parent), capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        git_hash = None
    return {
        'condition': 'donor_detail',
        'subject_id': subject_id,
        'donor_id': donor_id,
        'n_keep': int(n_keep),
        'target_ear': target_ear,
        'donor_ear': donor_ear,
        'match_level': bool(match_level),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_hash': git_hash,
        **extra,
    }


# ---------------------------------------------------------------------------
# Build one subject's modified SOFA — run this file directly
# ---------------------------------------------------------------------------

SUB_ID = 'CO'          # participant with a measured <id>.sofa
OUT_SUFFIX = 'donor'   # writes <SUB_ID>_donor_<DONOR>.sofa

# Nothing else is set per participant. n_keep, the target dissimilarity, the
# band, the filter bank and the eligibility guards all live in
# hrtf.analysis.donor_selection as protocol constants and are identical for
# everyone; the donor is the only per-subject choice, and it is made by the
# rule in donor_selection.select_donor.

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('tkagg')
    import slab

    from hrtf_relearning.utils import paths
    from hrtf_relearning.hrtf.analysis import donor_selection as selection
    from hrtf_relearning.hrtf.analysis.vsi import vsi as vsi_of
    from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
    from hrtf_relearning.hrtf.modify.edge_shift import _embed_modification_params

    sofa_dir = paths.SOFA_DIR / SUB_ID
    own_path = sofa_dir / f'{SUB_ID}.sofa'
    if not own_path.exists():
        raise FileNotFoundError(f'no SOFA at {own_path} — check SUB_ID')
    own = slab.HRTF(str(own_path))
    own.name = SUB_ID
    print(f'loaded {own_path.name}')

    candidates = selection.load_candidates(SUB_ID)
    print(f'{len(candidates)} candidate donors: {", ".join(candidates)}')

    chosen, rows = selection.select_donor(own, candidates)
    reference, _ = selection.pairwise_reference({SUB_ID: own, **candidates})
    selection.report(rows, reference)
    print(f'\nchosen donor: {chosen["donor"]}  '
          f'(VSI dissimilarity {chosen["vsi_dissimilarity"]:.3f}, target '
          f'{selection.TARGET_DISSIMILARITY:.2f}, ridge slope '
          f'{chosen["ridge_slope"]:+.2f}; VSI {chosen["vsi"]:.2f} vs own '
          f'{chosen["own_vsi"]:.2f}, diagnostic only)')
    if chosen['fallback']:
        print('  !! FALLBACK: no candidate reached the ridge criterion; this is '
              'the lowest slope available. Report it.')

    donor = candidates[chosen['donor']]
    modified = donor_detail_dtf(own, donor, n_keep=selection.N_KEEP)
    print(f'built composite: own envelope (n_keep={selection.N_KEEP}) + '
          f'{chosen["donor"]} detail, both ears, own phase and level')
    print(f'VSI  own={vsi_of(own):.3f}  composite={vsi_of(modified):.3f}')

    stem = f'{SUB_ID}_{OUT_SUFFIX}_{chosen["donor"]}'
    fig = plot_ears(own, modified, band=selection.DEFAULT_BAND,
                    suptitle=f'{SUB_ID}  own envelope + {chosen["donor"]} detail')
    print(f'about to write {stem}.sofa')

    input('press enter to save (ctrl-c to discard)')
    plot_dir = paths.subject_acoustic_dir(SUB_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f'{stem}.png', bbox_inches='tight')

    out_path = sofa_dir / f'{stem}.sofa'
    modified.write_sofa(str(out_path))
    params = modification_params(
        SUB_ID, chosen['donor'], n_keep=selection.N_KEEP,
        target_dissimilarity=selection.TARGET_DISSIMILARITY,
        band=selection.DEFAULT_BAND, resolution=selection.DEFAULT_RESOLUTION,
        scores={k: chosen[k] for k in ('vsi_dissimilarity', 'vsi', 'own_vsi',
                                       'i_sim', 'peak_r', 'ridge_slope')},
        ranking=[{k: row[k] for k in ('donor', 'vsi_dissimilarity', 'vsi',
                                      'ridge_slope', 'eligible')} for row in rows])
    _embed_modification_params(out_path, params)
    print(f'wrote {out_path}')

    # the full ranking next to the SOFA, for the supplement
    import csv
    csv_path = plot_dir / f'{stem}_donor_ranking.csv'
    with open(csv_path, 'w', newline='') as handle:
        fields = ['donor', 'vsi_dissimilarity', 'vsi', 'own_vsi', 'i_sim',
                  'peak_r', 'ridge_slope', 'ridge_bias', 'distance',
                  'eligible', 'fallback']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    print(f'wrote {csv_path}')
