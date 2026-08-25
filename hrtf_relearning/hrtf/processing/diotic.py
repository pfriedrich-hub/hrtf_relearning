"""
diotic.py

Copy one ear's measured DTF onto BOTH ears, so the interaural spectral
difference (ISD) carries no elevation information.

Why this exists
---------------
The received spectrum is the product of an unknown source spectrum and the
direction-dependent filter, P(f) = S(f) * H(f, dir). One equation, two
unknowns. The interaural ratio is the one channel that solves it for free:

    P_L(f) / P_R(f) = H_L(f) / H_R(f)

-- the source cancels exactly, at every frequency and every ripple density.
Whether a listener actually uses that channel is an empirical question, and
this module builds the SOFA that answers it: with the same DTF on both ears
the ratio is frequency-flat, so an ISD read-out has nothing left to read and
the elevation cue is available only monaurally.

How much is being removed is measurable, not assumed -- see
`localization_helpers.spectral_metrics.isd_depth`. Measured 2026-08-18 in the
0.5-2 ripples/oct band at az 0: AS 1.21 dB rms against a 2.02 dB monaural cue,
GS 1.37/2.47, FS 2.05/2.62. A symmetric dummy head (KEMAR) has exactly 0.00,
which is the point -- the channel exists only because real ears are asymmetric.

What it is NOT
--------------
Not a monaural condition: both ears still receive a full elevation cue, and
off the midline they still receive the spherical-head ILD and ITD, so the
sound lateralizes and externalizes normally. Only the *difference* between the
two ears' spectra is made uninformative.

The unavoidable side effect: the ear that is overwritten now carries the other
ear's pattern, which the listener has never heard at that ear. The duplicated
ear's own monaural cue is untouched, so the clean comparison is against the
listener's better/trained ear; counterbalance `ear` across subjects if that
side effect needs ruling out.

Where the manipulation is applied
---------------------------------
On the measured az=0 arc, before azimuth expansion -- the same
extract -> modify -> re-expand workflow every other manipulation uses (see
processing/midline.py). Only that arc is measured; the other 456 directions in
a subject SOFA are the arc times a spherical-head model, so modifying all 475
directly would edit synthesised data.
"""
import copy
import datetime
import logging
import subprocess

from hrtf_relearning.hrtf.processing.midline import midline_arc, expand_from_midline

CHANNELS = {'left': 0, 'right': 1}


def diotic_arc(arc, ear='left'):
    """Return a copy of `arc` with `ear`'s channel duplicated into both channels.

    arc : slab.HRTF, the az=0 arc (e.g. from processing.midline.midline_arc).
    ear : 'left' or 'right', the ear whose DTF is kept.
    """
    if ear not in CHANNELS:
        raise ValueError(f"ear must be 'left' or 'right', got {ear!r}")
    channel = CHANNELS[ear]
    out = copy.deepcopy(arc)
    for tf in out:
        tf.data = tf.data[:, [channel, channel]]  # same DTF, ITD and ILD both zero
    return out


def diotic_hrtf(hrtf, ear='left', head_radius=None, az_range=(-50, 50),
                **expand_kwargs):
    """Full-sphere SOFA whose two ears carry the same measured DTF.

    Extracts the measured az=0 arc, duplicates `ear` into both channels, and
    re-expands across azimuth so the spherical-head ILD and ITD are imposed by
    the model exactly as they are for the native set (itd_method='phase', so
    native and diotic share their ITD bit-for-bit).

    head_radius : REQUIRED, metres. It must be the radius the native SOFA was
        built with, not a default. The whole point of this manipulation is that
        the diotic and natural conditions differ ONLY in the interaural
        spectral difference; a different sphere radius would change their ITD
        and ILD as well and confound the comparison. Recover it from the file
        the subject is actually being tested against:

            from hrtf_relearning.hrtf.record.fit_head_radius import fit_from_sofa
            head_radius = fit_from_sofa(native_sofa_path)['head_radius']

        which returns the imposed radius with ~0 residual for any SOFA this
        pipeline built. A large residual means the native set was expanded with
        the legacy itd_method='onset' and re-expanding it here would change the
        ITD; rebuild the native SOFA before running the experiment.

    Returns a slab.HRTF named `<base name>_diotic_<ear>`.
    """
    if head_radius is None:
        raise ValueError(
            'head_radius is required -- pass the radius the native SOFA was '
            'built with (fit_head_radius.fit_from_sofa), not a default, or the '
            'diotic set will differ from it in ITD and ILD as well as ISD.')
    arc = diotic_arc(midline_arc(hrtf), ear=ear)
    arc.name = f'{getattr(hrtf, "name", "hrtf")}_diotic_{ear}_midline'
    out = expand_from_midline(arc, az_range=az_range, head_radius=head_radius,
                              **expand_kwargs)
    logging.info('diotic: duplicated %s ear across %d directions (r=%.4f m) -> %s',
                 ear, out.n_sources, head_radius, out.name)
    return out


def _git_hash():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def modification_params(base_hrtf, ear, **kw):
    """Self-documenting record embedded in the SOFA (GLOBAL_ModificationParams)."""
    return {
        'condition': 'diotic',
        'ear': ear,
        'base_hrtf': (getattr(base_hrtf, 'name', None)
                      or str(getattr(base_hrtf, 'sofa_path', '')) or None),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_hash': _git_hash(),
        'module': 'diotic',
        **kw,
    }


def save_diotic_sofa(base_hrtf, path, ear='left', head_radius=None, **kw):
    """Write the diotic variant to `path` and embed its params. Returns the HRTF.

    `head_radius` is required -- see `diotic_hrtf`. Extra keyword arguments are
    forwarded to it (e.g. az_range) and also recorded in the embedded params.
    """
    hrtf_new = diotic_hrtf(base_hrtf, ear=ear, head_radius=head_radius, **kw)
    kw['head_radius'] = head_radius
    hrtf_new.write_sofa(str(path))
    from hrtf_relearning.hrtf.modify.edge_shift import embed_modification_params
    embed_modification_params(path, modification_params(base_hrtf, ear, **kw))
    return hrtf_new
