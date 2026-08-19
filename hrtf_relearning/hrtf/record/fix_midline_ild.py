"""Retroactive midline-ILD correction for SOFAs built before the 2026-08-18 fix.

Background
----------
Until 2026-08-18 `record_hrir` passed `align_interaural=True` down to
`compute_ir`, which zeroed the frontal ITD/ILD on the subject IRs AND on the
reference IRs *before* `equalize` divided one by the other. A broadband level
match is an energy measure and does not commute with a per-frequency division,
so instead of removing the reference's channel imbalance it discarded the
cancellation the division would have performed and left a residual: a flat
-1.35 dB (left quieter) at every frequency, the same sign for every subject
sharing a reference. `expand_azimuths_with_binaural_cues` then copied the
frontal arc to every azimuth, turning it into a constant ILD bias over the
whole +-50 deg field -- worth ~+6 deg of rightward localization bias.

The pipeline is fixed (`equalize(align_interaural=True)` ->
`zero_frontal_interaural`). Subjects recorded from now on need nothing. This
script repairs SOFAs that were already built, WITHOUT the raw sweeps: because
the artifact is a per-ear scalar, the same end state is reached by applying the
inverse scalar to the finished SOFA.

Method
------
For each elevation, the frontal (az=0) direction sets a per-ear gain pair that
drives the broadband energy ILD over `ild_band` to zero while preserving
total power (see `frontal_ild_db` for why energy and not a per-octave
weighted log-mean). That pair is applied to EVERY azimuth at that
elevation, because the azimuth expansion built those directions as copies of
the frontal response times a per-ear relative model correction -- so a scalar
on the frontal response propagates to the whole column.

Exactness
---------
Above 800 Hz this is identical to rebuilding from the raw sweeps with the fixed
pipeline. Below 800 Hz it is not: `lowfreq_extrapolate` anchors that region to
the spherical-head model's absolute magnitude, which does not scale with the
measured level, so a scalar applied afterwards also scales the model-anchored
part. The discrepancy is at most the correction itself at the very bottom of
the spectrum, tapering to zero at 800 Hz -- a band where ILD contributes
essentially nothing to lateralization. Rebuild from raw where the sweeps exist.

The frontal ITD needs no repair: `expand_azimuths_with_binaural_cues` with
`itd_method='phase'` (step 3a) already zeroes it after equalization, and
measurement confirms 0.00 us phase-slope ITD at the midline in the affected
SOFAs.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy

ILD_BAND_HZ = (200.0, 16000.0)
EPS = 1e-30


def frontal_ild_db(spectrum, freqs, ild_band=ILD_BAND_HZ):
    """Broadband ENERGY interaural level difference over `ild_band`, in dB.

    `spectrum` is (2, n_freq), left ear first. Positive means left is louder.

    Energy, not a per-octave weighted log-mean. Both were tried against the one
    behavioural anchor available: AS's midline offset predicts +5.93 deg of
    rightward bias under the energy measure and +8.82 deg under the per-octave
    log-mean (the ILD-vs-azimuth slopes are nearly identical, 0.212 vs 0.220
    dB/deg, so this is a real disagreement about the intercept, not a change of
    units). Her measured bias was +5.62 deg. Zeroing the log-mean would have
    left her about 3 deg biased the other way. Lateralization of a broadband
    stimulus integrates energy across frequency, so the loud bands should
    dominate; a per-octave average deliberately upweights the quiet ones.
    """
    band = (freqs >= ild_band[0]) & (freqs <= ild_band[1])
    if not band.any():
        raise ValueError(f"ild_band {ild_band} selects no frequency bins")
    power_l = float(numpy.mean(numpy.abs(spectrum[0, band]) ** 2))
    power_r = float(numpy.mean(numpy.abs(spectrum[1, band]) ** 2))
    return float(10 * numpy.log10((power_l + EPS) / (power_r + EPS)))


def _energy_preserving_gains(spectrum, ild_db):
    """Gain pair with gain_L/gain_R = 10**(-ild_db/20) that keeps total power."""
    ratio = 10 ** (-ild_db / 20.0)
    p_l = float(numpy.mean(numpy.abs(spectrum[0]) ** 2))
    p_r = float(numpy.mean(numpy.abs(spectrum[1]) ** 2))
    gain_r = numpy.sqrt((p_l + p_r) / (ratio ** 2 * p_l + p_r))
    return ratio * gain_r, gain_r


def fix_sofa(sofa_file, ild_band=ILD_BAND_HZ, archive_dir=None, dry_run=False):
    """Zero the midline ILD of an existing SOFA, in place.

    Returns {elevation: (gain_L, gain_R, ild_before_dB)}. With dry_run=True
    nothing is written and the corrections are only reported.
    """
    sofa_file = Path(sofa_file)
    with h5py.File(sofa_file, "r") as sofa:
        ir = numpy.array(sofa["Data.IR"])
        pos = numpy.array(sofa["SourcePosition"])
        fs = float(numpy.array(sofa["Data.SamplingRate"]).ravel()[0])

    az = (pos[:, 0] + 180) % 360 - 180
    el = numpy.round(pos[:, 1], 6)
    freqs = numpy.fft.rfftfreq(ir.shape[-1], 1 / fs)

    corrections = {}
    for elevation in numpy.unique(el):
        column = numpy.where(el == elevation)[0]
        frontal = column[numpy.argmin(numpy.abs(az[column]))]
        if abs(az[frontal]) > 1.5:
            logging.warning("No frontal direction at elevation %.2f - skipped.", elevation)
            continue

        spectrum = numpy.fft.rfft(ir[frontal], axis=-1)
        ild_db = frontal_ild_db(spectrum, freqs, ild_band)
        gain_l, gain_r = _energy_preserving_gains(spectrum, ild_db)

        ir[column, 0, :] *= gain_l
        ir[column, 1, :] *= gain_r
        corrections[float(elevation)] = (float(gain_l), float(gain_r), ild_db)

    if dry_run:
        return corrections

    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        archived = archive_dir / f"{sofa_file.stem}_pre_ildfix_{stamp}.sofa"
        shutil.copy2(sofa_file, archived)
        logging.info("Archived original to %s", archived)

    mean_db = float(numpy.mean([c[2] for c in corrections.values()]))
    with h5py.File(sofa_file, "r+") as sofa:
        sofa["Data.IR"][...] = ir
        sofa.attrs["DateModified"] = datetime.now().isoformat(sep=" ")
        sofa.attrs["GLOBAL_MidlineILDFix"] = (
            f"midline ILD zeroed post hoc {datetime.now().isoformat(timespec='seconds')}; "
            f"criterion=broadband energy ILD {ild_band[0]:.0f}-{ild_band[1]:.0f} Hz, "
            f"energy preserving, per-elevation gains applied across the whole azimuth "
            f"column; mean ILD removed {mean_db:+.3f} dB. Corrects the pre-2026-08-18 "
            f"align_ild-before-referencing artifact. Frontal ITD was already 0."
        )
    logging.info("%s: removed %+.2f dB mean midline ILD over %d elevations.",
                 sofa_file.name, mean_db, len(corrections))
    return corrections


# %% apply to the subjects currently running
if __name__ == "__main__":
    from hrtf_relearning.utils import paths

    logging.basicConfig(level=logging.INFO)
    for subject_id in ("AS", "NW"):
        subject_sofa = paths.SOFA_DIR / subject_id / f"{subject_id}.sofa"
        fix_sofa(subject_sofa, archive_dir=subject_sofa.parent / "old")
