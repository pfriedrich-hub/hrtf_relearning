"""
midline.py — modify the 19 measured DTFs, not the 475 synthesised ones.

WHY THIS EXISTS. Only the az=0 arc is measured. Everything else in a subject's
SOFA is that arc times a spherical-head model:

    |H(az, el, f)| = |H(0, el, f)| * |H_sphere(az, el, f)| / |H_sphere(0, el, f)|
     IPD(az, el, f) = IPD_sphere(az, el, f)

(:func:`hrtf_relearning.hrtf.record.processing.expand_azimuths_with_binaural_cues`).
So a 475-direction file has 19 independent measurements in it, and a cue
manipulation applied to all 475 is applied 25 times over to the same data --
once per azimuth, each time to a spectrum that already has the head shadow
baked into it.

Doing it on the arc instead and re-expanding has three consequences:

* The ILD at az!=0 is the midline ILD plus the model's, exactly, because the
  model contributes nothing at az=0 by symmetry. Any ILD error the
  manipulation introduces is therefore a MIDLINE error, propagated as a
  constant offset -- so :func:`qc_midline` on 19 directions is a complete
  check, not a sample of 475.
* The ITD is a function of geometry alone (``itd_method='phase'``), so a
  magnitude-only manipulation cannot perturb it and the native and modified
  sets share their ITD bit-for-bit.
* The envelope is fitted to the listener's own pinna response rather than to
  the pinna response times the head shadow, so the n_keep coefficients are
  spent on the thing they are meant to describe.

The arc is read back out of the finished SOFA rather than from the npz: the
expansion never touches az=0 magnitudes (step 2 skips them) and only
time-shifts the right ear there, so the arc in the SOFA is the same one that
went in, and no reference recording or deconvolution replay is needed.
"""

import copy
import logging

import numpy
import slab

logger = logging.getLogger(__name__)

MIDLINE_TOL_DEG = 1e-2

#: Bands qc_midline reports the ILD in, and the tolerance on each. The
#: broadband figure is exact by construction (both match_level rescales), so a
#: non-zero value there means something upstream stopped preserving energy.
QC_BANDS = {'broadband': (None, None), 'low': (200.0, 2000.0),
            'high': (2000.0, 16000.0)}

#: The low-band tolerance is not a round number, it comes from a criterion: the
#: modified midline ILD has to stay INSIDE the range of ILDs the listener's own
#: HRTF produces at some real direction. Measured over this cohort that range is
#: roughly +-6 to +-9 dB in 200-2000 Hz, and the midline sits 0.6-3.2 dB off
#: centre, so ~5 dB of deviation is the point at which az=0 starts to look more
#: lateral than any direction that physically exists. Beyond that the ILD and
#: the ITD -- which is forced to exactly zero at the midline -- disagree by more
#: than the whole physical range of the azimuth axis, which is the in-head
#: localization configuration.
QC_TOLERANCE_DB = {'broadband': 0.05, 'low': 5.0, 'high': 2.0}
QC_ITD_TOLERANCE_US = 1.0


# ---------------------------------------------------------------------------
# arc <-> full set
# ---------------------------------------------------------------------------

def midline_sources(hrtf, tol=MIDLINE_TOL_DEG):
    """Indices of the az=0 arc, sorted by elevation."""
    azimuth = numpy.mod(numpy.asarray(hrtf.sources.vertical_polar[:, 0], dtype=float), 360.0)
    idx = numpy.where(numpy.isclose(azimuth, 0.0, atol=tol)
                      | numpy.isclose(azimuth, 360.0, atol=tol))[0]
    if idx.size == 0:
        raise ValueError('no az=0 sources found — is this an expanded SOFA?')
    return idx[numpy.argsort(numpy.asarray(hrtf.sources.vertical_polar[idx, 1], dtype=float))]


def midline_arc(hrtf, tol=MIDLINE_TOL_DEG):
    """The az=0 arc as a standalone ``slab.HRTF``.

    The manipulations in :mod:`hrtf_relearning.hrtf.modify` take a slab.HRTF
    and do not care how many directions are in it, so they work on the arc
    unchanged — including :func:`...donor_detail.donor_detail_dtf`, whose
    ``source_index_map`` will match the donor's arc to this one.
    """
    idx = midline_sources(hrtf, tol=tol)
    data = numpy.stack([numpy.asarray(hrtf[i].data, dtype=float) for i in idx])
    arc = slab.HRTF(data=data,
                    sources=numpy.asarray(hrtf.sources.vertical_polar[idx], dtype=float),
                    samplerate=hrtf.samplerate, datatype='FIR')
    arc.name = f'{getattr(hrtf, "name", "hrtf")}_midline'
    logger.debug('midline arc: %d directions, el %.1f..%.1f',
                 len(idx), arc.sources.vertical_polar[:, 1].min(),
                 arc.sources.vertical_polar[:, 1].max())
    return arc


class _ArcContainer:
    """Duck-typed stand-in for ``record.processing.ImpulseResponses``.

    Only the four things ``expand_azimuths_with_binaural_cues`` touches are
    implemented. This exists so this module does not import
    ``record.recordings``, which imports ``freefield`` at module level and so
    is unavailable on the cue-editing-only install.
    """

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def __getitem__(self, key):
        return self.data[key]

    def get_sources(self, distance=1.4):
        out = []
        for key in self.data:
            _speaker, azimuth, elevation = key.split('_')
            out.append([float(azimuth), float(elevation), distance])
        return numpy.asarray(out, dtype=float)


def expand_from_midline(arc, az_range=(-50, 50), head_radius=0.0875,
                        itd_method='phase', speaker='0', **kwargs):
    """Rebuild the full azimuth set from an az=0 arc.

    Same code path the recording pipeline uses — this only wraps the arc in the
    container that function expects and unwraps the result. ``itd_method``
    defaults to ``'phase'`` here (unlike the recording pipeline, where the
    default is still the legacy ``'onset'``) because the whole point of
    modifying on the arc is that native and modified sets are expanded
    separately and must come out with identical ITDs.
    """
    import pyfar
    from hrtf_relearning.hrtf.record.processing import expand_azimuths_with_binaural_cues

    fs = int(arc.samplerate)
    data = {}
    for i in range(arc.n_sources):
        azimuth, elevation, _ = numpy.asarray(arc.sources.vertical_polar[i], dtype=float)
        key = f'{speaker}_{numpy.mod(azimuth, 360.0):.1f}_{elevation:.1f}'
        data[key] = pyfar.Signal(numpy.asarray(arc[i].data, dtype=float).T, fs)

    expanded = expand_azimuths_with_binaural_cues(
        _ArcContainer(data, {'fs': fs}), az_range=az_range,
        head_radius=head_radius, itd_method=itd_method, **kwargs)

    keys = list(expanded.data.keys())
    out = slab.HRTF(
        data=numpy.stack([expanded.data[k].time.T for k in keys]),
        sources=expanded.get_sources(), samplerate=fs, datatype='FIR')
    out.name = getattr(arc, 'name', 'hrtf').replace('_midline', '')
    logger.info('expanded %d midline directions -> %d sources (itd_method=%s)',
                arc.n_sources, out.n_sources, itd_method)
    return out


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def _ild_db(hrtf, band):
    """Per-direction ILD (left minus right) in dB. ``band=(None, None)`` -> broadband L2."""
    low, high = band
    out = []
    for i in range(hrtf.n_sources):
        data = numpy.asarray(hrtf[i].data, dtype=float)
        if low is None:
            out.append(20.0 * numpy.log10(numpy.linalg.norm(data[:, 0])
                                          / max(numpy.linalg.norm(data[:, 1]), 1e-30)))
            continue
        freqs = numpy.fft.rfftfreq(data.shape[0], d=1.0 / float(hrtf[i].samplerate))
        mask = (freqs >= low) & (freqs <= high)
        left = numpy.abs(numpy.fft.rfft(data[:, 0]))[mask]
        right = numpy.abs(numpy.fft.rfft(data[:, 1]))[mask]
        out.append(float(numpy.mean(20.0 * numpy.log10(
            numpy.maximum(left, 1e-30) / numpy.maximum(right, 1e-30)))))
    return numpy.asarray(out)


def _itd_us(hrtf, band=(200.0, 1500.0)):
    """Per-direction ITD from the interaural phase slope, microseconds.

    Magnitude-invariant, unlike a cross-correlation or an onset: it measures
    the delay that was imposed, so a magnitude-only edit must leave it at zero
    change. Use this, not xcorr, to check that a manipulation left ITD alone.
    """
    out = []
    for i in range(hrtf.n_sources):
        data = numpy.asarray(hrtf[i].data, dtype=float)
        freqs = numpy.fft.rfftfreq(data.shape[0], d=1.0 / float(hrtf[i].samplerate))
        ipd = numpy.unwrap(numpy.angle(numpy.fft.rfft(data[:, 1])
                                       * numpy.conj(numpy.fft.rfft(data[:, 0]))))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        out.append(-numpy.polyfit(2 * numpy.pi * freqs[mask], ipd[mask], 1)[0] * 1e6)
    return numpy.asarray(out)


def _spectral_sd_db(hrtf, ear, band=(4000.0, 16000.0)):
    """SD of the log-magnitude across directions, averaged over ``band``, one ear."""
    channel = 0 if ear == 'left' else 1
    spectra = []
    for i in range(hrtf.n_sources):
        data = numpy.asarray(hrtf[i].data, dtype=float)
        freqs = numpy.fft.rfftfreq(data.shape[0], d=1.0 / float(hrtf[i].samplerate))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        spectra.append(20.0 * numpy.log10(
            numpy.maximum(numpy.abs(numpy.fft.rfft(data[:, channel])), 1e-30))[mask])
    return float(numpy.mean(numpy.std(numpy.asarray(spectra), axis=0)))


def qc_midline(native_arc, modified_arc, processed_ear=None, raise_on_fail=True,
               tolerance_db=None, itd_tolerance_us=QC_ITD_TOLERANCE_US):
    """Check a modified midline arc against the native one.

    This is the complete ILD/ITD check for the whole subject: the expansion
    adds the model's contribution on top of whatever is here, and contributes
    nothing at az=0, so every deviation the listener will encounter at any
    azimuth originates in these 19 directions.

    Parameters
    ----------
    native_arc, modified_arc : slab.HRTF
        Same directions, same order — use :func:`midline_arc` on both.
    processed_ear : {'left', 'right', None}
        Ear the monaural reduction was applied to. If given, the elevation and
        azimuth spectral SD of that ear are reported: elevation SD must FALL
        (the cue is being removed) while azimuth SD should survive (head
        shadow, which is what keeps the ear plausible enough to externalize).
    raise_on_fail : bool
        Raise instead of warning when a tolerance is exceeded.

    Returns
    -------
    dict
        ``ild_<band>`` mean/max absolute deviation in dB, ``itd`` likewise in
        microseconds, plus ``elevation_sd``/``azimuth_sd`` when
        ``processed_ear`` is given, and ``passed``.
    """
    if native_arc.n_sources != modified_arc.n_sources:
        raise ValueError(f'arc length differs: {native_arc.n_sources} vs '
                         f'{modified_arc.n_sources}')
    tolerance_db = dict(QC_TOLERANCE_DB, **(tolerance_db or {}))

    report, failures = {}, []
    for name, band in QC_BANDS.items():
        deviation = numpy.abs(_ild_db(modified_arc, band) - _ild_db(native_arc, band))
        report[f'ild_{name}_mean'] = float(deviation.mean())
        report[f'ild_{name}_max'] = float(deviation.max())
        if deviation.mean() > tolerance_db[name]:
            failures.append(f'ILD {name}: {deviation.mean():.2f} dB mean '
                            f'(tolerance {tolerance_db[name]:.2f})')

    itd_deviation = numpy.abs(_itd_us(modified_arc) - _itd_us(native_arc))
    report['itd_mean_us'] = float(itd_deviation.mean())
    report['itd_max_us'] = float(itd_deviation.max())
    if itd_deviation.max() > itd_tolerance_us:
        failures.append(f'ITD: {itd_deviation.max():.2f} us max '
                        f'(tolerance {itd_tolerance_us:.2f}) — a magnitude-only '
                        f'modification must not move it at all')

    if processed_ear is not None:
        for label, hrtf in (('native', native_arc), ('modified', modified_arc)):
            report[f'elevation_sd_{label}'] = _spectral_sd_db(hrtf, processed_ear)
        if report['elevation_sd_modified'] >= report['elevation_sd_native']:
            failures.append(
                f"elevation SD on the {processed_ear} ear did not fall "
                f"({report['elevation_sd_native']:.2f} -> "
                f"{report['elevation_sd_modified']:.2f} dB) — the cue is still there")

    report['passed'] = not failures
    if failures:
        message = 'midline QC failed:\n  ' + '\n  '.join(failures)
        if raise_on_fail:
            raise ValueError(message)
        logger.warning(message)
    return report


def format_qc(report):
    """One-line-per-metric rendering of :func:`qc_midline`, for build logs."""
    lines = ['midline QC (modified vs native, az=0 arc)']
    for name in QC_BANDS:
        lines.append(f'  ILD {name:<10} {report[f"ild_{name}_mean"]:7.3f} dB mean '
                     f'{report[f"ild_{name}_max"]:7.3f} max')
    lines.append(f'  ITD            {report["itd_mean_us"]:7.3f} us mean '
                 f'{report["itd_max_us"]:7.3f} max')
    if 'elevation_sd_native' in report:
        lines.append(f'  elevation SD   {report["elevation_sd_native"]:7.3f} -> '
                     f'{report["elevation_sd_modified"]:7.3f} dB  (processed ear)')
    lines.append(f'  {"PASS" if report["passed"] else "FAIL"}')
    return '\n'.join(lines)
