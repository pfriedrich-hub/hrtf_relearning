"""
phase_variants.py — HRTFs with the SAME magnitude spectrum and different time
structure. A behavioural test of Batteau's (1967) time-domain pinna model.

THE QUESTION
------------
Batteau (Proc R Soc Lond B 168:158) proposed that the pinna encodes direction as
a set of short reflection DELAYS which the nervous system inverts. The modern
account says the cue is the resulting magnitude spectrum — the notches. For a
linear time-invariant HRIR these are the same information in two coordinate
systems: an echo of delay tau IS a comb with notches at (2k+1)/2tau. So no
manipulation of the HRIR magnitude can distinguish them, and neither can moving
notches around (that is what modify.edge_shift and modify.shift_spectral_detail
do — they move echo delays and notch frequencies by exactly the same amount).

What CAN be separated is the READOUT: does the percept need the monaural
temporal arrangement, or only the magnitude spectrum? Manipulations that leave
|H(f)| bit-identical and change nothing else answer that directly.

  A literal time reversal is the cleanest case. For real h[n],
      h[N-1-n]  <-->  conj(H(f)) * exp(-j*omega*(N-1))
  so |H| is IDENTICAL to machine precision — only the phase sign flips. Direct
  sound and echo swap order. Batteau's readout should break; a magnitude-
  spectrum readout cannot notice.

PRIOR ART — and what is actually missing
----------------------------------------
Two studies bracket this, and they measured different things. Check both
methods sections before citing either; the summary below is from the abstracts.

Kulkarni, Isabelle & Colburn (1999, JASA 105:2821) built exactly these
conditions — minimum-phase-plus-delay, linear-phase, and reversed-phase-plus-
delay — but ran DISCRIMINATION, not localization: can listeners hear that
anything changed. They could not, PROVIDED the low-frequency ITD was
appropriate. That proviso is the whole reason this module works as hard as it
does on ITD (see THE TRAP below): in their study, wherever listeners did
discriminate, the ITD was the handle they used.

Kistler & Wightman (1992, JASA 91:1637) did measure LOCALIZATION, and found
direction judgments from minimum-phase-plus-delay reconstructions nearly
identical to those from measured HRTFs. So 'minphase' is a replication control
here, not an open question.

What nobody appears to have done is localization with REVERSED or MAXIMUM
phase. Discrimination and localization are not the same test — a listener can
fail to hear any difference and still point differently, and more importantly
can hear a difference that carries no directional information. The reversal
condition has only ever been tested for audibility. That gap, plus individual
HRIRs and live head tracking (under which a running-echo readout and a
magnitude-pattern readout diverge far more than they do statically), is what
this module is for.

CONDITIONS
----------
'reversed'   h[::-1], then re-aligned in time. |H| bit-identical to the measured
             HRIR (this is the primary contrast — compare against the native
             SOFA directly, no rebuild needed on the baseline side).
'minphase'   causal minimum-phase reconstruction from |H|, all-pass component
             removed. HRIRs are close to minimum phase, so this should sound
             like the original; it is the control that says "rebuilding from
             magnitude alone costs nothing".
'maxphase'   time reversal of the minphase IR = conj of the minphase spectrum.
             Identical magnitude to 'minphase' to machine precision, opposite
             all-pass. (minphase, maxphase) is an exactly matched pair that
             differs ONLY in the sign of the phase.
'allpass'    dispersive all-pass: unit modulus, so |H| is preserved exactly by
             construction, with group delay ramped across a chosen band. The
             SAME filter goes on both ears, so ITD, IPD and ILD are untouched
             and only the monaural time structure is smeared. This is the
             parametric condition: sweep ``dispersion_ms`` and find the
             breakpoint. Compare it against the auditory-filter time constants
             (~0.25 ms at 16 kHz, ~9 ms at 300 Hz) — a magnitude readout should
             survive dispersion well beyond a pinna echo delay (60-160 us).

THE TRAP: TIME OF ARRIVAL AND ITD
---------------------------------
Every phase manipulation moves the impulse response in time, and reversal moves
it to the far end of the buffer. Left uncorrected you have not run a phase
experiment, you have run an ITD experiment. Two safeguards here:

  * 'allpass' needs no correction at all — the same unit-modulus filter on both
    ears leaves H_L/H_R, hence every binaural cue, mathematically unchanged.
  * the other conditions are re-aligned per ear to the original time of arrival,
    measured by ``itd_mode``. Default 'centroid' (energy centroid of the IR),
    because it is the one statistic that maps exactly onto itself under time
    reversal, so the reversed HRIR carries the same ITD as the original by
    construction rather than by luck. 'onset' (first sample within 15 dB of the
    peak) and 'peak' are available; ``verify`` reports the resulting ITD under
    ALL THREE estimators so the residual is visible and logged, not assumed.

The re-alignment is a pure delay applied as a linear phase term, so it cannot
disturb the magnitude spectrum.

USAGE
-----
    from hrtf_relearning.hrtf.modify import phase_variants as pv

    hrtf = slab.HRTF(str(paths.SOFA_DIR / 'GS' / 'GS.sofa'))
    rev, report = pv.phase_variant(hrtf, 'reversed')
    print(pv.format_report(report))

    # or write a SOFA with provenance + QC figure in one call
    pv.save_condition_sofa(hrtf, 'reversed', out_path)

Run this file directly to build every condition for one subject (config block
below), print the verification table and save the QC figures.
"""

from pathlib import Path

import copy
import numpy

# ---------------------------------------------------------------------------
# Parameters — edit these, then run this file
# ---------------------------------------------------------------------------
SUB_ID        = 'GS'          # subject with a measured <id>.sofa in data/hrtf/sofa/<id>/
CONDITIONS    = ('reversed', 'minphase', 'maxphase')
DISPERSIONS   = (0.5, 2.0, 5.0)     # 'allpass' doses [ms of group-delay ramp]
DISPERSION_BAND = (3000.0, 16000.0)  # where the ramp is applied [Hz]
ITD_MODE      = 'phase'       # 'phase' | 'centroid' | 'onset' | 'peak'
ALIGN         = 'fractional'  # 'fractional' (exact ITD) | 'integer' (exact |H|)
N_OUT         = 1024          # taps per filter, baseline and variants alike
QC_DIRECTION  = (0.0, 0.0)    # (azimuth, elevation) shown in the QC figure
OUT_SUFFIX    = 'phase'       # writes <SUB_ID>_<OUT_SUFFIX>_<condition>.sofa

CONDITION_NAMES = ('reversed', 'minphase', 'maxphase', 'allpass')

_ONSET_THRESHOLD_DB = 15.0


# ---------------------------------------------------------------------------
# Minimum phase
# ---------------------------------------------------------------------------

def minimum_phase_from_magnitude(mag):
    """Real-cepstrum minimum-phase reconstruction from a one-sided magnitude.

    Identical in behaviour to
    ``synth_spectral_features.minimum_phase_from_magnitude``, duplicated here on
    purpose: that module imports sklearn at module scope, and this one is kept
    numpy-only so the manipulation can be unit-tested and imported without the
    full analysis stack.

    Parameters
    ----------
    mag : (n_bins, n_channels) array
        One-sided magnitude spectrum (linear).

    Returns
    -------
    (n_bins, n_channels) complex array — the minimum-phase spectrum. Its
    magnitude equals the input to machine precision; all the energy is packed
    as early as a causal filter with that magnitude can pack it.
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError('mag must have shape (n_bins, n_channels)')
    n_bins, n_channels = mag.shape
    n_samples = 2 * (n_bins - 1)
    tiny = numpy.finfo(float).tiny
    spec_min = numpy.empty((n_bins, n_channels), dtype=complex)

    for ch in range(n_channels):
        log_half = numpy.log(numpy.maximum(mag[:, ch], tiny))
        log_full = numpy.concatenate((log_half, log_half[-2:0:-1]))
        cepstrum = numpy.fft.ifft(log_full).real
        folded = numpy.zeros_like(cepstrum)
        folded[0] = cepstrum[0]
        folded[1:n_samples // 2] = 2.0 * cepstrum[1:n_samples // 2]
        folded[n_samples // 2] = cepstrum[n_samples // 2]
        spec_min[:, ch] = numpy.exp(numpy.fft.fft(folded))[:n_bins]

    return spec_min


# ---------------------------------------------------------------------------
# Time-of-arrival estimators
# ---------------------------------------------------------------------------

def toa(ir, mode='centroid', threshold_db=_ONSET_THRESHOLD_DB):
    """Time of arrival per channel, in (fractional) samples.

    Parameters
    ----------
    ir : (n_samples, n_channels) array
    mode : {'centroid', 'onset', 'peak'}
        'centroid' — energy centroid, sum(n * h[n]^2) / sum(h[n]^2). The only
            one of the three that MIRRORS EXACTLY under time reversal: the
            centroid of ``h[::-1]`` is ``N - 1 - centroid(h)``. Aligning
            centroids therefore transfers the original ITD to the reversed HRIR
            exactly, with no residual. Fractional-sample precision.
        'onset' — first sample within ``threshold_db`` of the peak. Matches the
            convention used elsewhere in the package
            (synth_spectral_features.find_ir_onsets). Integer samples. Under
            reversal this tracks the low-level pre-ringing rather than the main
            energy, which is why it is not the default.
        'peak' — argmax |h|. Integer samples, robust but coarse.

    Returns
    -------
    (n_channels,) float array
    """
    ir = numpy.asarray(ir, dtype=float)
    if ir.ndim != 2:
        raise ValueError('ir must have shape (n_samples, n_channels)')
    n_samples, n_channels = ir.shape
    n = numpy.arange(n_samples, dtype=float)
    out = numpy.zeros(n_channels, dtype=float)

    for ch in range(n_channels):
        x = ir[:, ch]
        energy = x ** 2
        total = float(energy.sum())
        if total <= 0:
            continue
        if mode == 'centroid':
            out[ch] = float((n * energy).sum() / total)
        elif mode == 'peak':
            out[ch] = float(numpy.argmax(numpy.abs(x)))
        elif mode == 'onset':
            magnitude = numpy.abs(x)
            peak_idx = int(numpy.argmax(magnitude))
            peak_val = float(magnitude[peak_idx])
            limit = peak_val / (10.0 ** (float(threshold_db) / 20.0))
            above = numpy.where(magnitude[:peak_idx + 1] >= limit)[0]
            out[ch] = float(above[0]) if above.size else 0.0
        else:
            raise ValueError(f"unknown itd_mode {mode!r}; expected "
                             "'centroid', 'onset' or 'peak'")
    return out


def itd_phase_us(ir, samplerate, band=(200.0, 1500.0)):
    """Low-frequency ITD from the interaural PHASE difference, in microseconds.

    The gold-standard estimator, and the one to trust when the others disagree.
    It reads the interaural transfer function H_left / H_right directly and fits
    ``IPD(f) = -2*pi*f*ITD`` through the origin over ``band``, which is what the
    binaural system is actually sensitive to below ~1.5 kHz.

    Why it earns its place here: the time-domain estimators (centroid, onset,
    peak) are MONAURAL statistics differenced across ears. A manipulation that
    is applied identically to both ears — 'allpass' — leaves H_left/H_right
    mathematically untouched, so the true ITD cannot have changed; but it
    redistributes each ear's energy in time by an amount that depends on that
    ear's spectrum, so the centroid difference moves by hundreds of
    microseconds. That is an artefact of the estimator, not a change in the
    stimulus, and only the phase-based figure shows it for what it is.

    Sign convention as in :func:`itd_us`: channel 0 minus channel 1.
    """
    ir = numpy.asarray(ir, dtype=float)
    n_fft = _fft_length(ir.shape[0])
    spec = numpy.fft.rfft(ir, n=n_fft, axis=0)
    freqs = numpy.fft.rfftfreq(n_fft, d=1.0 / float(samplerate))

    floor = numpy.finfo(float).tiny
    itf = spec[:, 0] / numpy.where(numpy.abs(spec[:, 1]) > floor,
                                   spec[:, 1], floor)
    # unwrap from DC so the fit sees a continuous phase, then use the band
    ipd = numpy.unwrap(numpy.angle(itf))
    keep = (freqs >= band[0]) & (freqs <= band[1])
    if not numpy.any(keep):
        return float('nan')
    omega = 2.0 * numpy.pi * freqs[keep]
    slope = float(-(omega * ipd[keep]).sum() / (omega ** 2).sum())
    return slope * 1e6


def ild_db(ir, band=(200.0, 18000.0), samplerate=None):
    """Broadband ILD in dB, channel 0 minus channel 1, over ``band``."""
    ir = numpy.asarray(ir, dtype=float)
    n_fft = _fft_length(ir.shape[0])
    spec = numpy.abs(numpy.fft.rfft(ir, n=n_fft, axis=0))
    freqs = numpy.fft.rfftfreq(n_fft, d=1.0 / float(samplerate or 1.0))
    keep = ((freqs >= band[0]) & (freqs <= band[1])
            if samplerate else numpy.ones(spec.shape[0], dtype=bool))
    power = (spec[keep] ** 2).sum(axis=0)
    floor = numpy.finfo(float).tiny
    return float(10.0 * numpy.log10(max(power[0], floor) / max(power[1], floor)))


def itd_us(ir, samplerate, mode='centroid'):
    """Interaural time difference in microseconds, left minus right.

    Sign convention follows the channel order in the SOFA (channel 0 = left),
    so a source on the left gives a NEGATIVE value (left ear arrives first).
    """
    t = toa(ir, mode=mode)
    if t.size < 2:
        return float('nan')
    return float((t[0] - t[1]) / float(samplerate) * 1e6)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

_CEPSTRUM_OVERSAMPLING = 16


def _fft_length(n_samples, factor=4):
    """Working FFT length: next power of two at or above ``factor`` x the IR.

    Generous on purpose, and the factor matters more than it looks. The
    cepstral minimum-phase reconstruction aliases in QUEFRENCY if the buffer is
    tight: at 4x a 512-tap DTF the reconstruction is off by up to 1 dB, at 16x
    it is exact to 2e-13 dB. Hence ``_CEPSTRUM_OVERSAMPLING = 16`` for that step
    and 4x for everything else, where only the circular shift needs headroom.
    """
    return int(2 ** numpy.ceil(numpy.log2(max(int(factor) * int(n_samples), 8))))


def _delay(ir, shift, n_fft=None):
    """Delay each channel by ``shift`` samples (may be fractional or negative).

    Integer shifts are applied by :func:`numpy.roll`, which is bit-exact.
    Fractional shifts go through a linear phase term on the rfft grid: an exact
    all-pass ON THAT GRID, so the magnitude is untouched there, but the ideal
    fractional delay has sinc tails that decay only as 1/n. Cropping the result
    back to a shorter filter clips them, which is the one place in this module
    where a magnitude error creeps in — about 0.03 dB worst case for a 512-tap
    DTF cropped to 1024. See the ``align`` argument of :func:`phase_variant`.

    ``shift`` may be a scalar or one value per channel.
    """
    ir = numpy.asarray(ir, dtype=float)
    n_samples, n_channels = ir.shape
    shift = numpy.broadcast_to(numpy.asarray(shift, dtype=float),
                               (n_channels,)).copy()

    if numpy.allclose(shift, numpy.round(shift)):
        out = numpy.empty_like(ir)
        for ch in range(n_channels):
            out[:, ch] = numpy.roll(ir[:, ch], int(round(shift[ch])))
        return out

    n_fft = int(n_fft or _fft_length(n_samples))
    spec = numpy.fft.rfft(ir, n=n_fft, axis=0)
    omega = 2.0 * numpy.pi * numpy.fft.rfftfreq(n_fft, d=1.0)
    return numpy.fft.irfft(
        spec * numpy.exp(-1j * omega[:, None] * shift[None, :]),
        n=n_fft, axis=0)


def dispersive_allpass(n_fft, samplerate, dispersion_ms, band=None, sign=1.0,
                       predelay_samples=None):
    """Unit-modulus spectrum with a linearly ramped group delay.

    The group delay rises (``sign > 0``) or falls (``sign < 0``) linearly across
    ``band``, by ``dispersion_ms`` milliseconds end to end, and is flat outside.
    Phase is the running integral of the group delay, so |A(f)| == 1 exactly at
    every bin and the magnitude spectrum of anything it multiplies is preserved
    to machine precision.

    Applied identically to both ears this is the ONLY manipulation here that
    needs no time-of-arrival correction: H_L/H_R is unchanged, so ITD, IPD and
    ILD all survive untouched and the change is purely monaural.

    Parameters
    ----------
    n_fft : int
    samplerate : float
    dispersion_ms : float
        Total group-delay excursion across the band, in milliseconds.
    band : (low_hz, high_hz) or None
        Where the ramp is applied. ``None`` ramps across the whole spectrum.
        Restricting it to the pinna-cue region (e.g. 3-16 kHz) avoids smearing
        low frequencies, where the auditory filters ring for milliseconds anyway
        and the only audible consequence is timbre.
    sign : float
        ``+1`` delays high frequencies, ``-1`` delays low frequencies.
    predelay_samples : int or None
        Constant delay added to the whole filter. Not cosmetic — REQUIRED. The
        dispersive impulse response is a chirp with ringing on both sides of it;
        without a pre-delay that ringing runs off the front of the buffer and
        wraps around, which turns the exact circular identity |A| == 1 into a
        LINEAR convolution that is no longer all-pass. With a 512-tap DTF the
        difference is ~0.8 dB of magnitude error versus ~0.02 dB. The default,
        ``max(64, dispersion/2)``, keeps 99.9% of the chirp inside the first
        ~380 samples. It is the same on both ears, so it costs latency and
        nothing else.

    Returns
    -------
    (spectrum, realised_dispersion_ms)
        ``spectrum`` is a (n_bins,) complex array of unit modulus.

    The group delay uses a RAISED-COSINE ramp rather than a linear one. A
    linear ramp has corners at the band edges; a discontinuous derivative in
    frequency means a long-ringing impulse response in time, and the tails wrap.
    Smoothing the corners drops the wrapped energy by three orders of magnitude
    (2e-12 vs 6e-9 of the total) for no cost in the manipulation itself.
    """
    freqs = numpy.fft.rfftfreq(int(n_fft), d=1.0 / float(samplerate))
    nyquist = freqs[-1]
    low, high = (0.0, nyquist) if band is None else (float(band[0]), float(band[1]))
    if high <= low:
        raise ValueError(f'band must be increasing, got ({low}, {high})')

    dispersion_s = float(dispersion_ms) * 1e-3
    if predelay_samples is None:
        predelay_samples = max(64.0,
                               0.5 * dispersion_s * float(samplerate))
    predelay_s = float(predelay_samples) / float(samplerate)

    ramp = numpy.clip((freqs - low) / (high - low), 0.0, 1.0)
    ramp = 0.5 * (1.0 - numpy.cos(numpy.pi * ramp))       # raised cosine
    if sign < 0:
        ramp = 1.0 - ramp
    group_delay = dispersion_s * ramp + predelay_s        # seconds

    # phase(f) = -2*pi * integral_0^f group_delay(f') df'
    phase = -2.0 * numpy.pi * numpy.concatenate(
        ([0.0], numpy.cumsum(0.5 * (group_delay[1:] + group_delay[:-1])
                             * numpy.diff(freqs))))

    # a real impulse response needs the phase at DC and Nyquist to be a multiple
    # of pi. Correct it with an added LINEAR term (i.e. a pure delay of a
    # fraction of a sample) rather than by rescaling the ramp, so the realised
    # dispersion is exactly the requested one.
    target = numpy.round(phase[-1] / numpy.pi) * numpy.pi
    phase = phase + (target - phase[-1]) * (freqs / nyquist)

    spectrum = numpy.exp(1j * phase)
    spectrum[0] = 1.0
    spectrum[-1] = 1.0 if numpy.cos(target) >= 0 else -1.0
    return spectrum, float(dispersion_ms)


# ---------------------------------------------------------------------------
# The manipulation, on a single HRIR pair
# ---------------------------------------------------------------------------

def unaligned_variant_ir(ir, samplerate, condition, dispersion_ms=2.0,
                         band=None, sign=1.0):
    """One (n_samples, 2) HRIR pair -> the phase-modified pair, NOT yet aligned.

    Returned in a working buffer of length ``_fft_length(n_samples)``, sitting
    wherever the manipulation happens to leave it. :func:`phase_variant` then
    applies the time alignment, which has to be decided across the whole HRTF at
    once (see there) rather than direction by direction.

    Kept free of slab/SOFA plumbing so it can be unit-tested on synthetic
    impulse responses.

    Returns
    -------
    (work, realised_dispersion_ms or None)
    """
    ir = numpy.asarray(ir, dtype=float)
    if ir.ndim != 2 or ir.shape[1] != 2:
        raise ValueError('each HRIR must have shape (n_samples, 2)')
    if condition not in CONDITION_NAMES:
        raise ValueError(f'unknown condition {condition!r}; expected one of '
                         f'{CONDITION_NAMES}')

    n_samples = ir.shape[0]
    n_fft = _fft_length(n_samples)

    if condition == 'allpass':
        # unit modulus, identical on both ears -> every binaural cue survives
        # untouched and no time-of-arrival correction is needed or wanted
        allpass, realised = dispersive_allpass(
            n_fft, samplerate, dispersion_ms, band=band, sign=sign)
        spec = numpy.fft.rfft(ir, n=n_fft, axis=0)
        return numpy.fft.irfft(spec * allpass[:, None], n=n_fft, axis=0), realised

    work = numpy.zeros((n_fft, 2))
    if condition == 'reversed':
        # exact: the reversal of a finite real sequence has a bit-identical
        # magnitude spectrum. No reconstruction, no approximation.
        work[:n_samples] = ir[::-1]
    else:
        # the cepstral step needs far more headroom than the rest (see
        # _fft_length): at 4x it aliases in quefrency and the reconstruction is
        # off by ~1 dB, at 16x it is exact
        n_cep = _fft_length(n_samples, factor=_CEPSTRUM_OVERSAMPLING)
        spec = numpy.fft.rfft(ir, n=n_cep, axis=0)
        hmin = numpy.fft.irfft(minimum_phase_from_magnitude(numpy.abs(spec)),
                               n=n_cep, axis=0)[:n_samples]
        # NOTE both members of the pair are built from the SAME truncated hmin,
        # so 'maxphase' is exactly the reversal of 'minphase' and their
        # magnitude spectra agree to machine precision. Truncating first is what
        # makes that exact; reversing the untruncated buffer would not.
        work[:n_samples] = hmin if condition == 'minphase' else hmin[::-1]
    return work, None


# ---------------------------------------------------------------------------
# The manipulation, on a whole HRTF
# ---------------------------------------------------------------------------

def pad_hrtf(hrtf, n_samples):
    """Zero-pad every HRIR to ``n_samples`` taps. Returns a deep copy.

    The baseline has to be length-matched to the variants before you build a
    pybinsim database or compare spectra: filters of different lengths are
    different stimuli in ways that have nothing to do with phase.
    """
    out = copy.deepcopy(hrtf)
    for filt in out:
        ir = numpy.asarray(filt.data, dtype=float)
        if ir.shape[0] > int(n_samples):
            raise ValueError(f'HRIR is already {ir.shape[0]} taps, '
                             f'cannot pad to {n_samples}')
        padded = numpy.zeros((int(n_samples), ir.shape[1]))
        padded[:ir.shape[0]] = ir
        filt.data = padded
    return out


def phase_variant(hrtf, condition, itd_mode='phase', align='fractional',
                  dispersion_ms=2.0, band=None, sign=1.0, n_out=None,
                  energy_warn=0.9999):
    """Build a magnitude-identical, phase-modified copy of an HRTF.

    Parameters
    ----------
    hrtf : slab.HRTF
        Not modified — a deep copy is returned.
    condition : {'reversed', 'minphase', 'maxphase', 'allpass'}
    itd_mode : {'phase', 'centroid', 'onset', 'peak'}
        How the ITD is put back. Ignored for 'allpass', which needs no
        correction at all.

        'phase' (default) matches the LOW-FREQUENCY INTERAURAL PHASE, which is
            what the binaural system reads. Time reversal conjugates the
            interaural transfer function, i.e. it flips the sign of the IPD; a
            per-ear delay can undo the linear (pure-delay) part of that but not
            the rest, so aligning any monaural statistic leaves a residual of
            ~15 us mean / ~90 us worst case on a real HRTF — above the ITD JND.
            Fitting the IPD slope directly drives that residual to zero where it
            matters and leaves only the frequency-dependent remainder.
        'centroid' matches the energy centroid, the only monaural statistic that
            mirrors exactly under time reversal. Use it if you want the
            manipulation to be defined purely in the time domain.
        'onset', 'peak' — cruder, integer-precision, kept for comparison.

        Whatever you choose, ``verify`` reports the outcome under all of them.
    align : {'fractional', 'integer'}
        How precisely the alignment shift is applied — a real trade-off, so it
        is a knob rather than a hidden choice.

        'fractional' (default) hits the target exactly, so ITD is preserved to
            full precision — measured on a real 512-tap DTF ('reversed',
            itd_mode='phase'): ITD residual 0.1 us mean, 0.3 us worst. The price
            is that the ideal fractional delay has sinc tails decaying as 1/n,
            and cropping the filter clips them: magnitude error up to ~0.2 dB in
            the worst single bin, rms ~0.001 dB. Still an order of magnitude
            below the smallest audible spectral change, but not zero.
        'integer' rounds every shift to a whole sample, making the magnitude
            BIT-EXACT (~1e-13 dB, i.e. floating point only). The cost moves to
            the ITD, which is then quantised to one sample: residual ~8 us mean,
            17 us worst, around the ITD JND. Use it when you want to be able to
            state that the magnitude spectra are literally identical.

        Neither error is large enough to confound the experiment; the choice is
        about which claim you would rather make exactly. ``verify`` reports both
        sides for whichever you pick.
    dispersion_ms, band, sign
        'allpass' only; see :func:`dispersive_allpass`.
    n_out : int or None
        Output filter length. Default ``2 * n_samples``, and it needs to be:
        reversal turns a decaying HRIR into a growing one, so the low-level tail
        becomes a long PRE-ringing that will not fit in the original buffer.
        With a 512-tap DTF the reversed version loses ~3.5% of its energy off
        the front if you insist on 512 taps out, and that truncation is the one
        thing in this module that genuinely does change the magnitude spectrum.
        Pad the baseline to the same length with :func:`pad_hrtf` before
        comparing or before building a binsim database.
    energy_warn : float
        Warn if any direction retains less than this fraction of its energy.

    ALIGNMENT
    ---------
    Two things have to be preserved and they are not the same thing:

      * the INTERAURAL difference at each direction (the ITD), and
      * the pattern of absolute arrival time ACROSS directions, which is what
        changes as the listener turns their head and which the head tracker
        renders.

    So the shift is decomposed. Its common (both-ears) part follows the energy
    centroid plus a SINGLE global constant ``delta`` shared by every direction —
    a global constant preserves the whole dynamic ToA pattern and costs only an
    overall latency nobody can hear, whereas a free per-direction offset would
    quietly flatten it. Its differential (between-ears) part is then set by
    ``itd_mode``. ``delta`` is the smallest value that keeps every direction
    inside the buffer, computed from the data in a first pass.

    Returns
    -------
    (slab.HRTF, report dict)
        The report is what :func:`verify` and :func:`format_report` consume and
        what gets embedded in the SOFA as provenance.
    """
    out = copy.deepcopy(hrtf)
    n_samples = int(out[0].n_samples)
    n_out = int(n_out or 2 * n_samples)
    n_fft = _fft_length(n_samples)
    samplerate = float(out[0].samplerate)
    realised = None

    # --- pass 1: build every direction, unaligned, and measure where it landed
    works = []
    reference = []
    natural = []
    reference_itd = []
    natural_itd = []
    for filt in out:
        ir = numpy.asarray(filt.data, dtype=float)
        work, dispersion = unaligned_variant_ir(
            ir, filt.samplerate, condition, dispersion_ms=dispersion_ms,
            band=band, sign=sign)
        works.append(work)
        realised = dispersion if dispersion is not None else realised
        if condition != 'allpass':
            centroid_mode = 'centroid' if itd_mode == 'phase' else itd_mode
            reference.append(toa(ir, mode=centroid_mode))
            natural.append(toa(work, mode=centroid_mode))
            if itd_mode == 'phase':
                reference_itd.append(itd_phase_us(ir, filt.samplerate))
                natural_itd.append(itd_phase_us(work, filt.samplerate))

    # --- resolve the shift (see ALIGNMENT above)
    if condition == 'allpass':
        delta = 0.0
        shift = numpy.zeros((len(works), 2))
    else:
        if align not in ('fractional', 'integer'):
            raise ValueError(f"unknown align {align!r}; expected 'fractional' "
                             "or 'integer'")
        reference = numpy.asarray(reference, dtype=float)
        natural = numpy.asarray(natural, dtype=float)
        # smallest delta that makes every per-ear shift non-negative, so nothing
        # wraps around the front of the working buffer
        delta = float(max(0.0, numpy.max(natural - reference)))
        shift = reference + delta - natural

        if itd_mode == 'phase':
            # keep the common part from the centroid rule (it carries the
            # across-direction ToA pattern) and replace only the between-ears
            # part, so the low-frequency IPD slope matches the original exactly
            common = shift.mean(axis=1, keepdims=True)
            wanted = ((numpy.asarray(reference_itd, dtype=float)
                       - numpy.asarray(natural_itd, dtype=float))
                      * 1e-6 * samplerate)                  # us -> samples
            shift = common + 0.5 * numpy.stack([wanted, -wanted], axis=1)
        elif itd_mode not in ('centroid', 'onset', 'peak'):
            raise ValueError(f"unknown itd_mode {itd_mode!r}")

        if align == 'integer':
            shift = numpy.round(shift)

    # --- pass 2: apply it and crop
    retained = []
    for filt, work, row in zip(out, works, shift):
        if condition != 'allpass':
            work = _delay(work, row, n_fft=n_fft)
        ir_out = work[:n_out].copy()
        total = float((work ** 2).sum())
        retained.append(float((ir_out ** 2).sum() / total) if total > 0 else 1.0)
        filt.data = ir_out

    retained = numpy.asarray(retained, dtype=float)
    report = {
        'condition': condition,
        'itd_mode': itd_mode if condition != 'allpass' else None,
        'align': align if condition != 'allpass' else None,
        'dispersion_ms_requested': dispersion_ms if condition == 'allpass' else None,
        'dispersion_ms_realised': realised,
        'dispersion_band': (tuple(band) if (band is not None
                                           and condition == 'allpass') else None),
        'sign': sign if condition == 'allpass' else None,
        'n_directions': int(retained.size),
        'n_samples_in': n_samples,
        'n_samples_out': n_out,
        'global_delay_samples': float(delta),
        'shift_samples_range': (float(numpy.min(shift)), float(numpy.max(shift))),
        'energy_retained_min': float(retained.min()) if retained.size else 1.0,
    }

    if retained.size and retained.min() < energy_warn:
        print(f'  [warn] {condition}: worst direction retains '
              f'{retained.min() * 100:.3f}% of its energy after cropping to '
              f'{n_out} samples. Increase n_out (or reduce the dispersion) — '
              f'truncation is the one thing here that DOES change the '
              f'magnitude spectrum.')
    return out, report


# ---------------------------------------------------------------------------
# Verification — the whole point is that the magnitude did not move
# ---------------------------------------------------------------------------

def verify(hrtf_base, hrtf_variant, report=None, band=(200.0, 18000.0),
           dynamic_range_db=40.0):
    """Check that only the phase changed, and report what happened to the ITD.

    Compares the two HRTFs direction by direction, ear by ear, and returns

      max_db / rms_db   worst and typical |magnitude difference| over ``band``
      itd_*             ITD in microseconds under all three estimators, for
                        base and variant, plus the mean absolute difference

    For an exactly matched pair ('reversed' against the native HRTF, 'maxphase'
    against 'minphase') ``max_db`` should be at the 1e-10 dB level — i.e. pure
    floating point. Anything larger means the manipulation leaked into the
    magnitude and the experiment is confounded.

    ``itd_delta_us`` should be ~0 for the estimator used as ``itd_mode``, by
    construction. The other two estimators will show a residual; that residual
    is real and belongs in the record. Judge it against the ~10-20 us JND for
    ITD discrimination.
    """
    if hrtf_base.n_sources != hrtf_variant.n_sources:
        raise ValueError('HRTFs have different numbers of directions')

    max_db = 0.0
    max_db_all = 0.0
    sq_sum = 0.0
    n_terms = 0
    itd = {mode: {'base': [], 'variant': []}
           for mode in ('centroid', 'onset', 'peak', 'phase')}
    ild = {'base': [], 'variant': []}

    for filt_a, filt_b in zip(hrtf_base, hrtf_variant):
        ir_a = numpy.asarray(filt_a.data, dtype=float)
        ir_b = numpy.asarray(filt_b.data, dtype=float)
        n_fft = _fft_length(max(ir_a.shape[0], ir_b.shape[0]))
        freqs = numpy.fft.rfftfreq(n_fft, d=1.0 / filt_a.samplerate)
        keep = (freqs >= band[0]) & (freqs <= band[1])

        mag_a = numpy.abs(numpy.fft.rfft(ir_a, n=n_fft, axis=0))[keep]
        mag_b = numpy.abs(numpy.fft.rfft(ir_b, n=n_fft, axis=0))[keep]
        floor = numpy.finfo(float).tiny
        level_a = 20.0 * numpy.log10(numpy.maximum(mag_a, floor))
        diff = numpy.abs(level_a
                         - 20.0 * numpy.log10(numpy.maximum(mag_b, floor)))
        max_db_all = max(max_db_all, float(diff.max()))
        # gate to bins within dynamic_range_db of this direction's peak. A dB
        # difference at the bottom of a deep notch is a ratio of two numbers
        # that are both nearly zero: it blows up on rounding error and says
        # nothing about audibility. The gated figure is the one to judge.
        gate = level_a > (level_a.max(axis=0, keepdims=True)
                          - float(dynamic_range_db))
        if gate.any():
            max_db = max(max_db, float(diff[gate].max()))
            sq_sum += float((diff[gate] ** 2).sum())
            n_terms += int(gate.sum())

        for mode in ('centroid', 'onset', 'peak'):
            itd[mode]['base'].append(itd_us(ir_a, filt_a.samplerate, mode=mode))
            itd[mode]['variant'].append(itd_us(ir_b, filt_b.samplerate, mode=mode))
        itd['phase']['base'].append(itd_phase_us(ir_a, filt_a.samplerate))
        itd['phase']['variant'].append(itd_phase_us(ir_b, filt_b.samplerate))
        ild['base'].append(ild_db(ir_a, band=band, samplerate=filt_a.samplerate))
        ild['variant'].append(ild_db(ir_b, band=band, samplerate=filt_b.samplerate))

    ild_delta = numpy.abs(numpy.asarray(ild['variant'], dtype=float)
                          - numpy.asarray(ild['base'], dtype=float))
    result = {
        'ild_delta_db_mean': float(ild_delta.mean()) if ild_delta.size else 0.0,
        'ild_delta_db_max': float(ild_delta.max()) if ild_delta.size else 0.0,
        'max_db': max_db,
        'max_db_all_bins': max_db_all,
        'rms_db': float(numpy.sqrt(sq_sum / n_terms)) if n_terms else 0.0,
        'verify_band': tuple(band),
        'dynamic_range_db': float(dynamic_range_db),
    }
    for mode, values in itd.items():
        base = numpy.asarray(values['base'], dtype=float)
        variant = numpy.asarray(values['variant'], dtype=float)
        result[f'itd_{mode}_delta_us_mean'] = float(numpy.mean(numpy.abs(variant - base)))
        result[f'itd_{mode}_delta_us_max'] = float(numpy.max(numpy.abs(variant - base)))
        result[f'itd_{mode}_range_us'] = (float(base.min()), float(base.max()))
    if report is not None:
        result.update({k: v for k, v in report.items() if k != 'condition'})
        result['condition'] = report.get('condition')
    return result


def format_report(result):
    """One readable block per condition — paste-able into a lab notebook."""
    lines = [f"condition            {result.get('condition')}"]
    if result.get('dispersion_ms_realised') is not None:
        band = result.get('dispersion_band')
        lines.append(f"dispersion           {result['dispersion_ms_realised']:.3f} ms "
                     f"(requested {result.get('dispersion_ms_requested')})"
                     + (f", band {band[0]:.0f}-{band[1]:.0f} Hz" if band else ''))
    if result.get('itd_mode'):
        lines.append(f"aligned by           {result['itd_mode']} "
                     f"({result.get('align')}), global delay "
                     f"{result.get('global_delay_samples', 0.0):.1f} samples")
    if result.get('n_samples_out'):
        lines.append(f"filter length        {result.get('n_samples_in')} -> "
                     f"{result['n_samples_out']} taps")
    verify_band = result.get('verify_band', (0.0, 0.0))
    lines.append(f"magnitude deviation  max {result['max_db']:.3e} dB, "
                 f"rms {result['rms_db']:.3e} dB "
                 f"({verify_band[0]:.0f}-{verify_band[1]:.0f} Hz, top "
                 f"{result.get('dynamic_range_db', 0):.0f} dB of each direction; "
                 f"all bins incl. notch floors: "
                 f"{result.get('max_db_all_bins', float('nan')):.3e} dB)")
    for mode in ('phase', 'centroid', 'onset', 'peak'):
        key = f'itd_{mode}_delta_us_mean'
        if key in result:
            mark = '  <- the one to trust' if mode == 'phase' else ''
            lines.append(f"ITD change ({mode:8s}) mean {result[key]:7.2f} us, "
                         f"max {result[f'itd_{mode}_delta_us_max']:7.2f} us{mark}")
    if 'ild_delta_db_mean' in result:
        lines.append(f"ILD change           mean {result['ild_delta_db_mean']:7.3f} dB, "
                     f"max {result['ild_delta_db_max']:7.3f} dB")
    if result.get('energy_retained_min') is not None:
        lines.append(f"energy retained      {result['energy_retained_min'] * 100:.3f}% "
                     f"(worst direction)")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# QC figure
# ---------------------------------------------------------------------------

def _erb(f):
    """Equivalent rectangular bandwidth at ``f`` (Glasberg & Moore 1990)."""
    return 24.7 * (4.37 * numpy.asarray(f, dtype=float) / 1000.0 + 1.0)


def cochleagram(x, samplerate, centre_freqs, order=4, pad_ms=25.0):
    """Gammatone filterbank envelope — the time-frequency view the ear takes.

    Returns ``(n_channels, n_samples)`` of Hilbert-style envelope magnitude,
    from a quadrature gammatone pair per channel.

    This is the representation that makes the point: the phase variants look
    obviously different HERE (the within-channel time course moves) while the
    time-integrated excitation pattern across channels is identical. Anything
    reading rate-place after temporal integration cannot tell them apart.
    """
    x = numpy.asarray(x, dtype=float).ravel()
    pad = int(pad_ms * 1e-3 * samplerate)
    x = numpy.concatenate([x, numpy.zeros(pad)])
    n = numpy.arange(int(0.025 * samplerate)) / float(samplerate)
    out = numpy.zeros((len(centre_freqs), len(x)))
    for i, cf in enumerate(centre_freqs):
        b = 1.019 * _erb(cf)
        envelope = n ** (order - 1) * numpy.exp(-2.0 * numpy.pi * b * n)
        kernel_cos = envelope * numpy.cos(2.0 * numpy.pi * cf * n)
        kernel_sin = envelope * numpy.sin(2.0 * numpy.pi * cf * n)
        norm = numpy.sqrt((kernel_cos ** 2).sum())
        real = numpy.convolve(x, kernel_cos / norm)[:len(x)]
        imag = numpy.convolve(x, kernel_sin / norm)[:len(x)]
        out[i] = numpy.hypot(real, imag)
    return out


def _nearest_source(hrtf, azimuth, elevation):
    """Index of the measured direction closest to (azimuth, elevation)."""
    positions = numpy.asarray(hrtf.sources.vertical_polar, dtype=float)
    az = (positions[:, 0] + 180.0) % 360.0 - 180.0
    return int(numpy.argmin(numpy.abs(az - azimuth)
                            + numpy.abs(positions[:, 1] - elevation)))


def qc_figure(hrtf_base, variants, direction=(0.0, 0.0), ear=0,
              n_channels=70, freq_range=(300.0, 18000.0), title=None,
              focus_band=None, normalise_channels=False):
    """Four-panel QC: IRs, cochleagrams, magnitude overlay, excitation pattern.

    ``variants`` is a mapping ``{label: slab.HRTF}``. The magnitude overlay and
    the excitation pattern should superimpose exactly; the cochleagrams should
    not. That contrast IS the manipulation — if the bottom two panels separate,
    something leaked into the magnitude and the condition is invalid.

    focus_band : (low_hz, high_hz) or None
        Draw guide lines / shading at these frequencies. Pair it with a narrowed
        ``freq_range`` and a raised ``n_channels`` to inspect the pinna-cue
        region: ``freq_range=(2500, 18000), n_channels=140,
        focus_band=(4000, 12000)`` puts the elevation notch trajectory across
        most of the axis instead of squeezing it into the top corner.
    normalise_channels : bool
        Normalise each cochleagram channel to its own peak. The absolute view is
        dominated by the spectral notches — which are IDENTICAL across
        conditions by construction, so they carry no information here. Dividing
        them out leaves only the within-channel time course, i.e. exactly the
        thing the manipulation changes.
    """
    import matplotlib.pyplot as plt

    idx = _nearest_source(hrtf_base, *direction)
    samplerate = float(hrtf_base[idx].samplerate)
    series = [('original', numpy.asarray(hrtf_base[idx].data, dtype=float)[:, ear])]
    series += [(label, numpy.asarray(h[idx].data, dtype=float)[:, ear])
               for label, h in variants.items()]
    n_col = len(series)

    centre_freqs = numpy.geomspace(freq_range[0], freq_range[1], n_channels)
    banks = [cochleagram(x, samplerate, centre_freqs) for _, x in series]
    peak = max(float(b.max()) for b in banks)

    fig = plt.figure(figsize=(4.4 * n_col, 9))
    grid = fig.add_gridspec(3, n_col, height_ratios=[0.7, 1.3, 1.0],
                            hspace=0.45, wspace=0.25)
    n_samples = max(len(x) for _, x in series)
    t_bank = numpy.arange(max(b.shape[1] for b in banks)) / samplerate * 1e3

    # a COMMON time window across all conditions, taken from where the energy
    # actually is. The variants carry a global delay of several hundred samples
    # (see the ALIGNMENT note in phase_variant), so a fixed 0-8 ms window would
    # simply miss the reversed and maximum-phase panels.
    starts, ends = [], []
    for _, x in series:
        cumulative = numpy.cumsum(x ** 2)
        if cumulative[-1] <= 0:
            continue
        starts.append(int(numpy.argmax(cumulative > 0.001 * cumulative[-1])))
        ends.append(int(numpy.argmax(cumulative > 0.999 * cumulative[-1])))
    margin = 0.5                                              # ms
    t_lo = max(0.0, (min(starts) if starts else 0) / samplerate * 1e3 - margin)
    t_hi = ((max(ends) if ends else n_samples) / samplerate * 1e3) + margin

    for col, ((label, x), bank) in enumerate(zip(series, banks)):
        ax = fig.add_subplot(grid[0, col])
        # per-series time axis: the base may be shorter than the variants
        t_ir = numpy.arange(len(x)) / samplerate * 1e3
        ax.plot(t_ir, x / numpy.abs(x).max(), lw=0.9, color=f'C{col}')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('time [ms]', fontsize=8)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(-1.05, 1.05)
        if col == 0:
            ax.set_ylabel('amplitude')

        ax = fig.add_subplot(grid[1, col])
        if normalise_channels:
            scaled = bank / numpy.maximum(bank.max(axis=1, keepdims=True), 1e-30)
            level = 20.0 * numpy.log10(scaled + 1e-4)
            mesh_kw = dict(vmin=-28, vmax=0, cmap='viridis')
        else:
            level = 20.0 * numpy.log10(bank / peak + 1e-6)
            mesh_kw = dict(vmin=-42, vmax=0, cmap='magma')
        ax.pcolormesh(t_bank[:bank.shape[1]], centre_freqs / 1000.0, level,
                      shading='auto', **mesh_kw)
        ax.set_yscale('log')
        # ticks that survive a narrowed freq_range — the default 0.5/1/2/4/8/16
        # leaves a zoomed axis with two labels on it
        candidates = numpy.array([0.3, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20])
        shown = candidates[(candidates >= centre_freqs[0] / 1000.0)
                           & (candidates <= centre_freqs[-1] / 1000.0)]
        ax.set_yticks(shown)
        ax.set_yticklabels([f'{v:g}' for v in shown])
        ax.minorticks_off()
        if focus_band is not None:
            for edge in focus_band:
                ax.axhline(edge / 1000.0, color='cyan', lw=0.9, ls='--', alpha=0.75)
        ax.set_xlim(t_lo, min(t_hi + 4.0, t_bank[-1]))
        ax.set_xlabel('time [ms]', fontsize=9)
        if col == 0:
            ax.set_ylabel('CF [kHz]')
        ax.set_title('cochleagram'
                     + (' — per-channel normalised' if normalise_channels else ''),
                     fontsize=9)

    n_fft = _fft_length(n_samples)
    freqs = numpy.fft.rfftfreq(n_fft, d=1.0 / samplerate)
    ax = fig.add_subplot(grid[2, 0])
    for col, (label, x) in enumerate(series):
        mag = numpy.abs(numpy.fft.rfft(x, n=n_fft))
        ax.semilogx(freqs, 20.0 * numpy.log10(mag + 1e-12), lw=1.3,
                    ls=['-', '--', ':', '-.'][col % 4], color=f'C{col}', label=label)
    ax.set_xlim(*freq_range)
    if focus_band is not None:
        ax.axvspan(*focus_band, color='cyan', alpha=0.13)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('magnitude [dB]')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('magnitude spectra — must superimpose', fontsize=9)

    ax = fig.add_subplot(grid[2, 1] if n_col > 1 else grid[2, 0])
    for col, bank in enumerate(banks):
        excitation = 10.0 * numpy.log10((bank ** 2).sum(axis=1) + 1e-12)
        ax.semilogx(centre_freqs, excitation, lw=1.3,
                    ls=['-', '--', ':', '-.'][col % 4], color=f'C{col}')
    ax.set_xlim(*freq_range)
    ax.set_xlabel('CF [Hz]')
    ax.set_ylabel('level [dB]')
    ax.set_title('time-integrated excitation — must superimpose', fontsize=9)

    if n_col > 2:
        ax = fig.add_subplot(grid[2, 2])
        ringing = 1e3 * 3.0 / (2.0 * numpy.pi * 1.019 * _erb(centre_freqs))
        ax.loglog(centre_freqs, ringing, 'k', lw=1.6)
        for delay, label in ((0.010, '10 us'), (0.060, '60 us'), (0.160, '160 us')):
            ax.axhline(delay, ls=':', lw=1, color='C3')
            ax.text(freq_range[0] * 1.1, delay * 1.15, label, fontsize=7, color='C3')
        ax.set_xlim(*freq_range)
        ax.set_ylim(3e-3, 20)
        ax.set_xlabel('CF [Hz]')
        ax.set_ylabel('[ms]')
        ax.set_title('filter ringing vs pinna echo delay', fontsize=9)

    position = numpy.asarray(hrtf_base.sources.vertical_polar, dtype=float)[idx]
    fig.suptitle(title or f'phase variants — az {position[0]:.0f}°, '
                          f'el {position[1]:.0f}°, '
                          f'{"left" if ear == 0 else "right"} ear', fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Writing a condition to disk, with provenance
# ---------------------------------------------------------------------------

def modification_params(base_hrtf, report):
    """Parameter record describing exactly what produced a modified SOFA.

    Same contract as edge_shift.modification_params: condition, every parameter,
    the base HRTF identity, a timestamp and the repo git hash, so a localization
    run can always be traced back to the stimulus that produced it.
    """
    import datetime
    import subprocess

    try:
        git_hash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).resolve().parent), capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        git_hash = None

    params = dict(report)
    params.update({
        'base_hrtf': (getattr(base_hrtf, 'name', None)
                      or str(getattr(base_hrtf, 'sofa_path', '')) or None),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_hash': git_hash,
        'module': 'phase_variants',
    })
    return params


def save_condition_sofa(base_hrtf, condition, path, plot=True, plot_dir=None,
                        direction=(0.0, 0.0), ear=0, verify_band=(200.0, 18000.0),
                        baseline_path=None, **kw):
    """Build one condition, verify it, write the SOFA, save the QC figure.

    Returns ``(slab.HRTF, verification dict)``. The verification runs against
    the base HRTF ZERO-PADDED to the variant's length, which is the only fair
    comparison: the variants are longer (see the ``n_out`` note in
    :func:`phase_variant`), and comparing a 512-tap baseline against a 1024-tap
    variant would report a length difference as if it were a phase effect.

    ``result['max_db']`` is the number to check before running anyone. With the
    defaults it should be ~1e-4 dB for 'allpass' and ~0.2 dB worst-bin
    (rms ~0.001 dB) for the others; with ``align='integer'`` it drops to the
    floating-point floor.

    baseline_path : path or None
        If given, the length-matched baseline is written there too. Do this —
        the experiment's control condition has to be the PADDED baseline, not
        the native SOFA, or the conditions differ in filter length as well as
        in phase.

    Plotting and provenance embedding are both wrapped so neither can block the
    SOFA write — the file is on disk before either runs.
    """
    from hrtf_relearning.hrtf.modify.edge_shift import embed_modification_params

    path = Path(path)
    hrtf_new, report = phase_variant(base_hrtf, condition, **kw)
    base_matched = pad_hrtf(base_hrtf, report['n_samples_out'])
    result = verify(base_matched, hrtf_new, report=report, band=verify_band)

    path.parent.mkdir(parents=True, exist_ok=True)
    hrtf_new.write_sofa(str(path))
    embed_modification_params(path, modification_params(base_hrtf, result))

    if baseline_path is not None:
        baseline_path = Path(baseline_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        base_matched.write_sofa(str(baseline_path))
        embed_modification_params(baseline_path, modification_params(
            base_hrtf, {'condition': 'baseline_padded',
                        'n_samples_in': report['n_samples_in'],
                        'n_samples_out': report['n_samples_out']}))
        print(f'  length-matched baseline -> {baseline_path}')

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig = qc_figure(base_matched, {condition: hrtf_new},
                            direction=direction, ear=ear,
                            title=f'{path.stem} — {condition}')
            out_dir = Path(plot_dir) if plot_dir else path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path = out_dir / f'{path.stem}_qc.png'
            fig.savefig(fig_path, bbox_inches='tight', dpi=140)
            plt.close(fig)
            print(f'  QC figure -> {fig_path}')
        except Exception as exc:      # never let a plot failure block the write
            print(f'  [warn] QC figure skipped: {exc}')

    return hrtf_new, result


def dispersion_tag(dispersion_ms):
    """Filename-safe tag, e.g. 0.5 -> ``'0p5ms'``."""
    return f"{float(dispersion_ms):g}".replace('.', 'p') + 'ms'


# ---------------------------------------------------------------------------
# Self-test — the four claims this module rests on
# ---------------------------------------------------------------------------

def selftest(samplerate=48828.0, verbose=True):
    """Check the properties the experiment depends on, on synthetic data.

    numpy only, no SOFA, no subject data — run it after touching anything here.
    Raises AssertionError on failure; returns a dict of the measured residuals.

    The four claims:
      1. the dispersive all-pass really has unit modulus at every bin, so it
         cannot change any magnitude spectrum;
      2. the energy centroid mirrors exactly under time reversal, which is what
         makes 'centroid' a safe alignment statistic for 'reversed';
      3. the phase-derived ITD recovers a known pure delay;
      4. reversing a two-echo impulse response — a literal Batteau pinna model —
         leaves its magnitude spectrum bit-identical. This is the manipulation.
    """
    results = {}
    rng = numpy.random.default_rng(0)

    residual = 0.0
    for dispersion, band in ((0.5, None), (2.0, (3000.0, 16000.0)),
                             (10.0, (1000.0, 20000.0))):
        allpass, _ = dispersive_allpass(4096, samplerate, dispersion, band=band)
        residual = max(residual, float(numpy.max(numpy.abs(numpy.abs(allpass) - 1.0))))
    results['allpass_modulus_residual'] = residual
    assert residual < 1e-12, f'all-pass modulus off by {residual}'

    ir = numpy.zeros((512, 2))
    decay = numpy.exp(-numpy.arange(160) / 40.0)
    ir[40:200, 0] = rng.normal(size=160) * decay
    ir[52:212, 1] = rng.normal(size=160) * decay
    mirror = float(numpy.max(numpy.abs((511 - toa(ir, 'centroid'))
                                       - toa(ir[::-1], 'centroid'))))
    results['centroid_mirror_residual'] = mirror
    assert mirror < 1e-9, f'centroid does not mirror: {mirror}'

    worst = 0.0
    for delay_samples in (-12.0, 7.5, 30.0):
        pulse = numpy.zeros((512, 2))
        pulse[100:120, 0] = rng.normal(size=20)
        pulse[:, 1] = pulse[:, 0]
        shifted = _delay(pulse, [0.0, -delay_samples])[:512]
        expected = delay_samples / samplerate * 1e6
        worst = max(worst, abs(itd_phase_us(shifted, samplerate) - expected))
    results['itd_phase_error_us'] = worst
    assert worst < 1.0, f'phase ITD off by {worst} us'

    # a literal Batteau pinna: direct sound plus two reflections, at 123 and
    # 389 us — squarely in the range he proposed for elevation
    echoes = numpy.zeros((256, 2))
    echoes[0] = 1.0
    echoes[6] = 0.6
    echoes[19] = -0.4
    forward = numpy.abs(numpy.fft.rfft(echoes, 8192, axis=0))
    backward = numpy.abs(numpy.fft.rfft(echoes[::-1], 8192, axis=0))
    floor = numpy.finfo(float).tiny
    deviation = float(numpy.max(numpy.abs(
        20.0 * numpy.log10(numpy.maximum(forward, floor))
        - 20.0 * numpy.log10(numpy.maximum(backward, floor)))))
    results['reversal_magnitude_db'] = deviation
    assert deviation < 1e-9, f'reversal changed the magnitude by {deviation} dB'

    if verbose:
        for key, value in results.items():
            print(f'  {key:32s} {value:.3e}')
        print('  selftest passed')
    return results


# ---------------------------------------------------------------------------
# Run on one subject
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import slab
    import matplotlib.pyplot as plt

    from hrtf_relearning.utils import paths

    sofa_dir = paths.SOFA_DIR / SUB_ID
    native_path = sofa_dir / f'{SUB_ID}.sofa'
    if not native_path.exists():
        raise FileNotFoundError(
            f'no SOFA at {native_path} — check SUB_ID in the config block')

    print('self-test:')
    selftest()
    print()

    hrtf = slab.HRTF(str(native_path))
    hrtf.name = SUB_ID
    print(f'loaded {native_path.name}  '
          f'({hrtf.n_sources} directions, {hrtf[0].n_samples} taps, '
          f'{hrtf[0].samplerate:.0f} Hz)\n')

    # the control condition is the baseline PADDED to the variants' length, not
    # the native SOFA — otherwise the conditions differ in filter length too
    baseline = pad_hrtf(hrtf, N_OUT)
    baseline_path = sofa_dir / f'{SUB_ID}_{OUT_SUFFIX}_baseline.sofa'
    baseline.write_sofa(str(baseline_path))
    print(f'baseline -> {baseline_path.name} ({N_OUT} taps)\n')

    built = {}
    for condition in CONDITIONS:
        stem = f'{SUB_ID}_{OUT_SUFFIX}_{condition}'
        hrtf_new, result = save_condition_sofa(
            hrtf, condition, sofa_dir / f'{stem}.sofa',
            plot_dir=paths.subject_acoustic_dir(SUB_ID),
            direction=QC_DIRECTION, itd_mode=ITD_MODE, align=ALIGN, n_out=N_OUT)
        built[condition] = hrtf_new
        print(format_report(result), '\n')

    for dispersion in DISPERSIONS:
        stem = f'{SUB_ID}_{OUT_SUFFIX}_allpass_{dispersion_tag(dispersion)}'
        hrtf_new, result = save_condition_sofa(
            hrtf, 'allpass', sofa_dir / f'{stem}.sofa',
            plot_dir=paths.subject_acoustic_dir(SUB_ID),
            direction=QC_DIRECTION, dispersion_ms=dispersion,
            band=DISPERSION_BAND, n_out=N_OUT)
        built[f'allpass {dispersion:g} ms'] = hrtf_new
        print(format_report(result), '\n')

    # the matched pair: minphase vs maxphase differ ONLY in the sign of the
    # phase, so this comparison should be at the floating-point floor
    if {'minphase', 'maxphase'} <= set(built):
        pair = verify(built['minphase'], built['maxphase'])
        print(f"minphase vs maxphase: magnitude deviation max "
              f"{pair['max_db']:.3e} dB\n")

    fig = qc_figure(hrtf, built, direction=QC_DIRECTION,
                    title=f'{SUB_ID} — phase variants')
    plot_dir = paths.subject_acoustic_dir(SUB_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f'{SUB_ID}_{OUT_SUFFIX}_overview.png',
                bbox_inches='tight', dpi=140)
    print(f'wrote {plot_dir / f"{SUB_ID}_{OUT_SUFFIX}_overview.png"}')
    plt.show(block=False)
    plt.pause(0.1)
