"""
shift_spectral_detail.py — move a subject's own fine spectral cues up in frequency.

  1. SELECT a frequency window (default: the Trapeau et al. 2016 peak-VSI octave,
     5.7-11.3 kHz).
  2. SEPARATE features from broad shape by cepstral smoothing: a truncated cosine
     series on log-magnitude gives the coarse envelope (Kulkarni & Colburn 1998,
     Nature 396:747); what is left over is the fine detail — the pinna peaks and
     notches that carry the elevation cue.
  3. MOVE the selected features along the ERB-number axis by ``shift_erb``
     (> 0 = up in frequency). The envelope stays exactly where it is.

Magnitude-only: the original phase is kept, so onset/ITD structure is untouched.
The input HRTF already carries its binaural cues and they pass through intact:
ITD is untouched, and because both ears get the same ERB transport the broadband
ILD is preserved while its fine structure travels with the cue.

Why a constant ERB step: it preserves notch/peak SPACING on the auditory scale,
so the pattern is displaced rather than distorted. Every direction keeps a unique
pattern — a bijective, relearnable remap rather than a destroyed or conflicting
cue. Above ~1 kHz a constant ERB step is roughly a constant frequency factor:
1.3x ~= 2.4 ERB, 1.4x ~= 3.0 ERB.

TRANSPORT
---------
The features are selected FIRST and then carried to their new frequencies::

    d_sel   = w * detail                     # features inside the window
    d_rest  = detail - d_sel                 # everything else, left in place
    d_moved = d_sel  translated by +shift_erb on the ERB axis
    detail' = d_rest + d_moved

So THE MODIFIED REGION MOVES WITH THE CONTENT. The window says which features to
take, not where the result is allowed to live: they land in :func:`target_band`,
which lies above ``band`` for a positive shift. Nothing is dropped at the band
edge, and nothing is duplicated. Use :func:`describe` to print both bands and the
shift in Hz at each edge — the same ERB step spans far more Hz at 11 kHz than at
5.7 kHz.

EDGES: ``skirt_octaves``
------------------------
Hard edges (the default, 0) leave a step in the detail at the old and new band
edge, as deep as the detail happened to be there — up to ~9 dB if a notch sits on
the edge. A raised-cosine ramp of ``skirt_octaves`` softens it: 0.05 octaves takes
that 9 dB step down to ~0.9 dB, 0.1 octaves to ~0.2 dB.

Where the ramp ends up: it multiplies the detail BEFORE the transport, so it is
carried along and the same soft edge appears on the sides of the NEW window,
at ``target_band`` of the ramp span. There is no separate target-side parameter,
and you do not want one — a ramp applied only at the target would fade features
that had already been removed from their origin, i.e. destroy them.

The ramp sits OUTSIDE the nominal band (the window is flat across
``[low, high]`` and rolls off beyond it), so everything strictly inside
SHIFT_BAND still moves whole. Only content within the ramp span is split: it
keeps ``1 - a`` of its depth at the old frequency and arrives with ``a`` at the
new one — the same notch, shallower, in two places. Energy is conserved and
nothing vanishes, but it is a weak conflicting cue, so keep the ramp short and
put the band edges where the detail is quiet. :func:`describe` prints the ramp
spans at both ends so you can check what falls inside them.

Run this file directly to apply the shift to one subject's SOFA: set SUB_ID and
the parameters in the config block below, run, eyeball the before/after image,
then press enter to write <SUB_ID>_shift_<shift>erb.sofa next to the native one
(the shift is in the filename, so different settings never overwrite each other).
"""

import copy
import numpy

# ---------------------------------------------------------------------------
# Parameters — edit these, then run this file
# ---------------------------------------------------------------------------
SUB_ID     = 'FD'            # subject with a measured <id>.sofa in data/hrtf/sofa/<id>/

# SHIFT_BAND selects WHICH features move. They are transported to
# target_band(SHIFT_BAND, SHIFT_ERB) — above the band for a positive shift.
SHIFT_BAND = (5700, 11300)   # Trapeau peak-VSI octave [Hz]; None -> whole spectrum
SHIFT_ERB  = 1               # ERB displacement; > 0 up, < 0 down, 0 = no-op
N_KEEP     = 4               # cosine coeffs kept for the coarse envelope (M)
SKIRT      = 0.1            # taper on the selection window [octaves]; 0 = hard edges
EQ_RMS     = True            # match per-ERB detail RMS between source and target
FILL_GAP   = True            # pad the vacated strip from its own edges, so it is
                             # filled uniformly instead of dropping to envelope-only
                             # (no new notch is created). False -> leave it flat.

PLOT_KIND  = 'image'         # 'image' (before/after heatmap) | 'waterfall' | 'surface'
EAR        = 'right'         # ear shown in the QC plot
VSI_BW     = (5700, 11300)   # band for the VSI / VSI-dissimilarity readout
OUT_SUFFIX = 'shift'         # writes <SUB_ID>_<OUT_SUFFIX>_<shift tag>.sofa,
                             # e.g. JF_shift_2p5erb.sofa (m = minus, p = point)


# ---------------------------------------------------------------------------
# ERB scale (Glasberg & Moore 1990)
# ---------------------------------------------------------------------------

def hz_to_erb(f):
    """ERB-number for a frequency in Hz."""
    return 21.4 * numpy.log10(4.37 * numpy.asarray(f, dtype=float) / 1000.0 + 1.0)


def erb_to_hz(e):
    """Inverse of :func:`hz_to_erb`: frequency in Hz for an ERB-number."""
    return (10.0 ** (numpy.asarray(e, dtype=float) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_density(f):
    """d(ERB-number)/df at ``f`` [1/Hz].

    Used to weight linear-frequency bins so energy is measured per ERB, which is
    the scale the shift is constant on.
    """
    f = numpy.asarray(f, dtype=float)
    return (21.4 * 4.37 / 1000.0) / (numpy.log(10.0) * (4.37 * f / 1000.0 + 1.0))


def erb_shift_to_hz(f_center, shift_erb):
    """Hz equivalent of a ``shift_erb``-sized ERB step evaluated at ``f_center``.

    The same ERB step spans far more Hz at 12 kHz than at 6 kHz, so report a
    shift in Hz at the frequency it is actually applied to.
    """
    return float(erb_to_hz(hz_to_erb(f_center) + shift_erb) - f_center)


def target_band(band, shift_erb):
    """Where the features selected by ``band`` end up, in Hz.

    Returns ``None`` when ``band`` is ``None`` (whole spectrum shifted).
    """
    if band is None:
        return None
    low, high = float(band[0]), float(band[1])
    return (float(erb_to_hz(hz_to_erb(low) + shift_erb)),
            float(erb_to_hz(hz_to_erb(high) + shift_erb)))


# ---------------------------------------------------------------------------
# Cepstral envelope / detail split (Kulkarni & Colburn 1998)
# ---------------------------------------------------------------------------

_COS_BASIS_CACHE = {}


def _cos_basis(n_bins):
    """Cached cosine basis for the truncated-series smoother."""
    basis = _COS_BASIS_CACHE.get(n_bins)
    if basis is None:
        n_samples = 2 * (n_bins - 1)
        k = numpy.arange(n_bins, dtype=float)[:, None]
        n = numpy.arange(n_bins, dtype=float)[None, :]
        basis = numpy.cos(2.0 * numpy.pi * k * n / float(n_samples))
        _COS_BASIS_CACHE[n_bins] = basis
    return basis


def smooth_magnitude(mag, n_keep):
    """Coarse envelope of a one-sided magnitude spectrum.

    Truncated cosine-series reconstruction of log-magnitude: the first
    ``n_keep`` coefficients (M in Kulkarni & Colburn 1998) are kept, the rest
    discarded. Lower ``n_keep`` -> more of the spectrum counts as shiftable
    detail; higher -> only the sharpest peaks/notches move.

    Parameters
    ----------
    mag : (n_bins, n_channels) array
    n_keep : int

    Returns
    -------
    (n_bins, n_channels) array — the smoothed magnitude.
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")
    n_bins, n_channels = mag.shape
    n_keep = int(n_keep)
    if n_keep < 1 or n_keep > n_bins:
        raise ValueError(f"n_keep must be between 1 and {n_bins}, got {n_keep}")

    log_mag = numpy.log(numpy.maximum(mag, numpy.finfo(float).tiny))
    basis = _cos_basis(n_bins)

    log_mag_smooth = numpy.empty_like(log_mag)
    for ch in range(n_channels):
        coeffs, _, _, _ = numpy.linalg.lstsq(basis, log_mag[:, ch], rcond=None)
        coeffs[n_keep:] = 0.0
        log_mag_smooth[:, ch] = basis @ coeffs

    return numpy.exp(log_mag_smooth)


# ---------------------------------------------------------------------------
# Selection window
# ---------------------------------------------------------------------------

def band_window(freqs, low_hz, high_hz, skirt_octaves=0.0):
    """Selection weight in [0, 1] marking which detail gets transported.

    ``skirt_octaves = 0`` (default) gives hard edges: every selected feature is
    moved whole. A raised-cosine skirt of ``skirt_octaves`` on each side softens
    the edge, and since the window is applied before the transport that soft edge
    is carried to the new window too — at the price of splitting any feature
    inside the ramp between the old and new position. Keep it short (0.05-0.1
    octaves) and put the band edges where the detail is quiet.
    """
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError(f"band edges must be positive, got ({low_hz}, {high_hz})")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz must exceed low_hz, got ({low_hz}, {high_hz})")
    if skirt_octaves < 0:
        raise ValueError(f"skirt_octaves must be >= 0, got {skirt_octaves}")

    f = numpy.asarray(freqs, dtype=float)
    w = numpy.zeros_like(f)

    pos = f > 0
    if not numpy.any(pos):
        return w

    f_pos = f[pos]
    log_f = numpy.log2(f_pos)
    log_lo = numpy.log2(low_hz)
    log_hi = numpy.log2(high_hz)
    skirt = float(skirt_octaves)

    if skirt == 0:
        w_pos = ((log_f >= log_lo) & (log_f <= log_hi)).astype(float)
    else:
        w_pos = numpy.zeros_like(f_pos)

        ramp_up = (log_f >= log_lo - skirt) & (log_f < log_lo)
        x = (log_f[ramp_up] - (log_lo - skirt)) / skirt
        w_pos[ramp_up] = 0.5 * (1 - numpy.cos(numpy.pi * x))

        w_pos[(log_f >= log_lo) & (log_f <= log_hi)] = 1.0

        ramp_dn = (log_f > log_hi) & (log_f <= log_hi + skirt)
        x = (log_f[ramp_dn] - log_hi) / skirt
        w_pos[ramp_dn] = 0.5 * (1 + numpy.cos(numpy.pi * x))

    w[pos] = w_pos
    w[0] = 0.0
    if f.size and f[-1] > 0:
        w[-1] = 0.0
    return w


# ---------------------------------------------------------------------------
# The manipulation, on a single spectrum
# ---------------------------------------------------------------------------

def shift_detail_spectrum(freqs, mag, shift_erb, band=(5700.0, 11300.0),
                          envelope_n_keep=4, skirt_octaves=0.0,
                          equalize_rms=True, fill_gap=True):
    """Select, split, and transport — the whole manipulation on one spectrum.

    Kept free of any HRTF/slab plumbing so it can be unit-tested directly on
    synthetic spectra; :func:`shift_spectral_detail` is the HRTF wrapper.

    Parameters
    ----------
    freqs : (n_bins,) array
        One-sided frequency axis in Hz (``numpy.fft.rfftfreq``).
    mag : (n_bins, n_channels) array
        One-sided magnitude spectrum (linear, not dB).
    shift_erb : float
        Displacement along the ERB-number axis. ``> 0`` moves features UP in
        frequency, ``< 0`` down, ``0`` is a no-op rebuild.
    band : (low_hz, high_hz) or None
        Which features to move. ``None`` transports the whole detail.
    envelope_n_keep : int
        Cosine coefficients kept for the envelope (M).
    skirt_octaves : float
        Raised-cosine taper on the selection window, in octaves. See
        :func:`band_window` — leave at 0 unless you know the edges are quiet.
    equalize_rms : bool
        Rescale the transported detail so its per-ERB RMS matches the selected
        detail. Because the shift is a pure translation on the ERB axis this is
        already true up to interpolation loss, so it is a small correction.
    fill_gap : bool, default True
        Moving a finite band up vacates its bottom ``shift_erb`` worth of
        frequency: detail is taken out and nothing lands there, leaving
        envelope-only — a flat, elevation-independent stripe in the TF image.
        With ``fill_gap`` the gap is padded from its OWN edges: the detail is
        interpolated linearly (on the ERB axis) between the values bounding the
        gap. That is continuous at both ends and monotone in between, so the
        region is filled uniformly without introducing a new notch or peak.
        ``False`` leaves the gap at envelope-only.

    Returns
    -------
    (n_bins, n_channels) array — the new magnitude spectrum.
    """
    freqs = numpy.asarray(freqs, dtype=float)
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")
    if freqs.shape[0] != mag.shape[0]:
        raise ValueError("freqs and mag must have the same number of bins")
    if envelope_n_keep < 1:
        raise ValueError(f"envelope_n_keep must be >= 1, got {envelope_n_keep}")

    eps = numpy.finfo(float).tiny
    log_mag_db = 20.0 * numpy.log10(numpy.maximum(mag, eps))

    # 2a) coarse envelope — held fixed
    envelope_db = 20.0 * numpy.log10(
        numpy.maximum(smooth_magnitude(mag, n_keep=int(envelope_n_keep)), eps))

    # 2b) fine detail — the peaks/notches that carry the elevation cue
    detail_db = log_mag_db - envelope_db

    # 1) select which features to move; the rest stay exactly where they are
    w = (numpy.ones_like(freqs) if band is None
         else band_window(freqs, band[0], band[1], skirt_octaves=skirt_octaves))
    selected = w[:, None] * detail_db
    residual = detail_db - selected

    # 3) transport the SELECTED detail along the ERB axis. Sampling at
    #    ERB(f) - shift_erb means the pattern lands at ERB(f0) + shift_erb, i.e.
    #    higher frequencies for shift_erb > 0. Outside the source support the
    #    selected detail is zero, so extrapolation is zero — nothing is invented
    #    and nothing is clipped off, the features simply arrive at their new
    #    frequencies.
    src_freqs = erb_to_hz(hz_to_erb(freqs) - shift_erb)
    moved = numpy.empty_like(selected)
    for ch in range(selected.shape[1]):
        moved[:, ch] = numpy.interp(src_freqs, freqs, selected[:, ch],
                                    left=0.0, right=0.0)

    # 4) clear the LANDING zone before depositing. Transporting the window the
    #    same way gives the exact footprint the moved detail occupies (ramps
    #    included). Without this, the part of the landing zone that lies outside
    #    the source band still holds its native detail and the two SUM — the
    #    overlap ends up ~2.6x too deep, which reads as a band of exaggerated
    #    contrast just above the source band.
    w_moved = numpy.interp(src_freqs, freqs, w, left=0.0, right=0.0)
    residual *= (1.0 - w_moved)[:, None]

    if equalize_rms:
        weight = erb_density(freqs)
        for ch in range(selected.shape[1]):
            e_sel = float(numpy.sum(weight * selected[:, ch] ** 2))
            e_mov = float(numpy.sum(weight * moved[:, ch] ** 2))
            if e_mov > 0 and e_sel > 0:
                moved[:, ch] *= numpy.sqrt(e_sel / e_mov)

    new_detail = residual + moved

    # 5) fill the vacated gap by padding from its own edges. The gap is where
    #    the detail was taken out and nothing landed: coverage counts how much
    #    of a bin is accounted for by kept-native plus transported content, so
    #    the shortfall isolates the gap (partial ramps included). Filling by
    #    linear interpolation across the gap on the ERB axis is continuous at
    #    both edges and monotone between them, so it cannot create a notch or
    #    peak that was not there. Where a gap runs to the end of the spectrum
    #    numpy.interp clamps, i.e. it holds the edge value flat.
    if fill_gap:
        coverage = (1.0 - w) * (1.0 - w_moved) + w_moved
        shortfall = numpy.clip(1.0 - coverage, 0.0, 1.0)
        if numpy.any(shortfall > 1e-6):
            erb = hz_to_erb(freqs)
            known = shortfall <= 1e-6
            if numpy.any(known):
                for ch in range(new_detail.shape[1]):
                    filler = numpy.interp(erb, erb[known], new_detail[known, ch])
                    new_detail[:, ch] = ((1.0 - shortfall) * new_detail[:, ch]
                                         + shortfall * filler)

    new_log_mag = envelope_db + new_detail
    return 10.0 ** (new_log_mag / 20.0)


# ---------------------------------------------------------------------------
# The manipulation, on a whole HRTF
# ---------------------------------------------------------------------------

def shift_spectral_detail(hrtf, shift_erb, band=(5700.0, 11300.0),
                          envelope_n_keep=4, skirt_octaves=0.0,
                          equalize_rms=True, fill_gap=True, warn_nyquist=True):
    """Move each direction's fine spectral cues up (or down) the ERB axis.

    Select the features inside ``band``, separate them from the coarse envelope
    by cepstral smoothing, and carry them to ``target_band(band, shift_erb)``.
    Magnitude-only; the original phase is kept.

    Parameters are as in :func:`shift_detail_spectrum`, plus:

    warn_nyquist : bool, default True
        Print a warning if the transported features would land above Nyquist,
        where they cannot be represented and WOULD be lost.

    Returns
    -------
    slab.HRTF — a deep copy; the input is not modified.
    """
    out = copy.deepcopy(hrtf)

    if warn_nyquist and band is not None and shift_erb > 0:
        first = out[0]
        nyquist = float(first.samplerate) / 2.0
        top = target_band(band, shift_erb)[1]
        if top > nyquist:
            print(f"WARNING: transported features reach {top:.0f} Hz, above "
                  f"Nyquist ({nyquist:.0f} Hz) — the top of the band will be "
                  f"lost. Reduce shift_erb or lower the band.")

    for filt in out:
        ir_original = numpy.asarray(filt.data, dtype=float)
        if ir_original.ndim != 2 or ir_original.shape[1] != 2:
            raise ValueError("Each HRIR must have shape (n_samples, 2)")

        n_samples = ir_original.shape[0]
        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / filt.samplerate)

        spec_original = numpy.fft.rfft(ir_original, axis=0)
        mag_out = shift_detail_spectrum(
            freqs, numpy.abs(spec_original), shift_erb,
            band=band, envelope_n_keep=envelope_n_keep,
            skirt_octaves=skirt_octaves, equalize_rms=equalize_rms,
            fill_gap=fill_gap,
        )

        # magnitude-only edit: keep the ORIGINAL phase, so onset / ITD structure
        # and the HRTF's existing binaural cues pass through intact.
        spec_processed = mag_out * numpy.exp(1j * numpy.angle(spec_original))
        filt.data = numpy.fft.irfft(spec_processed, n=n_samples, axis=0)

    return out


def shift_tag(shift_erb):
    """Filename-safe tag for a shift, e.g. 2.5 -> ``'2p5erb'``, -1 -> ``'m1erb'``.

    Put this in output names so a modified SOFA states how far its cues were
    moved: a set built with a different ``shift_erb`` is a different stimulus and
    must not silently overwrite another.
    """
    value = float(shift_erb)
    text = f'{abs(value):g}'.replace('.', 'p')
    return f"{'m' if value < 0 else ''}{text}erb"


def describe(band, shift_erb, skirt_octaves=0.0):
    """Summary of what a given (band, shift, skirt) actually does.

    Prints the selection window, where the features land, and the shift in Hz at
    both edges -- the ERB step is far wider in Hz at the top of the band than at
    the bottom, which is easy to forget when tuning ``shift_erb``.

    With ``skirt_octaves > 0`` it also prints the ramp spans, at the source and
    where they land. Content inside those spans is split between the old and new
    position, so check what sits there before committing to a ramp length.
    """
    if band is None:
        text = (f"shift_spectral_detail: whole spectrum, {shift_erb:+.2f} ERB "
                f"({erb_shift_to_hz(8000, shift_erb):+.0f} Hz at 8 kHz)")
        print(text)
        return text

    low, high = float(band[0]), float(band[1])
    t_low, t_high = target_band(band, shift_erb)
    lines = [
        f"shift_spectral_detail: select {low:.0f}-{high:.0f} Hz, "
        f"{shift_erb:+.2f} ERB -> features land at {t_low:.0f}-{t_high:.0f} Hz "
        f"({erb_shift_to_hz(low, shift_erb):+.0f} Hz at the low edge, "
        f"{erb_shift_to_hz(high, shift_erb):+.0f} Hz at the high edge)"
    ]

    if skirt_octaves > 0:
        skirt = float(skirt_octaves)
        ramp_lo = (low * 2.0 ** -skirt, low)
        ramp_hi = (high, high * 2.0 ** skirt)
        t_lo = target_band(ramp_lo, shift_erb)
        t_hi = target_band(ramp_hi, shift_erb)
        lines.append(
            f"  {skirt:g}-oct ramp: {ramp_lo[0]:.0f}-{ramp_lo[1]:.0f} and "
            f"{ramp_hi[0]:.0f}-{ramp_hi[1]:.0f} Hz, landing at "
            f"{t_lo[0]:.0f}-{t_lo[1]:.0f} and {t_hi[0]:.0f}-{t_hi[1]:.0f} Hz")
        lines.append(
            "  content inside the ramps is split between old and new position; "
            "everything strictly inside the band still moves whole")
    else:
        lines.append("  hard edges: every selected feature moves whole")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Run on one subject
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # imported here, not at module scope, so the manipulation above stays
    # dependency-free (numpy only) and importable from modify.py without cycles
    import slab
    from hrtf_relearning.utils import paths
    from hrtf_relearning.hrtf.modify.plot_compare import plot
    from hrtf_relearning.hrtf.analysis.vsi import (
        vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity,
    )

    sofa_dir = paths.SOFA_DIR / SUB_ID
    native_path = sofa_dir / f'{SUB_ID}.sofa'
    if not native_path.exists():
        raise FileNotFoundError(
            f'no SOFA at {native_path} — check SUB_ID in the config block')

    hrtf = slab.HRTF(str(native_path))
    hrtf.name = SUB_ID
    print(f'loaded {native_path.name}')

    describe(SHIFT_BAND, SHIFT_ERB, skirt_octaves=SKIRT)
    hrtf_shift = shift_spectral_detail(
        hrtf,
        shift_erb=SHIFT_ERB,
        band=SHIFT_BAND,
        envelope_n_keep=N_KEEP,
        skirt_octaves=SKIRT,
        equalize_rms=EQ_RMS,
        fill_gap=FILL_GAP,
    )
    print(f'M={N_KEEP}, skirt={SKIRT} oct, eq_rms={EQ_RMS}, fill_gap={FILL_GAP}')

    vsi_o = _vsi(hrtf,       bandwidth=VSI_BW)
    vsi_m = _vsi(hrtf_shift, bandwidth=VSI_BW)
    vsi_d = _vsi_dissimilarity(hrtf, hrtf_shift, bandwidth=VSI_BW)
    print(f'VSI  native={vsi_o:.3f}  shifted={vsi_m:.3f}  dissimilarity={vsi_d:.3f}')

    # before/after median-plane transfer function. Every feature selected inside
    # SHIFT_BAND should reappear in the target band, none missing.
    fig = plot(hrtf, hrtf_shift, PLOT_KIND, ear=EAR,
               vsi_orig=vsi_o, vsi_mod=vsi_m, vsi_dis=vsi_d, vsi_bw=VSI_BW)

    # the shift goes in the filename: a set built with a different SHIFT_ERB is a
    # different stimulus and must not overwrite the previous one
    stem = f'{SUB_ID}_{OUT_SUFFIX}_{shift_tag(SHIFT_ERB)}'
    print(f'about to write {stem}.sofa')

    input('press enter to save (ctrl-c to discard)')
    plot_dir = paths.subject_acoustic_dir(SUB_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f'{stem}.png', bbox_inches='tight')
    sofa_dir.mkdir(parents=True, exist_ok=True)
    out_path = sofa_dir / f'{stem}.sofa'
    hrtf_shift.write_sofa(str(out_path))
    print(f'wrote {out_path}')
