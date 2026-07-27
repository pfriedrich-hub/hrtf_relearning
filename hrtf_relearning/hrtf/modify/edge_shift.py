"""
edge_shift.py

All-notch-consistent spectral manipulations of an individual HRTF for the
elevation edge-coding experiment. Three conditions share ONE notch-detection
pass and ONE shift magnitude (Δ, in ERB):

    'rising'  : the notch's rising edge moved UP   by Δ, minimum pinned
    'falling' : the notch's falling edge moved DOWN by Δ, minimum pinned (control)
    'whole'   : the notch minimum moved UP by Δ, both bracketing saddles pinned
                (all-models-agree reference)

Notch/peak identity and their center frequencies are found by ELIMINATING the
microscopic fluctuations of the log-magnitude with a light Gaussian (Iida et
al. 2007, Eq. 2-3; sigma ~122 Hz) and taking the true minima/maxima of that
curve. A narrow, symmetric Gaussian de-ripples WITHOUT displacing the extrema,
so the notch center frequencies (N1, N2, ...) are read at their true positions
and the two distinct pinna notches stay resolved. This follows Iida et al.
(2007), our primary reference (N1 AND N2 are both required to reproduce
median-plane localization). An earlier version smoothed via cepstral
cosine-series truncation (Kulkarni & Colburn 1998, n_keep): fine for coarse
identity but it reshapes the curve and SHIFTS/MERGES the minima (verified on
JS/CA/Aachen, dev/verify_notch_method.py), so it is retained only as an
explicit legacy fallback (pass n_keep=... to detect_notches). See
experiment/protocols/documentation/cue_perception_synthesis.md for the
cross-study evidence basis and where the accounts agree/conflict. This identity
step is deliberately SEPARATE from a rising/falling edge's actual EXTENT: the
edge span itself is read off the derivative of the RAW (unsmoothed) spectrum,
searched only within the window the coarse step already bounded (a notch's
minimum out to its nearest saddle), as the outermost points where |dL/df|
clears 15% of that window's own peak slope -- not the first contiguous run
from the peak, so a secondary ripple/shoulder on the same flank is bridged
rather than truncating the detected edge early. See
experiment/protocols/documentation/edge_detection_and_shift_methods.md for
the full derivation, parameter justification, and validation numbers.

An earlier version of this module used the pinna spectral-notch extraction
method of Raykar, Duraiswami & Yegnanarayana (2005, JASA 118(1):364-374) --
group delay of the LP-residual autocorrelation of the pinna-isolated HRIR --
for the same identity job. It gives equivalent notch-merging behavior but
needs substantially more machinery (onset detection, LPC, autocorrelation,
two windowing stages) for no gain in correctness over the cepstral approach,
which also has the advantage of operating in the same representation (the
log-magnitude spectrum) already used for the manipulation itself.

'rising'/'falling' keep the notch MINIMUM pinned and rigidly TRANSLATE the
detected edge span (not the whole minimum-to-saddle range) by s ERB -- both
boundaries of that span move by the identical amount, so the manipulated
flank is an exact, undistorted copy of the measured one, only moved, not
stretched to span a gap. The sliver between the pinned minimum and the
translated span's new near boundary is filled with a flat hold at the
minimum's own dB value (see _notch_knots) -- this is a small, close-to-free
edit, since that sliver already had close-to-zero slope in the original
curve (that is exactly why it wasn't part of the detected edge). The
bracketing saddle on that side is NEVER moved and never even approached --
because the detected edge span generically terminates well before reaching
it (a notch's flank typically flattens into a broad shoulder near the peak
rather than staying steep all the way up to it), whatever compensating
stretch is needed to land back on the saddle is confined to that small,
already-shallow leftover sub-plateau between the edge's far boundary and the
saddle -- entirely within this notch's own bracket, never touching a
neighboring notch's territory, even though adjacent notches structurally
share that saddle (see build_warped_logmag).

Design invariants (per notch): notch MINIMUM is pinned; bracketing SADDLES
never move under 'rising'/'falling' (so depth can be verified directly at
the original i_lo/i_hi, no post-hoc bookkeeping needed); in-band RMS is
matched (removes the in-notch-power confound); original PHASE is kept (ITD
unchanged). 'whole' is a distinct manipulation -- the minimum itself moves
with both saddles pinned, reshaping both flanks around a recentered notch --
used as an all-models-agree reference condition, not to dissociate
rising/falling accounts. verify_binned/verify_condition still check
realized shifts against the actual Baumgartner model resolution
(baumgartner_filterbank) -- that's a separate question (does THIS model
register the manipulation) from whether the manipulation itself is well
defined and perceptually real, which is what detect_notches is for.

Which notches get shifted can be decided per direction (select_features gate,
the default) or, more robustly, from ELEVATION-CONTINUITY TRACKS
(stabilized_valid_cfs, opt-in via edge_shift_set/manipulate_hrtf use_tracking).
Per-direction gating flickers: a real notch that momentarily dips below the
depth gate is shifted at some elevations but not adjacent ones, and the
"deepest" label swaps between unrelated notches (measured on real DTFs: 21
interior gate-dropouts across 18 subject-ears, primary-CF jumps up to ~1 oct
between 4-deg-apart directions). Median-plane notches are in fact smooth,
continuous elevation trajectories, so tracking links detected minima into
tracks within each constant-azimuth arc, decides each track's validity ONCE
(passes the depth/width gate at >= min_valid_frac of the directions where it is
detected), and applies that decision to every member -- shifting the same
physical notch at all its elevations. detect_notches itself is unchanged.

Core works on numpy arrays; slab adapters (manipulate_hrtf, save_condition_sofa,
compare_tf) sit on top for the SOFA-per-condition workflow. See
experiment/protocols/cue_shift.py for the cell-by-cell experiment protocol
that sequences this module, and
experiment/protocols/documentation/elevation_spectral_cue_models.md for the
model background.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

EPS = 1e-9

# --- perceptual gates / defaults (documentation/cue_perception_synthesis.md) ----
# Iida et al. 2007 (primary): eliminate the microscopic fluctuations of the
# amplitude spectrum with a LIGHT Gaussian (their Eq. 2-3: window +-4 bins,
# sigma 1.3 bins on a 512-pt/48-kHz grid ~ sigma ~ 122 Hz), then take the true
# maxima/minima as peaks/notches (center frequency, depth, sharpness). A light
# symmetric Gaussian de-ripples WITHOUT moving the extrema; a coarse cepstral
# truncation (n_keep) instead reshapes the curve and SHIFTS/MERGES the minima
# (verified on JS/CA/Aachen, dev/verify_notch_method.py) -- so cepstral is kept
# only as an explicit legacy fallback (pass n_keep=... to detect_notches).
SIGMA_HZ_DEFAULT = 122.0   # Iida Eq 2-3 microscopic-fluctuation Gaussian (Hz)
NKEEP_LEGACY = 30          # cepstral fallback only (biases CF; do not use for detection)
SADDLE_PROM_DB = 4.0    # Moore et al. 1989: peak/saddle detection ~2.5-5 dB @8 kHz.
                        # Used for peak/saddle prominence (e.g. the P1 reference
                        # peak); NOT the notch-depth cue gate -- see below.
MIN_NOTCH_DEPTH_DB = 3.0  # depth (prominence) a detected minimum needs to count as a
                        # real elevation cue for select_features / stabilized_valid_cfs.
                        # Set to the detect_notches detection floor (prominence_db=3.0)
                        # so borderline 3-4 dB notches -- which flicker in/out of the
                        # 4 dB gate between adjacent elevations -- are kept rather than
                        # dropped (Moore's detectability range is ~2.5-5 dB, so 3 dB is
                        # still a defensible cue floor). Decoupled from SADDLE_PROM_DB so
                        # lowering the cue gate doesn't also loosen peak/P1 detection.
SEP_MIN_OCT = 0.5       # Macpherson & Middlebrooks 2003: usable to ~2 ripples/oct
WIDTH_MIN_OCT = 0.17    # Moore et al. 1989: bw < ~0.25*fc undetectable (~0.17 oct at
                        # half depth); also the ~6 ripples/oct timbre limit. NOTE:
                        # this perceptual-detectability threshold is NOT the
                        # detection gate -- see DETECT_WIDTH_MIN_OCT below.
DETECT_WIDTH_MIN_OCT = 0.05  # real-feature width floor for select_features. The
                        # Iida Gaussian (sigma~122 Hz) has already removed the
                        # bin-level wiggles, so a surviving minimum wider than
                        # ~this (several x the smoothing width) is a real measured
                        # feature; DEPTH (min_depth_db) is the wiggle-vs-cue
                        # discriminator, not width. Deep-but-narrow notches (e.g.
                        # CA 6.9 kHz, 24 dB, 0.13 oct) are genuine cues and were
                        # being wrongly dropped by the 0.17 perceptual threshold.
CF_JND_FRAC = 0.08      # Moore et al. 1989: notch center-frequency JND ~8% @8 kHz
NOTCH_BAND = (5000.0, 11000.0)  # Asano/Suzuki/Sone 1990 (6-10 kHz) + van Opstal
                                # (6-9 kHz): the elevation spectral-notch region


# --- ERB scale (Glasberg & Moore 1990) ----
def hz_to_erb(f):
    return 21.4 * np.log10(4.37 * np.asarray(f, float) / 1000.0 + 1.0)


def erb_to_hz(e):
    return (10.0 ** (np.asarray(e, float) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_shift_to_hz(f_center, shift_erb):
    """Hz equivalent of a `shift_erb`-sized ERB step, evaluated at f_center.
    This is what gives "1 ERB" a concrete meaning: it's location-dependent --
    the same ERB step spans far more Hz at 12 kHz than at 6 kHz, since ERB
    bandwidth grows with frequency. Use this to report/interpret a shift in
    Hz at the frequency it's actually applied to (e.g. a notch's own f_hz),
    not as a single global Hz-per-ERB constant."""
    e_center = float(hz_to_erb(f_center))
    return float(erb_to_hz(e_center + shift_erb) - f_center)


# --- spectrum <-> ERB grid ----
def _to_erb_grid(L_db, freqs, f_lo, f_hi, step=0.05):
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fb = freqs[band]
    e_lo, e_hi = float(hz_to_erb(f_lo)), float(hz_to_erb(f_hi))
    M = max(256, int((e_hi - e_lo) / step))
    e = np.linspace(e_lo, e_hi, M)
    Le = np.interp(erb_to_hz(e), fb, L_db[band])
    return e, Le, band, fb


# --- cepstral smoothing (Kulkarni & Colburn 1998) ----
_COS_BASIS_CACHE = {}


def _cos_basis(n_bins):
    """Cache the square cosine basis and its inverse for a given bin count. The
    basis depends only on n_bins, so building/factorising it once and reusing it
    turns each _cepstral_smooth_db call into two matrix-vector products instead of
    an O(n^3) lstsq -- essential when smoothing a whole HRIR set (hundreds of
    directions x ears)."""
    hit = _COS_BASIS_CACHE.get(n_bins)
    if hit is None:
        n_samples = 2 * (n_bins - 1)
        k = np.arange(n_bins, dtype=float)[:, None]
        n_ = np.arange(n_bins, dtype=float)[None, :]
        basis = np.cos(2.0 * np.pi * k * n_ / float(n_samples))
        hit = (basis, np.linalg.inv(basis))
        _COS_BASIS_CACHE[n_bins] = hit
    return hit


def _cepstral_smooth_db(mag, n_keep):
    """Truncated cosine-series reconstruction of a one-sided magnitude
    spectrum: expand log|H| in a cosine series over frequency-bin index and
    keep only the first n_keep coefficients. Same technique as modify.py's
    `_smooth` (used there for the shift_band envelope/detail split);
    reimplemented locally so this module stays numpy-only and doesn't pull
    in modify.py's heavier slab/sklearn/matplotlib import chain just for one
    shared helper. Returns smoothed log-magnitude (dB).

    The cosine basis is square and full-rank, so the exact cosine coefficients
    are basis^{-1} @ log_mag (identical to the old lstsq solution); both the
    basis and its inverse are cached per bin count (see _cos_basis)."""
    mag = np.asarray(mag, float)
    n_bins = len(mag)
    n_keep = int(np.clip(n_keep, 1, n_bins))
    log_mag = np.log(np.maximum(mag, np.finfo(float).tiny))
    basis, basis_inv = _cos_basis(n_bins)
    coeffs = basis_inv @ log_mag
    coeffs[n_keep:] = 0.0
    log_mag_smooth = basis @ coeffs
    return 20.0 * np.log10(np.maximum(np.exp(log_mag_smooth), EPS))


def _gaussian_smooth_db(mag, freqs, sigma_hz=SIGMA_HZ_DEFAULT):
    """Iida et al. (2007) Eq. 2-3 microscopic-fluctuation smoothing: convolve the
    log-magnitude with a light Gaussian of sigma `sigma_hz` (default ~122 Hz,
    truncated at +-3.08 sigma as in Iida's n=4/sigma=1.3-bin kernel). Being
    narrow and symmetric it removes bin-level ripple WITHOUT displacing the
    extrema, so the notch/peak center frequencies read off it are the true ones
    -- the intended behavior for detect_notches. Returns smoothed log-mag (dB).
    Reflect padding avoids edge roll-off at the band ends."""
    L = 20.0 * np.log10(np.maximum(np.asarray(mag, float), EPS))
    df = float(freqs[1] - freqs[0])
    sig = max(float(sigma_hz) / df, 1e-6)
    half = max(int(np.ceil(3.08 * sig)), 1)
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k /= k.sum()
    Lp = np.pad(L, half, mode="reflect")
    return np.convolve(Lp, k, mode="valid")


def _edge_span(seg, lo, eps_frac):
    """Within a signed derivative slice `seg` (positive = the direction of
    interest, e.g. dL/df for a rising edge or -dL/df for falling), return
    the (lo_idx, hi_idx) absolute indices of the OUTERMOST samples whose
    value clears eps_frac of `seg`'s own peak -- not the first contiguous
    run from the peak. This is what lets a secondary ripple/shoulder on the
    same flank get bridged rather than truncating the detected edge early
    (see module docstring / edge_detection_and_shift_methods.md). Returns
    None if `seg` never goes positive (no usable slope in this window)."""
    if seg.max() <= 0:
        return None
    eps = eps_frac * seg.max()
    above = np.where(seg >= eps)[0]
    return (lo + int(above.min()), lo + int(above.max()))


def _level_at(fb, L, f_hz):
    return float(L[int(np.argmin(np.abs(fb - f_hz)))])


def _halfdepth_bounds(L, i, depth, lo, hi):
    """Indices bracketing the half-depth width of the notch at index i, searched
    within [lo, hi]. Half-depth level = L[i] + depth/2; used to report each
    notch's width/sharpness (Iida 2007 parameterises notches by CF, depth, and
    sharpness)."""
    thr = L[i] + 0.5 * float(depth)
    a = i
    while a > lo and L[a] < thr:
        a -= 1
    b = i
    while b < hi and L[b] < thr:
        b += 1
    return a, b


def detect_notches(e, freqs, mag, f_lo=3000.0, f_hi=17500.0, sigma_hz=SIGMA_HZ_DEFAULT,
                   n_keep=None, prominence_db=3.0, eps_frac=0.15):
    """Notch minima, bracketing saddle peaks, and rising/falling edge spans.

    IDENTITY (which frequencies are notches/saddles at all): local extrema
    of the cepstrally-smoothed (n_keep) log-magnitude spectrum with
    prominence >= prominence_db (scipy.signal.find_peaks on +/- the smoothed
    curve). Coarse enough that fine ripple genuinely belonging to one
    auditory feature merges into a single trough; fine enough that distinct
    notches survive as separate minima.

    EXTENT (the rising/falling edge span for each notch): derivative of that
    SAME cepstrally-smoothed curve (not the raw spectrum -- see below),
    searched only within the window the identity step already bounded (a
    notch's minimum out to its nearest saddle on that side). The edge span
    is the frequency range between the outermost samples whose |derivative|
    clears eps_frac of that window's own peak derivative -- see _edge_span.

    Using the raw spectrum's derivative for extent (an earlier version of
    this function) is vulnerable to individual, very narrow spectral
    ripples: because eps is a *fraction of that window's own peak*
    derivative, a single hyper-narrow dip (e.g. a 1-bin, -30 dB spike with
    no smoothed counterpart at all) can dominate the peak and pull the
    detected edge onto the ripple's flanks rather than the broad, real
    structure around it -- confirmed by direct comparison on real DTFs
    (JS, el=8: raw derivative peaks at ~0.36 driven by one such spike vs.
    ~0.03 for the smoothed curve in the same window). Reading extent off
    the same n_keep-smoothed curve already used for identity removes this
    failure mode and keeps identity and extent in one consistent
    representation, which is also the more defensible modeling choice if
    fine ripple below the coarse-smoothing resolution isn't perceptually
    resolved in the first place.

    Returns a list of dicts (frequency-sorted), each with:
      i_min, i_lo, i_hi : ERB-grid indices (into a grid with the same
        hz_to_erb/erb_to_hz mapping used elsewhere in this module) of the
        notch minimum and its bracketing saddles.
      f_hz : notch center frequency.
      edge_rise, edge_fall : (i_near_min, i_near_saddle) ERB-grid index pairs
        for the rising/falling edge span, or None if that side has no
        usable (positive-going, for rising; negative-going, for falling)
        slope in its window (e.g. a monotonic flank truncated by the
        analysis band's own edge).
    """
    # smooth the FULL one-sided spectrum (DC..Nyquist) -- the cosine basis
    # assumes bin index 0 is DC, so smoothing an already band-cropped
    # sub-array would use the wrong basis entirely -- then crop to the
    # analysis band afterward.
    # Iida Eq 2-3 light Gaussian by default (CF-preserving); cepstral only if
    # n_keep is explicitly passed (legacy, shifts/merges minima -- see header).
    Lc_full = (_cepstral_smooth_db(mag, n_keep) if n_keep is not None
               else _gaussian_smooth_db(mag, freqs, sigma_hz))
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fb = freqs[band]
    Lc = Lc_full[band]
    e_fb = hz_to_erb(fb)

    notch_idx, notch_props = find_peaks(-Lc, prominence=prominence_db)
    peak_idx, _ = find_peaks(Lc, prominence=prominence_db)
    if len(notch_idx) == 0:
        return []
    notch_prom = {int(i): float(p) for i, p in zip(notch_idx, notch_props["prominences"])}
    peak_idx_sorted = np.sort(peak_idx)
    # extent uses the derivative of the SAME smoothed curve as identity (see
    # docstring) -- not the raw spectrum, which is vulnerable to single
    # narrow ripples dominating the window's own peak derivative.
    dLc = np.gradient(Lc, fb)

    def _to_e_idx(fb_idx):
        return int(np.clip(np.searchsorted(e, e_fb[fb_idx]), 0, len(e) - 1))

    out = []
    for ni in np.sort(notch_idx):
        after = peak_idx_sorted[peak_idx_sorted > ni]
        before = peak_idx_sorted[peak_idx_sorted < ni]
        i_hi = int(after[0]) if len(after) else len(fb) - 1
        i_lo = int(before[-1]) if len(before) else 0
        if i_hi <= ni or ni <= i_lo:
            continue  # degenerate (rare) -- no usable bracket for this notch

        rise = _edge_span(dLc[ni:i_hi + 1], ni, eps_frac)
        fall = _edge_span(-dLc[i_lo:ni + 1], i_lo, eps_frac)
        # _edge_span always returns (low-freq-end-of-window, high-freq-end-of-
        # window). For 'rise' the window runs min->saddle (increasing freq),
        # so that's already (near_min, near_saddle). For 'fall' the window
        # runs saddle->min (increasing freq), so the raw return is
        # (near_saddle, near_min) -- reversed here so edge_fall follows the
        # same (near_min, near_saddle) convention as edge_rise, which is what
        # _notch_knots's falling branch assumes when it unpacks
        # e_near_min, e_near_saddle = e[edge[0]], e[edge[1]].

        # parametric characterization (Iida 2007: CF, depth, sharpness)
        depth = notch_prom.get(int(ni), float(max(Lc[i_lo], Lc[i_hi]) - Lc[ni]))
        w_lo, w_hi = _halfdepth_bounds(Lc, ni, depth, i_lo, i_hi)
        bw_hz = float(fb[w_hi] - fb[w_lo])
        # half-height crossing on the rising (upper) flank -- the compact "edge"
        # reference (midpoint of the low->high attenuation contrast). More stable
        # across elevation than the steepest point, and the anchor for mode='edge'
        # (translate this crossing up by s, notch minimum and upper peak pinned).
        half_r = Lc[ni] + 0.5 * (Lc[i_hi] - Lc[ni])
        xr = np.where(Lc[ni:i_hi + 1] >= half_r)[0]
        i_edge_r = ni + int(xr[0]) if len(xr) else ni
        out.append(dict(
            i_min=_to_e_idx(ni), i_lo=_to_e_idx(i_lo), i_hi=_to_e_idx(i_hi),
            i_edge_rise=_to_e_idx(i_edge_r),
            f_hz=round(float(fb[ni]), 0),
            f_edge_rise_hz=round(float(fb[i_edge_r]), 0),
            depth_db=round(depth, 1),
            width_oct=round(float(np.log2(fb[w_hi] / fb[w_lo])) if w_hi > w_lo else 0.0, 3),
            q=round(float(fb[ni] / bw_hz), 1) if bw_hz > 0 else None,
            edge_rise=None if rise is None else (_to_e_idx(rise[0]), _to_e_idx(rise[1])),
            edge_fall=None if fall is None else (_to_e_idx(fall[1]), _to_e_idx(fall[0])),
        ))
    # label notches N1, N2, ... by ascending frequency above ~5 kHz (Iida)
    out.sort(key=lambda d: d["f_hz"])
    k = 0
    for d in out:
        if d["f_hz"] >= 5000.0:
            k += 1
            d["label"] = f"N{k}"
        else:
            d["label"] = "lo"
    return out


# --- per-notch warp (minimum pinned, saddles never move) ----
def _notch_knots(e, notch, s_target, mode, margin=0.05, strict=True, ceiling_e=None):
    """Return (knots_out, knots_src, realized_shift) for one notch, or None
    if this notch has no usable edge on the requested side.

    The notch MINIMUM is pinned exactly (the notch's identity/CF anchor) and
    never moves. 'rising'/'falling' rigidly TRANSLATE the detected edge span
    (notch['edge_rise'] or notch['edge_fall'], see detect_notches) by s ERB
    -- both boundaries of that span move by the identical amount, so the
    manipulated flank is an exact, undistorted copy of the measured one,
    only moved, not stretched to span a gap. The sliver between the pinned
    minimum and the translated span's new near boundary is filled with a
    flat hold at the minimum's own dB value (zero-order hold): this is a
    small, close-to-free edit, since that sliver already had close-to-zero
    slope in the original curve (that's exactly why it fell outside the
    detected edge in the first place).

    The bracketing saddle on that side (notch['i_hi'] for rising,
    notch['i_lo'] for falling) is the far boundary and NEVER MOVES: because
    the detected edge span generically stops well short of it, whatever
    compensating stretch is needed to still land exactly back on the saddle
    is confined to the small, already-shallow sub-plateau between the
    edge's far boundary and the saddle -- entirely within this notch's own
    bracket. This is what lets the room/clamp calculation below be purely
    local (no lookup into neighboring notches needed): two adjacent notches
    structurally share that saddle, but each notch's own warp only ever
    touches the interval up to it, mapping the shared point to itself, so
    there is no cross-notch interference to special-case.

    'whole' is unchanged: moves the minimum itself (a point feature, not a
    slope) with both bracketing peaks pinned -- that mode deliberately
    reshapes both flanks around a re-centered notch, so no edge-span/flat-pad
    logic applies there.

    strict: see module docstring's clamping convention --
      strict=True  (default) -- raise if the notch can't reach s_target.
      strict=False -- clamp and return the reduced shift.
    """
    e_min = e[notch['i_min']]
    if mode == 'rising':
        edge = notch.get('edge_rise')
        if edge is None:
            return None
        e_near_min, e_near_saddle = e[edge[0]], e[edge[1]]
        e_saddle = e[notch['i_hi']]
        room = (e_saddle - e_near_saddle) - margin
    elif mode == 'falling':
        edge = notch.get('edge_fall')
        if edge is None:
            return None
        e_near_min, e_near_saddle = e[edge[0]], e[edge[1]]
        e_saddle = e[notch['i_lo']]
        room = (e_near_saddle - e_saddle) - margin
    elif mode == 'whole':
        e_lo, e_hi = e[notch['i_lo']], e[notch['i_hi']]
        room = (e_hi - e_min) - margin
    elif mode == 'edge':
        # rigid, SHAPE-PRESERVING translation of the [onset, shoulder] edge
        # segment up by s (onset and shoulder both move by exactly s, so the
        # measured edge shape is copied, not stretched). The notch MINIMUM is
        # pinned; the gap opened between it and the shifted onset is padded at
        # the floor level; the plateau+peak above the shoulder is compressed up
        # to `ceiling_e` -- the NEXT notch's CENTRE (its minimum), not this
        # notch's own peak. So the peak rides up with the edge but the next
        # cue's centre frequency stays fixed; the room is shoulder->next-centre,
        # not the (near-zero) shoulder->peak gap.
        er = notch.get('edge_rise')
        if er is None:
            return None
        e_onset, e_shoulder = e[er[0]], e[er[1]]
        e_saddle = ceiling_e if ceiling_e is not None else e[notch['i_hi']]
        room = (e_saddle - e_shoulder) - margin
    else:
        raise ValueError("mode must be 'rising', 'falling', 'whole' or 'edge'")
    s = max(min(s_target, room), 0.0)
    if strict and s < s_target - 1e-9:
        f_hz = float(erb_to_hz(e_min))
        anchor = 'minimum' if mode == 'whole' else 'edge'
        raise ValueError(
            f"shift_erb={s_target:.2f} exceeds available room ({max(room, 0.0):.2f} ERB) "
            f"between the {anchor} and the bracketing saddle at the notch near {f_hz:.0f} Hz "
            f"(mode='{mode}'). Pass strict=False to allow clamping to the available room instead."
        )
    if mode == 'rising':
        # flat pad [e_min, e_near_min+s] (src held at e_min) then a pure
        # translation [e_near_min+s, e_near_saddle+s] -> [e_near_min, e_near_saddle]
        # (both knot pairs differ by exactly s, so the line through them IS
        # src=e'-s, not a stretch), then compensation up to the fixed saddle.
        ko = [e_min, e_near_min + s, e_near_saddle + s, e_saddle]
        ks = [e_min, e_min,          e_near_saddle,      e_saddle]
    elif mode == 'falling':
        ko = [e_saddle, e_near_saddle - s, e_near_min - s, e_min]
        ks = [e_saddle, e_near_saddle,     e_min,           e_min]
    elif mode == 'whole':
        ko = [e_lo, e_min + s, e_hi]
        ks = [e_lo, e_min,     e_hi]
    elif mode == 'edge':
        # [min,onset]: IDENTITY -- the notch centre and its floor are left
        # exactly untouched, so the centre frequency cannot drift.
        # [onset,onset+s]: flat pad at the onset level (fills the gap opened by
        # the shift). [onset+s,shoulder+s]<-[onset,shoulder]: RIGID edge (both
        # differ by s -> fixed shape). [shoulder+s,ceil]<-[shoulder,ceil]:
        # falling flank compressed (steepened) up to the next cue's centre,
        # which is pinned so the neighbour's centre stays fixed too.
        if e_onset - e_min > 1e-9:
            ko = [e_min, e_onset, e_onset + s, e_shoulder + s, e_saddle]
            ks = [e_min, e_onset, e_onset,     e_shoulder,     e_saddle]
        else:  # onset sits on the minimum: no identity segment to add
            ko = [e_min, e_onset + s, e_shoulder + s, e_saddle]
            ks = [e_min, e_onset,     e_shoulder,     e_saddle]
    return np.array(ko), np.array(ks), s


def build_warped_logmag(e, Le, notches, s_target, mode, strict=True, ceiling_centers_e=None):
    """Apply every notch's warp, then resample Le through the composed
    monotone warp. Returns (Le_new, realized).

    For 'rising'/'falling'/'whole' each notch's warp is confined to its own
    [i_lo, i_hi] bracket and maps the shared boundary with a neighbor to itself,
    so no neighbor lookup is needed. 'edge' is the exception: its ceiling is the
    NEXT notch's rising-edge onset (so the fixed-shape edge can ride the peak up
    without reaching the next cue), computed here from the frequency-sorted
    `notches` list and passed to _notch_knots as ceiling_e.

    strict: see _notch_knots -- raise (default) vs silently clamp when a
    notch doesn't have room for the full requested shift."""
    src = e.copy()
    realized = []
    # per-notch ceiling for 'edge' mode = next notch's rising-edge onset (or the
    # grid top for the highest notch); None for the other modes.
    ceilings = [None] * len(notches)
    if mode == 'edge':
        # ceiling = centre of the next REAL notch above this edge's shoulder,
        # taken from ceiling_centers_e (all valid minima, incl. ones merged out
        # of `notches`), so the compression never rolls over a neighbour's
        # centre. Falls back to the next entry in `notches`, else the grid top.
        centres = np.array(sorted(ceiling_centers_e)) if ceiling_centers_e else None
        for i, n in enumerate(notches):
            er = n.get('edge_rise')
            e_shoulder = e[er[1]] if er is not None else e[n['i_min']]
            ceil = e[-1]
            if centres is not None:
                above = centres[centres > e_shoulder + 1e-9]
                if len(above):
                    ceil = float(above[0])
            elif i + 1 < len(notches):
                ceil = e[notches[i + 1]['i_min']]
            ceilings[i] = ceil
    for i, n in enumerate(notches):
        result = _notch_knots(e, n, s_target, mode, strict=strict, ceiling_e=ceilings[i])
        if result is None:
            realized.append(0.0)
            continue
        ko, ks, s = result
        if np.any(np.diff(ko) <= 0) or np.any(np.diff(ks) < 0):
            realized.append(0.0)
            continue
        m = (e >= ko[0]) & (e <= ko[-1])
        src[m] = np.interp(e[m], ko, ks)
        realized.append(s)
    src = np.clip(src, e[0], e[-1])
    Le_new = PchipInterpolator(e, Le)(src)
    return Le_new, realized


# --- confound matching ----
def match_band_rms(e, Le_base, Le_new, rms_band):
    f = erb_to_hz(e)
    m = (f >= rms_band[0]) & (f <= rms_band[1])
    p0 = np.mean((10.0 ** (Le_base[m] / 20.0)) ** 2)
    p1 = np.mean((10.0 ** (Le_new[m] / 20.0)) ** 2)
    return Le_new + 10.0 * np.log10(p0 / max(p1, EPS))  # constant dB: depth kept


# --- verification (fine grid) ----
def _depth_db(e, Le, a, b):
    """Peak-to-trough depth (dB) of Le between grid indices a and b
    (inclusive). Since 'rising'/'falling' shifts never move the bracketing
    saddles (see _notch_knots), a notch's own i_lo/i_hi are valid bounds for
    reading depth on BOTH the baseline and the shifted curve directly -- no
    post-hoc relocation of the bounds is needed (unlike an earlier version
    of this module, where the saddle itself moved and depth checks had to
    track its new position)."""
    return max(Le[a], Le[b]) - float(np.min(Le[a:b + 1]))


def _notch_centroid_e(e, Le, a, b):
    ref = max(Le[a], Le[b])
    d = np.maximum(ref - Le[a:b + 1], 0.0)
    return float(np.sum(e[a:b + 1] * d) / (d.sum() + EPS))


# --- single IR ----
def edge_shift_ir(ir, fs, shift_erb, mode='rising', f_lo=3000.0, f_hi=17500.0,
                  rms_band=(3000.0, 16000.0), sigma_hz=SIGMA_HZ_DEFAULT, n_keep=None,
                  prominence_db=3.0, eps_frac=0.15, nfft=None, match_power=True,
                  return_report=False, strict=True, features_only=True, feature_kw=None,
                  valid_cfs=None, valid_tol_oct=0.03):
    """Notches/saddles/edges are found via detect_notches (cepstral-smoothed
    identity + raw-spectrum edge extent -- see that function and the module
    docstring). n_keep, prominence_db, eps_frac are passed straight through
    to it. f_lo/f_hi bound both the notch search and the ERB grid used for
    the warp/output.

    features_only (default True): warp only the perceptually valid notch(es)
    from select_features (in-band, depth >= MIN_NOTCH_DEPTH_DB, width >=
    WIDTH_MIN_OCT), not every detected minimum -- so sub-perceptual ripple and
    high-frequency measurement spikes (e.g. a sharp ~16 kHz artifact) are left
    untouched. Pass feature_kw={...} to tune the gate, or features_only=False
    to warp the full detected set (legacy).

    valid_cfs (optional): explicit list of notch centre frequencies (Hz) to warp
    for THIS direction, overriding the per-direction select_features gate. Used
    by edge_shift_set(use_tracking=True), which decides validity from
    elevation-continuity tracks (stabilized_valid_cfs) so the same physical notch
    is shifted at all its elevations rather than flickering in and out. Detected
    minima within valid_tol_oct octaves of any listed CF are warped; an empty
    list means 'no valid notch here' (no manipulation)."""
    ir = np.asarray(ir, float)
    n = len(ir)
    # 4x oversampling beyond the next power of 2: detect_notches' derivative
    # (np.gradient of the raw spectrum) and the cepstral cosine basis are
    # both resolution-sensitive, and this is the frequency grid density the
    # detection scheme was actually validated against (see
    # edge_detection_and_shift_methods.md) -- the bare next-power-of-2 grid
    # is too coarse for the edge-extent rule to behave as characterized.
    nfft = nfft or int(2 ** np.ceil(np.log2(n)) * 4)
    H = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mag = np.abs(H)
    L = 20.0 * np.log10(np.maximum(mag, EPS))
    phase = np.angle(H)

    e, Le, band, fb = _to_erb_grid(L, freqs, f_lo, f_hi)
    notches_all = detect_notches(e, freqs, mag, f_lo=f_lo, f_hi=f_hi, sigma_hz=sigma_hz,
                                 n_keep=n_keep, prominence_db=prominence_db, eps_frac=eps_frac)
    ceiling_centers_e = None
    if valid_cfs is not None:
        # elevation-continuity path: the caller (edge_shift_set with
        # use_tracking) has already decided, per direction, WHICH notch centre
        # frequencies are valid cues (stabilized_valid_cfs), so bypass the
        # per-direction select_features gate entirely and warp exactly the
        # detected minima matching those CFs. Same detect params were used to
        # compute valid_cfs, so a small octave tolerance matches them robustly.
        cfs = list(valid_cfs)
        notches = [d for d in notches_all
                   if any(abs(np.log2(d['f_hz'] / c)) < valid_tol_oct for c in cfs)]
        ceiling_centers_e = sorted(float(e[d['i_min']]) for d in notches)
    elif features_only:
        fkw = feature_kw or {}
        # manipulate EVERY real notch's edge -- the sep-merge (Macpherson 0.5 oct)
        # is about counting separately-resolvable CUES for the viability screen,
        # not about which edges to shift: two deep notches 0.4 oct apart (e.g. JS
        # 8 & 10.7 kHz) are distinct features with non-overlapping edge regions,
        # so both should be shifted. Use the UNMERGED valid set (sep_min_oct=0)
        # for both the manipulated notches and the compression ceilings.
        notches = select_features(notches_all, **{**fkw, 'sep_min_oct': 0.0})
        ceiling_centers_e = sorted(float(e[d['i_min']]) for d in notches)
    else:
        notches = notches_all
    if not notches:
        rep = dict(status='no_notch_found', n_notches=0)
        return (ir.copy(), rep) if return_report else ir.copy()

    Le_new, realized = build_warped_logmag(e, Le, notches, abs(shift_erb), mode,
                                           strict=strict, ceiling_centers_e=ceiling_centers_e)
    if match_power:
        Le_new = match_band_rms(e, Le, Le_new, rms_band)

    L_full = L.copy()
    L_full[band] = np.interp(fb, erb_to_hz(e), Le_new)
    ir_new = np.fft.irfft(10.0 ** (L_full / 20.0) * np.exp(1j * phase), nfft)[:n]

    if not return_report:
        return ir_new
    per_notch = []
    for k, nn in enumerate(notches):
        f_hz = float(erb_to_hz(e[nn['i_min']]))
        per_notch.append(dict(
            f_hz=round(f_hz, 0),
            target_erb=round(abs(shift_erb), 2),
            target_hz=round(erb_shift_to_hz(f_hz, abs(shift_erb)), 0),
            realized_erb=round(realized[k], 2),
            realized_hz=round(erb_shift_to_hz(f_hz, realized[k]), 0),
            depth_base_db=round(_depth_db(e, Le, nn['i_lo'], nn['i_hi']), 1),
            depth_new_db=round(_depth_db(e, Le_new, nn['i_lo'], nn['i_hi']), 1)))
    f = erb_to_hz(e)
    mrb = (f >= rms_band[0]) & (f <= rms_band[1])
    rms_d = 10 * np.log10(np.mean((10 ** (Le_new[mrb] / 20)) ** 2)
                          / np.mean((10 ** (Le[mrb] / 20)) ** 2))
    return ir_new, dict(status='ok', mode=mode, n_notches=len(notches),
                        rms_delta_db=round(rms_d, 2), notches=per_notch)


# --- edge-only (coarse baseline + relevant-cue rising edges) ----
def edge_only_ir(ir, fs, n_keep_baseline=4, f_lo=3000.0, f_hi=17500.0,
                 sigma_hz=SIGMA_HZ_DEFAULT, prominence_db=3.0, feature_kw=None,
                 n_cues=None, tilt_db_per_oct=0.0, edge_skirt_oct=0.30,
                 match_power=True, rms_band=(3000.0, 16000.0), nfft=None,
                 return_report=False, **_legacy):
    """Coarse externalisation baseline + the rising edge(s) of the RELEVANT
    elevation cue(s), each reproduced at its measured shape.

    Rationale (modify.py shift paradigm + the n_keep perceptual-scale notes): a
    low-n_keep cepstral baseline (n_keep_baseline ~ 4) keeps the broad concha
    envelope that drives EXTERNALISATION -- so the stimulus still sounds real and
    out-of-head -- but discards the fine spectral structure that carries
    ELEVATION, so on its own it externalises WITHOUT localising (the low-n_keep
    coarse-landscape condition). On top of that landscape we add back ONLY the
    RISING edge of each perceptually relevant notch, keeping the edge's shape and
    discarding the falling flank (the descent INTO the notch) -- so the notch
    itself never reappears, only its low->high recovery contrast.

    Which cues, and where each edge is, are defined EXACTLY as in the edge-shift
    paradigm (detect_notches + select_features), not by a raw positive-slope mask:

        base_db  = _cepstral_smooth_db(|H|, n_keep_baseline)   # coarse landscape
        ref_db   = _gaussian_smooth_db(|H|, sigma_hz)          # de-rippled original
        notches  = detect_notches(...)          # minima, saddles, edge_rise spans
        feats    = select_features(notches, **feature_kw)  # the RELEVANT cue(s)
        for each feat (frequency-sorted), take its edge_rise span
        [near_min .. near_saddle] and copy the de-rippled original's rise over
        that span verbatim as a step (shape conserved), holding a flat PLATEAU
        above the shoulder so no falling edge is introduced. Successive edges
        stack into a cascade.
        L_new    = base_db + overlay

    Because the overlay is built only inside each cue's own edge_rise span and is
    flat everywhere else, the manipulated DTF is the smooth landscape plus one (or
    a few) genuine rising edges at their true frequencies and shapes -- nothing is
    reintroduced away from a relevant cue.

    feature_kw : passed to select_features (band, min_depth_db, min_width_oct,
        sep_min_oct) to choose which notch(es) count as relevant cues. Defaults
        gate to the elevation-notch band with the Iida/Moore depth/width criteria.
        Pass sep_min_oct=0.0 to keep separately-resolvable notches distinct.
    n_cues : if set, keep only the n_cues most prominent (deepest) relevant cues
        per direction; fewer are kept when fewer are detected.
    tilt_db_per_oct : log-frequency attenuation subtracted from the overlay,
        -tilt_db_per_oct * log2(f / f_lo) dB -- offsets the cascade's upward step
        accumulation and mimics the natural HF roll-off. 0 leaves the raw edges.
    edge_skirt_oct : octaves over which the overlay tapers back to the baseline at
        f_hi (continuity). Original PHASE is kept (ITD unchanged); match_power
        rescales to the original in-band RMS."""
    ir = np.asarray(ir, float)
    n = len(ir)
    nfft = nfft or int(2 ** np.ceil(np.log2(n)) * 4)
    H = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mag = np.abs(H)
    L = 20.0 * np.log10(np.maximum(mag, EPS))
    phase = np.angle(H)

    base_db = _cepstral_smooth_db(mag, n_keep_baseline)      # externalisation landscape
    ref_db = _gaussian_smooth_db(mag, freqs, sigma_hz)       # de-rippled original (edge shape)

    # ERB grid (same mapping detect_notches indexes into) for baseline, edge ref
    e, _Le, band, fb = _to_erb_grid(L, freqs, f_lo, f_hi)
    ehz = erb_to_hz(e)
    Le_base = np.interp(ehz, freqs, base_db)
    Le_ref = np.interp(ehz, freqs, ref_db)

    notches = detect_notches(e, freqs, mag, f_lo=f_lo, f_hi=f_hi, sigma_hz=sigma_hz,
                             prominence_db=prominence_db)
    feats = select_features(notches, **(feature_kw or {}))
    # keep only the n_cues most prominent (deepest) relevant cues, then restore
    # frequency order so the edges still stack low->high.
    if n_cues is not None and len(feats) > n_cues:
        feats = sorted(feats, key=lambda d: (d.get('depth_db') or 0.0), reverse=True)[:n_cues]
        feats.sort(key=lambda d: d['f_hz'])

    overlay = np.zeros_like(e)
    running = 0.0
    edges = []
    for d in feats:                                  # frequency-sorted (select_features)
        er = d.get('edge_rise')
        if er is None:
            continue
        a, b = int(er[0]), int(er[1])                # (near_min, near_saddle) ERB indices
        if b <= a:
            continue
        seg = Le_ref[a:b + 1] - Le_ref[a]            # verbatim rising-edge shape, 0 at onset
        overlay[a:b + 1] = running + seg
        overlay[b + 1:] = running + seg[-1]          # hold plateau above the shoulder
        running += float(seg[-1])
        edges.append(dict(f_hz=d.get('f_hz'), f_edge_rise_hz=d.get('f_edge_rise_hz'),
                          rise_db=round(float(seg[-1]), 1), label=d.get('label')))

    # log-frequency detilt + top-edge taper, then rejoin the baseline
    if tilt_db_per_oct and overlay.any():
        overlay = np.maximum(overlay - tilt_db_per_oct * np.log2(ehz / f_lo), 0.0)
    if edge_skirt_oct > 0 and overlay.any():
        top = np.log2(ehz[-1])
        k = np.log2(ehz) > (top - edge_skirt_oct)
        taper = np.ones_like(ehz)
        taper[k] = 0.5 * (1 + np.cos(np.pi * (np.log2(ehz[k]) - (top - edge_skirt_oct)) / edge_skirt_oct))
        overlay = overlay * taper

    Le_new = Le_base + overlay
    L_new = L.copy()
    L_new[band] = np.interp(fb, ehz, Le_new)         # in-band replace; keep original outside
    # outside the analysis band, fall back to the smooth landscape too
    L_new[~band] = base_db[~band]

    if match_power:
        m = (freqs >= rms_band[0]) & (freqs <= rms_band[1])
        p0 = np.mean((10.0 ** (L[m] / 20.0)) ** 2)
        p1 = np.mean((10.0 ** (L_new[m] / 20.0)) ** 2)
        L_new = L_new + 10.0 * np.log10(p0 / max(p1, EPS))

    ir_new = np.fft.irfft(10.0 ** (L_new / 20.0) * np.exp(1j * phase), nfft)[:n]
    if not return_report:
        return ir_new
    return ir_new, dict(status='ok' if edges else 'no_relevant_cue', mode='edge_only',
                        n_keep_baseline=n_keep_baseline, n_edges=len(edges),
                        total_rise_db=round(running, 1), edges=edges)


# --- Iida-style parametric description + perceptual gates ----
def select_features(notches, band=NOTCH_BAND, min_depth_db=MIN_NOTCH_DEPTH_DB,
                    min_width_oct=DETECT_WIDTH_MIN_OCT, sep_min_oct=SEP_MIN_OCT):
    """Reduce the full detected-minima set to the real elevation notches,
    treating each separately resolvable notch as one feature.

    A detected minimum counts as a notch if its centre lies in the elevation-
    notch band (Asano/Suzuki/Sone 1990, van Opstal), its depth relative to the
    bracketing maxima is >= min_depth_db (Moore et al. 1989 peak/saddle
    detection ~2.5-5 dB @8 kHz -- this is the wiggle-vs-cue discriminator), and
    its half-depth width clears a small real-feature floor min_width_oct
    (DETECT_WIDTH_MIN_OCT ~0.05 oct). The width floor is deliberately well below
    the Moore perceptual-detectability threshold (WIDTH_MIN_OCT 0.17): the Iida
    Gaussian has already removed bin-level wiggles, so depth -- not width --
    separates real notches from noise, and deep-but-narrow notches are genuine
    cues (raise min_width_oct toward 0.17 to re-impose perceptual gating).

    Two surviving notches are kept as SEPARATE features only if their centres
    are at least sep_min_oct apart (~2 ripples/oct, Macpherson & Middlebrooks
    2003); a pair closer than that is not separately resolvable, so only the
    more prominent (deeper) one is kept. Single notch is therefore the common
    outcome, two notches the exception. Returns the feature list, frequency-
    sorted and relabelled N1, N2, ..."""
    cand = [d for d in notches
            if band[0] <= d["f_hz"] <= band[1]
            and (d.get("depth_db") or 0.0) >= min_depth_db
            and (d.get("width_oct") or 0.0) >= min_width_oct]
    cand.sort(key=lambda d: d["f_hz"])
    feats = []
    for d in cand:
        if feats and np.log2(d["f_hz"] / feats[-1]["f_hz"]) < sep_min_oct:
            # not separately resolvable from the previous feature: keep deeper
            if (d.get("depth_db") or 0.0) > (feats[-1].get("depth_db") or 0.0):
                feats[-1] = d
        else:
            feats.append(d)
    out = []
    for i, d in enumerate(feats, 1):
        d = dict(d)
        d["label"] = f"N{i}"
        out.append(d)
    return out


# --- elevation-continuity tracking (robust notch identity across a cone) ----
# Per-direction detection + gating flickers: a real notch that momentarily dips
# below the depth gate pops in and out between adjacent elevations, and a
# "deepest = primary" label swaps between unrelated notches when two are close in
# depth (measured on real DTFs: up to ~1 oct primary-CF jumps between 4-deg-apart
# directions, and ~17% of physical notch tracks shifted at some elevations but
# not adjacent ones). Median-plane notches are in fact smooth, continuous
# elevation trajectories, so the fix is to link detected minima into tracks and
# decide each track's identity/validity ONCE, then apply it to every member
# direction. detect_notches itself is unchanged; this is a layer on top.
def _link_tracks(seqs, band=NOTCH_BAND, tol_oct=0.22, max_gap=3):
    """Link per-direction detected notch minima into elevation trajectories.

    seqs : list (one per direction, ordered by elevation) of the detect_notches
        output for that direction. Only minima with a centre in `band` are linked.
    A detected minimum extends the nearest active track whose last centre is
    within tol_oct octaves; a track may skip up to max_gap directions (a minimum
    momentarily undetected) and still continue. Unclaimed minima start new tracks.

    tol_oct (0.22) and max_gap (3) are the knobs against OVER-SEGMENTATION (one
    physical trough chopped into several tracks). Both defaults were raised from
    the initial 0.14/1 after the database overview showed single notches split
    where (a) the CF steps locally steeply between two elevations (>0.14 oct,
    e.g. CO 5.94->6.72 kHz) -- tol_oct handles this -- or (b) the notch fades
    below detection for a couple of directions before returning -- max_gap
    handles this. tol_oct stays well under the ~0.5 oct inter-notch spacing, so
    distinct notches are not merged (verified: no track picks up two minima at
    one elevation across the measured database). A residual split remains only
    where a notch genuinely fades for more than max_gap directions -- there the
    split is correct, not an artifact, and must not be bridged.

    Returns a list of tracks, each a list of (row, notch_dict) in elevation
    order. Greedy nearest-CF association -- adequate because median-plane notch
    tracks are well separated (>~0.5 oct) and move smoothly (< ~0.06 oct/step
    typically; steeper local segments up to ~0.19 oct occur, hence tol_oct 0.22)."""
    tracks = []  # dict(last_row, last_cf, pts=[(row, notch)])
    for row, notches in enumerate(seqs):
        cand = sorted([d for d in notches if band[0] <= d['f_hz'] <= band[1]],
                      key=lambda d: d['f_hz'])
        used = set()
        for tr in tracks:
            if row - tr['last_row'] > max_gap + 1:
                continue  # track went cold -- can't bridge this large a gap
            best, best_d = None, tol_oct
            for j, d in enumerate(cand):
                if j in used:
                    continue
                dd = abs(np.log2(d['f_hz'] / tr['last_cf']))
                if dd < best_d:
                    best, best_d = j, dd
            if best is not None:
                used.add(best)
                tr['pts'].append((row, cand[best]))
                tr['last_row'], tr['last_cf'] = row, cand[best]['f_hz']
        for j, d in enumerate(cand):
            if j not in used:
                tracks.append(dict(last_row=row, last_cf=d['f_hz'], pts=[(row, d)]))
    return [tr['pts'] for tr in tracks]


def _arcs(sources_vp, az_tol=5.0, el_range=(-90.0, 90.0)):
    """Group direction indices into constant-azimuth elevation arcs (cones of
    confusion), each sorted by elevation. Azimuth is binned to az_tol degrees;
    only elevations within el_range are kept. Returns {az_bin: [dir_idx, ...]}.
    A midline-only SOFA collapses to a single arc; a full HRTF yields one arc
    per measured azimuth."""
    sources_vp = np.asarray(sources_vp, float)
    az = (sources_vp[:, 0] + 180) % 360 - 180
    el = sources_vp[:, 1]
    keys = np.round(az / az_tol) * az_tol
    arcs = {}
    for i in range(len(sources_vp)):
        if el_range[0] <= el[i] <= el_range[1]:
            arcs.setdefault(float(keys[i]), []).append(i)
    for k in arcs:
        arcs[k].sort(key=lambda i: el[i])
    return arcs


def track_notches(arr, fs, sources_vp, ear, az_bin=None, band=NOTCH_BAND,
                  tol_oct=0.22, max_gap=3, az_tol=5.0, el_range=(-90.0, 90.0),
                  sigma_hz=SIGMA_HZ_DEFAULT, n_keep=None, prominence_db=3.0,
                  eps_frac=0.15, nfft=None, f_lo=3000.0, f_hi=17500.0):
    """Detect + link notch tracks for one ear along each constant-azimuth arc.
    Returns {az_bin: [track, ...]}, each track a list of (dir_idx, notch_dict) in
    elevation order (dir_idx are indices into arr's direction axis). Pass az_bin
    to restrict to one arc. Mostly a QC/plotting entry point; the manipulation
    uses stabilized_valid_cfs (which wraps this)."""
    arr = np.asarray(arr, float)
    arcs = _arcs(sources_vp, az_tol=az_tol, el_range=el_range)
    if az_bin is not None:
        arcs = {az_bin: arcs.get(az_bin, [])}

    def detect(ir1d):
        n = len(ir1d)
        _nfft = nfft or int(2 ** np.ceil(np.log2(n)) * 4)
        freqs = np.fft.rfftfreq(_nfft, 1.0 / fs)
        mag = np.abs(np.fft.rfft(ir1d, _nfft))
        L = 20.0 * np.log10(np.maximum(mag, EPS))
        e, _Le, _b, _f = _to_erb_grid(L, freqs, f_lo, f_hi)
        return detect_notches(e, freqs, mag, f_lo=f_lo, f_hi=f_hi, sigma_hz=sigma_hz,
                              n_keep=n_keep, prominence_db=prominence_db, eps_frac=eps_frac)

    out = {}
    for az, idxs in arcs.items():
        seqs = [detect(arr[i, :, ear]) for i in idxs]
        tracks = _link_tracks(seqs, band=band, tol_oct=tol_oct, max_gap=max_gap)
        # remap row -> original direction index
        out[az] = [[(idxs[row], d) for row, d in pts] for pts in tracks]
    return out


def stabilized_valid_cfs(arr, fs, sources_vp, band=NOTCH_BAND,
                         min_depth_db=MIN_NOTCH_DEPTH_DB, min_width_oct=DETECT_WIDTH_MIN_OCT,
                         min_valid_frac=0.5, min_len=3, tol_oct=0.22, max_gap=3,
                         az_tol=5.0, el_range=(-90.0, 90.0), sigma_hz=SIGMA_HZ_DEFAULT,
                         n_keep=None, prominence_db=3.0, eps_frac=0.15, nfft=None,
                         f_lo=3000.0, f_hi=17500.0, return_labels=False):
    """Per-direction/ear set of notch centre frequencies to treat as VALID,
    decided by elevation-continuity tracks instead of independent per-direction
    gating. This is the robust replacement for select_features' per-direction
    gate when building a whole condition (see the tracking note above and the
    detection-robustness diagnosis).

    Within each constant-azimuth elevation arc, detected minima are linked into
    tracks (_link_tracks). A track counts as a real elevation cue when it passes
    the per-direction depth/width gate (min_depth_db, min_width_oct, in band) at
    >= min_valid_frac of the directions where a minimum was actually detected --
    a fraction rule that is robust to a single-direction dip below the gate. That
    one decision is then applied to EVERY detected member of the track, so:
      * a notch that momentarily dips below the depth gate is still shifted at
        that elevation (elevation-consistent manipulation), and
      * a track that is only briefly deep, or spans fewer than min_len detected
        directions, is excluded everywhere (rejects transient/noise minima).
    Only directions where the minimum was actually detected get a CF (a bridged
    gap has no minimum to shift, so none is fabricated).

    Returns {(dir, ear): [cf_hz, ...]}; with return_labels, also
    {(dir, ear): {cf_hz: 'N1'|'N2'|...}} labelling tracks by ascending mean CF
    (stable across the whole arc, unlike a per-direction depth rank)."""
    arr = np.asarray(arr, float)
    n_ears = arr.shape[2]
    valid = {}
    labels = {}
    det_kw = dict(band=band, tol_oct=tol_oct, max_gap=max_gap, az_tol=az_tol,
                  el_range=el_range, sigma_hz=sigma_hz, n_keep=n_keep,
                  prominence_db=prominence_db, eps_frac=eps_frac, nfft=nfft,
                  f_lo=f_lo, f_hi=f_hi)
    for ear in range(n_ears):
        arc_tracks = track_notches(arr, fs, sources_vp, ear, **det_kw)
        for az, tracks in arc_tracks.items():
            kept = []  # (mean_cf, [(dir, notch)]) for labelling
            for pts in tracks:
                if len(pts) < min_len:
                    continue
                passes = [((d.get('depth_db') or 0.0) >= min_depth_db
                           and (d.get('width_oct') or 0.0) >= min_width_oct)
                          for _i, d in pts]
                if np.mean(passes) >= min_valid_frac:
                    kept.append((float(np.mean([d['f_hz'] for _i, d in pts])), pts))
            kept.sort(key=lambda t: t[0])
            for rank, (_mcf, pts) in enumerate(kept, 1):
                for i, d in pts:
                    valid.setdefault((i, ear), []).append(d['f_hz'])
                    labels.setdefault((i, ear), {})[d['f_hz']] = f"N{rank}"
    return (valid, labels) if return_labels else valid


def parametric_summary(ir, fs, f_lo=3000.0, f_hi=17500.0, sigma_hz=SIGMA_HZ_DEFAULT,
                       n_keep=None, prominence_db=3.0, nfft=None,
                       feature_kw=None):
    """Parametric description of one DTF magnitude (Iida et al. 2007) plus the
    perceptual viability of its elevation notch(es). See
    experiment/protocols/documentation/cue_perception_synthesis.md.

    The full detected-minima set is reduced to perceptually valid FEATURES by
    select_features (band + depth + width gates; two notches split into
    separate features only when >= SEP_MIN_OCT apart). Single notch is the
    common outcome. A DTF is usable for the edge-shift manipulation as soon as
    it has one valid notch -- no N1/N2 separation is required.

    Returns dict:
      notches   : full detected set {label, f_hz, depth_db, width_oct, q}
      features  : perceptually valid notches (select_features), frequency-sorted
      p1_hz     : first spectral peak in ~3.5-6 kHz (elevation-independent ref)
      primary   : most prominent (deepest) valid notch, or None
      N1, N2    : first/second feature by frequency (N2 is None when single) -
                  kept for backwards compatibility
      gates     : {n_features, primary_f_hz, primary_depth_db, primary_width_oct,
                   separation_oct (only when two features), usable}

    usable = at least one valid notch (depth >= MIN_NOTCH_DEPTH_DB and width >=
    DETECT_WIDTH_MIN_OCT within NOTCH_BAND). Pass feature_kw={...} to tune the
    band/depth/width/separation of select_features.
    """
    ir = np.asarray(ir, float)
    n = len(ir)
    nfft = nfft or int(2 ** np.ceil(np.log2(n)) * 4)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mag = np.abs(np.fft.rfft(ir, nfft))
    L = 20.0 * np.log10(np.maximum(mag, EPS))
    e, _Le, _band, _fb = _to_erb_grid(L, freqs, f_lo, f_hi)
    notches = detect_notches(e, freqs, mag, f_lo=f_lo, f_hi=f_hi, sigma_hz=sigma_hz,
                             n_keep=n_keep, prominence_db=prominence_db)

    Lc = (_cepstral_smooth_db(mag, n_keep) if n_keep is not None
          else _gaussian_smooth_db(mag, freqs, sigma_hz))

    # P1 reference peak (concha resonance, elevation-independent): first 3.5-6 kHz peak
    pk, _ = find_peaks(Lc, prominence=SADDLE_PROM_DB)
    p1 = next((round(float(freqs[i]), 0) for i in pk if 3500.0 <= freqs[i] <= 6000.0), None)

    features = select_features(notches, **(feature_kw or {}))
    primary = max(features, key=lambda d: d.get("depth_db") or 0.0) if features else None
    N1 = features[0] if len(features) >= 1 else None
    N2 = features[1] if len(features) >= 2 else None

    gates = dict(n_features=len(features), primary_f_hz=None,
                 primary_depth_db=None, primary_width_oct=None,
                 separation_oct=None, usable=False)
    if primary is not None:
        gates["primary_f_hz"] = primary["f_hz"]
        gates["primary_depth_db"] = primary["depth_db"]
        gates["primary_width_oct"] = primary["width_oct"]
        gates["usable"] = True
    if N1 is not None and N2 is not None:
        gates["separation_oct"] = round(float(np.log2(N2["f_hz"] / N1["f_hz"])), 3)

    keys = ("label", "f_hz", "f_edge_rise_hz", "depth_db", "width_oct", "q")
    return dict(
        notches=[{k: d.get(k) for k in keys} for d in notches],
        features=[{k: d.get(k) for k in keys} for d in features],
        p1_hz=p1, primary=primary, N1=N1, N2=N2, gates=gates)


def shift_is_perceptible(f_hz, shift_erb):
    """True if an ERB-sized notch-CF shift at f_hz clears Moore et al. (1989)'s
    ~8% notch center-frequency JND (CF_JND_FRAC) -- i.e. the manipulation is large
    enough to be heard as a spectral change. Check before building a condition."""
    return abs(erb_shift_to_hz(f_hz, shift_erb)) >= CF_JND_FRAC * float(f_hz)


# --- full set (numpy core) ----
def edge_shift_set(hrirs, fs, shift_erb, mode='rising', verbose=False, strict=False,
                   use_tracking=False, sources_vp=None, track_kw=None, **kw):
    """hrirs: (n_dir, n_taps, n_ears). Returns (out_array, reports).

    strict=False (default, unlike edge_shift_ir's own default): one
    direction/ear with a too-close bracketing peak shouldn't abort a whole
    condition build, so this always clamps rather than raising -- but it
    never does so silently: any clamped notch is printed below, itemized by
    (dir, ear, frequency, requested vs realized ERB/Hz). Pass strict=True to
    raise on the first one instead (e.g. for interactively checking one
    direction with edge_shift_ir directly).

    use_tracking (opt-in): decide which notches to shift from elevation-continuity
    tracks (stabilized_valid_cfs) instead of the per-direction select_features
    gate. This removes the count flicker / inconsistent-manipulation artifact
    (a real notch shifted at some elevations but not adjacent ones). Requires
    sources_vp (n_dir x 3, az/el/r -- e.g. base_hrtf.sources.vertical_polar);
    track_kw overrides tracking params (min_valid_frac, min_len, band, tol_oct,
    max_gap, az_tol, el_range). The band defaults to feature_kw['band'] if given.
    Tracking uses the same detect params (sigma_hz/n_keep/prominence_db/eps_frac/
    f_lo/f_hi) as the per-IR warp, taken from kw so the CFs align."""
    hrirs = np.asarray(hrirs, float)
    out = np.empty_like(hrirs)
    reports = []

    valid_map = None
    if use_tracking:
        if sources_vp is None:
            raise ValueError("edge_shift_set(use_tracking=True) requires sources_vp "
                             "(n_dir x 3 az/el/r, e.g. base_hrtf.sources.vertical_polar)")
        tkw = dict(track_kw or {})
        # keep detection params in sync with the per-IR warp so track CFs match
        for k in ('f_lo', 'f_hi', 'sigma_hz', 'n_keep', 'prominence_db', 'eps_frac', 'nfft'):
            if k in kw and k not in tkw:
                tkw[k] = kw[k]
        fkw = kw.get('feature_kw') or {}
        if 'band' in fkw and 'band' not in tkw:
            tkw['band'] = fkw['band']
        valid_map = stabilized_valid_cfs(hrirs, fs, sources_vp, **tkw)

    for d in range(hrirs.shape[0]):
        for ear in range(hrirs.shape[2]):
            vc = None if valid_map is None else valid_map.get((d, ear), [])
            ir_new, rep = edge_shift_ir(hrirs[d, :, ear], fs, shift_erb,
                                        mode=mode, return_report=True, strict=strict,
                                        valid_cfs=vc, **kw)
            out[d, :, ear] = ir_new
            rep.update(dir=d, ear=ear)
            reports.append(rep)
            if verbose and rep['status'] != 'ok':
                print(f"dir {d} ear {ear}: {rep['status']}")
    if verbose:
        print_notch_summary(reports, label=f"{mode} notch counts")
    print_clamped_notches(reports)
    return out, reports


def edge_only_set(hrirs, fs, verbose=False, **kw):
    """edge-only (flatten, keep rising edges) over a full HRIR set.
    hrirs: (n_dir, n_taps, n_ears). Returns (out_array, reports). kw passes to
    edge_only_ir (f_lo/f_hi/feature_kw/match_power/...)."""
    hrirs = np.asarray(hrirs, float)
    out = np.empty_like(hrirs)
    reports = []
    for d in range(hrirs.shape[0]):
        for ear in range(hrirs.shape[2]):
            ir_new, rep = edge_only_ir(hrirs[d, :, ear], fs, return_report=True, **kw)
            out[d, :, ear] = ir_new
            rep.update(dir=d, ear=ear)
            reports.append(rep)
    if verbose:
        print(f"edge_only: {len(reports)} directions x ears flattened")
    return out, reports


def print_clamped_notches(reports):
    """Loudly report any notch that couldn't reach its full target_erb
    (clamped by the bracketing peak) -- always printed, not just verbose,
    so a batch run (edge_shift_set with strict=False) never silently waters
    down a shift without you noticing.

    Grouped by (ear, f_hz, target_erb, realized_erb) rather than printed one
    line per direction: many HRIR sets copy the midline DTF magnitude across
    azimuths with only the ITD changed, so those directions are identical
    manipulation targets and would otherwise repeat the same line dozens of
    times. `dirs` lists which direction indices share each group."""
    clamped = [
        (r.get('dir'), r.get('ear'), nn)
        for r in reports if r.get('status') == 'ok'
        for nn in r['notches']
        if abs(nn['realized_erb'] - nn['target_erb']) > 1e-6
    ]
    if not clamped:
        return
    from collections import defaultdict
    groups = defaultdict(list)
    for d, ear, nn in clamped:
        key = (ear, nn['f_hz'], nn['target_erb'], nn['target_hz'],
              nn['realized_erb'], nn['realized_hz'])
        groups[key].append(d)
    print(f"WARNING: {len(clamped)} notch(es) clamped short of the requested shift "
         f"across {len(groups)} distinct spectra (bracketing peak too close):")
    for (ear, f_hz, tgt_e, tgt_hz, real_e, real_hz), dirs in groups.items():
        print(f"  ear {ear} @ {f_hz:.0f} Hz ({len(dirs)} dir(s)): requested "
             f"{tgt_e:.2f} ERB ({tgt_hz:.0f} Hz), realized {real_e:.2f} ERB ({real_hz:.0f} Hz)")


def notch_count_summary(reports):
    """{n_notches: count} breakdown across a list of edge_shift_ir reports
    (e.g. edge_shift_set's `reports`), so you can see the full distribution --
    e.g. {0: 2, 1: 8, 2: 40, 3: 10} -- instead of just an ok/fail count."""
    from collections import Counter
    return dict(sorted(Counter(r['n_notches'] for r in reports).items()))


def print_notch_summary(reports, label=None):
    """Print notch_count_summary() as one readable line, e.g.:
    'rising notch counts: 0 notches: 2/60, 1 notch: 8/60, 2 notches: 40/60, 3 notches: 10/60'"""
    counts = notch_count_summary(reports)
    total = len(reports)
    parts = ', '.join(f"{n} notch{'es' if n != 1 else ''}: {c}/{total}"
                      for n, c in counts.items())
    print(f"{label + ': ' if label else ''}{parts}")


# --- model-resolution (~28-bin) verification ----
def erb_filterbank_rms(freqs, mag, flow=700.0, fhigh=18000.0, spacing=1.0):
    """RMS (dB) in ERB-spaced gammatone-weighted filters ~ Baumgartner front end.
    Replace with the project's /hrtf/models bank for exact model-side numbers."""
    ec = np.arange(float(hz_to_erb(flow)), float(hz_to_erb(fhigh)), spacing)
    fc = erb_to_hz(ec)
    P = mag ** 2
    out = np.empty(len(fc))
    for i, c in enumerate(fc):
        bw = 24.7 * (4.37 * c / 1000.0 + 1.0)
        w = (1.0 + 4.0 * ((freqs - c) / (1.019 * bw)) ** 2) ** (-2)
        out[i] = 10 * np.log10(np.sum(w * P) / np.sum(w) + EPS)
    return fc, ec, out


def baumgartner_filterbank(fs, flow=700.0, fhigh=20000.0):
    """Factory returning a filterbank(freqs, mag) callable bound to `fs`, for
    verify_binned's filterbank= argument, built on the project's actual model
    front end (pf.dsp.filter.GammatoneBands, same call as
    hrtf.models.baumgartner2014.eq_2: freq_range=[700, 20e3]) instead of the
    hand-rolled ERB-gammatone approximation in erb_filterbank_rms. Use this
    when the verification number needs to match what Baumgartner2014 /
    predict_polar actually reads, not just an ERB-spaced proxy.

    mag is reconstructed to a zero-phase IR before filtering (phase does not
    affect per-band RMS, and verify_binned only ever passes magnitude in)."""
    import pyfar as pf
    bands = pf.dsp.filter.GammatoneBands(freq_range=[flow, fhigh], sampling_rate=fs)

    def _filterbank(freqs, mag):
        nfft = 2 * (len(freqs) - 1)
        ir = np.fft.irfft(mag.astype(complex), nfft)
        filtered = bands.process(pf.Signal(ir, fs))[0]
        rms = np.sqrt(np.mean(filtered.time ** 2, axis=-1)).squeeze(axis=-1)
        Lb = 20.0 * np.log10(np.maximum(rms, EPS))
        fc = bands.frequencies
        return fc, hz_to_erb(fc), Lb

    return _filterbank


def verify_binned(ir_base, ir_new, fs, search_hz=(5000.0, 11000.0), nfft=None,
                  filterbank=erb_filterbank_rms):
    nfft = nfft or int(2 ** np.ceil(np.log2(len(ir_base))))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    fc, ec, Lb = filterbank(f, np.abs(np.fft.rfft(ir_base, nfft)))
    _, _, Ln = filterbank(f, np.abs(np.fft.rfft(ir_new, nfft)))

    def ec_edge(L):
        win = (fc >= search_hz[0]) & (fc <= search_hz[1])
        im = int(np.where(win)[0][np.argmin(L[win])])
        g = np.maximum(np.gradient(L, ec), 0.0)
        b = min(len(L) - 1, im + 5)
        ge = g[im:b + 1]
        lo = max(0, im - 4)
        ref = max(L[lo], L[b])
        d = np.maximum(ref - L[lo:b + 1], 0.0)
        return (np.sum(ec[im:b + 1] * ge) / (ge.sum() + EPS),
                np.sum(ec[lo:b + 1] * d) / (d.sum() + EPS))

    e0, c0 = ec_edge(Lb)
    e1, c1 = ec_edge(Ln)
    return dict(n_bins=len(fc), edge_shift_binned=round(e1 - e0, 3),
                centroid_shift_binned=round(c1 - c0, 3))


# --- slab adapters ----
def hrtf_to_array(hrtf):
    return np.stack([f.data for f in hrtf.data]), float(hrtf.samplerate)


def array_to_hrtf(arr, base_hrtf):
    import slab
    return slab.HRTF(data=arr, datatype='FIR', samplerate=base_hrtf.samplerate,
                     sources=base_hrtf.sources.vertical_polar,
                     listener=base_hrtf.listener)


def manipulate_hrtf(base_hrtf, condition, shift_erb, use_tracking=False, **kw):
    """condition in {'baseline','whole','edge','edge_only','rising','falling'}.
    Returns (manipulated slab.HRTF, reports). 'baseline' clones unchanged.

    'whole' translates each notch up by shift_erb (both saddles pinned; the
    dose/sanity arm). 'edge' translates each notch's half-height rising edge
    up by shift_erb (minimum and upper peak pinned; the DCN rising-edge arm --
    has real headroom, unlike 'rising' which rigidly moves the whole steep
    flank and clamps). 'edge_only' flattens the DTF keeping ONLY the rising
    edges (shift_erb ignored; the DCN edge-isolation stimulus). Pass
    feature_kw={'band': (4000., 15000.)} to select all perceptually-valid
    notches in the elevation range (default NOTCH_BAND is narrower).

    use_tracking (opt-in, whole/edge/rising/falling only): pick the notches to
    shift from elevation-continuity tracks rather than per-direction gating, so
    the same physical notch is shifted at all its elevations (no flicker). The
    source geometry is taken from base_hrtf; pass track_kw={...} in kw to tune
    the tracker (min_valid_frac, min_len, ...). Ignored for 'edge_only'."""
    if condition == 'baseline':
        arr, _ = hrtf_to_array(base_hrtf)
        return array_to_hrtf(arr, base_hrtf), [dict(status='baseline')]
    arr, fs = hrtf_to_array(base_hrtf)
    if condition == 'edge_only':
        out, reports = edge_only_set(arr, fs, **kw)
    else:
        sources_vp = np.asarray(base_hrtf.sources.vertical_polar) if use_tracking else None
        out, reports = edge_shift_set(arr, fs, shift_erb, mode=condition,
                                      use_tracking=use_tracking, sources_vp=sources_vp, **kw)
    return array_to_hrtf(out, base_hrtf), reports


def modification_params(base_hrtf, condition, shift_erb, **kw):
    """Assemble the parameter record that fully describes a manipulation.

    This is the ground truth of *what was done* to produce a modified SOFA:
    the condition, the shift magnitude, and every keyword forwarded to
    manipulate_hrtf (use_tracking, feature_kw, track_kw, n_keep, ...), plus the
    base HRTF identity, a timestamp and the repo git hash so the exact code can
    be recovered too. Written into the SOFA (see _embed_modification_params)
    and echoed onto the localization test so a run can always be traced back to
    its modification -- which was impossible for e.g. FD's 12:13 test."""
    import datetime, subprocess
    from pathlib import Path
    try:
        git_hash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).resolve().parent), capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        git_hash = None
    return {
        'condition': condition,
        'shift_erb': shift_erb,
        'kw': kw,
        'base_hrtf': (getattr(base_hrtf, 'name', None)
                      or str(getattr(base_hrtf, 'sofa_path', '')) or None),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_hash': git_hash,
        'module': 'edge_shift',
    }


def _embed_modification_params(path, params):
    """Best-effort: write `params` as a JSON GLOBAL_ModificationParams attribute
    into the just-written SOFA (HDF5). Wrapped so a failure never blocks the
    SOFA write -- the file is already on disk before this runs."""
    import json
    try:
        import h5py
        blob = json.dumps(params, default=str)
        with h5py.File(str(path), 'r+') as h:
            h.attrs['GLOBAL_ModificationParams'] = blob
        return blob
    except Exception as exc:
        print(f"  [warn] could not embed modification params in SOFA: {exc}")
        return None


def read_modification_params(path):
    """Return the modification-params dict embedded in a SOFA, or None.

    None means the file predates param embedding (e.g. FD_shift.sofa) or was
    written by another tool -- the modification is then not self-documented."""
    import json
    try:
        import h5py
        with h5py.File(str(path), 'r') as h:
            blob = h.attrs.get('GLOBAL_ModificationParams')
        if blob is None:
            return None
        if isinstance(blob, bytes):
            blob = blob.decode()
        return json.loads(blob)
    except Exception:
        return None


def save_condition_sofa(base_hrtf, condition, shift_erb, path, plot=True,
                        plot_dir=None, plot_ears=('left', 'right'),
                        plot_smoothing='raw', **kw):
    """Write one manipulated condition to a SOFA file (metadata cloned from base,
    only the IR data replaced). Returns (manipulated slab.HRTF, reports).

    The modification parameters (condition, shift_erb, forwarded kw, base HRTF,
    timestamp, git hash) are embedded into the SOFA as a JSON
    GLOBAL_ModificationParams attribute, so a modified HRTF is self-documenting
    and any localization run against it can be traced back to exactly what was
    done (read them back with read_modification_params).

    plot : if True (default), also save a baseline-vs-manipulated WATERFALL QC
        figure (one column per ear) via save_waterfall_qc -- the overlay view
        for eyeballing an edge/notch shift, produced automatically whenever a
        shifted copy is generated. Plotting is wrapped so a failure never blocks
        the SOFA write. Pass plot=False to skip.
    plot_dir : override output directory (default subject_plot_dir(<subject>)/hrtf/).
    plot_ears, plot_smoothing : forwarded to save_waterfall_qc."""
    hrtf_new, reports = manipulate_hrtf(base_hrtf, condition, shift_erb, **kw)
    hrtf_new.write_sofa(str(path))
    _embed_modification_params(
        path, modification_params(base_hrtf, condition, shift_erb, **kw))
    if plot:
        try:
            fig_path = save_waterfall_qc(base_hrtf, hrtf_new, path, plot_dir=plot_dir,
                                         ears=plot_ears, smoothing=plot_smoothing)
            print(f"  waterfall QC -> {fig_path}")
        except Exception as exc:  # never let a plot failure block the experiment
            print(f"  [warn] waterfall QC plot skipped: {exc}")
    return hrtf_new, reports


def verify_condition(base_hrtf, manip_hrtf, base_sofa_path, cond_sofa_path,
                     search_hz=(5000.0, 11000.0)):
    """Two-level verification of one manipulated condition against baseline,
    per the predict-then-test loop (elevation_spectral_cue_models.md sec. 5):

    1. Model-resolution edge/centroid shift (verify_binned, using
       baumgartner_filterbank so it reads exactly what the model reads),
       run per direction/ear on the raw HRIRs — fast, no SOFA write needed.
    2. The actual Baumgartner2014 polar-response prediction (predict_polar)
       for baseline vs condition, both using the *baseline* SOFA as template
       (self-template, per sec. 3), so mean_qe/mean_pe deltas reflect only
       the manipulation.

    base_hrtf, manip_hrtf : slab.HRTF (same directions/order, e.g. from
        manipulate_hrtf) — used for the fast per-direction binned check.
    base_sofa_path, cond_sofa_path : str or Path — already-written SOFA files
        (e.g. from save_condition_sofa) — used for the full model prediction.

    Returns dict(binned=[...], baseline=predict_polar dict,
                condition=predict_polar dict, delta_pe, delta_qe).
    """
    from hrtf_relearning.hrtf.models.baumgartner2014 import predict_polar

    base_arr, fs = hrtf_to_array(base_hrtf)
    manip_arr, _ = hrtf_to_array(manip_hrtf)
    fb = baumgartner_filterbank(fs)
    binned = []
    for d in range(base_arr.shape[0]):
        for ear in range(base_arr.shape[2]):
            rep = verify_binned(base_arr[d, :, ear], manip_arr[d, :, ear], fs,
                                search_hz=search_hz, filterbank=fb)
            rep.update(dir=d, ear=ear)
            binned.append(rep)

    base_pred = predict_polar(str(base_sofa_path), str(base_sofa_path))
    cond_pred = predict_polar(str(cond_sofa_path), str(base_sofa_path))

    return dict(binned=binned, baseline=base_pred, condition=cond_pred,
               delta_pe=cond_pred['mean_pe'] - base_pred['mean_pe'],
               delta_qe=cond_pred['mean_qe'] - base_pred['mean_qe'])


# --- visualization ----
def median_plane_indices(hrtf, az_tol=2.0):
    vp = hrtf.sources.vertical_polar
    az = (vp[:, 0] + 180) % 360 - 180
    idx = np.where(np.abs(az) <= az_tol)[0]
    return list(idx[np.argsort(vp[idx, 1])])  # sorted by elevation


def compare_tf(base_hrtf, manip_hrtf, sourceidx=None, ear='left', kind='image',
               xlim=(3000, 16000), show=True):
    """Side-by-side plot_tf of baseline vs manipulated for the same sources.
    sourceidx defaults to the median-plane arc. Returns the figure."""
    import matplotlib.pyplot as plt
    if sourceidx is None:
        sourceidx = median_plane_indices(base_hrtf)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    base_hrtf.plot_tf(sourceidx, ear=ear, kind=kind, xlim=xlim, show=False, axis=ax[0])
    manip_hrtf.plot_tf(sourceidx, ear=ear, kind=kind, xlim=xlim, show=False, axis=ax[1])
    ax[0].set_title('baseline'); ax[1].set_title('manipulated')
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def compare_waterfall(base_hrtf, manip_hrtf=None, ear='left', xlim=(3000.0, 16000.0),
                      linesep=40.0, smoothing='raw', axis=None, show=True,
                      labels=('original', 'modified')):
    """Overlay original (grey) vs manipulated (red) median-plane DTFs stacked by
    elevation -- one pair of curves per elevation, the same view as the QC
    montage (dev/waterfall_edge_qc.py). Companion to compare_tf (side-by-side
    images). Use this to eyeball any manipulated condition against baseline.

    manip_hrtf=None plots the baseline alone (single-HRTF waterfall, e.g. for a
    recorded HRTF).
    smoothing : 'raw' (default -- the actual presented DTF, fine structure kept),
        'gaussian' (Iida de-ripple, for readability), or an int n_keep (cepstral).
    Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ear_i = 0 if ear == 'left' else 1
    arr0, fs = hrtf_to_array(base_hrtf)
    arr1 = None if manip_hrtf is None else hrtf_to_array(manip_hrtf)[0]
    vp = base_hrtf.sources.vertical_polar
    az = (vp[:, 0] + 180) % 360 - 180
    idx = sorted([i for i in range(len(vp)) if abs(az[i]) <= 2.0], key=lambda i: vp[i, 1])
    if axis is None:
        fig, axis = plt.subplots(figsize=(7, 8))
    else:
        fig = axis.figure
    nfft = int(2 ** np.ceil(np.log2(arr0.shape[1])) * 4)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= xlim[0]) & (freqs <= xlim[1])

    def curve(x):
        mag = np.abs(np.fft.rfft(x, nfft))
        if smoothing == 'gaussian':
            return _gaussian_smooth_db(mag, freqs)
        if isinstance(smoothing, (int, np.integer)):
            return _cepstral_smooth_db(mag, int(smoothing))
        return 20.0 * np.log10(np.maximum(mag, EPS))

    off = 0.0
    for i in idx:
        L0 = curve(arr0[i, :, ear_i])
        axis.plot(freqs[band], L0[band] + off, color='0.6', lw=1.0)
        if arr1 is not None:
            L1 = curve(arr1[i, :, ear_i])
            axis.plot(freqs[band], L1[band] + off, color='#D62728', lw=1.1)
        axis.text(xlim[1] * 1.01, L0[band][-1] + off, f"{vp[i, 1]:+.0f}", fontsize=7, va='center')
        off += linesep
    axis.set_xscale('log'); axis.set_xlim(*xlim)
    axis.set_xticks([3000, 5000, 8000, 12000, 16000])
    axis.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel('frequency (Hz)'); axis.set_yticks([])
    axis.set_ylabel('magnitude (dB, offset by elevation)')
    handles = [Line2D([], [], color='0.6', lw=1, label=labels[0])]
    if arr1 is not None:
        handles.append(Line2D([], [], color='#D62728', lw=1, label=labels[1]))
    axis.legend(handles=handles, fontsize=8, loc='upper left')
    if show:
        plt.show()
    return fig


def _agg_figure(ncols, figsize):
    """A standalone Agg-backed Figure (+ axes row) for file output that does
    NOT touch the global matplotlib backend -- so saving a QC figure never
    disturbs an interactive session (e.g. the run_ar HRIR preview on the rig)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    return fig, axes


def save_waterfall_qc(base_hrtf, manip_hrtf, sofa_path, plot_dir=None,
                      ears=('left', 'right'), smoothing='raw',
                      xlim=(3000.0, 16000.0), linesep=40.0):
    """Save a baseline-vs-manipulated WATERFALL figure (one column per ear) as
    QC for a written condition SOFA. This is the plot called automatically by
    save_condition_sofa(plot=True) whenever a shifted copy is generated.

    Output: <plot_dir>/<sofa_stem>_waterfall.png, where plot_dir defaults to
    subject_plot_dir(<subject>)/hrtf/ (subject = base_hrtf.name or the SOFA's parent
    folder name). Renders on an isolated Agg canvas (file only, no window, and
    the interactive backend is left untouched). Returns the saved Path."""
    from pathlib import Path
    from hrtf_relearning.utils import paths

    sofa_path = Path(sofa_path)
    label = sofa_path.stem
    subject_id = getattr(base_hrtf, 'name', None) or sofa_path.parent.name
    if plot_dir is None:
        plot_dir = paths.subject_plot_dir(subject_id) / 'hrtf'
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = _agg_figure(len(ears), (7 * len(ears), 8))
    for ax, ear in zip(axes, ears):
        compare_waterfall(base_hrtf, manip_hrtf, ear=ear, xlim=xlim, linesep=linesep,
                          smoothing=smoothing, axis=ax, show=False,
                          labels=('baseline', label))
        ax.set_title(f'{ear} ear')
    fig.suptitle(label)
    fig.tight_layout()
    out = plot_dir / f'{label}_waterfall.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return out


def save_recorded_hrtf_waterfall(hrtf, subject_id=None, plot_dir=None,
                                 ears=('left', 'right'), smoothing='raw',
                                 xlim=(3000.0, 16000.0), linesep=40.0):
    """Save a single-HRTF median-plane WATERFALL (one column per ear) for a
    recorded HRTF -- the QC image for an existing individual HRIR.

    Output: <plot_dir>/<name>_recorded_waterfall.png, plot_dir defaults to
    subject_plot_dir(<name>)/hrtf/ (name = subject_id or hrtf.name). Renders on
    an isolated Agg canvas. Returns the saved Path."""
    from pathlib import Path
    from hrtf_relearning.utils import paths

    name = subject_id or getattr(hrtf, 'name', None) or 'hrtf'
    if plot_dir is None:
        plot_dir = paths.subject_plot_dir(name) / 'hrtf'
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = _agg_figure(len(ears), (7 * len(ears), 8))
    for ax, ear in zip(axes, ears):
        compare_waterfall(hrtf, None, ear=ear, xlim=xlim, linesep=linesep,
                          smoothing=smoothing, axis=ax, show=False,
                          labels=(f'{name} (recorded)', None))
        ax.set_title(f'{ear} ear')
    fig.suptitle(f'{name} — recorded HRTF')
    fig.tight_layout()
    out = plot_dir / f'{name}_recorded_waterfall.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    return out


if __name__ == "__main__":
    fs = 48000
    n = 256
    f = np.fft.rfftfreq(2048, 1 / fs)
    # synthetic: two in-band notches ~6.5 and ~10 kHz (0.62 oct apart -> two
    # resolved features), peaks between/around. detect_notches only ever reads
    # the magnitude spectrum, so a zero-phase reconstruction is fine here -- no
    # onset/LPC machinery needed for this detector.
    L = (-15 * np.exp(-((f - 6500) ** 2) / (2 * 800 ** 2))
         - 9 * np.exp(-((f - 10000) ** 2) / (2 * 1000 ** 2))
         + 5 * np.exp(-((f - 8200) ** 2) / (2 * 900 ** 2))
         + 4 * np.exp(-((f - 5000) ** 2) / (2 * 700 ** 2)))
    ir = np.fft.irfft(10 ** (L / 20), 2048)[:n]

    summary = parametric_summary(ir, fs)
    print("parametric summary (Iida-style):")
    print(" all detected minima:")
    for nd in summary['notches']:
        print(f"   {nd['label']:>3s}: {nd['f_hz']:.0f} Hz  depth={nd['depth_db']}dB  "
              f"width={nd['width_oct']}oct  Q={nd['q']}")
    print(f" valid features: {[(d['label'], d['f_hz']) for d in summary['features']]}")
    print(f"   P1 (reference peak): {summary['p1_hz']} Hz")
    print(f"   primary notch: {summary['primary']['f_hz'] if summary['primary'] else None} Hz")
    print(f"   gates: {summary['gates']}")
    if summary['primary']:
        print(f"   0.5-ERB shift perceptible @primary (Moore JND)? "
              f"{shift_is_perceptible(summary['primary']['f_hz'], 0.5)}")

    for mode in ('whole', 'edge', 'rising', 'falling'):
        _, rep = edge_shift_ir(ir, fs, 0.5, mode=mode, return_report=True, strict=False)
        print(f"\n{mode:8s}  n_notches={rep['n_notches']}  rmsΔ={rep['rms_delta_db']:+.2f}dB")
        for nn in rep['notches']:
            print(f"   {nn['f_hz']:.0f}Hz: target={nn['target_erb']:.2f} "
                  f"realized={nn['realized_erb']:.2f} "
                  f"depth {nn['depth_base_db']:.1f}->{nn['depth_new_db']:.1f}dB")
