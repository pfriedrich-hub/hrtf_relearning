"""
edge_shift.py

All-notch-consistent spectral manipulations of an individual HRTF for the
elevation edge-coding experiment. Three conditions share ONE notch-detection
pass and ONE shift magnitude (Δ, in ERB):

    'rising'  : the rising edge moved UP   by Δ, minimum AND peak pinned
    'falling' : the falling edge moved DOWN by Δ, minimum AND peak pinned (control)
    'whole'   : the notch minimum moved UP by Δ, both peaks pinned (all-models-agree reference)

The edge (for 'rising'/'falling') is defined as the point of steepest
positive/negative log-magnitude gradient between the notch minimum and the
neighboring peak (_edge_index) -- matching the DCN type-IV / Baumgartner
positive-gradient definition of a spectral edge. It is the one thing that
moves; the minimum and the bracketing peak both stay exactly where they were.

Design invariants (per notch): bracketing PEAKS are pinned (so notches never
interfere and manipulations stay local); notch DEPTH is preserved; in-band RMS
is matched (removes the in-notch-power confound); original PHASE is kept (ITD
unchanged). Manipulation itself is on the fine ERB grid, but WHICH notches get
manipulated -- and WHERE their bracketing peaks are pinned -- is gated
directly against the Baumgartner gammatone-band (~1 ERB spacing) RMS
representation of the same spectrum (baumgartner_filterbank) -- the actual
auditory-filter-resolution front end, not a distance proxy. Two fine-grid
dips the auditory system can't tell apart (inside the same coarse notch)
collapse into ONE manipulation target: the deepest fine minimum in that
coarse region. Likewise, the peaks a shift is pinned against (and thus the
room available for it) are the COARSE peaks, not the nearest fine ripple --
a fine ripple the auditory system can't resolve as a separate peak isn't a
real perceptual boundary, and pinning to it would silently starve the warp
of room it should actually have. verify_binned then re-checks realized
shifts in that same ~28-bin domain, since that is what the model reads.

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


# --- edge point: steepest spectral gradient between a notch minimum and its
#     bracketing peak. This IS "the edge" -- matches the DCN type-IV /
#     Baumgartner positive-gradient definition of a spectral edge -- rather
#     than an implicit side effect of warping the whole minimum-to-peak span.
def _edge_index(e, Le, i_min, i_bound, flank):
    """Index of steepest positive ('rising') or negative ('falling')
    log-magnitude gradient between the notch minimum (i_min) and the
    bracketing peak (i_bound)."""
    g = np.gradient(Le, e)
    if flank == 'rising':
        a, b = i_min, max(i_min + 1, i_bound)
        seg = np.maximum(g[a:b + 1], 0.0)
    else:
        a, b = min(i_bound, i_min - 1), i_min
        seg = np.maximum(-g[a:b + 1], 0.0)
    return a + int(np.argmax(seg))


# --- multi-notch detection ----
def detect_notches(e, Le, fs, freqs, mag, search_hz=(4000.0, 16000.0),
                   min_prom_db=2.0, coarse_min_prom_db=0.5, max_notches=3,
                   coarse_filterbank=None):
    """Notches gated by auditory resolvability, with bracketing peaks and
    identified edge points on the fine grid.

    Resolvability is no longer a distance proxy -- it's checked directly
    against the Baumgartner gammatone-band (~1 ERB spacing) RMS
    representation of this SAME spectrum (baumgartner_filterbank), i.e. the
    actual auditory-filter-resolution front end the model (and, as a
    physiological model of the periphery, the ear) uses. A coarse notch is
    found first; the deepest FINE-grid minimum inside that coarse notch's own
    bracketing region becomes the notch to manipulate. Multiple close
    fine-grid candidates that fall inside the same coarse notch (e.g. two
    sub-ERB-spaced dips the auditory system can't tell apart) collapse into
    that ONE notch, rather than being treated as separate (and separately
    failing) manipulation targets. max_notches caps the deepest COARSE
    notches (the auditory-relevant N1/N2/... count), not fine-grid ripples.

    Each notch's rising/falling edge is likewise found on the coarse
    representation: the point of steepest positive/negative gradient in
    (ec, Lb) between the coarse notch and its coarse bracketing peak
    (_edge_index), mapped back onto the fine grid. This is the point
    'rising'/'falling' modes actually shift by Delta -- using a fine-grid
    gradient here instead would lock onto whichever fine ripple's flank is
    locally steepest (e.g. the first of two collapsed sub-notches), not the
    rising trend the auditory system actually resolves across the notch.

    Adjacent notches share a bracketing peak, so their [i_lo, i_hi] intervals
    have disjoint interiors -> warps compose without interfering.

    fs, freqs, mag: the caller's sample rate and rfft frequency/magnitude
    arrays (freqs, mag = np.fft.rfftfreq(...), np.abs(np.fft.rfft(...))),
    needed to build the coarse gammatone-band representation."""
    f = erb_to_hz(e)
    fine_notches, _ = find_peaks(-Le, prominence=min_prom_db)

    fb = coarse_filterbank or baumgartner_filterbank(fs)
    fc, ec, Lb = fb(freqs, mag)
    coarse_notches, _ = find_peaks(-Lb, prominence=coarse_min_prom_db)
    coarse_peaks, _ = find_peaks(Lb, prominence=coarse_min_prom_db)
    cwin = (fc[coarse_notches] >= search_hz[0]) & (fc[coarse_notches] <= search_hz[1])
    coarse_notches = coarse_notches[cwin]
    if len(coarse_notches) == 0:
        return []
    if len(coarse_notches) > max_notches:
        coarse_notches = coarse_notches[np.argsort(Lb[coarse_notches])[:max_notches]]
    coarse_notches = np.sort(coarse_notches)

    out = []
    for cm in coarse_notches:
        c_above = coarse_peaks[coarse_peaks > cm]
        c_below = coarse_peaks[coarse_peaks < cm]
        ec_lo = ec[c_below[-1]] if len(c_below) else ec[0]
        ec_hi = ec[c_above[0]] if len(c_above) else ec[-1]
        cand = fine_notches[(e[fine_notches] >= ec_lo) & (e[fine_notches] <= ec_hi)]
        if len(cand) == 0:
            continue  # coarse dip has no clean fine-grid minimum underneath (rare)
        im = int(cand[np.argmin(Le[cand])])  # deepest fine minimum in this coarse region
        # Bracketing peaks are pinned at the COARSE (auditory-resolved) peak
        # position (ec_lo/ec_hi, already found above), not the nearest fine
        # ripple. A fine-grid peak the auditory system can't resolve as
        # distinct from its neighboring coarse peak isn't a real perceptual
        # boundary -- pinning to it starves the warp of room it should
        # actually have (this is the same principle that collapses close
        # fine-grid dips into one coarse manipulation target, above, applied
        # to the peaks instead of the minima).
        i_lo = int(np.clip(np.searchsorted(e, ec_lo), 0, len(e) - 1))
        i_hi = int(np.clip(np.searchsorted(e, ec_hi), 0, len(e) - 1))
        # The edge point is likewise located on the COARSE trend from the
        # coarse notch to its coarse bracketing peak, not via steepest FINE
        # gradient. When two sub-ERB dips collapse into one coarse notch
        # (above), the fine spectrum still has real ripple structure between
        # them -- a fine-grid gradient search over the whole [i_min, i_hi]
        # span locks onto whichever ripple's flank is locally steepest
        # (typically the first one), not the rising trend the auditory
        # system actually resolves across the full merged notch. Finding the
        # edge on (ec, Lb) first and mapping that ERB position back onto the
        # fine grid keeps edge placement consistent with the same coarse
        # representation already used to gate detection and pin peaks.
        i_c_hi = int(c_above[0]) if len(c_above) else len(ec) - 1
        i_c_lo = int(c_below[-1]) if len(c_below) else 0
        e_edge_rise = ec[_edge_index(ec, Lb, cm, i_c_hi, 'rising')]
        e_edge_fall = ec[_edge_index(ec, Lb, cm, i_c_lo, 'falling')]
        i_edge_rise = int(np.clip(np.searchsorted(e, e_edge_rise), 0, len(e) - 1))
        i_edge_fall = int(np.clip(np.searchsorted(e, e_edge_fall), 0, len(e) - 1))
        out.append(dict(i_min=im, i_lo=i_lo, i_hi=i_hi,
                        i_edge_rise=i_edge_rise, i_edge_fall=i_edge_fall,
                        coarse_f_hz=round(float(erb_to_hz(ec[cm])), 0)))
    return out


# --- per-notch warp (peaks pinned) ----
def _notch_knots(e, notch, s_target, mode, margin=0.05, strict=True, edge_window_erb=1.0):
    """Return (knots_out, knots_src, realized_shift) for one notch.

    'rising'/'falling' translate a WINDOW of the log-magnitude curve --
    +/-edge_window_erb/2 around the notch's identified edge point (steepest
    gradient, see _edge_index / detect_notches) -- by s ERB, with the notch
    minimum and the bracketing peak pinned exactly at their original
    positions. Both boundaries of that window shift by the identical amount
    s, so the piecewise-linear knot map degenerates to an exact rigid
    translation across the window (np.interp between two knots offset by a
    constant is a pure shift, not a stretch) -- the edge's actual shape is
    preserved, not smoothly re-warped between two fixed points. Whatever
    stretch/compression is needed to still land exactly on the pinned
    minimum/peak is pushed entirely into the flanking sub-segments outside
    the window, i.e. into the locally flat floor/roof regions rather than
    into the steep part of the edge itself.

    edge_window_erb defaults to 1 ERB, matching the auditory-filter
    resolution already used elsewhere (detect_notches' coarse gating): finer
    structure than that isn't independently resolved anyway, so treating the
    edge as a rigid ~1-ERB-wide template is the natural cutoff, not an
    arbitrary one. The window is clipped so it never crosses the minimum or
    the peak (relevant for very narrow notches); if the available span is
    narrower than the window itself, it collapses to the single edge point
    (identical to the old point-anchor behaviour).

    'whole' still moves the minimum itself (the notch's characteristic
    frequency, a point feature, not a slope) with both peaks pinned, since
    that mode isn't an edge manipulation at all and has no "shape" to
    preserve.

    The shift is always clamped to stay short of the bracketing peak (by
    `margin` ERB), so peaks stay pinned and adjacent notches never interfere
    -- that invariant isn't negotiable here. What `strict` controls is
    whether hitting that limit is silent:
      strict=True  (default) -- raise if the notch can't reach s_target, so a
        request that quietly gets watered down is never invisible.
      strict=False -- clamp and return the reduced shift, same as before.
    """
    e_lo, e_min, e_hi = e[notch['i_lo']], e[notch['i_min']], e[notch['i_hi']]
    hw = edge_window_erb / 2.0
    if mode == 'rising':
        e_edge = e[notch['i_edge_rise']]
        e_w_lo = max(e_edge - hw, e_min)
        e_w_hi = min(e_edge + hw, e_hi)
        if e_w_hi <= e_w_lo:
            e_w_lo = e_w_hi = e_edge  # window doesn't fit -- fall back to a point anchor
        room = (e_hi - e_w_hi) - margin
    elif mode == 'falling':
        e_edge = e[notch['i_edge_fall']]
        e_w_lo = max(e_edge - hw, e_lo)
        e_w_hi = min(e_edge + hw, e_min)
        if e_w_hi <= e_w_lo:
            e_w_lo = e_w_hi = e_edge
        room = (e_w_lo - e_lo) - margin
    elif mode == 'whole':
        room = (e_hi - e_min) - margin
    else:
        raise ValueError("mode must be 'rising', 'falling' or 'whole'")
    s = max(min(s_target, room), 0.0)
    if strict and s < s_target - 1e-9:
        f_hz = float(erb_to_hz(e_min))
        anchor = 'edge window' if mode in ('rising', 'falling') else 'minimum'
        raise ValueError(
            f"shift_erb={s_target:.2f} exceeds available room ({max(room, 0.0):.2f} ERB) "
            f"between the {anchor} and the bracketing peak at the notch near {f_hz:.0f} Hz "
            f"(mode='{mode}'). Pass strict=False to allow clamping to the available room instead."
        )
    if mode == 'rising':
        ko = [e_lo, e_min, e_w_lo + s, e_w_hi + s, e_hi]
        ks = [e_lo, e_min, e_w_lo,     e_w_hi,     e_hi]
    elif mode == 'falling':
        ko = [e_lo, e_w_lo - s, e_w_hi - s, e_min, e_hi]
        ks = [e_lo, e_w_lo,     e_w_hi,     e_min, e_hi]
    elif mode == 'whole':
        ko = [e_lo, e_min + s, e_hi]
        ks = [e_lo, e_min,     e_hi]
    return np.array(ko), np.array(ks), s


def build_warped_logmag(e, Le, notches, s_target, mode, strict=True, edge_window_erb=1.0):
    """Apply every notch's warp into its own [i_lo, i_hi] span (disjoint), then
    resample Le through the composed monotone warp. Returns (Le_new, realized).
    strict: see _notch_knots -- raise (default) vs silently clamp when a
    notch doesn't have room for the full requested shift.
    edge_window_erb: see _notch_knots -- width of the rigidly-translated
    edge-shape template for 'rising'/'falling' (ignored for 'whole')."""
    src = e.copy()
    realized = []
    for n in notches:
        ko, ks, s = _notch_knots(e, n, s_target, mode, strict=strict,
                                 edge_window_erb=edge_window_erb)
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
def _depth_db(e, Le, n):
    a, b = n['i_lo'], n['i_hi']
    return max(Le[a], Le[b]) - float(np.min(Le[a:b + 1]))


def _flank_edge_e(e, Le, n, flank):
    """E-value of the edge point on `Le` (see _edge_index) -- used both for
    the pre-warp edge (should match n['i_edge_rise']/'i_edge_fall' from
    detect_notches) and, on the post-warp spectrum, to verify the edge
    actually landed where intended."""
    i_bound = n['i_hi'] if flank == 'rising' else n['i_lo']
    return e[_edge_index(e, Le, n['i_min'], i_bound, flank)]


def _notch_centroid_e(e, Le, a, b):
    ref = max(Le[a], Le[b])
    d = np.maximum(ref - Le[a:b + 1], 0.0)
    return float(np.sum(e[a:b + 1] * d) / (d.sum() + EPS))


# --- single IR ----
def edge_shift_ir(ir, fs, shift_erb, mode='rising', f_lo=3000.0, f_hi=16000.0,
                  search_hz=(4000.0, 16000.0), rms_band=(3000.0, 16000.0),
                  min_prom_db=2.0, coarse_min_prom_db=0.5, max_notches=3, nfft=None,
                  match_power=True, return_report=False, strict=True, edge_window_erb=1.0):
    """edge_window_erb: width of the rigidly-translated edge-shape template
    for 'rising'/'falling' (see _notch_knots) -- defaults to 1 ERB, the same
    auditory-resolution cutoff used by detect_notches' coarse gating."""
    ir = np.asarray(ir, float)
    n = len(ir)
    nfft = nfft or int(2 ** np.ceil(np.log2(n)))
    H = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    L = 20.0 * np.log10(np.maximum(np.abs(H), EPS))
    phase = np.angle(H)

    e, Le, band, fb = _to_erb_grid(L, freqs, f_lo, f_hi)
    notches = detect_notches(e, Le, fs, freqs, np.abs(H), search_hz, min_prom_db,
                             coarse_min_prom_db, max_notches)
    if not notches:
        rep = dict(status='no_notch_found', n_notches=0)
        return (ir.copy(), rep) if return_report else ir.copy()

    Le_new, realized = build_warped_logmag(e, Le, notches, abs(shift_erb), mode, strict=strict,
                                           edge_window_erb=edge_window_erb)
    if match_power:
        Le_new = match_band_rms(e, Le, Le_new, rms_band)

    L_full = L.copy()
    L_full[band] = np.interp(fb, erb_to_hz(e), Le_new)
    ir_new = np.fft.irfft(10.0 ** (L_full / 20.0) * np.exp(1j * phase), nfft)[:n]

    if not return_report:
        return ir_new
    flank = 'falling' if mode == 'falling' else 'rising'
    per_notch = []
    for k, nn in enumerate(notches):
        sign = 1 if flank == 'rising' else -1
        pad = int(np.ceil(abs(shift_erb) / (e[1] - e[0]))) + 2
        a = max(0, nn['i_lo'] - pad)
        b = min(len(e) - 1, nn['i_hi'] + pad)
        f_hz = float(erb_to_hz(e[nn['i_min']]))
        per_notch.append(dict(
            f_hz=round(f_hz, 0),
            coarse_f_hz=nn.get('coarse_f_hz'),
            target_erb=round(abs(shift_erb), 2),
            target_hz=round(erb_shift_to_hz(f_hz, abs(shift_erb)), 0),
            realized_erb=round(realized[k], 2),
            realized_hz=round(erb_shift_to_hz(f_hz, realized[k]), 0),
            edge_shift_erb=round(sign * (_flank_edge_e(e, Le_new, nn, flank)
                                         - _flank_edge_e(e, Le, nn, flank)), 2),
            centroid_shift_erb=round(sign * (_notch_centroid_e(e, Le_new, a, b)
                                             - _notch_centroid_e(e, Le, a, b)), 2),
            depth_base_db=round(_depth_db(e, Le, nn), 1),
            depth_new_db=round(_depth_db(e, Le_new, nn), 1)))
    f = erb_to_hz(e)
    mrb = (f >= rms_band[0]) & (f <= rms_band[1])
    rms_d = 10 * np.log10(np.mean((10 ** (Le_new[mrb] / 20)) ** 2)
                          / np.mean((10 ** (Le[mrb] / 20)) ** 2))
    return ir_new, dict(status='ok', mode=mode, n_notches=len(notches),
                        rms_delta_db=round(rms_d, 2), notches=per_notch)


# --- full set (numpy core) ----
def edge_shift_set(hrirs, fs, shift_erb, mode='rising', verbose=False, strict=False, **kw):
    """hrirs: (n_dir, n_taps, n_ears). Returns (out_array, reports).

    strict=False (default, unlike edge_shift_ir's own default): one
    direction/ear with a too-close bracketing peak shouldn't abort a whole
    condition build, so this always clamps rather than raising -- but it
    never does so silently: any clamped notch is printed below, itemized by
    (dir, ear, frequency, requested vs realized ERB/Hz). Pass strict=True to
    raise on the first one instead (e.g. for interactively checking one
    direction with edge_shift_ir directly)."""
    hrirs = np.asarray(hrirs, float)
    out = np.empty_like(hrirs)
    reports = []
    for d in range(hrirs.shape[0]):
        for ear in range(hrirs.shape[2]):
            ir_new, rep = edge_shift_ir(hrirs[d, :, ear], fs, shift_erb,
                                        mode=mode, return_report=True, strict=strict, **kw)
            out[d, :, ear] = ir_new
            rep.update(dir=d, ear=ear)
            reports.append(rep)
            if verbose and rep['status'] != 'ok':
                print(f"dir {d} ear {ear}: {rep['status']}")
    if verbose:
        print_notch_summary(reports, label=f"{mode} notch counts")
    print_clamped_notches(reports)
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


def manipulate_hrtf(base_hrtf, condition, shift_erb, **kw):
    """condition in {'baseline','rising','falling','whole'}. Returns
    (manipulated slab.HRTF, reports). 'baseline' clones unchanged."""
    if condition == 'baseline':
        arr, _ = hrtf_to_array(base_hrtf)
        return array_to_hrtf(arr, base_hrtf), [dict(status='baseline')]
    arr, fs = hrtf_to_array(base_hrtf)
    out, reports = edge_shift_set(arr, fs, shift_erb, mode=condition, **kw)
    return array_to_hrtf(out, base_hrtf), reports


def save_condition_sofa(base_hrtf, condition, shift_erb, path, **kw):
    """Write one manipulated condition to a SOFA file (metadata cloned from base,
    only the IR data replaced). Returns (manipulated slab.HRTF, reports)."""
    hrtf_new, reports = manipulate_hrtf(base_hrtf, condition, shift_erb, **kw)
    hrtf_new.write_sofa(str(path))
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


if __name__ == "__main__":
    fs = 48000
    n = 256
    f = np.fft.rfftfreq(2048, 1 / fs)
    # synthetic: N1 ~8 kHz, N2 ~13 kHz, peaks between/around
    L = (-15 * np.exp(-((f - 8000) ** 2) / (2 * 800 ** 2))
         - 9 * np.exp(-((f - 13000) ** 2) / (2 * 1000 ** 2))
         + 5 * np.exp(-((f - 10500) ** 2) / (2 * 900 ** 2))
         + 4 * np.exp(-((f - 5500) ** 2) / (2 * 900 ** 2)))
    ir = np.fft.irfft(10 ** (L / 20), 2048)[:n]

    for mode in ('rising', 'falling', 'whole'):
        _, rep = edge_shift_ir(ir, fs, 1.5, mode=mode, return_report=True)
        print(f"\n{mode:8s}  n_notches={rep['n_notches']}  rmsΔ={rep['rms_delta_db']:+.2f}dB")
        for nn in rep['notches']:
            print(f"   {nn['f_hz']:.0f}Hz: edge={nn['edge_shift_erb']:+.2f} "
                  f"cen={nn['centroid_shift_erb']:+.2f} realized={nn['realized_erb']:.2f} "
                  f"depth {nn['depth_base_db']:.1f}->{nn['depth_new_db']:.1f}dB")
