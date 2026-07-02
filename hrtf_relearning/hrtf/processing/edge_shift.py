"""
edge_shift.py

All-notch-consistent spectral manipulations of an individual HRTF for the
elevation edge-coding experiment. Three conditions share ONE notch-detection
pass and ONE shift magnitude (Δ, in ERB):

    'rising'  : every rising  edge moved UP   by Δ (notch minima onsets pinned)
    'falling' : every falling edge moved DOWN by Δ (control)
    'whole'   : every whole notch moved UP    by Δ (all-models-agree reference)

Design invariants (per notch): bracketing PEAKS are pinned (so notches never
interfere and manipulations stay local); notch DEPTH is preserved; in-band RMS
is matched (removes the in-notch-power confound); original PHASE is kept (ITD
unchanged). Manipulation is on a fine ERB grid; verify realized shifts in the
~28-bin filterbank domain (verify_binned) because that is what the model reads.

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


# --- spectrum <-> ERB grid ----
def _to_erb_grid(L_db, freqs, f_lo, f_hi, step=0.05):
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fb = freqs[band]
    e_lo, e_hi = float(hz_to_erb(f_lo)), float(hz_to_erb(f_hi))
    M = max(256, int((e_hi - e_lo) / step))
    e = np.linspace(e_lo, e_hi, M)
    Le = np.interp(erb_to_hz(e), fb, L_db[band])
    return e, Le, band, fb


# --- multi-notch detection ----
def detect_notches(e, Le, search_hz=(4000.0, 16000.0), min_prom_db=2.0,
                   max_notches=3):
    """All notches (deepest first, then freq-sorted) with bracketing peaks.
    Adjacent notches share a bracketing peak, so their [i_lo, i_hi] intervals
    have disjoint interiors -> warps compose without interfering."""
    f = erb_to_hz(e)
    notches, _ = find_peaks(-Le, prominence=min_prom_db)
    peaks, _ = find_peaks(Le, prominence=min_prom_db)
    win = (f[notches] >= search_hz[0]) & (f[notches] <= search_hz[1])
    notches = notches[win]
    if len(notches) == 0:
        return []
    if len(notches) > max_notches:
        notches = notches[np.argsort(Le[notches])[:max_notches]]
    notches = np.sort(notches)
    out = []
    for im in notches:
        above = peaks[peaks > im]
        below = peaks[peaks < im]
        out.append(dict(i_min=int(im),
                        i_lo=int(below[-1]) if len(below) else 0,
                        i_hi=int(above[0]) if len(above) else len(e) - 1))
    return out


# --- per-notch warp (peaks pinned) ----
def _notch_knots(e, notch, s_target, mode, margin=0.05):
    """Return (knots_out, knots_src, realized_shift) for one notch."""
    e_lo, e_min, e_hi = e[notch['i_lo']], e[notch['i_min']], e[notch['i_hi']]
    if mode in ('rising', 'whole'):
        s = min(s_target, (e_hi - e_min) - margin)
    else:  # falling
        s = min(s_target, (e_min - e_lo) - margin)
    s = max(s, 0.0)
    if mode == 'rising':
        ko = [e_lo, e_min, e_min + s, e_hi]
        ks = [e_lo, e_min, e_min,     e_hi]
    elif mode == 'falling':
        ko = [e_lo, e_min - s, e_min, e_hi]
        ks = [e_lo, e_min,     e_min, e_hi]
    elif mode == 'whole':
        ko = [e_lo, e_min + s, e_hi]
        ks = [e_lo, e_min,     e_hi]
    else:
        raise ValueError("mode must be 'rising', 'falling' or 'whole'")
    return np.array(ko), np.array(ks), s


def build_warped_logmag(e, Le, notches, s_target, mode):
    """Apply every notch's warp into its own [i_lo, i_hi] span (disjoint), then
    resample Le through the composed monotone warp. Returns (Le_new, realized)."""
    src = e.copy()
    realized = []
    for n in notches:
        ko, ks, s = _notch_knots(e, n, s_target, mode)
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
    g = np.gradient(Le, e)
    if flank == 'rising':
        a, b = n['i_min'], max(n['i_min'] + 1, n['i_hi'])
        seg = np.maximum(g[a:b + 1], 0.0)
    else:
        a, b = min(n['i_lo'], n['i_min'] - 1), n['i_min']
        seg = np.maximum(-g[a:b + 1], 0.0)
    return e[a + int(np.argmax(seg))]


def _notch_centroid_e(e, Le, a, b):
    ref = max(Le[a], Le[b])
    d = np.maximum(ref - Le[a:b + 1], 0.0)
    return float(np.sum(e[a:b + 1] * d) / (d.sum() + EPS))


# --- single IR ----
def edge_shift_ir(ir, fs, shift_erb, mode='rising', f_lo=3000.0, f_hi=16000.0,
                  search_hz=(4000.0, 16000.0), rms_band=(3000.0, 16000.0),
                  min_prom_db=2.0, max_notches=3, nfft=None, match_power=True,
                  return_report=False):
    ir = np.asarray(ir, float)
    n = len(ir)
    nfft = nfft or int(2 ** np.ceil(np.log2(n)))
    H = np.fft.rfft(ir, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    L = 20.0 * np.log10(np.maximum(np.abs(H), EPS))
    phase = np.angle(H)

    e, Le, band, fb = _to_erb_grid(L, freqs, f_lo, f_hi)
    notches = detect_notches(e, Le, search_hz, min_prom_db, max_notches)
    if not notches:
        rep = dict(status='no_notch_found', n_notches=0)
        return (ir.copy(), rep) if return_report else ir.copy()

    Le_new, realized = build_warped_logmag(e, Le, notches, abs(shift_erb), mode)
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
        per_notch.append(dict(
            f_hz=round(float(erb_to_hz(e[nn['i_min']])), 0),
            target_erb=round(abs(shift_erb), 2),
            realized_erb=round(realized[k], 2),
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
def edge_shift_set(hrirs, fs, shift_erb, mode='rising', verbose=False, **kw):
    """hrirs: (n_dir, n_taps, n_ears). Returns (out_array, reports)."""
    hrirs = np.asarray(hrirs, float)
    out = np.empty_like(hrirs)
    reports = []
    for d in range(hrirs.shape[0]):
        for ear in range(hrirs.shape[2]):
            ir_new, rep = edge_shift_ir(hrirs[d, :, ear], fs, shift_erb,
                                        mode=mode, return_report=True, **kw)
            out[d, :, ear] = ir_new
            rep.update(dir=d, ear=ear)
            reports.append(rep)
            if verbose and rep['status'] != 'ok':
                print(f"dir {d} ear {ear}: {rep['status']}")
    return out, reports


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
