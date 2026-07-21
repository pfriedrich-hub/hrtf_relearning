"""
cue_warp.py

Spectral-cue manipulation for the elevation experiment, as ONE primitive:
a frequency warp of the DTF log-magnitude on the auditory (ERB) axis,

    output(f) = cleaned_input( f - d(f) )        d in ERB, applied then inverted

where the displacement field d(f) defines the condition. Because the manipulation
resamples the (lightly cleaned) full-resolution magnitude rather than smoothing it,
notch depth, width and edge slope are preserved by construction. Two conditions:

    'whole'  : translate the whole notch (both flanks + minimum) up by `shift_erb`.
               d = shift_erb across the notch bracket [saddle_lo, saddle_hi],
               ramped to 0 in the plateaus to either side.
    'rising' : translate only the upper (rising) flank up by `shift_erb`, minimum
               pinned. d = 0 up to the minimum, shift_erb from just above it to the
               upper saddle. The notch floor widens; CF is unchanged.

Both share the same rising-edge displacement, so they are indistinguishable if the
elevation percept tracks the rising edge alone, and diverge if it needs the whole
feature (notch CF / spectral pattern). Manipulation is applied to DETECTABLE
notches only (Moore et al. 1989 depth/width; Macpherson & Middlebrooks 2003
resolvability): a pair closer than SEP_MIN_OCT is treated as one feature and
warped via its OUTER bracket, so no unresolvable interior edge is ever moved.

Magnitude-only: original phase (ITD/onset) is kept; in-band power is matched.
Light artifact clean = Iida et al. 2007 microscopic-fluctuation Gaussian (~120 Hz):
removes single-bin measurement spikes without touching the notch.
"""
import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

EPS = 1e-12
NOTCH_BAND = (5000.0, 11000.0)   # Asano 1990 / van Opstal: elevation-notch region
SEARCH = (3000.0, 16000.0)       # bracket search range
MIN_DEPTH_DB = 4.0               # Moore 1989: saddle/notch detectability ~2.5-5 dB @8 kHz
MIN_WIDTH_OCT = 0.10             # real-feature width floor (raise to ~0.17 for strict Moore)
SEP_MIN_OCT = 0.5                # Macpherson 2003: separable to ~2 ripples/oct
CLEAN_HZ = 120.0                 # Iida 2007 light clean
SKIRT_ERB = 0.3                  # raised-cosine ramp width of the displacement field


def erb(f):
    return 21.4 * np.log10(4.37e-3 * np.asarray(f, float) + 1.0)


def erb_inv(e):
    return (10.0 ** (np.asarray(e, float) / 21.4) - 1.0) / 4.37e-3


def _clean(logmag, df):
    return gaussian_filter1d(logmag, CLEAN_HZ / df, mode="nearest")


# ---- detectable-notch detection (simple, robust) --------------------------
def detect_notches(clean, f):
    """Detectable notches in `clean` (cleaned log-mag), each as
    {f_min, f_lo, f_hi, depth, width_oct}. Gated by depth (Moore) and width,
    restricted to NOTCH_BAND centres, and merged across pairs closer than
    SEP_MIN_OCT (Macpherson) into a single outer-bracket feature."""
    band = (f >= SEARCH[0]) & (f <= SEARCH[1])
    fb, L = f[band], clean[band]
    mins, props = find_peaks(-L, prominence=MIN_DEPTH_DB)
    maxs, _ = find_peaks(L, prominence=1.0)
    feats = []
    for k, i in enumerate(mins):
        fc = fb[i]
        if not (NOTCH_BAND[0] <= fc <= NOTCH_BAND[1]):
            continue
        above, below = maxs[maxs > i], maxs[maxs < i]
        i_hi = int(above[0]) if len(above) else len(fb) - 1
        i_lo = int(below[-1]) if len(below) else 0
        depth = float(props["prominences"][k])
        thr = L[i] + 0.5 * depth                       # half-depth width
        a, b = i, i
        while a > i_lo and L[a] < thr:
            a -= 1
        while b < i_hi and L[b] < thr:
            b += 1
        width = float(np.log2(fb[b] / fb[a])) if b > a else 0.0
        if width < MIN_WIDTH_OCT:
            continue
        feats.append(dict(f_min=float(fc), f_lo=float(fb[i_lo]),
                          f_hi=float(fb[i_hi]), depth=depth, width_oct=width))
    feats.sort(key=lambda d: d["f_min"])
    merged = []
    for d in feats:
        if merged and np.log2(d["f_min"] / merged[-1]["f_min"]) < SEP_MIN_OCT:
            m = merged[-1]
            deep = d if d["depth"] > m["depth"] else m
            lo, hi = min(m["f_lo"], d["f_lo"]), max(m["f_hi"], d["f_hi"])
            merged[-1] = dict(f_min=deep["f_min"], f_lo=lo, f_hi=hi,
                              depth=max(m["depth"], d["depth"]),
                              width_oct=float(np.log2(hi / lo)))
        else:
            merged.append(dict(d))
    return merged


# ---- displacement fields --------------------------------------------------
def _ramp(e, e0, e1):
    """0 for e<=e0, 1 for e>=e1, raised-cosine between (e1 may be < e0)."""
    x = np.clip((e - e0) / (e1 - e0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def displacement(e, feats, shift_erb, mode, skirt=SKIRT_ERB):
    """ERB displacement field d(e) for the given detectable features."""
    d = np.zeros_like(e)
    for ft in feats:
        e_lo, e_min, e_hi = erb(ft["f_lo"]), erb(ft["f_min"]), erb(ft["f_hi"])
        if mode == "whole":
            w = _ramp(e, e_lo - skirt, e_lo) * _ramp(e, e_hi + skirt, e_hi)
        elif mode == "rising":
            w = _ramp(e, e_min, e_min + skirt) * _ramp(e, e_hi + skirt, e_hi)
        else:
            raise ValueError(f"mode must be 'whole' or 'rising', got {mode!r}")
        d = np.maximum(d, shift_erb * w)   # features are separated -> no overlap
    return d


# ---- core warp ------------------------------------------------------------
def warp_ir(ir, fs, condition, shift_erb, return_feats=False):
    """Apply the `condition` ('whole' | 'rising') warp of `shift_erb` ERB to one
    HRIR. Detectable notches only; unchanged (no detectable notch) is returned
    verbatim. Returns the manipulated HRIR (and the feature list if requested)."""
    ir = np.asarray(ir, float)
    n = len(ir)
    H = np.fft.rfft(ir)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    df = f[1] - f[0]
    logmag = 20.0 * np.log10(np.abs(H) + EPS)
    clean = _clean(logmag, df)
    feats = detect_notches(clean, f)
    if not feats:
        return (ir.copy(), feats) if return_feats else ir.copy()

    d = displacement(erb(f), feats, float(shift_erb), condition)
    new = np.interp(erb_inv(erb(f) - d), f, clean, left=clean[0], right=clean[-1])

    aff = d > 1e-6                                    # match power over affected band
    if aff.any():
        g = np.sqrt(np.sum(10.0 ** (clean[aff] / 10.0))
                    / np.sum(10.0 ** (new[aff] / 10.0)))
        new = new + 20.0 * np.log10(g)
    out = np.fft.irfft(10.0 ** (new / 20.0) * np.exp(1j * np.angle(H)), n)
    return (out, feats) if return_feats else out


def warp_hrtf(hrtf, condition, shift_erb):
    """Apply warp_ir to every direction/ear of a slab.HRTF (deep copy)."""
    out = copy.deepcopy(hrtf)
    for filt in out:
        data = np.asarray(filt.data, float)
        for ch in range(data.shape[1]):
            data[:, ch] = warp_ir(data[:, ch], filt.samplerate, condition, shift_erb)
        filt.data = data
    return out
