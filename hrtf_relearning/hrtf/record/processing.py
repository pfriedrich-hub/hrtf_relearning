# processing.py
"""
Signal-processing layer for the HRTF relearning project.

Responsibilities:
- sweep deconvolution -> IRs
- alignment & averaging
- equalization
- low-frequency extrapolation
- azimuth expansion & binaural cue imposition

No I/O. No hardware. No FreeField.
"""

from __future__ import annotations

import copy
import numpy
from datetime import datetime
import slab
import pyfar
import warnings
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)
from .recordings import Recordings, SpeakerGridBase


# =====================================================================
# Utilities
# =====================================================================

def _ensure_time_and_freq(sig: pyfar.Signal) -> pyfar.Signal:
    """Force pyfar.Signal to have both time and freq cached."""
    _ = sig.time
    _ = sig.freq
    return sig


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


# =====================================================================
# IR alignment + averaging (SINGLE canonical implementation)
# =====================================================================

def align_and_average_irs(
    irs: list[pyfar.Signal],
    onset_threshold_db: float = 15.0,
    reference: str = "median",
) -> pyfar.Signal:
    """
    Align binaural IRs by onset and average in time domain.

    Parameters
    ----------
    irs
        List of pyfar.Signal, shape (2, n_samples)
    reference
        "median" (recommended) or "min"

    Returns
    -------
    pyfar.Signal
        Averaged IR, full length, no windowing
    """
    if not irs:
        raise ValueError("align_and_average_irs: empty input")

    fs = irs[0].sampling_rate
    n_samples = irs[0].n_samples

    for i, ir in enumerate(irs):
        if ir.sampling_rate != fs or ir.n_samples != n_samples:
            raise ValueError("IR list must have equal fs and length")
        if ir.cshape[0] != 2:
            raise ValueError("Expected binaural IRs (2 ears)")

    onsets = numpy.array([
        pyfar.dsp.find_impulse_response_start(
            _ensure_time_and_freq(ir), threshold=onset_threshold_db
        ).min()
        for ir in irs
    ])

    if reference == "median":
        ref_onset = numpy.median(onsets)
    elif reference == "min":
        ref_onset = numpy.min(onsets)
    else:
        raise ValueError("reference must be 'median' or 'min'")

    aligned = []
    for ir, onset in zip(irs, onsets):
        shift = ref_onset - onset
        aligned.append(
            pyfar.dsp.time_shift(ir, shift, unit="samples").time
        )

    avg_time = numpy.mean(numpy.stack(aligned, axis=0), axis=0)
    return pyfar.Signal(avg_time, fs)


# =====================================================================
# Deconvolution: Recordings -> ImpulseResponses
# =====================================================================

def compute_ir(
    recordings: Recordings,
    onset_threshold_db: float = 15.0,
    inversion_range_hz: tuple[float, float] | None = None,
) -> "ImpulseResponses":
    """
    Deconvolve sweep recordings into impulse responses.

    Pipeline:
    - regularized spectrum inversion of excitation
    - per-take deconvolution
    - onset alignment
    - time-domain averaging

    No windowing, no cropping.
    """
    if not recordings.data:
        raise ValueError("compute_ir: empty recordings")

    if recordings.signal is None:
        raise ValueError("compute_ir: missing excitation signal")

    fs = int(recordings.params["fs"])
    sig_params = recordings.params["signal"]

    if inversion_range_hz is None:
        inversion_range_hz = (
            sig_params["from_frequency"],
            sig_params["to_frequency"],
        )

    # --- build binaural excitation ---
    sig = recordings.signal
    exc = pyfar.Signal(
        numpy.stack([sig.data, sig.data], axis=0),
        fs,
    )

    ref_inv = pyfar.dsp.regularized_spectrum_inversion(
        exc,
        frequency_range=inversion_range_hz,
    )

    ir_dict = {}

    for key, rec_list in recordings.data.items():
        irs = []
        for rec in _as_list(rec_list):
            y = pyfar.Signal(rec.data.T, fs)
            irs.append(y * ref_inv)

        ir_avg = align_and_average_irs(
            irs,
            onset_threshold_db=onset_threshold_db,
            reference="median",
        )

        ir_dict[key] = slab.Filter(
            data=ir_avg.time.T,
            samplerate=fs,
            fir="IR",
        )

    params = {
        "fs": fs,
        "signal": sig_params,
        "compute_ir": {
            "onset_threshold_db": onset_threshold_db,
            "inversion_range_hz": list(inversion_range_hz),
            "date": datetime.now().isoformat(),
        },
    }

    return ImpulseResponses(data=ir_dict, params=params)


# =====================================================================
# Equalization (single canonical version)
# =====================================================================

def equalize(
    measured: "ImpulseResponses",
    reference: "ImpulseResponses",
    n_samples_out: int,
    onset_threshold_db: float = 15.0,
) -> "ImpulseResponses":
    """
    Loudspeaker-wise equalization using reference IRs.

    Assumes:
    - measured and reference share speaker IDs
    - both already window-free IRs
    """
    fs = int(measured.params["fs"])
    signal_params = measured.params["signal"]

    out = {}

    for key, filt in measured.data.items():
        spk_id = key.split("_")[0]
        ref_key = next(
            k for k in reference.data.keys()
            if k.startswith(spk_id + "_")
        )

        H = pyfar.Signal(filt.data.T, fs)
        R = pyfar.Signal(reference.data[ref_key].data.T, fs)

        R_inv = pyfar.dsp.regularized_spectrum_inversion(
            R, frequency_range=(200, 18000)
        )

        H_eq = H * R_inv
        out[key] = slab.Filter(
            data=H_eq.time[:, :n_samples_out].T,
            samplerate=fs,
            fir="IR",
        )

    params = {
        "fs": fs,
        "signal": signal_params,
        "equalize": {
            "n_samples_out": n_samples_out,
            "date": datetime.now().isoformat(),
        },
    }

    return ImpulseResponses(data=out, params=params)


# =====================================================================
# Low-frequency extrapolation (SINGLE implementation)
# =====================================================================

def lowfreq_extrapolate(
    irs: "ImpulseResponses",
    f_extrap: float = 400.0,
    f_target: float = 150.0,
    head_radius: float | None = None,
) -> "ImpulseResponses":
    """
    Replace low-frequency magnitude using spherical-head anchors.
    Phase is preserved.
    """
    hrir, coords, keys, fs = _irs_to_pyfar(irs)
    shtf = _spherical_head_for(coords, hrir.n_samples, fs, head_radius)

    freqs = hrir.frequencies
    mag_meas = numpy.abs(hrir.freq)
    phase = numpy.angle(hrir.freq)
    mag_head = numpy.abs(shtf.freq)

    idx_target = hrir.find_nearest_frequency(f_target)
    mask = freqs >= f_extrap

    mag_interp = numpy.empty_like(mag_meas)

    for i in range(mag_meas.shape[0]):
        for ear in range(2):
            freqs_anchor = numpy.concatenate(
                ([0.0, freqs[idx_target]], freqs[mask])
            )
            mags_anchor = numpy.concatenate(
                (
                    [mag_head[i, ear, 0], mag_head[i, ear, idx_target]],
                    mag_meas[i, ear, mask],
                )
            )
            mag_interp[i, ear] = numpy.interp(freqs, freqs_anchor, mags_anchor)

    hrir.freq = mag_interp * numpy.exp(1j * phase)
    return _pyfar_to_irs(irs, keys, hrir.time, fs)

def expand_azimuths_with_binaural_cues(
    hrir,
    az_range: tuple[float, float] = (-50, 50),
    head_radius: float | None = None,
    onset_threshold_db: float = 15.0,
    show: bool = False,
    probe_az: float = 45.0,
):
    """
     Extend vertical-arc HRIRs across azimuths and impose binaural cues.

     This combines three processing steps:
       1) **Azimuth expansion** – Duplicates all IRs near `center_az` across
          a grid within `az_range`, spaced by the mean elevation step.
       2) **Full-band ILD shaping** – Applies spherical-head ILDs to all
          off-midline sources while preserving per-frequency power and phase.
       3) **ITD alignment** – Shifts the right-ear IR so measured ITDs match
          those predicted by the spherical-head model.

     If `show=True`, plots one example (closest to `probe_az`) using
     `pyfar.plot.time_freq` to visualize ITD and ILD.

     Parameters
     ----------
     irs : ImpulseResponses
         Binaural input (2 ch) in pyfar spherical coordinates
         (`domain="sph"`, `convention="top_elev"`).
     az_range : (float, float)
         Azimuth range (deg) for duplication, e.g. (-35, 35).
     head_radius : float or None
         Sphere radius in meters; default uses model internal value.
     center_az : float
         Midline azimuth (deg), typically 0.
     show : bool
         Plot diagnostic time/freq view at `probe_az`.

     Returns
     -------
     ImpulseResponses
         New object with expanded azimuths, spherical-head ILDs,
         and ITD-aligned IRs.

     Notes
     -----
     ILDs are applied per frequency bin via R/L magnitude ratios from the
     spherical-head model; ITDs are adjusted by time-shifting the right ear.
     """

    # ---------------------------------------------------------------------
    # STEP 2: AZIMUTH EXPANSION
    # ---------------------------------------------------------------------
    # We start from a vertical arc at center_az (e.g., 0°). We’ll duplicate
    # those IRs across an azimuth grid covering az_range. The azimuth grid
    # step equals the mean *elevation* step in your existing arc, so the
    # az grid density matches your vertical sampling density.
    # New entries are deep-copied so later edits won’t affect originals.
    # ---------------------------------------------------------------------
    sources0 = hrir.get_sources()  # shape (n_pos, 3): [az, el, r]
    elevations = numpy.unique(sources0[:, 1])
    if len(elevations) > 1:
        vertical_res = float(numpy.mean(numpy.diff(numpy.sort(elevations))))
    else:
        # Fallback (single elevation present): create a single az step
        vertical_res = az_range[1] - az_range[0]

    # --- build azimuth grid based on vertical spacing ---
    azimuths = numpy.arange(az_range[0], az_range[1] + vertical_res / 2, vertical_res)
    # wrap to [0, 360) for pyfar convention (0°=front, 90°=left)
    azimuths_wrapped = _wrap_az_deg_ccw(azimuths)

    # --- find midline IRs at az = 0 (wrapped 0°) ---
    def _key_az_wrapped(k: str) -> float:
        # keys look like "spk_az_el"
        return float(_wrap_az_deg_ccw(float(k.split("_")[1])))

    # --- duplicate midline IRs across azimuth grid, inserting wrapped az into keys ---
    out = copy.deepcopy(hrir)
    new_entries = {}

    for key in hrir.data.keys():
        spk, _az_str, el_str = key.split("_")  # reuse elevation
        for az_w in azimuths_wrapped:
            az_s = f"{float(az_w):.1f}"
            new_key = f"{spk}_{az_s}_{el_str}"
            if new_key not in out.data and new_key not in new_entries:
                new_entries[new_key] = copy.deepcopy(out.data[key])  # independent copy

    # --- update and sort dictionary for stable downstream behavior ---
    out.data.update(new_entries)

    try:
        from collections import OrderedDict
        def _parse_key_triple(k):
            spk, az_s, el_s = k.split("_")
            return (float(az_s), float(el_s), spk)

        out.data = OrderedDict(sorted(out.data.items(), key=lambda kv: _parse_key_triple(kv[0])))
    except Exception:
        pass  # sorting is optional

    # --- convert to pyfar + compute spherical-head reference ---
    hrir, coords, keys, fs = _irs_to_pyfar(out)
    shtf = _spherical_head_for(coords, hrir.n_samples, fs, head_radius)

    # ---------------------------------------------------------------------
    # STEP 3: FULL-BAND ILD SHAPING (OFF-MIDLINE ONLY)
    # ---------------------------------------------------------------------
    # For each synthesized direction, we impose the spherical-head ILD per
    # frequency bin. We preserve the measured *power average* per bin:
    #   r(f)  = H_R_head / H_L_head    (magnitude ratio)
    #   A(f)  = sqrt((|H_L|^2 + |H_R|^2) / 2)
    #   mL'   = A * sqrt(2/(1+r^2))
    #   mR'   = r * mL'
    # Phases are preserved (ILD affects magnitudes only).
    # Midline (≈ center_az) directions are left untouched.
    # ---------------------------------------------------------------------
    H_meas = hrir.freq  # complex spectrum, shape (n_pos, 2, n_bins)
    mag_meas = numpy.abs(H_meas)
    phase_meas = numpy.angle(H_meas)
    mag_head = numpy.abs(shtf.freq)

    n_pos, n_ears, _ = mag_meas.shape
    if n_ears != 2:
        raise ValueError("Binaural data expected (2 ears).")

    sources = out.get_sources()
    az_all = sources[:, 0]
    is_midline = sources[:, 0] == 0

    # Copy magnitudes; we will overwrite off-midline entries
    mag_new = mag_meas.copy()

    for i in range(n_pos):
        if is_midline[i]:
            # Keep measured magnitudes on the midline as-is
            continue

        # Head-model ILD ratio r(f) = R/L
        mL_h = mag_head[i, 0, :]
        mR_h = mag_head[i, 1, :]
        r = mR_h / numpy.maximum(mL_h, 1e-12)  # protect division

        # Measured magnitudes and power average
        mL = mag_meas[i, 0, :]
        mR = mag_meas[i, 1, :]
        A = numpy.sqrt((mL**2 + mR**2) / 2.0)

        # Apply ILD while preserving A and phases
        mL_new = A * numpy.sqrt(2.0 / (1.0 + r**2))
        mR_new = r * mL_new

        mag_new[i, 0, :] = mL_new
        mag_new[i, 1, :] = mR_new

    # Recombine with original phases
    H_new = mag_new * numpy.exp(1j * phase_meas)
    hrir.freq = H_new  # pyfar will update time on demand

    # ---------------------------------------------------------------------
    # STEP 4: ITD ALIGNMENT (GLOBAL TIME SHIFT OF RIGHT EAR)
    # ---------------------------------------------------------------------
    # We want the *onset difference* (right minus left) to match the model
    # in the *time domain*. Compute onsets for both (model & processed),
    # then time-shift the *entire* right ear by ΔITD per direction.
    # ---------------------------------------------------------------------
    _ = hrir.time  # ensure time cache
    _ = shtf.time

    on_mod = pyfar.dsp.find_impulse_response_start(shtf, threshold=onset_threshold_db)
    on_mea = pyfar.dsp.find_impulse_response_start(hrir, threshold=onset_threshold_db)

    time_data = hrir.time  # shape (n_pos, 2, n_samples)
    out_time = numpy.empty_like(time_data)

    for i in range(time_data.shape[0]):
        # Convert sample offsets to seconds
        itd_model = (on_mod[i, 1] - on_mod[i, 0]) / fs
        itd_meas  = (on_mea[i, 1] - on_mea[i, 0]) / fs
        delta_itd = itd_model - itd_meas  # desired additional shift for right ear

        # Left ear unchanged
        out_time[i, 0, :] = time_data[i, 0, :]

        # Shift right ear in time using pyfar; preserves spectrum consistency
        sig_r = pyfar.Signal(time_data[i, 1:2, :], fs)
        sig_rs = pyfar.dsp.time_shift(sig_r, delta_itd, unit="s")
        out_time[i, 1, :] = sig_rs.time[0]

    # ---------------------------------------------------------------------
    # OPTIONAL: Quick diagnostic plot at ~probe_az
    # ---------------------------------------------------------------------
    if show:
        idx = int(numpy.argmin(numpy.abs(az_all - float(probe_az))))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,10))
        ax_t, ax_f = pyfar.plot.time_freq(pyfar.Signal(out_time[idx], fs))
        ax_t.get_lines()[0].set_label('left')
        ax_t.get_lines()[1].set_label('right')
        ax_t.legend()
        ax_f.get_lines()[0].set_label('left')
        ax_f.get_lines()[1].set_label('right')
        ax_f.legend()
        ax_t.set_title('time')
        ax_f.set_title("magnitude")
        plt.suptitle(f"Result @ az≈{az_all[idx]:.1f}°")
        plt.show()

    # ---------------------------------------------------------------------
    # Return as a fresh ImpulseResponses object with provenance
    # ---------------------------------------------------------------------
    out_final = _pyfar_to_irs(out, keys, out_time, fs)
    out_final.params.setdefault("processing", {})
    out_final.params["processing"]["expand_azimuths_with_binaural_cues"] = {
        "az_range": [float(az_range[0]), float(az_range[1])],
        "head_radius": float(head_radius) if head_radius is not None else None,
        "onset_threshold_db": float(onset_threshold_db),
        "date": datetime.now().isoformat(),
    }
    return out_final

def _wrap_az_deg_ccw(az):
    """Wrap azimuth(s) to [0, 360) with CCW-positive (pyfar 'sph/top_elev')."""
    az = numpy.asarray(az, dtype=float)
    az = numpy.mod(az, 360.0)
    az[az < 0] += 360.0
    return az

# =====================================================================
# Containers (imported here to avoid circulars)
# =====================================================================

class ImpulseResponses(SpeakerGridBase):
    pass


# =====================================================================
# pyfar <-> slab helpers (single source of truth)
# =====================================================================

def _irs_to_pyfar(irs: ImpulseResponses):
    fs = int(irs.params["fs"])
    keys = list(irs.data.keys())
    data = numpy.stack([irs.data[k].data.T for k in keys], axis=0)
    sig = pyfar.Signal(data, fs)

    sources = irs.get_sources()
    coords = pyfar.Coordinates(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        domain="sph",
        convention="top_elev",
        unit="deg",
    )
    return sig, coords, keys, fs


def _pyfar_to_irs(template: ImpulseResponses, keys, time_data, fs):
    out = copy.deepcopy(template)
    for key, td in zip(keys, time_data):
        out.data[key] = slab.Filter(td.T, fs, fir="IR")
    return out


def _spherical_head_for(coords, n_samples, fs, head_radius=None):
    from hrtf_relearning.hrtf.processing.spherical_head import spherical_head
    return spherical_head(
        coords,
        n_samples=n_samples,
        sampling_rate=fs,
        head=None if head_radius is None else head_radius,
    )
