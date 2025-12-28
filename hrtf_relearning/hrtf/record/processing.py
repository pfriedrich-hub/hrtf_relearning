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
# Deconvolution: Lists of Recordings -> ImpulseResponses
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
    exc = pyfar.Signal(sig.data.T, fs)
    ref_inv = pyfar.dsp.regularized_spectrum_inversion(exc, frequency_range=inversion_range_hz)
    ir_dict = {}
    for key, rec_list in recordings.data.items():
        # time align and average recordings for each loudspeaker
        recs = pyfar.Signal(numpy.stack([rec.data.T for rec in rec_list], axis=0), fs)  # convert to pyfar
        recs_aligned, shifts = align_recordings(recs)
        recs_averaged = pyfar.Signal(numpy.mean(recs_aligned.time, axis=0), sampling_rate=fs)

        # convolve with excitation signal to compute IR and store as slab Filter
        ir = recs_averaged * ref_inv
        ir_dict[key] = slab.Filter(
            data=ir.time.T,
            samplerate=fs,
            fir="IR")

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

def align_recordings(
    recs: pyfar.Signal,
    max_shift: int = 10,
    ref_index: int = 0,
):
    """
    Align multiple binaural recordings by local (small-shift) correlation.

    Parameters
    ----------
    recs : pyfar.Signal
        Shape (n_rec, 2, n_samples)
    max_shift : int
        Maximum absolute shift (in samples) to test, e.g. 1 or 2
    ref_index : int
        Index of reference recording

    Returns
    -------
    recs_aligned : pyfar.Signal
        Time-aligned recordings, same shape as input
    shifts : ndarray
        Applied shifts in samples, shape (n_rec, 2)
    """

    if recs.cshape is None or len(recs.cshape) != 2:
        raise ValueError("Expected recs with cshape (n_rec, 2)")

    fs = recs.sampling_rate
    x = recs.time                    # (n_rec, 2, n_samples)
    n_rec, n_ears, n_samples = x.shape

    ref = x[ref_index]               # (2, n_samples)

    shifts = numpy.zeros((n_rec, n_ears), dtype=int)
    aligned = numpy.empty_like(x)

    # predefine shift candidates
    shift_candidates = numpy.arange(-max_shift, max_shift + 1)

    for i in range(n_rec):
        for ear in range(n_ears):

            # reference and target
            r = ref[ear]
            y = x[i, ear]

            # compute correlation scores only for small shifts
            scores = []
            for s in shift_candidates:
                if s < 0:
                    score = numpy.dot(r[-s:], y[: n_samples + s])
                elif s > 0:
                    score = numpy.dot(r[: n_samples - s], y[s:])
                else:
                    score = numpy.dot(r, y)
                scores.append(score)

            best_shift = shift_candidates[numpy.argmax(scores)]
            shifts[i, ear] = best_shift

            # apply shift using pyfar (safe, future-proof)
            sig = pyfar.Signal(y[None, :], fs)
            sig_shifted = pyfar.dsp.time_shift(
                sig,
                -best_shift,
                unit="samples",
            )
            aligned[i, ear] = sig_shifted.time[0]

    recs_aligned = pyfar.Signal(aligned, fs)
    if (abs(shifts) > 2).any():
        logging.warning(f'Time shifts > 2 samples when averaging recordings: \n{shifts}')
    return recs_aligned, shifts

# =====================================================================
# Equalization
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
    
    # =====================================================================
    # plotting
    # =====================================================================

    def tf(
            self,
            azimuth=0,
            linesep=20,
            xscale="log",
            axis=None,
    ):
        """
        Waterfall plot of left + right ear spectra from in-ear recordings.
        Elevations determine vertical offset (one curve per elevation).
        Left ear = dark gray, right = lighter gray.

        Parameters
        ----------
        xlim : tuple
            Frequency axis limits.
        n_bins : int or None
            FFT resolution for spectrum().
        linesep : float
            Vertical separation in dB between stacked spectra.
        xscale : 'linear' or 'log'
            Frequency axis scaling.
        show : bool
            Show the plot immediately.
        axis : matplotlib axis or None
            Insert plot into an existing axis.
        """
        xlim = (self.params['signal']['from_frequency'], self.params['signal']['to_frequency'])
        # Create axis
        if axis is None:
            fig, axis = plt.subplots(figsize=(7, 6))
        else:
            fig = axis.figure

        keys = list(self.data.keys())

        elevations = []
        specs_L = []  # left ear spectra
        specs_R = []  # right ear spectra
        freqs_saved = None

        # ------------------------------------------------------------------
        # Extract spectra from all recordings
        # ------------------------------------------------------------------
        for key in keys:
            _, _, el = self.parse_key(key)
            filt = self.data[key]  # slab.Binaural

            # left ear
            freqs, Hl = filt.channel(0).tf(show=False)
            # right ear
            _, Hr = filt.channel(1).tf(show=False)

            if freqs_saved is None:
                freqs_saved = freqs

            elevations.append(el)
            specs_L.append(Hl)
            specs_R.append(Hr)

        elevations = numpy.asarray(elevations)
        specs_L = numpy.asarray(specs_L)
        specs_R = numpy.asarray(specs_R)

        # baseline correction (compute common baseline from original data)
        baseline = numpy.mean((specs_L + specs_R) / 2)

        specs_L = specs_L - baseline
        specs_R = specs_R - baseline

        # ------------------------------------------------------------------
        # Sort by elevation
        # ------------------------------------------------------------------
        idx = numpy.argsort(elevations)
        elevations = elevations[idx]
        specs_L = specs_L[idx]
        specs_R = specs_R[idx]

        # Compute vertical offsets
        vlines = numpy.arange(len(elevations)) * (linesep + 20)

        # ------------------------------------------------------------------
        # Plot waterfall curves
        # ------------------------------------------------------------------
        for i, (Hl, Hr) in enumerate(zip(specs_L, specs_R)):
            axis.plot(
                freqs_saved, Hl + vlines[i],
                color="0.25", linewidth=0.8, alpha=0.9, label="Left" if i == 0 else None,
            )
            axis.plot(
                freqs_saved, Hr + vlines[i],
                color="0.65", linewidth=0.8, alpha=0.9, label="Right" if i == 0 else None,
            )

        # ------------------------------------------------------------------
        # Elevation labels
        # ------------------------------------------------------------------
        ticks = vlines[::2]
        labels = elevations[::2].astype(int)

        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
        axis.set_ylabel("Elevation (°)")

        # ------------------------------------------------------------------
        # dB scale bar (height = linesep)
        # ------------------------------------------------------------------
        scale_x = xlim[0] + 300
        scale_y0 = vlines[-1] + 40
        scale_y1 = scale_y0 + linesep

        axis.plot([scale_x, scale_x], [scale_y0, scale_y1],
                  color="0.1", linewidth=1.2)
        axis.text(
            scale_x + 90,
            scale_y0 + linesep / 2,
            f"{linesep} dB",
            va="center", fontsize=7, color="0.1"
        )

        # ------------------------------------------------------------------
        # Formatting utilities
        # ------------------------------------------------------------------
        axis.set_xlim(xlim)
        axis.set_xscale(xscale)
        if xscale == "log":
            axis.set_xscale("log")
            # optionally: let matplotlib choose ticks/formatter
            axis.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs="all"))
            axis.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(base=10.0,
                                                                          labelOnlyBase=False))
            axis.grid(axis='x', which='both', linestyle=':', linewidth=0.3, alpha=1)
            axis.set_xticks([20, 40, 60, 100, 200, 400, 600, 1000, 2e3, 4e3, 6e3, 10e3, 20e3])
            axis.set_xticklabels([20, 40, 60, 100, 200, 400, 600, '1k', '2k', '4k', '6k', '10k', '20k'])
            axis.set_xlim(1e3, 18e3)  # works better

        # axis.set_xscale(xscale)
        # # Format frequency labels as kHz
        # axis.xaxis.set_major_formatter(
        #     matplotlib.ticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}")
        # )
        # axis.set_xlabel("Frequency [kHz]")

        axis.grid(axis="y", linestyle=":", linewidth=0.3, alpha=1)
        axis.legend(loc="upper right", fontsize=7)
        plt.show()
        return fig
    


# =====================================================================
# pyfar <-> slab helpers
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


