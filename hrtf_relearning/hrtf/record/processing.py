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

Azimuth convention
------------------
ONE convention holds everywhere in this module and in every SOFA it writes:

    azimuth in [0, 360), COUNTERCLOCKWISE-positive seen from above
    (pyfar 'sph/top_elev'):  0 = front, 90 = LEFT, 270 = right.
    elevation in [-90, 90], positive up.

Measured dome keys arrive on the frontal arc (azimuth 0.00), which is the same
number under either sign convention. `expand_azimuths_with_binaural_cues` wraps
the expanded grid with `_wrap_az_deg_ccw` and deliberately does NOT re-emit the
frontal column, so 0 and 360 never both appear. `validate_source_grid` enforces
that at the point of conversion to `slab.HRTF`: a negative azimuth, an azimuth
at or past 360, or two sources at the same (az, el) is a bug here, not something
downstream should be asked to tolerate.

Consumers convert explicitly and must keep doing so:
  * `Localization_AR` / `Localization_VR` mirror the head-tracker azimuth, which
    is CLOCKWISE-positive, via ``(-az + 360) % 360``.
  * `make_sequence` works in signed ``[-180, 180)``.
  * `hrir2mat.frontal_index` folds ``az > 180`` to negative to find straight
    ahead; `write_filters` passes the [0, 360) value to pyBinSim unchanged.
"""
from __future__ import annotations
import copy
import numpy
from datetime import datetime
import pyfar
import warnings
import logging
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)
from .recordings import Recordings, SpeakerGridBase
import slab

# =====================================================================
# Pipeline constants
# =====================================================================
# --- spectrum inversion ----------------------------------------------
# TWO different signals get inverted in this pipeline, which is why there are
# two upper bounds. They are not an inconsistency waiting to be reconciled:
#
#   compute_ir  inverts the EXCITATION SWEEP -- analytic, known, full-band
#               (120 Hz to `signal['to_frequency']`, currently 22 kHz, against a
#               Nyquist of ~24.4 kHz). Nothing about the sweep is unreliable, so
#               the only reason to stop below its own top edge is to stay clear
#               of Nyquist. This bound sets where the finished DTFs fall off a
#               cliff: measured across the 14 subject SOFAs, level holds near
#               0 dB through 18-20 kHz and then drops to -26 dB at 20-22 kHz.
#
#   equalize    inverts the measured REFERENCE IR -- band-limited by speaker,
#               microphone and room, sitting on a noise floor. Above the
#               speaker's usable band 1/R asks for unbounded boost, so the top
#               must be set where the reference stops being trustworthy. That
#               is a judgement call about the rig, not a property of a signal.
#
# On whether 18 kHz is low enough: the three KEMAR re-seats give a
# reproducibility floor of 3.0-3.9 dB rms in 16-20 kHz against an
# across-direction spread of 6.3-8.9 dB, i.e. a cue-to-noise ratio near 2.0 --
# the SAME ratio as 1-4 kHz. So the data do NOT show a frequency above which the
# DTF collapses into noise, and there is no measurement-driven case for pulling
# either bound down. The separate argument for a lower top is perceptual, not
# acoustic: the elevation cues this project manipulates live in 4-15 kHz, and
# broadband measures are band-limited to 16 kHz for exactly that reason (see
# `equalize`'s `ild_band`). Changing either bound re-scales every SOFA, so treat
# it as an experiment-wide decision, not a tuning knob.
EXCITATION_INVERSION_TOP_HZ = 20e3
REFERENCE_INVERSION_TOP_HZ = 18e3

# --- post-division time window ---------------------------------------
# Applied in `equalize` to the equalized IR, relative to its detected onset:
# fade in, plateau, fade out. Total pass band onset-0.25 ms .. onset+2.5 ms.
#
# 2.5 ms after the direct sound is ~0.86 m of extra path, so this keeps the
# direct sound and the earliest pinna/torso detail and rejects the first room
# boundary (floor and the dome frame both arrive later than that). The previous
# setting was 4.8/5.8 ms -- kept commented at the call site -- which let the
# first floor reflection into the DTF.
WINDOW_FADE_IN_S = 0.00025
WINDOW_PLATEAU_S = 0.0015
WINDOW_FADE_OUT_S = 0.0010

# =====================================================================
# Deconvolution: Lists of Recordings -> ImpulseResponses
# =====================================================================
def compute_ir(
    recordings: Recordings,
    onset_threshold_db: float = 10.0,
    inversion_range_hz: tuple[float, float] | None = None,
    align_interaural: bool = False,
) -> "ImpulseResponses":
    """
    Deconvolve sweep recordings into impulse responses.

    Pipeline:
    - temporally align recordings per speaker and ear
    - time-domain average recordings
    - regularized spectrum inversion of excitation signal
    - deconvolution -> IR
    - time alignment of impulse responses

    No windowing, no cropping.
    """
    fs = int(recordings.params["fs"])
    sig_params = recordings.params["signal"]

    # Default to the sweep's own band, capped at EXCITATION_INVERSION_TOP_HZ.
    # Previously this defaulted to None and then crashed on list(None); every
    # caller passed the range explicitly, which is why nobody noticed.
    if inversion_range_hz is None:
        inversion_range_hz = (
            float(sig_params["from_frequency"]),
            min(float(sig_params["to_frequency"]), EXCITATION_INVERSION_TOP_HZ),
        )

    params = copy.deepcopy(recordings.params)
    params.update({
        "fs": fs,
        "signal": sig_params,
        "compute_ir": {
            "onset_threshold_db": onset_threshold_db,
            "inversion_range_hz": list(inversion_range_hz),
            "date": datetime.now().isoformat(),
        },
    })

    # time align and average recordings per loudspeaker
    recordings = average_recordings(recordings)

    # --- invert excitation signal ---
    sig = recordings.signal
    exc = pyfar.Signal(sig.data.T, fs)
    ref_inv = pyfar.dsp.regularized_spectrum_inversion(exc, frequency_range=inversion_range_hz)

    # --- convolve to obtain IR ---
    ir_dict = {}
    for key, recording in recordings.data.items():
        ir = recording * ref_inv
        ir_dict[key] = ir
    irs = ImpulseResponses(data=ir_dict, params=params)

    # --- temporal alignment ---
    irs = time_align_irs(irs, center_key='23_0.0_0.0', desired_onset_s=.001, onset_threshold_db=onset_threshold_db,
                         align_itd=align_interaural, align_ild=align_interaural)
    return irs

def _match_direction_key(data, center_key, tol_deg=1e-3):
    """Find the key of `data` that names the same DIRECTION as `center_key`.

    Speaker-grid keys are 'index_azimuth_elevation' formatted with a varying
    number of decimals -- `record_dome` writes two ('23_0.00_0.00'), older
    reference folders one ('23_0.0_0.0'). Comparing them as strings is a silent
    trap, so compare the parsed azimuth/elevation instead, and fall back to the
    speaker index if the angles do not match anything.

    Returns the matching key, or None.
    """
    if center_key is None:
        return None
    if center_key in data:
        return center_key
    try:
        index, azimuth, elevation = center_key.split("_")
        index, azimuth, elevation = int(index), float(azimuth), float(elevation)
    except (ValueError, AttributeError):
        return None

    by_index = None
    for key in data:
        try:
            k_index, k_azimuth, k_elevation = key.split("_")
            k_index, k_azimuth, k_elevation = int(k_index), float(k_azimuth), float(k_elevation)
        except ValueError:
            continue
        if abs(k_azimuth - azimuth) <= tol_deg and abs(k_elevation - elevation) <= tol_deg:
            return key
        if k_index == index and by_index is None:
            by_index = key
    return by_index


def time_align_irs(
    irs,
    *,
    center_key: str | None = '23_0.0_0.0',
    onset_threshold_db: float = 15.0,
    desired_onset_s: float = 0.001,
    onset_mode: str = "earliest",   # "earliest" or "sum"
    # NEW: optional frontal-only modifications
    align_itd: bool = False,   # NEW
    align_ild: bool = False,   # NEW
    frontal_az_deg: float = 0.0,          # NEW
    frontal_tol_deg: float = 1e-6,        # NEW
):
    """
    Global time anchoring for a set of IRs:
    - pick one reference direction (center_key or first entry)
    - find onset on that reference
    - shift ALL IRs by the same amount so onset lands at desired_onset_s

    Optional:
    - For frontal sources (az==0 across elevations): remove ITD and/or ILD.

    DEPRECATED (2026-08-18): do NOT use align_itd/align_ild in the recording
    path. `compute_ir` runs on the subject AND the reference, so zeroing here
    happens BEFORE `equalize` divides one by the other, which leaves a
    systematic interaural residual rather than removing one -- see
    `zero_frontal_interaural`, which does the same job after the division and
    is what `equalize(align_interaural=True)` calls. These two flags are kept
    only to reproduce SOFAs built before that date.
    """

    fs = int(irs.params["fs"])
    keys = list(irs.data.keys())
    if len(keys) == 0:
        raise ValueError("time_align_irs_global: empty IR set")

    if align_itd or align_ild:
        logging.warning(
            "time_align_irs: align_itd/align_ild run BEFORE reference "
            "equalization and leave a systematic interaural residual. Use "
            "equalize(align_interaural=True) instead; these flags are kept "
            "only to reproduce pre-2026-08-18 SOFAs."
        )

    # --- select reference direction ---
    # Match by DIRECTION, not by string. Subject keys are written with two
    # decimals ('23_0.00_0.00') and reference keys with one ('23_0.0_0.0'), so
    # the plain `center_key in irs.data` test succeeded for references and
    # silently failed for every subject, anchoring subjects on keys[0] -- the
    # TOP ELEVATION -- while their reference was anchored on the frontal
    # direction. Fixed 2026-08-19.
    #
    # A MISS IS NOW FATAL. It used to fall back to keys[0] with a warning, which
    # is how the two-decimals bug survived months of recordings: the fallback is
    # silent in every practical sense (a log line in a long session) and it
    # anchors subject and reference independently, which shifts the DTF in time
    # relative to its own reference. Pass center_key=None to anchor on the first
    # key deliberately.
    if center_key is None:
        ref_key = keys[0]
    else:
        ref_key = _match_direction_key(irs.data, center_key)
        if ref_key is None:
            raise ValueError(
                f"time_align_irs: anchor '{center_key}' matches no direction in "
                f"this set (neither by azimuth/elevation nor by speaker index). "
                f"Available: {keys[:6]}{' ...' if len(keys) > 6 else ''}. "
                "Anchoring elsewhere would shift this set in time relative to "
                "the reference it will be divided by, so this is not recoverable "
                "here -- fix the anchor, or pass center_key=None if you really "
                "mean 'use the first key'.")

    ref_sig = irs.data[ref_key]
    if ref_sig.cshape != (2,):
        raise ValueError(f"Expected binaural pyfar.Signal with cshape (2,), got {ref_sig.cshape} for {ref_key}")

    # --- compute onset for reference direction ---
    if onset_mode == "earliest":
        on = pyfar.dsp.find_impulse_response_start(ref_sig, threshold=onset_threshold_db)
        onset_samples = int(numpy.min(on))
    elif onset_mode == "sum":
        mono = pyfar.Signal(ref_sig.time[0] + ref_sig.time[1], fs)
        on = pyfar.dsp.find_impulse_response_start(mono, threshold=onset_threshold_db)
        onset_samples = int(on)
    else:
        raise ValueError("onset_mode must be 'earliest' or 'sum'")

    onset_s = onset_samples / fs
    shift_s = desired_onset_s - onset_s

    # --- apply the same shift to ALL IRs ---
    out = copy.deepcopy(irs)
    for k in keys:
        out.data[k] = pyfar.dsp.time_shift(out.data[k], shift_s, unit="s")

    # ------------------------------------------------------------------
    # NEW: helper to detect frontal keys (az==0 across elevations)
    # ------------------------------------------------------------------
    def _is_frontal_key(k: str) -> bool:  # NEW
        try:
            _spk, az_s, _el_s = k.split("_")
            az = float(az_s)
        except Exception:
            return False
        # treat 0 and 360 as equivalent; you said frontal means az=0
        az_wrapped = float(numpy.mod(az, 360.0))
        return (abs(az_wrapped - frontal_az_deg) <= frontal_tol_deg) or (abs(az_wrapped - 360.0) <= frontal_tol_deg)

    # ------------------------------------------------------------------
    # NEW: frontal-only ITD removal (shift later ear to match earlier onset)
    # ------------------------------------------------------------------
    if align_itd:  # NEW
        for k in keys:
            if not _is_frontal_key(k):
                continue
            sig = out.data[k]
            on_lr = pyfar.dsp.find_impulse_response_start(sig, threshold=onset_threshold_db)
            onL, onR = int(on_lr[0]), int(on_lr[1])

            delta = onR - onL  # + => R later
            if delta == 0:
                continue

            time_data = sig.time.copy()  # (2, n_samples)
            if delta > 0:
                # R later -> shift R earlier
                r = pyfar.Signal(time_data[1:2, :], fs)
                r2 = pyfar.dsp.time_shift(r, -delta, unit="samples")
                time_data[1, :] = r2.time[0]
            else:
                # L later -> shift L earlier
                l = pyfar.Signal(time_data[0:1, :], fs)
                l2 = pyfar.dsp.time_shift(l, delta, unit="samples")  # delta negative => earlier
                time_data[0, :] = l2.time[0]

            out.data[k] = pyfar.Signal(time_data, fs)

    # ------------------------------------------------------------------
    # NEW: frontal-only ILD removal (broadband gain to equalize levels)
    #      Uses per-ear RMS over the full IR; preserves average power.
    # ------------------------------------------------------------------
    if align_ild:  # NEW
        eps = 1e-12  # NEW
        for k in keys:
            if not _is_frontal_key(k):
                continue
            sig = out.data[k]
            x = sig.time  # (2, n_samples)

            # RMS per ear (broadband level)
            rmsL = float(numpy.sqrt(numpy.mean(x[0] ** 2)))
            rmsR = float(numpy.sqrt(numpy.mean(x[1] ** 2)))

            # target preserves average power across ears
            target = float(numpy.sqrt((rmsL**2 + rmsR**2) / 2.0))

            gL = target / max(rmsL, eps)
            gR = target / max(rmsR, eps)

            x2 = x.copy()
            x2[0] *= gL
            x2[1] *= gR

            out.data[k] = pyfar.Signal(x2, fs)

    out.params.setdefault("time_alignment", {})
    out.params["time_alignment"].update({
        "method": "global_reference_onset",
        "reference_key": ref_key,
        "onset_mode": onset_mode,
        "onset_threshold_db": float(onset_threshold_db),
        "desired_onset_s": float(desired_onset_s),
        "reference_onset_samples": int(onset_samples),
        "applied_shift_s": float(shift_s),
        # NEW: provenance for optional steps
        "zero_itd_for_frontal": bool(align_itd),   # NEW
        "zero_ild_for_frontal": bool(align_ild),   # NEW
        "frontal_az_deg": float(frontal_az_deg),              # NEW
        "frontal_tol_deg": float(frontal_tol_deg),            # NEW
        "date": datetime.now().isoformat(),
    })
    return out


def average_recordings(recordings: Recordings):
    """
    Time align, average slab binaural recordings and convert to pyfar Signal.
    :param recordings:
    :return:
    """
    fs = recordings.params["fs"]
    out = copy.deepcopy(recordings)
    for key, rec_list in out.data.items():
        # time align and average recordings for each loudspeaker
        recs = pyfar.Signal(numpy.stack([rec.data.T for rec in rec_list], axis=0), fs)  # convert to pyfar
        recs_aligned, shifts = align_recordings(recs)
        recs_averaged = pyfar.Signal(numpy.mean(recs_aligned.time, axis=0), sampling_rate=fs)
        out.data[key] = recs_averaged
    return out

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
        # compute best_shift ONCE per recording
        r = ref[0] + ref[1]
        y = x[i, 0] + x[i, 1]

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

        # apply identical shift to both ears
        for ear in range(2):
            sig = pyfar.Signal(x[i, ear][None, :], fs)
            sig_shifted = pyfar.dsp.time_shift(sig, -best_shift, unit="samples")
            aligned[i, ear] = sig_shifted.time[0]

        shifts[i, :] = best_shift

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
    inversion_range_hz,
    onset_threshold_db: float = 20.0,
    align_interaural: bool = False,
    ild_band: tuple[float, float] = (200.0, 16000.0),
    itd_band: tuple[float, float] = (200.0, 1500.0),
) -> "ImpulseResponses":
    """
    Loudspeaker-wise equalization using reference IRs.

    Assumes:
    - measured and reference share speaker IDs
    - both already window-free IRs

    align_interaural
        Zero ITD and ILD on the frontal arc AFTER the division, via
        `zero_frontal_interaural`. This is the only correct place for it --
        see that function for why doing it in `compute_ir` (i.e. to the
        subject and the reference separately, before dividing) leaves a
        systematic residual instead of removing one.
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

        H = filt
        R = reference.data[ref_key]
        # convolve
        R_inv = pyfar.dsp.regularized_spectrum_inversion(
            R, frequency_range=inversion_range_hz
        )
        H_eq = H * R_inv

        # onset align
        onsets = pyfar.dsp.find_impulse_response_start(H_eq, threshold=onset_threshold_db)
        H_aligned = pyfar.dsp.time_shift(
            H_eq, -numpy.min(onsets) / H_eq.sampling_rate + .001,
            unit='s')
        # window
        onset = pyfar.dsp.find_impulse_response_start(H_aligned, threshold=onset_threshold_db)
        onset_min = numpy.min(onset) / H_aligned.sampling_rate  # onset in seconds
        # times = (onset_min - .00025 if (onset_min - .00025) > 0 else 0,  # start of fade-in
        #          onset_min,  # end if fade-in
        #          onset_min + .0048,  # start of fade_out
        #          onset_min + .0058)  # end of_fade_out
        times = (onset_min - WINDOW_FADE_IN_S,                    # start of fade-in
                 onset_min,                                       # end of fade-in
                 onset_min + WINDOW_PLATEAU_S,                    # start of fade-out
                 onset_min + WINDOW_PLATEAU_S + WINDOW_FADE_OUT_S)  # end of fade-out
        H_windowed, window = pyfar.dsp.time_window(
            H_aligned, times, 'hann', unit='s', crop='none', return_window=True)
        # print('win')

        # crop
        times = [0, 10, n_samples_out-10, n_samples_out-1]
        H_final = pyfar.dsp.time_window(
            H_windowed, times, 'boxcar', crop='end')

        out[key] = H_final

        # out[key] = slab.Filter(
        #     data=H_eq.time[:, :n_samples_out].T,
        #     samplerate=fs,
        #     fir="IR",
        # )

    params = {
        "fs": fs,
        "signal": signal_params,
        "equalize": {
            "n_samples_out": n_samples_out,
            "date": datetime.now().isoformat(),
            "align_interaural": bool(align_interaural),
        },
    }

    out_irs = ImpulseResponses(data=out, params=params)

    if align_interaural:
        out_irs = zero_frontal_interaural(
            out_irs, align_itd=True, align_ild=True,
            ild_band=ild_band, itd_band=itd_band,
        )
        params["equalize"]["ild_band"] = list(ild_band)
        params["equalize"]["itd_band"] = list(itd_band)
        out_irs.params = params

    return out_irs


# =====================================================================
# Low-frequency extrapolation (SINGLE implementation)
# =====================================================================

def lowfreq_extrapolate(
    irs: "ImpulseResponses",
    f_extrap: float = 400.0,
    f_target: float = 150.0,
    head_radius: float | None = 0.0875,
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

def _spherical_head_for(coords, n_samples, fs, head_radius=None):
    """
    Wrap spherical_head() and, if head_radius is given, construct the expected
    spharpy/SOFAR SamplingSphere with two ear nodes (±90° az, 0° el).
    """
    from hrtf_relearning.hrtf.processing.spherical_head import spherical_head as _spherical_head
    if head_radius is None:  # use default head radius: .0875 m
        return _spherical_head(coords, n_samples=n_samples, sampling_rate=fs)
    head = pyfar.Coordinates(0, [head_radius, -head_radius], 0)
    return _spherical_head(coords,head=head,n_samples=n_samples,sampling_rate=fs)

# =====================================================================
# Containers (imported here to avoid circulars)
# =====================================================================

class ImpulseResponses(SpeakerGridBase):
    """
      Directional impulse responses over the speaker dome.
      Values are `slab.Filter` (FIR).
      """

    def __init__(self, data=None, params=None):
        super().__init__(data=data, params=params)

    def to_slab_hrtf(
        self,
        fs: int | None = None,
        datatype: str = "FIR",
    ) -> slab.HRTF:
        """
        Convert this ImpulseResponses object into a slab.HRTF.

        Assumes that self.data is a dict mapping keys like
        '23_0.0_40.0' → slab.Filter with shape (n_samples, n_channels).

        The resulting HRTF has shape (n_positions, n_samples, n_channels)
        and sources from self.get_sources().

        Parameters
        ----------
        fs
            Samplerate for the HRTF. If None, tries self.params["fs"],
            then the samplerate of the first Filter.
        datatype
            Passed to slab.HRTF (e.g. 'FIR').

        Returns
        -------
        hrtf : slab.HRTF
        """
        if not self.data:
            raise ValueError("to_slab_hrtf: no filters in self.data")

        # samplerate
        if fs is None:
            if "fs" in self.params:
                fs = int(self.params["fs"])
            else:
                # fall back to first filter's samplerate
                first_key = next(iter(self.data.keys()))
                fs = int(self.data[first_key].samplerate)

        # ensure a stable order: use keys list once for data and coordinates
        keys = list(self.data.keys())

        # stack filters: (n_positions, n_samples, n_channels)
        data = numpy.stack([self.data[k].time for k in keys], axis=0)

        # sources: (n_positions, 3) -> [az, el, r]
        # uses the same internal order as self.data, so things stay aligned.
        # Validated here because this is the single gate every SOFA passes
        # through -- see the module docstring's "Azimuth convention".
        sources = validate_source_grid(self.get_sources())

        hrir = slab.HRTF(
            data=data,
            sources=sources,
            samplerate=fs,
            datatype=datatype,
        )

        return hrir
    # =====================================================================
    # plotting
    # =====================================================================

    def waterfall(self, azimuth=0, linesep=20, xscale="log", axis=None):
        """
        Waterfall plot of left + right ear spectra from in-ear impulse responses.
        Elevations determine vertical offset (one curve per elevation).
        Left ear = dark gray, right = lighter gray.
        
        INTERACTIVE USE ONLY -- nothing in the pipeline calls this. Kept
        deliberately for eyeballing a set in a console; do not assume it is
        exercised by any run.
        """

        import numpy
        import matplotlib
        from hrtf_relearning.utils.mpl_backend import use_interactive
        use_interactive()
        import matplotlib.pyplot as plt

        xlim = (
            self.params["signal"]["from_frequency"],
            self.params["signal"]["to_frequency"],
        )

        # ------------------------------------------------------------
        # Axis handling
        # ------------------------------------------------------------
        if axis is None:
            fig, axis = plt.subplots(figsize=(7, 6))
        else:
            fig = axis.figure

        keys = list(self.data.keys())

        elevations = []
        specs_L = []
        specs_R = []
        freqs_saved = None

        # ------------------------------------------------------------
        # Extract spectra
        # ------------------------------------------------------------
        for key in keys:
            _, _, el = self.parse_key(key)
            sig = self.data[key]  # pyfar.Signal, shape (2, n_samples)

            # FFT via pyfar
            H = sig.freq  # complex spectrum, shape (2, n_freqs)
            freqs = sig.frequencies

            # magnitude in dB
            Hl = 20 * numpy.log10(numpy.abs(H[0]) + 1e-12)
            Hr = 20 * numpy.log10(numpy.abs(H[1]) + 1e-12)

            if freqs_saved is None:
                freqs_saved = freqs

            elevations.append(el)
            specs_L.append(Hl)
            specs_R.append(Hr)

        elevations = numpy.asarray(elevations)
        specs_L = numpy.asarray(specs_L)
        specs_R = numpy.asarray(specs_R)

        # ------------------------------------------------------------
        # Baseline correction (common average)
        # ------------------------------------------------------------
        baseline = numpy.mean((specs_L + specs_R) / 2.0)
        specs_L = specs_L - baseline
        specs_R = specs_R - baseline

        # ------------------------------------------------------------
        # Sort by elevation
        # ------------------------------------------------------------
        idx = numpy.argsort(elevations)
        elevations = elevations[idx]
        specs_L = specs_L[idx]
        specs_R = specs_R[idx]

        # Vertical offsets
        vlines = numpy.arange(len(elevations)) * (linesep + 20)

        # ------------------------------------------------------------
        # Plot waterfall
        # ------------------------------------------------------------
        for i, (Hl, Hr) in enumerate(zip(specs_L, specs_R)):
            axis.plot(
                freqs_saved,
                Hl + vlines[i],
                color="0.25",
                linewidth=0.8,
                alpha=0.9,
                label="Left" if i == 0 else None,
            )
            axis.plot(
                freqs_saved,
                Hr + vlines[i],
                color="0.65",
                linewidth=0.8,
                alpha=0.9,
                label="Right" if i == 0 else None,
            )

        # ------------------------------------------------------------
        # Elevation labels
        # ------------------------------------------------------------
        ticks = vlines[::2]
        labels = elevations[::2].astype(int)

        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
        axis.set_ylabel("Elevation (°)")

        # ------------------------------------------------------------
        # dB scale bar
        # ------------------------------------------------------------
        scale_x = xlim[0] + 1e3
        scale_y0 = vlines[-1] + 40
        scale_y1 = scale_y0 + linesep

        axis.plot(
            [scale_x, scale_x],
            [scale_y0, scale_y1],
            color="0.1",
            linewidth=1.2,
        )
        axis.text(
            scale_x + 90,
            scale_y0 + linesep / 2,
            f"{linesep} dB",
            va="center",
            fontsize=7,
            color="0.1",
        )

        # ------------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------------
        axis.set_xlim(xlim)
        axis.set_xscale(xscale)

        if xscale == "log":
            axis.xaxis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=10.0, subs="all")
            )
            axis.xaxis.set_minor_formatter(
                matplotlib.ticker.LogFormatter(base=10.0, labelOnlyBase=False)
            )
            axis.grid(axis="x", which="both", linestyle=":", linewidth=0.3)
            axis.set_xticks(
                [20, 40, 60, 100, 200, 400, 600, 1000, 2000, 4000, 6000, 10000, 20000]
            )
            axis.set_xticklabels(
                [20, 40, 60, 100, 200, 400, 600, "1k", "2k", "4k", "6k", "10k", "20k"]
            )
            axis.set_xlim(1e3, xlim[1])

        axis.grid(axis="y", linestyle=":", linewidth=0.3)
        axis.legend(loc="upper right", fontsize=7)

        plt.show()
        return fig


    def time_freq(
            self,
            *,
            max_plots: int | None = None,
            title: str | None = None,
    ):
        """
        Plot all binaural impulse responses into a single figure using
        pyfar.plot.time_freq.

        Intended for debugging / inspection (not publication).

        Parameters
        ----------
        max_plots : int or None
            Optional limit on number of IRs to plot.
        title : str or None
            Optional figure title.
        """

        import matplotlib.pyplot as plt
        import pyfar

        keys = list(self.data.keys())
        if max_plots is not None:
            keys = keys[:max_plots]

        if len(keys) == 0:
            raise ValueError("No impulse responses to plot.")

        fig = plt.figure(figsize=(8, 10))

        for i, key in enumerate(keys):
            sig = self.data[key]

            if sig.cshape != (2,):
                raise ValueError(
                    f"{key}: expected binaural pyfar.Signal with cshape (2,), "
                    f"got {sig.cshape}"
                )

            ax_t, ax_f = pyfar.plot.time_freq(
                sig,
                figure=fig)

            # Label ears only once
            if i == 0:
                ax_t.get_lines()[0].set_label("Left")
                ax_t.get_lines()[1].set_label("Right")
                ax_t.legend(fontsize=8)
                ax_f.legend(fontsize=8)

        if title is not None:
            fig.suptitle(title)

        plt.show()
        return fig


# =====================================================================
# pyfar <-> slab helpers
# =====================================================================

def _irs_to_pyfar(irs: ImpulseResponses):
    fs = int(irs.params["fs"])
    keys = list(irs.data.keys())
    data = pyfar.Signal([irs[k].time for k in keys], sampling_rate=fs)
    sources = irs.get_sources()
    coords = pyfar.Coordinates(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        domain="sph",
        convention="top_elev",
        unit="deg",
    )
    # pyfar 0.8 dropped this constructor: it raises
    #   TypeError: Coordinates.__init__() got an unexpected keyword argument 'domain'
    # RESOLVED 2026-08-19 by pinning `pyfar>=0.7.5,<0.8` in pyproject.toml
    # rather than porting, because 0.7.5 is not merely a version that works --
    # it is the version the SOFAs on disk were built with. Rebuilding
    # Kemar_reseated_2 from raw sweeps reproduces the shipped file's magnitude
    # to 0.000000 dB rms under 0.7.5 and 0.001204 dB rms under 0.6.8.
    # If you ever do port this to the 0.8 API, treat it as a change to measured
    # data: rebuild an existing subject and diff before trusting it.
    return data, coords, keys, fs


def _pyfar_to_irs(template: ImpulseResponses, keys, time_data, fs):
    out = copy.deepcopy(template)
    for key, td in zip(keys, time_data):
        out.data[key] = pyfar.Signal(td, fs)
    return out

def _interaural_delay_s(freq_bin, freqs, band=(200.0, 1500.0)):
    """Broadband interaural delay of one binaural spectrum, in seconds.

    Taken from the slope of the unwrapped interaural phase difference over
    `band`: tau = -d(IPD)/d(omega). Positive means the right ear is later.

    Deterministic and fractional -- unlike an onset threshold, it does not
    depend on the magnitude spectrum, so a magnitude-only edit (the donor
    composite, the monaural envelope) cannot change it.
    """
    ipd = numpy.unwrap(numpy.angle(freq_bin[1] * numpy.conj(freq_bin[0])))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return -numpy.polyfit(2 * numpy.pi * freqs[mask], ipd[mask], 1)[0]


def zero_frontal_interaural(
    irs: "ImpulseResponses",
    *,
    align_itd: bool = True,
    align_ild: bool = True,
    ild_band: tuple[float, float] = (200.0, 16000.0),
    itd_band: tuple[float, float] = (200.0, 1500.0),
    frontal_az_deg: float = 0.0,
    frontal_tol_deg: float = 1e-6,
) -> "ImpulseResponses":
    """Zero ITD and ILD on the frontal arc, AFTER reference equalization.

    This must run on the equalized DTF, never on the subject and the reference
    IRs separately before dividing them (which is what `time_align_irs`
    align_itd/align_ild did until 2026-08-18). A broadband level match is an
    energy measure and does not commute with the per-frequency division in
    `equalize`: rms(S/R) != rms(S)/rms(R) unless |R| is flat. Normalising both
    operands first therefore does not zero the quotient -- it discards the
    cancellation the division would have performed on the reference's own
    channel imbalance and leaves a residual in its place. Measured on NW
    against ref_03.04 that residual was a flat -1.35 dB across 200 Hz-16 kHz,
    with the same sign for every subject sharing a reference, and it accounted
    for AS's +5.6 deg rightward localization bias. Done after the division the
    criterion is satisfied exactly, by construction.

    ILD criterion: broadband ENERGY interaural level difference over
    `ild_band`, removed with an energy-preserving pair of gains. Energy rather
    than a per-octave weighted log-mean because the two disagree by ~0.7 dB and
    the one behavioural anchor available picks energy: AS's midline offset
    predicts +5.93 deg of bias under the energy measure and +8.82 deg under the
    log-mean (ILD-vs-azimuth slopes 0.212 vs 0.220 dB/deg, so the disagreement
    is in the intercept, not the units), against +5.62 deg measured.
    Lateralization of a broadband stimulus integrates energy across frequency.
    ONE SCALAR PER DIRECTION, deliberately -- a per-frequency correction
    would put both ears on the same magnitude spectrum and destroy the
    elevation-dependent interaural spectral difference, which is real anatomy
    and measures larger (SD 2.6-3.6 dB) than the elevation-invariant part it
    would be removing.

    ITD criterion: interaural phase slope over `itd_band`, removed as an exact
    fractional frequency-domain delay -- same estimator, band and sign
    convention as `expand_azimuths_with_binaural_cues` step 3a, which repeats
    this operation when `itd_method='phase'`. The repeat is a no-op, and doing
    it here means the frontal ITD is also zero when `expand_az=False` or under
    the legacy `itd_method='onset'`.
    """
    eps = 1e-30

    def _is_frontal(key: str) -> bool:
        try:
            az = float(key.split("_")[1])
        except (IndexError, ValueError):
            return False
        az = float(numpy.mod(az, 360.0))
        return (abs(az - frontal_az_deg) <= frontal_tol_deg
                or abs(az - 360.0) <= frontal_tol_deg)

    out = copy.deepcopy(irs)
    n_frontal = 0

    for key in list(out.data.keys()):
        if not _is_frontal(key):
            continue
        sig = out.data[key]
        freqs = sig.frequencies
        spectrum = sig.freq.copy()

        if align_itd:
            tau = _interaural_delay_s(spectrum, freqs, band=itd_band)
            spectrum[1] *= numpy.exp(1j * 2 * numpy.pi * freqs * tau)

        if align_ild:
            band = (freqs >= ild_band[0]) & (freqs <= ild_band[1])
            if not band.any():
                raise ValueError(f"ild_band {ild_band} selects no frequency bins")
            ild_db = 10 * numpy.log10(
                (numpy.mean(numpy.abs(spectrum[0, band]) ** 2) + eps)
                / (numpy.mean(numpy.abs(spectrum[1, band]) ** 2) + eps))

            ratio = 10 ** (-ild_db / 20.0)             # = gain_L / gain_R
            p_l = float(numpy.mean(numpy.abs(spectrum[0]) ** 2))
            p_r = float(numpy.mean(numpy.abs(spectrum[1]) ** 2))
            gain_r = numpy.sqrt((p_l + p_r) / (ratio ** 2 * p_l + p_r))
            spectrum[0] *= ratio * gain_r              # total power preserved
            spectrum[1] *= gain_r

        # DC and Nyquist must stay real for a real impulse response
        spectrum[..., 0] = numpy.abs(spectrum[..., 0])
        if sig.n_samples % 2 == 0:
            spectrum[..., -1] = numpy.abs(spectrum[..., -1])

        sig.freq = spectrum
        n_frontal += 1

    if n_frontal == 0:
        logging.warning("zero_frontal_interaural: no frontal (az=0) directions found.")
    else:
        logging.info("zero_frontal_interaural: %d frontal directions zeroed "
                     "(itd=%s, ild=%s).", n_frontal, align_itd, align_ild)

    out.params.setdefault("frontal_interaural", {}).update({
        "align_itd": bool(align_itd),
        "align_ild": bool(align_ild),
        "ild_band_hz": list(ild_band),
        "ild_criterion": "broadband energy ILD, energy preserving",
        "itd_band_hz": list(itd_band),
        "itd_criterion": "interaural phase slope, fractional delay",
        "applied": "post-equalization",
        "n_frontal": int(n_frontal),
        "date": datetime.now().isoformat(),
    })
    return out


def expand_azimuths_with_binaural_cues(
    hrir,
    az_range: tuple[float, float] = (-50, 50),
    head_radius: float | None = None,
    onset_threshold_db: float = 15.0,
    show: bool = False,
    probe_az: float = 45.0,
    itd_method: str = "phase",
    itd_band: tuple[float, float] = (200.0, 1500.0),
):
    """
    Extend frontal vertical-arc HRIRs across azimuth and impose binaural cues.

    Processing steps
    ----------------
    1) Duplicate the measured frontal vertical arc across an azimuth grid.
    2) Apply spherical-head spectral head shadow as a *relative* magnitude
       correction with respect to the frontal direction at the same elevation.
       This preserves the recorded left/right spectral detail while avoiding
       a level discontinuity between frontal and off-frontal directions.
    3) Impose spherical-head ITDs (see `itd_method`).

    Parameters
    ----------
    hrir : ImpulseResponses
        Binaural input HRIRs on a frontal vertical arc.
    az_range : (float, float)
        Azimuth range in degrees for expansion, e.g. (-50, 50).
    head_radius : float or None
        Sphere radius in meters. If None, the spherical-head default is used.
    onset_threshold_db : float
        Threshold for pyfar onset detection. Only used by ``itd_method='onset'``.
    show : bool
        If True, show a diagnostic time/frequency plot at the azimuth nearest
        to `probe_az`.
    probe_az : float
        Azimuth used for the optional diagnostic plot.
    itd_method : {'phase', 'onset'}, default 'phase'
        How the spherical-head ITD is imposed.

        ``'phase'``
            Takes the model's interaural PHASE difference relative to frontal
            at the same elevation -- the exact phase analogue of the magnitude
            shaping in step 2 -- and applies it in the frequency domain. Exact,
            fractional, frequency-dependent, and a function of geometry alone,
            so a magnitude-only modification of the input cannot perturb it and
            native/modified sets share their ITD bit-for-bit. The measured
            midline ITD is still zeroed, but from the interaural phase slope
            rather than the onset.

        ``'onset'``
            Legacy, for reproducing SOFAs built before 2026-08. Matches the
            difference of the two ears' 15 dB onsets and applies it as an
            integer-sample cyclic roll. Four measured costs: the onset detector
            runs on the *measured* signal, so re-running the expansion on the
            same data lands up to 5 samples away; the roll quantises to 1/fs
            (20.5 us at 48828); a single broadband delay cannot carry the
            model's frequency dependence, so the low-frequency ITD comes out
            ~15% short (405 vs 478 us at az 50 for a 0.0875 m head); and the
            midline "zeroing" leaves ~108 us of interaural phase delay behind
            while reading 0.0 us by its own onset metric.
    itd_band : (float, float)
        Band used for the midline interaural-delay estimate under
        ``itd_method='phase'``.

    Returns
    -------
    ImpulseResponses
        Expanded HRIR set with relative spherical-head magnitude shaping and
        model-based ITD alignment.
    """

    # ------------------------------------------------------------------
    # STEP 1: AZIMUTH EXPANSION
    # ------------------------------------------------------------------
    sources0 = hrir.get_sources()  # [az, el, r]
    elevations = numpy.unique(sources0[:, 1])

    if len(elevations) > 1:
        vertical_res = float(numpy.mean(numpy.diff(numpy.sort(elevations))))
    else:
        vertical_res = az_range[1] - az_range[0]

    azimuths = numpy.arange(az_range[0], az_range[1] + vertical_res / 2, vertical_res)
    azimuths_wrapped = _wrap_az_deg_ccw(azimuths)

    out = copy.deepcopy(hrir)
    new_entries = {}

    for key in hrir.data.keys():
        spk, _az_str, el_str = key.split("_")
        for az_w in azimuths_wrapped:
            az_canonical = float(numpy.mod(az_w, 360.0))

            # keep the original frontal responses; do not duplicate 0/360
            if (
                numpy.isclose(az_canonical, 0.0, atol=1e-6)
                or numpy.isclose(az_canonical, 360.0, atol=1e-6)
            ):
                continue

            az_s = f"{az_canonical:.1f}"
            new_key = f"{spk}_{az_s}_{el_str}"

            if new_key not in out.data and new_key not in new_entries:
                new_entries[new_key] = copy.deepcopy(out.data[key])

    out.data.update(new_entries)

    try:
        from collections import OrderedDict

        def _parse_key_triple(k):
            spk, az_s, el_s = k.split("_")
            return (float(az_s), float(el_s), spk)

        out.data = OrderedDict(sorted(out.data.items(), key=lambda kv: _parse_key_triple(kv[0])))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # STEP 2: SPHERICAL-HEAD MAGNITUDE SHAPING RELATIVE TO FRONTAL
    # ------------------------------------------------------------------
    # Key idea:
    # The measured HRIRs are frontal recordings across elevation.
    # Therefore, for a synthesized off-frontal direction, we do not want to
    # impose the *absolute* spherical-head magnitude, because that would create
    # an overall gain discontinuity between frontal and lateral directions.
    #
    # Instead, for each ear separately, we apply only the spherical-head
    # directional change relative to the frontal direction at the same elevation:
    #
    #   cL(az, el, f) = |H_head_L(az, el, f)| / |H_head_L(0, el, f)|
    #   cR(az, el, f) = |H_head_R(az, el, f)| / |H_head_R(0, el, f)|
    #
    #   |H_new_L| = |H_meas_L| * cL
    #   |H_new_R| = |H_meas_R| * cR
    #
    # This preserves the recorded left/right spectral detail and keeps the
    # frontal response as the baseline.
    # ------------------------------------------------------------------
    hrir_pf, coords, keys, fs = _irs_to_pyfar(out)
    shtf = _spherical_head_for(coords, hrir_pf.n_samples, fs, head_radius)

    H_meas = hrir_pf.freq
    mag_meas = numpy.abs(H_meas)
    phase_meas = numpy.angle(H_meas)
    mag_head = numpy.abs(shtf.freq)

    n_pos, n_ears, _ = mag_meas.shape
    if n_ears != 2:
        raise ValueError("Binaural data expected (2 ears).")

    sources = out.get_sources()
    az_all = sources[:, 0]
    el_all = sources[:, 1]

    mag_new = mag_meas.copy()
    eps = 1e-12

    for i in range(n_pos):
        az = float(az_all[i])
        el = float(el_all[i])

        # Keep frontal directions unchanged
        if numpy.isclose(numpy.mod(az, 360.0), 0.0, atol=1e-6):
            continue

        # Find frontal reference at same elevation
        frontal_candidates = numpy.where(
            numpy.isclose(numpy.mod(az_all, 360.0), 0.0, atol=1e-6)
            & numpy.isclose(el_all, el, atol=1e-6)
        )[0]

        if len(frontal_candidates) == 0:
            raise ValueError(
                f"No frontal reference found for elevation {el:.6f} deg."
            )

        i0 = int(frontal_candidates[0])

        # Relative spherical-head correction w.r.t. frontal at same elevation
        cL = mag_head[i, 0, :] / numpy.maximum(mag_head[i0, 0, :], eps)
        cR = mag_head[i, 1, :] / numpy.maximum(mag_head[i0, 1, :], eps)

        # Apply ear-by-ear, preserving recorded monaural spectral detail
        mag_new[i, 0, :] = mag_meas[i, 0, :] * cL
        mag_new[i, 1, :] = mag_meas[i, 1, :] * cR

    H_new = mag_new * numpy.exp(1j * phase_meas)
    hrir_pf.freq = H_new

    # ------------------------------------------------------------------
    # STEP 3: ITD
    # ------------------------------------------------------------------
    if itd_method not in ("onset", "phase"):
        raise ValueError(f"itd_method must be 'onset' or 'phase', got {itd_method!r}")

    if itd_method == "onset":
        _ = hrir_pf.time
        _ = shtf.time

        on_mod = pyfar.dsp.find_impulse_response_start(shtf, threshold=onset_threshold_db)
        on_mea = pyfar.dsp.find_impulse_response_start(hrir_pf, threshold=onset_threshold_db)

        time_data = hrir_pf.time
        out_time = numpy.empty_like(time_data)

        for i in range(time_data.shape[0]):
            itd_model = (on_mod[i, 1] - on_mod[i, 0]) / fs
            itd_meas = (on_mea[i, 1] - on_mea[i, 0]) / fs
            delta_itd = itd_model - itd_meas

            out_time[i, 0, :] = time_data[i, 0, :]

            sig_r = pyfar.Signal(time_data[i, 1:2, :], fs)
            sig_rs = pyfar.dsp.time_shift(sig_r, delta_itd, unit="s")
            out_time[i, 1, :] = sig_rs.time[0]

    else:  # itd_method == 'phase'
        # Applied on the N-point spectrum, i.e. as a circular shift. Zero-padding
        # to 2N and truncating would make it linear, but truncation discards the
        # part of the response that moves past the buffer and that costs the
        # exactness this method exists for -- tried and measured: midline ITD
        # 10.4 us instead of 0, and the operation stops being idempotent. The
        # circular form is exact in ITD, magnitude and repeat application.
        #
        # What it costs: on a MODIFIED set ~0.5% of the energy wraps to the
        # buffer edges. Note the modified arc already carries ~1.3% there before
        # this step runs -- a magnitude-only edit that keeps the original phase
        # is not compact in time -- so this is a small addition to something
        # that is already true of every build. See the guard below.
        freqs = hrir_pf.frequencies
        spectrum = hrir_pf.freq.copy()

        # frontal reference index per elevation (built once; step 2 resolves the
        # same thing per direction)
        frontal_of = {}
        for i in range(n_pos):
            if numpy.isclose(numpy.mod(az_all[i], 360.0), 0.0, atol=1e-6):
                frontal_of[round(float(el_all[i]), 6)] = i

        # 3a) zero the measured midline ITD, from the interaural phase slope.
        #     One pure delay per elevation, removed from the right ear of that
        #     elevation's whole azimuth column (they are all copies of it).
        for elevation, i0 in frontal_of.items():
            tau = _interaural_delay_s(spectrum[i0], freqs, band=itd_band)
            advance = numpy.exp(1j * 2 * numpy.pi * freqs * tau)
            for i in range(n_pos):
                if round(float(el_all[i]), 6) == elevation:
                    spectrum[i, 1, :] *= advance

        # 3b) impose the model's interaural phase, relative to frontal at the
        #     same elevation. Zero at az=0 by symmetry, so 3a survives intact.
        #     The model is sampled at n_out; the IPD is smooth in frequency, so
        #     resample it onto the padded grid rather than recomputing the SHTF.
        ipd_model = numpy.unwrap(
            numpy.angle(shtf.freq[:, 1, :] * numpy.conj(shtf.freq[:, 0, :])), axis=-1)
        model_freqs = shtf.frequencies
        for i in range(n_pos):
            i0 = frontal_of[round(float(el_all[i]), 6)]
            delta = numpy.interp(freqs, model_freqs, ipd_model[i] - ipd_model[i0])
            spectrum[i, 1, :] *= numpy.exp(1j * delta)

        # DC and Nyquist must stay real for a real impulse response. Rotating
        # them and letting the inverse transform discard the imaginary part
        # would change their MAGNITUDE, which this step must not do.
        spectrum[..., 0] = numpy.abs(spectrum[..., 0])
        if hrir_pf.n_samples % 2 == 0:
            spectrum[..., -1] = numpy.abs(spectrum[..., -1])

        def _edge_fraction(x):
            energy = x ** 2
            return ((energy[..., -16:].sum(axis=-1) + energy[..., :8].sum(axis=-1))
                    / numpy.maximum(energy.sum(axis=-1), 1e-30))

        before = _edge_fraction(hrir_pf.time)
        hrir_pf.freq = spectrum
        out_time = hrir_pf.time.copy()

        # Report only the energy THIS step wraps, not the absolute edge content:
        # the input already carries ~1.3% there on a modified set, which is a
        # property of the magnitude-only edit and not of the ITD.
        growth = (_edge_fraction(out_time) - before).max()
        if growth > 1e-2:
            logging.warning(
                "expand_azimuths: the fractional ITD wrapped %.2f%% of the energy "
                "to the buffer edges at the worst direction. Increase "
                "desired_onset_s or n_samples_out.", 100 * growth)


    # ------------------------------------------------------------------
    # OPTIONAL DIAGNOSTIC PLOT
    # ------------------------------------------------------------------
    if show:
        idx = int(numpy.argmin(numpy.abs(az_all - float(probe_az))))
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 10))
        ax_t, ax_f = pyfar.plot.time_freq(pyfar.Signal(out_time[idx], fs))
        ax_t.get_lines()[0].set_label("left")
        ax_t.get_lines()[1].set_label("right")
        ax_t.legend()
        ax_f.get_lines()[0].set_label("left")
        ax_f.get_lines()[1].set_label("right")
        ax_f.legend()
        ax_t.set_title("time")
        ax_f.set_title("magnitude")
        plt.suptitle(f"Result @ az≈{az_all[idx]:.1f}°")
        plt.show()

    # ------------------------------------------------------------------
    # RETURN
    # ------------------------------------------------------------------
    out_final = _pyfar_to_irs(out, keys, out_time, fs)
    out_final.params.setdefault("processing", {})
    out_final.params["processing"]["expand_azimuths_with_binaural_cues"] = {
        "az_range": [float(az_range[0]), float(az_range[1])],
        "head_radius": float(head_radius) if head_radius is not None else None,
        "onset_threshold_db": float(onset_threshold_db),
        "magnitude_shaping": "relative_to_frontal_same_elevation",
        "itd_method": str(itd_method),
        "itd_band": [float(itd_band[0]), float(itd_band[1])],
        "date": datetime.now().isoformat(),
    }
    return out_final

# The previous absolute-ILD version of expand_azimuths_with_binaural_cues used
# to sit here, commented out (217 lines). Removed 2026-08-19 -- it is in git
# history if it is ever needed; the live version imposes ILD relative to the
# frontal direction instead.

def _wrap_az_deg_ccw(az):
    """Wrap azimuth(s) to [0, 360) with CCW-positive (pyfar 'sph/top_elev')."""
    az = numpy.asarray(az, dtype=float)
    az = numpy.mod(az, 360.0)
    az[az < 0] += 360.0
    return az


def validate_source_grid(sources, tol_deg=1e-3):
    """
    Assert the source grid obeys the module's single azimuth convention.

    See the "Azimuth convention" section of the module docstring. Checks, in
    order of how much damage each would do downstream:

    1. azimuth in [0, 360) -- a negative azimuth means an unwrapped measured key
       leaked past `expand_azimuths_with_binaural_cues`, so the SOFA would carry
       two conventions at once and `hrir2mat.frontal_index` (which folds
       ``az > 180`` to negative) would mislabel straight ahead.
    2. no duplicate (az, el) within `tol_deg` -- most likely the frontal column
       emitted twice, once as the measured 0 and once as a wrapped 360. Two
       filters at one direction is silently wasteful in pyBinSim and makes
       "the frontal DTF" ambiguous for every analysis that looks one up.
    3. elevation in [-90, 90].

    Parameters
    ----------
    sources : array_like, shape (n, 3)
        [azimuth, elevation, distance] rows, degrees.
    tol_deg : float
        Two directions closer than this in BOTH coordinates count as duplicates.

    Raises
    ------
    ValueError
        On any violation, naming the offending directions.
    """
    src = numpy.asarray(sources, dtype=float)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"validate_source_grid: expected (n, 3) [az, el, r], got {src.shape}")

    az, el = src[:, 0], src[:, 1]

    bad = numpy.flatnonzero((az < 0.0) | (az >= 360.0))
    if bad.size:
        raise ValueError(
            "validate_source_grid: azimuth must be in [0, 360) CCW-positive "
            f"(0=front, 90=left); {bad.size} source(s) violate it, e.g. "
            f"{[float(a) for a in az[bad[:5]]]}. An unwrapped measured azimuth "
            "has leaked through -- wrap it with _wrap_az_deg_ccw before the "
            "grid is assembled, do not fix it downstream.")

    bad = numpy.flatnonzero(numpy.abs(el) > 90.0)
    if bad.size:
        raise ValueError(
            "validate_source_grid: elevation must be in [-90, 90]; "
            f"offending values {[float(e) for e in el[bad[:5]]]}")

    # duplicates: quantize to the tolerance so 0.0 and 360.0-after-wrap collide
    quantized = numpy.round(src[:, :2] / max(tol_deg, 1e-12)).astype(numpy.int64)
    _, first, counts = numpy.unique(
        quantized, axis=0, return_index=True, return_counts=True)
    if numpy.any(counts > 1):
        dupes = [(float(az[i]), float(el[i]))
                 for i in first[counts > 1][:5]]
        raise ValueError(
            f"validate_source_grid: {int((counts - 1).sum())} duplicate "
            f"direction(s), e.g. (az, el) = {dupes}. The usual cause is the "
            "frontal column being emitted twice -- once as the measured 0 and "
            "once as a wrapped 360.")

    return src
