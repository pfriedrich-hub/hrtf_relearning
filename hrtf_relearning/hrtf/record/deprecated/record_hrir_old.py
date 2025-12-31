import matplotlib
matplotlib.use('tkagg')
import logging
from pathlib import Path
import copy
from datetime import datetime
from pynput import keyboard
import numpy
import freefield
import pyfar
import warnings
import slab
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)


# -------------------------------------------------------------------------
# Global settings
# -------------------------------------------------------------------------
subject_id = 'kemar_test'
overwrite = False
reference = 'kemar_reference'
n_directions = 1
n_recordings = 20
fs = 48828  # 97656
show = True
n_samples_out = 256

slab.set_default_samplerate(fs)
freefield.set_logger("info")



# -------------------------------------------------------------------------
# High-level: record HRIR for one subject
# -------------------------------------------------------------------------
def record_hrir(
    subject_id: str, reference: str, n_directions: int = 5, n_recordings: int = 5,
    fs: int = fs, overwrite: bool = False, n_samples_out: int = 256, show: bool = True):

    hrtf_dir = Path.cwd() / "data" / "hrtf"
    subject_dir = hrtf_dir / "rec" / subject_id
    ref_dir = hrtf_dir / "rec" / "reference" / reference

    # 1) in ear recordings
    if (not subject_dir.exists()) or overwrite:
        subject_dir.mkdir(exist_ok=True, parents=True)
        # todo make sure we record 2 channels
        ear_pressure = Recordings.record_dome(n_directions, n_recordings, hp_freq=120, fs=fs)
        ear_pressure.params["subject_id"] = subject_id
        ear_pressure.to_wav(subject_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject recordings from wav directory")
        ear_pressure = Recordings.from_wav(subject_dir)

    # 2) reference recordings
    if (not ref_dir.exists()) or overwrite:
        logging.info("Recording new reference")
        ref_dir.mkdir(exist_ok=True, parents=True)
        reference_pressure = Recordings.record_dome(n_directions=1, n_recordings=n_recordings, hp_freq=120, fs=fs)
        reference_pressure.params["subject_id"] = reference
        reference_pressure.to_wav(ref_dir, overwrite=overwrite)
    else:
        logging.info("Loading reference recordings")
        reference_pressure = Recordings.from_wav(ref_dir)

    # plot spectra
    # ear_pressure.plot_spectra()
    # reference_pressure.plot_spectra()

    # 3) Compute TF (deconvolve recordings with inverted excitation signal)
    # tf_recorded = ear_pressure.compute_tf(n_samp_out=n_samples_out, show=show)
    # tf_reference = reference_pressure.compute_tf(n_samp_out=n_samples_out, show=show)

    # 4) Equalize HRIR by reference IR
    hrir = equalize(ear_pressure, reference_pressure, n_samples_out=n_samples_out, show=show)
    # hrir = equalize(hrir_recorded, hrir_reference, n_samp_out=n_samp_out, show=show)
    # hrir = equalize_old(ear_pressure, reference_pressure, n_samp_out, show=show)

    # 5) Low frequency extrapolation
    hrir = lowfreq_extrapolate(hrir, f_extrap=400.0, f_target=150.0, head_radius=0.0875, show=False)

    # 6) Extend azimuths + add binaural cues (ILD full-band off-midline + ITD align)
    hrir = expand_azimuths_with_binaural_cues(hrir, az_range=(-50, 50), head_radius=0.0875, show=False)

    # 5) Export to slab.HRTF
    hrir = hrir.to_slab_hrtf(datatype="FIR")
    if show:
        fig, ax = plt.subplots()
        hrir.plot_tf(hrir.cone_sources(0), axis=ax)
    return hrir


# -------------------------------------------------------------------------
# Base grid container
# -------------------------------------------------------------------------
class SpeakerGridBase:
    """
    Base container for data defined on the loudspeaker grid.
    Keys are strings like '19_0.0_40.0' → (idx, az, el).
    Values are typically slab.Binaural or slab.Filter.
    """

    def __init__(self, data=None, params=None):
        self.data: dict[str, object] = data or {}
        self.params: dict[str, object] = params or {}

    # --- dict-like ---------------------------------------------------------
    def __getitem__(self, key):
        """
        Access entries in the grid.

        - grid['19_0.0_40.0'] -> value for that key (dict-style)
        - grid[0]             -> first value in insertion order
        - grid[-1]            -> last value
        - grid[0:3]           -> list of first three values (values only)
        """
        # int index -> single value
        if isinstance(key, int):
            values = list(self.data.values())
            try:
                return values[key]  # supports negative indices as usual
            except IndexError as exc:
                raise IndexError(
                    f"Index {key} out of range for SpeakerGridBase with "
                    f"{len(values)} elements."
                ) from exc

        # slice -> list of values
        if isinstance(key, slice):
            values = list(self.data.values())
            return values[key]

        # everything else -> behave like a normal dict
        return self.data[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __len__(self):
        return len(self.data)

    # --- key helpers -------------------------------------------------------
    @staticmethod
    def parse_key(key: str) -> tuple[int, float, float]:
        parts = key.split("_")
        if len(parts) != 3:
            raise ValueError(f"Cannot parse key '{key}', expected 'idx_az_el'")
        idx = int(parts[0])
        az = float(parts[1])
        el = float(parts[2])
        return idx, az, el

    # --- spatial helpers ---------------------------------------------------
    def get_sources(self, distance: float = 1.4) -> numpy.ndarray:
        coords = []
        for key in self.data.keys():
            _, az, el = self.parse_key(key)
            coords.append([az, el, distance])
        return numpy.array(coords, dtype="float16")

    def to_wav(self, path: Path | str, overwrite: bool = False) -> None:
        """
        Write all entries in self.data to WAV files.

        Parameters
        ----------
        path : base directory for saving the wav files
        overwrite : if False, existing wav files are kept and skipped.
                    if True, existing wav files are overwritten.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.write_params_file(path)  # write recording parameters
        for key, obj in self.data.items():
            if not hasattr(obj, "write"):
                raise TypeError(f"Object for key {key} has no `.write` method: {type(obj)}")
            fname = path / f"{key}.wav"
            if fname.exists() and not overwrite:
                logging.info(f"Skipping existing file (overwrite=False): {fname}")
                continue
            obj.write(fname)

    # --- parameter logging -------------------------------------------------
    def write_params_file(self, path: Path, filename: str = "params.txt") -> None:
        """
        Write all entries from self.params to a human-readable text file.
        """
        path.mkdir(parents=True, exist_ok=True)
        file = path / filename

        lines = []
        for key, val in self.params.items():
            if isinstance(val, dict):
                lines.append(f"{key}:")
                for sk, sv in val.items():
                    lines.append(f"  {sk}: {sv}")
            else:
                lines.append(f"{key}: {val}")

        lines += [
            "",
            "Software versions:",
            f"  slab:      {getattr(slab, '__version__', 'unknown')}",
            f"  pyfar:     {getattr(pyfar, '__version__', 'unknown')}",
            f"  freefield: {getattr(freefield, '__version__', 'unknown')}",
        ]

        with file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # --- selection ---------------------------------------------------------
    def select(
        self,
        azimuth: float | tuple[float, float] | None = None,
        elevation: float | tuple[float, float] | None = None,
    ) -> "SpeakerGridBase":
        if azimuth is None:
            az_low, az_high = -float("inf"), float("inf")
        elif isinstance(azimuth, (int, float)):
            az_low = az_high = float(azimuth)
        else:
            az_low, az_high = float(min(azimuth)), float(max(azimuth))

        if elevation is None:
            el_low, el_high = -float("inf"), float("inf")
        elif isinstance(elevation, (int, float)):
            el_low = el_high = float(elevation)
        else:
            el_low, el_high = float(min(elevation)), float(max(elevation))

        sub = {}
        for key, obj in self.data.items():
            _, az, el = self.parse_key(key)
            if az_low <= az <= az_high and el_low <= el <= el_high:
                sub[key] = obj

        # copy params reference
        return type(self)(data=sub, params=self.params.copy())


# -------------------------------------------------------------------------
# Recordings (binaural in-ear)
# -------------------------------------------------------------------------
class Recordings(SpeakerGridBase):
    """
    Binaural in-ear recordings over the speaker dome.
    Values are `slab.Binaural`.
    """

    def __init__(self, data=None, params=None, signal: slab.Sound | None = None):
        super().__init__(data=data, params=params)
        self.signal = signal

    @classmethod
    def record_dome(cls, n_directions=5, n_recordings=5, hp_freq=120, fs=48828):
        """Record across the dome and return a Recordings object with all parameters stored."""
        if freefield.PROCESSORS.mode != "play_birec":
            freefield.initialize("dome", "play_birec")

        # excitation signal  # todo 1 - adjust sweep parameters (freq range and ramp) to measure across 120-18 khz
        params = {"type": "slab.Sound.chirp", "duration": 0.2, "level": 85,
                  "from_frequency": 120, "to_frequency": fs/2, "samplerate": fs}
        # Orb Audio Mod1 frequency response: 120 Hz - 18 KHz, 5 ms ramp cuts off some frequencies
        signal = slab.Sound.chirp(duration=params["duration"], level=params["level"], samplerate=fs, kind='logarithmic',
                                  from_frequency=params["from_frequency"], to_frequency=params["to_frequency"])
        signal = signal.ramp(when="both", duration=0.001)  # matches the ramp in bi_play_buf.rcx
        signal.params = params

        speakers_all = freefield.read_speaker_table()
        speakers = cls.get_speakers(speakers_all, azimuth=0, elevation=(50, -37.5))  # omit speaker at -50°
        if len(speakers) < 2:
            raise RuntimeError("Need at least two speakers to infer vertical resolution.")

        res = abs(speakers[0].elevation - speakers[1].elevation) / n_directions
        min_el = min(spk.elevation for spk in speakers)

        recordings_dict = {}
        filt = slab.Filter.band(kind="hp", frequency=hp_freq, samplerate=fs)
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED

        for n in range(n_directions):
            elevation_step = n * res
            if n_directions > 1: # skip for reference recordings
                freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                                processors=led_speaker.digital_proc)  # illuminate LED
                input(f"Press Enter when head is at {0 + elevation_step}° elevation ...")
            for base_spk in speakers:
                spk = copy.deepcopy(base_spk)
                spk.elevation -= elevation_step
                if spk.elevation >= min_el:
                    logging.info(f"Recording from Speaker {spk.index} at {spk.elevation:.1f}° elevation")
                    key = f"{spk.index}_{spk.azimuth}_{spk.elevation}"
                    rec = cls.record_speaker(spk, signal, n_recordings, fs*2)
                    rec.data -= numpy.mean(rec.data, axis=0)  # remove DC
                    rec = filt.apply(rec)  # highpass filter
                    recordings_dict[key] = rec
            freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
        params = {
            "fs": fs,
            "n_recordings": n_recordings,
            "n_directions": n_directions,
            "signal": getattr(signal, "params", {}),
            "highpass frequency": hp_freq,
            "datetime": datetime.now().isoformat(),
        }
        return cls(data=recordings_dict, params=params, signal=signal)

    # --- conversion to IRs --------------------------------------------------
    def compute_tf(self, n_samp_out: int = 256, show: bool = False) -> "ImpulseResponses":
        """
        Convert this set of recordings to impulse responses.

        Steps:
        1. Stack all binaural in-ear recordings into a multi channel pyfar.Signal
        2. Compute regularized inverse of the excitation signal
        3. Deconvolve (Y * X^{-1}) to obtain HRIRs
        4. Align group delay relative to a center loudspeaker
        5. Window and crop to a fixed length
        6. Convert to slab.Filter and return an ImpulseResponses object

        If show is True, a summary figure of all processing steps is plotted.
        """

        # ------------------------------------------------------------------
        # 0) Basic parameters
        # ------------------------------------------------------------------
        if "fs" in self.params:
            fs = int(self.params["fs"])
        else:
            from record_hrir_old import fs as fs_global
            fs = int(fs_global)

        if not self.data:
            raise ValueError("Recordings.compute_tf(): self.data is empty.")

        # ------------------------------------------------------------------
        # 1) Convert recordings -> pyfar.Signal
        # ------------------------------------------------------------------
        rec_array = [rec.data.T for rec in self.data.values()]  # (2, n_samples) per rec
        recording = pyfar.Signal(rec_array, fs)

        # ------------------------------------------------------------------
        # 2) Invert the excitation chirp
        # ------------------------------------------------------------------
        if self.signal is None:
            raise ValueError("Recordings.compute_tf(): self.signal is None, "
                             "cannot perform deconvolution.")
        sig = self.signal
        sig_params = self.params.get("signal", {})
        from_f = sig_params.get("from_frequency")
        to_f = sig_params.get("to_frequency")

        # filter excitation signal, if recordings where filtered
        if "highpass_frequency" in self.params:
            hp_freq = self.params.get('highpass_frequency')
            filt = slab.Filter.band(kind="hp", frequency=hp_freq, samplerate=fs)
            sig = filt.apply(sig)

        excitation = pyfar.Signal(sig.data.T, fs)  # (2, n_samples)
        ref_inv = pyfar.dsp.regularized_spectrum_inversion(
            excitation,
            (from_f, to_f),)

        # ------------------------------------------------------------------
        # 3) Deconvolve: HRIR = Y * X^{-1}
        # ------------------------------------------------------------------
        hrir_deconvolved = recording * ref_inv

        # ------------------------------------------------------------------
        # 4) Align group delay
        # ------------------------------------------------------------------
        hrir_shifted = pyfar.dsp.time_shift(hrir_deconvolved, 2000, unit="samples")

        center_key = "23_0.0_0.0"  # adjust to your actual center key
        keys_list = list(self.data.keys())
        if center_key in self.data:
            center_idx = keys_list.index(center_key)
        else:
            logging.warning(
                "Center key %s not found in recordings, using first entry (%s) instead.",
                center_key, keys_list[0]
            )
            center_idx = 0

        center_onset = pyfar.dsp.find_impulse_response_start(
            hrir_shifted[center_idx], threshold=15
        )
        center_onset = float(numpy.min(center_onset))
        center_onset_s = center_onset / hrir_shifted.sampling_rate

        desired_onset_s = 0.001
        shift_s = desired_onset_s - center_onset_s
        hrir_aligned = pyfar.dsp.time_shift(hrir_shifted, shift_s, unit="s")

        # ------------------------------------------------------------------
        # 5) Window around the earliest onset in the whole dataset
        # ------------------------------------------------------------------
        onsets = pyfar.dsp.find_impulse_response_start(hrir_aligned, threshold=10)
        onsets_min = numpy.min(onsets) / hrir_aligned.sampling_rate  # seconds

        times = (onsets_min - .00025,  # start of fade-in
                 onsets_min,  # end if fade-in
                 onsets_min + .0048,  # start of fade_out
                 onsets_min + .0058)  # end of_fade_out

        hrir_windowed, window = pyfar.dsp.time_window(
            hrir_aligned,
            times,
            "hann",
            unit="s",
            crop="end",
            return_window=True,
        )

        # cut to final length
        times_samples = [0, 10, 246, n_samp_out-1]
        hrir_final = pyfar.dsp.time_window(
            hrir_windowed,
            times_samples,
            "hann",
            crop="end",
        )
        # hrir_final = hrir_windowed

        # ------------------------------------------------------------------
        # 6) Optionally: plot processing steps
        # ------------------------------------------------------------------
        if show:
            _plot_processing_pyfar(
                excitation=excitation,
                ref_inv=ref_inv,
                recording=recording,
                hrir_deconvolved=hrir_deconvolved,
                hrir_shifted=hrir_shifted,
                hrir_aligned=hrir_aligned,
                hrir_windowed=hrir_windowed,
                hrir_final=hrir_final,
                window=window,
                center_idx=center_idx,
            )

        # ------------------------------------------------------------------
        # 7) Convert to slab.Filter and wrap in ImpulseResponses
        # ------------------------------------------------------------------
        params = {
            "fs": fs,
            "signal": self.params.get("signal", {}),
            "n_samp": n_samp_out,
            "date": datetime.now().isoformat(),
        }

        out = copy.deepcopy(self)
        for key, h in zip(out.data.keys(), hrir_final):
            out.data[key] = slab.Filter(
                data=h.time.T,
                samplerate=fs,
                fir="IR",
            )

        return ImpulseResponses(data=out.data, params=params)

    @staticmethod
    def record_speaker(speaker, signal: slab.Sound, n_recordings: int, fs: int) -> slab.Binaural:
        recordings = []
        for _ in range(n_recordings):
            recordings.append(
                freefield.play_and_record(
                    speaker=speaker,
                    sound=signal,
                    compensate_delay=True,
                    equalize=False,
                    recording_samplerate=fs,
                )
            )  # todo test this to make sure both channels are recorded!
        return slab.Binaural(numpy.mean(recordings, axis=0))

    @staticmethod
    def get_speakers(speakers, azimuth=None, elevation=None):
        out = []
        if azimuth is None:
            az_low, az_high = -float("inf"), float("inf")
        elif isinstance(azimuth, (int, float)):
            az_low = az_high = float(azimuth)
        else:
            az_low, az_high = float(min(azimuth)), float(max(azimuth))

        if elevation is None:
            el_low, el_high = -float("inf"), float("inf")
        elif isinstance(elevation, (int, float)):
            el_low = el_high = float(elevation)
        else:
            el_low, el_high = float(min(elevation)), float(max(elevation))

        for spk in speakers:
            if az_low <= spk.azimuth <= az_high and el_low <= spk.elevation <= el_high:
                out.append(spk)
        return out

        # --- I/O: always Binaural --------------------------------------

    @classmethod
    def from_wav(cls, path: Path | str):
        """
        Load binaural recordings from wav and params.txt into a Recordings object.
        Also reconstructs the excitation signal if params contain it.
        """
        path = Path(path)
        data: dict[str, slab.Binaural] = {}
        for wav in path.glob("*.wav"):
            data[wav.stem] = slab.Binaural(wav)
        if not data:
            raise FileNotFoundError(f"No .wav files found in {path}")
        data = dict(sorted(data.items(), key=lambda kv: (float(kv[0].rsplit("_", 2)[-1]))))  # sort by elevation

        params = parse_params_file(path)
        signal = None
        sig_params = params.get("signal", {})
        if isinstance(sig_params, dict) and sig_params.get("type") == "slab.Sound.chirp":
            dur = sig_params.get("duration", 0.2)
            level = sig_params.get("level", 90)
            fs_sig = sig_params.get("samplerate", fs)  # fall back to global fs
            f_start = sig_params.get("from_frequency", 50)
            f_end = sig_params.get("to_frequency", fs_sig / 2)

            signal = slab.Sound.chirp(
                duration=dur,
                level=level,
                samplerate=fs_sig,
                from_frequency=f_start,
                to_frequency=f_end,
            )
            # if you always ramped originally, you can ramp again here:
            signal = signal.ramp(when="both", duration=0.01)
            signal.params = sig_params

        return cls(data=data, params=params, signal=signal)

    # --- plotting of spectra -----------------------------------------------
    def spectrum(
            self,
            azimuth=0,
            linesep=20,
            xscale="linear",
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
            rec = self.data[key]  # slab.Binaural

            # left ear
            Hl, freqs = rec.channel(0).spectrum(show=False)
            # right ear
            Hr, _ = rec.channel(1).spectrum(show=False)

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

        # Format frequency labels as kHz
        axis.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, pos: f"{int(x / 1000)}")
        )
        axis.set_xlabel("Frequency [kHz]")

        axis.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.5)
        axis.legend(loc="upper right", fontsize=7)
        plt.show()
        return fig


# -------------------------------------------------------------------------
# Impulse Responses (slab.Filter grid)
# -------------------------------------------------------------------------
class ImpulseResponses(SpeakerGridBase):
    """
    Directional impulse responses over the speaker dome.
    Values are `slab.Filter` (FIR).
    """
    def __init__(self, data=None, params=None):
        super().__init__(data=data, params=params)

    def apply_spherical_head_lowfreq(
            self,
            f_extrap: float = 400.0,
            f_target: float = 150.0,
            head_radius: float | None = None,
            onset_threshold_db: float = 20.0,
            center_az: float = 0.0,
            az_tol: float = 1e-6,
    ) -> "ImpulseResponses":
        """
        Use a spherical head model as

        - low-frequency magnitude anchor (LF extrapolation, externalisation),
        - full-band ILD template for off-midline azimuths,
        - ITD target (via time shift of the whole IR).

        Pipeline inside this method:

        1) Convert current IRs (slab.Filter) -> pyfar.Signal (hrir_meas).
        2) Build pyfar.Coordinates from the source grid and compute spherical-
           head HRTFs (shtf) for the same positions.
        3) LOW-FREQ MAGNITUDE EXTRAPOLATION (notebook-style):
           - anchor magnitudes at 0 Hz and f_target from spherical head,
           - use measured magnitudes for f >= f_extrap,
           - interpolate smoothly over the full frequency axis,
           - keep measured phase.
        4) FULL-BAND ILD SHAPING (off-midline only):
           - for each non-midline azimuth, impose spherical-head ILD per
             frequency bin, preserving the measured average level.
        5) ITD ALIGNMENT:
           - compute ITD from spherical head and from the modified HRIRs,
           - time-shift the entire right-ear IR per position so ITD matches
             the spherical head across the band.
        6) Convert back to slab.Filter and store in self.data.

        Parameters
        ----------
        f_extrap
            Frequency (Hz) above which measured magnitude is trusted. Below
            this, magnitude is extrapolated between spherical head and
            measured data.
        f_target
            Frequency (Hz) at which the spherical-head magnitude is used as a
            second anchor for the low-frequency extrapolation.
        head_radius
            Optional head radius (meters) for spherical_head(). If None,
            the default head from spherical_head.py is used.
        onset_threshold_db
            Threshold in dB for onset detection when estimating ITD.
        center_az
            Azimuth (deg) considered the vertical midline (e.g. 0°).
        az_tol
            Tolerance (deg) when deciding if a source is on the midline.

        Returns
        -------
        self : ImpulseResponses
            Modified in-place and returned for chaining.
        """

        fs = int(self.params["fs"])

        # --- 1) Convert ImpulseResponses -> pyfar.Signal -------------------
        keys = list(self.data.keys())
        if not keys:
            raise ValueError("apply_spherical_head_lowfreq: no IRs in self.data")

        filters = [self.data[k] for k in keys]  # slab.Filter
        # slab.Filter.data: (n_samples, n_channels)
        # -> (n_pos, n_ears, n_samples)
        data = numpy.stack([filt.data.T for filt in filters], axis=0)
        hrir_meas = pyfar.Signal(data, fs)

        # --- 2) Build pyfar.Coordinates from your grid ---------------------
        # get_sources() should return [az, el, r] in degrees / meters
        sources = self.get_sources()  # shape (n_pos, 3)
        coords = pyfar.Coordinates(
            sources[:, 0],  # azimuth in deg
            sources[:, 1],  # elevation in deg
            sources[:, 2],  # radius in m
            domain="sph",
            convention = 'top_elev',
            unit="deg",
        )

        # --- 3) Spherical-head HRTFs for same positions --------------------
        shtf = _spherical_head_for(coords, hrir_meas.n_samples, fs, head_radius)

        # --- 4) Low-frequency magnitude extrapolation (notebook-style) -----
        freqs = hrir_meas.frequencies  # (n_freqs,)
        mag_meas = numpy.abs(hrir_meas.freq)
        phase_meas = numpy.angle(hrir_meas.freq)
        mag_sph = numpy.abs(shtf.freq)

        # frequencies we trust from the measurement
        mask_extrap = freqs >= f_extrap

        # frequency below which magnitude is assumed constant (second anchor)
        f_target_idx = hrir_meas.find_nearest_frequency(f_target)
        f_target = freqs[f_target_idx]  # snap to exact grid

        # valid (trusted) measurement magnitudes and freqs
        freqs_valid = freqs[mask_extrap]  # (n_valid,)
        mag_valid = mag_meas[..., mask_extrap]  # (n_pos, n_ears, n_valid)

        # spherical head magnitudes at anchors
        mag0 = mag_sph[..., 0:1]  # 0 Hz
        mag_ft = mag_sph[..., f_target_idx:f_target_idx + 1]  # f_target

        # concatenate along frequency axis: 0 Hz, f_target, f >= f_extrap
        mag_anchor = numpy.concatenate((mag0, mag_ft, mag_valid), axis=-1)
        freqs_anchor = numpy.concatenate((
            numpy.array([0.0]),
            numpy.array([f_target]),
            freqs_valid,
        ))

        # interpolate magnitude over full freq grid
        mag_interp = numpy.empty_like(hrir_meas.freq)
        for src in range(mag_anchor.shape[0]):
            for ear in range(mag_anchor.shape[1]):
                mag_interp[src, ear] = numpy.interp(
                    freqs,
                    freqs_anchor,
                    mag_anchor[src, ear],
                )

        # apply new magnitude, keep measured phase for now
        hrir_meas.freq = mag_interp * numpy.exp(1j * phase_meas)

        # --- 5) Full-band ILD shaping for off-midline azimuths ------------
        H_meas = hrir_meas.freq  # updated complex HRTFs
        mag_meas = numpy.abs(H_meas)  # updated magnitudes
        phase_meas = numpy.angle(H_meas)  # updated phases
        mag_head = mag_sph  # spherical-head magnitudes

        n_pos, n_ears, n_freqs = mag_meas.shape
        if n_ears != 2:
            raise ValueError("apply_spherical_head_lowfreq expects binaural data (2 ears).")

        az = sources[:, 0]
        is_midline = numpy.abs(az - center_az) <= az_tol

        mag_new = mag_meas.copy()

        for i in range(n_pos):
            if is_midline[i]:
                # keep original ILD on the vertical midline
                continue

            # spherical head ILD ratio R/L
            mag_head_L = mag_head[i, 0, :]
            mag_head_R = mag_head[i, 1, :]
            r = mag_head_R / numpy.maximum(mag_head_L, 1e-12)  # R/L

            # average magnitude from measured HRTF (power average)
            mag_L_meas = mag_meas[i, 0, :]
            mag_R_meas = mag_meas[i, 1, :]
            A = numpy.sqrt((mag_L_meas ** 2 + mag_R_meas ** 2) / 2.0)

            # new magnitudes: preserve A, enforce ILD ratio r
            mL = A * numpy.sqrt(2.0 / (1.0 + r ** 2))
            mR = r * mL

            mag_new[i, 0, :] = mL
            mag_new[i, 1, :] = mR

        # rebuild complex spectrum with new magnitudes and original phase
        H_new = mag_new * numpy.exp(1j * phase_meas)
        hrir_meas.freq = H_new

        # --- 6) ITD alignment: impose spherical-head ITD via time shift ----
        # force both into time domain (no iteration over Signal)
        _ = shtf.time
        _ = hrir_meas.time

        # onsets in samples: (n_pos, n_ears)
        onsets_model = pyfar.dsp.find_impulse_response_start(
            shtf, threshold=onset_threshold_db
        )
        onsets_meas = pyfar.dsp.find_impulse_response_start(
            hrir_meas, threshold=onset_threshold_db
        )

        # use numpy arrays for signal data to avoid domain-iteration issues
        time_data = hrir_meas.time  # (n_pos, n_ears, n_samples)
        time_data_shifted = numpy.empty_like(time_data)

        for i in range(time_data.shape[0]):  # over positions
            onset_m_L = onsets_model[i, 0]
            onset_m_R = onsets_model[i, 1]
            onset_h_L = onsets_meas[i, 0]
            onset_h_R = onsets_meas[i, 1]

            itd_model = (onset_m_R - onset_m_L) / fs
            itd_meas = (onset_h_R - onset_h_L) / fs
            delta_itd = itd_model - itd_meas

            # keep left ear fixed, shift right ear by delta_itd
            time_data_shifted[i, 0, :] = time_data[i, 0, :]

            sig_r = pyfar.Signal(time_data[i, 1:2, :], fs)  # (1, n_samples)
            sig_r_shifted = pyfar.dsp.time_shift(sig_r, delta_itd, unit="s")
            time_data_shifted[i, 1, :] = sig_r_shifted.time[0]

        # --- 7) Write back to self.data as slab.Filter --------------------
        for key, sig_time in zip(keys, time_data_shifted):
            # sig_time: (n_ears, n_samples) -> slab.Filter expects (n_samples, n_ears)
            self.data[key] = slab.Filter(
                data=sig_time.T,
                samplerate=fs,
                fir="IR",
            )

        # bookkeeping
        self.params.setdefault("spherical_head", {})
        self.params["spherical_head"].update(
            {
                "f_extrap": float(f_extrap),
                "f_target": float(f_target),
                "head_radius": float(head_radius) if head_radius is not None else None,
                "onset_threshold_db": float(onset_threshold_db),
                "center_az": float(center_az),
                "az_tol": float(az_tol),
                "date": datetime.now().isoformat(),
            }
        )

        return self

    # --- export to slab.HRTF ----------------------------------------------
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
        data = numpy.stack([self.data[k].data for k in keys], axis=0)

        # sources: (n_positions, 3) -> [az, el, r]
        # uses the same internal order as self.data, so things stay aligned
        sources = self.get_sources()

        hrir = slab.HRTF(
            data=data,
            sources=sources,
            samplerate=fs,
            datatype=datatype,
        )

        return hrir

    @classmethod
    def from_wav(cls, path: Path | str, fs: int | None = None, meta: dict | None = None):
        """
        Load IRs from WAV into slab.Filter objects.
        """
        path = Path(path)
        data: dict[str, slab.Filter] = {}
        for wav in path.glob("*.wav"):
            snd = slab.Binaural(wav)  # or slab.Sound
            data[wav.stem] = slab.Filter(data=snd.data, samplerate=snd.samplerate, fir="IR")
        if not data:
            raise FileNotFoundError(f"No .wav files found in {path}")
        if fs is None:
            first = next(iter(data.values()))
            fs = first.samplerate
        return cls(data=data, fs=fs, meta=meta or {})

    # --- plotting -----------------------------------------------------------
    def plot(self, kind: str = "tf"):
        """
        Plot TF or IRs offset by elevation.
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        for key, ir in self.data.items():
            _, _, el = self.parse_key(key)
            if kind.upper() == "TF":
                _ = ir.tf(show=True, axis=ax)
            else:
                ax.plot(ir.data)
            line = ax.lines[-1]
            x, y = line.get_xdata(), line.get_ydata()
            line.set_ydata(y + el)
            line.set_label(f"el={el:.1f}°")

        if kind.upper() == "TF":
            ax.set_xlim([2e3, 18e3])
            ax.set_xscale("linear")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylim(80, -80)
        else:
            ax.set_xlabel("Samples")

        ax.set_ylabel("Amplitude (offset by elevation)")
        ax.set_title(f"Impulse responses ({kind.upper()})")
        ax.legend(title="Elevation (°)")
        plt.show()

    # --- plotting of transfer functions -----------------------------------------------
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
            axis.set_xticklabels([20,40,60,100,200,400,600,'1k','2k','4k','6k','10k','20k'])
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


# -------------------------------------------------------------------------
# HRTF processing
# -------------------------------------------------------------------------
def equalize(
    recorded: ImpulseResponses | Recordings,
    reference: ImpulseResponses | Recordings,
    n_samples_out: int = 256,
    show: bool = False,
) -> ImpulseResponses:
    """
    Equalize IRs (per loudspeaker) using reference measurements.

    Steps:
    1. Convert recorded IRs (slab.Filter) to pyfar.Signal objects
    2. For each loudspeaker ID:
       - pick matching reference IR
       - remove DC
       - compute regularized inverse of the reference spectrum
       - convolve all IRs from this loudspeaker with the inverse
    3. Pack all equalized IRs into a single pyfar.Signal
    4. Time-align using the central loudspeaker (23_0.0_0.0) to 1 ms
    5. Window all HRIRs
    6. Convert back to ImpulseResponses (slab.Filter grid)
    7. Optionally plot intermediate steps (show=True)
    """
    logging.info("Applying equalization")

    signal_params = recorded.params["signal"]
    fs = int(recorded.params["fs"])
    from_f = signal_params["from_frequency"]
    to_f = signal_params["to_frequency"]
    center_key = "23_0.0_0.0"
    # save the center loudspeaker’s reference + inverse for plotting
    center_ref = None
    center_ref_inv = None

    # ------------------------------------------------------------------
    # 1) Convert recorded HRIRs to pyfar.Signal
    # ------------------------------------------------------------------
    equalized = ImpulseResponses()
    for key, filt in recorded.items():
        # slab.Filter data: (n_samples, n_channels)
        equalized[key] = pyfar.Signal(filt.data.T, filt.samplerate)

    # ------------------------------------------------------------------
    # 2) Loudspeaker-wise equalization using reference recordings
    # ------------------------------------------------------------------
    # speaker ID = first two chars of key (e.g. "19", "23", ...)
    speaker_ids = list(set(key[:2] for key in equalized.keys()))
    for spk_id in speaker_ids:

        # find the reference entry for this loudspeaker
        ref_candidates = [k for k in reference.keys() if k.startswith(spk_id + "_")]
        if not ref_candidates:
            raise KeyError(f"No matching reference for loudspeaker ID '{spk_id}'")
        ref_key = ref_candidates[0]
        ref = pyfar.Signal(reference[ref_key].data.T, fs)
        # remove DC
        # ref.time -= numpy.mean(reference.time, axis=-1, keepdims=True)
        # regularized inverse of reference
        ref_inv = pyfar.dsp.regularized_spectrum_inversion(ref, frequency_range=(200, 18000))
        # save center recordings for plotting
        if ref_key == center_key:
            center_ref = ref
            center_ref_inv = ref_inv

        # equalize all recordings from this loudspeaker
        rec_keys = [k for k in equalized.keys() if k.startswith(spk_id + "_")]
        for key in rec_keys:
            ir = equalized[key]
            # ir.time -= numpy.mean(ir.time, axis=-1, keepdims=True)  # remove DC
            ir_equalized = ir * ref_inv
            equalized[key] = ir_equalized

    # ------------------------------------------------------------------
    # 3) Pack equalized HRIRs into one pyfar.Signal
    # ------------------------------------------------------------------
    keys_list = list(equalized.keys())
    sig_list = [equalized[k] for k in keys_list]
    # shape: (n_positions, n_channels, n_samples)
    hrir_equalized = pyfar.Signal(numpy.stack([s.time for s in sig_list], axis=0), fs)

    # ------------------------------------------------------------------
    # 4) Temporal alignment using central loudspeaker
    # ------------------------------------------------------------------
    hrir_shifted = pyfar.dsp.time_shift(hrir_equalized, 2000)

    # find index of central loudspeaker in keys_list
    center_idx = keys_list.index(center_key)
    center_onset = pyfar.dsp.find_impulse_response_start(
        hrir_shifted[center_idx], threshold=15
    )
    center_onset = float(numpy.min(center_onset))
    center_onset_s = center_onset / fs

    # shift so that center onset is at 1 ms
    desired_onset_s = 0.001
    shift_s = desired_onset_s - center_onset_s
    hrir_aligned = pyfar.dsp.time_shift(hrir_shifted, shift_s, unit="s")

    # ------------------------------------------------------------------
    # 5) Window the HRIRs (short asymmetric Hanning window) and cut to final length
    # ------------------------------------------------------------------
    onsets = pyfar.dsp.find_impulse_response_start(hrir_aligned, threshold=15)
    onsets_min = numpy.min(onsets) / fs  # earliest onset in seconds
    times = (onsets_min - .00025,  # start of fade-in
             onsets_min,  # end if fade-in
             onsets_min + .0048,  # start of fade_out
             onsets_min + .0058)  # end of_fade_out
    hrir_windowed, window = pyfar.dsp.time_window(
        hrir_aligned, times, "hann", unit="s", crop="none", return_window=True)

    # cut to n_samp out if raw recordings where provided
    if isinstance(recorded, Recordings):
        times_samples = [0, 10, 246, n_samples_out - 1]
        hrir_cropped = pyfar.dsp.time_window(
            hrir_windowed,
            times_samples,
            "hann",
            crop="end",
        )
        equalized.params["n_samples"] = n_samples_out
        hrir_final = hrir_cropped
    else:
        hrir_final = hrir_windowed

    # ------------------------------------------------------------------
    # 6) Convert to slab filters and return ImpulseResponses
    # ------------------------------------------------------------------
    for key, filt in zip(equalized.keys(), hrir_final):
        # pyfar.Signal: (n_channels, n_samples) -> slab.Filter expects (n_samples, n_channels)
        equalized[key] = slab.Filter(data=filt.time.T, samplerate=fs, fir="IR")
    equalized.params = {
        "fs": fs,
        "signal": signal_params,
        "n_samp_out": n_samples_out,
        "date": datetime.now().isoformat(),
    }

    # ------------------------------------------------------------------
    # Plot equalization stages (optional)
    # ------------------------------------------------------------------
    if show:
        # raw recorded center IR as pyfar.Signal
        raw_center = pyfar.Signal(recorded[center_key].data.T, fs)
        # equalized but not windowed center signal
        _plot_equalization_pyfar(
            raw_center=raw_center,
            center_ref=center_ref,
            center_ref_inv=center_ref_inv,
            hrir_equalized=hrir_equalized,
            hrir_shifted=hrir_shifted,
            hrir_aligned=hrir_aligned,
            hrir_windowed=hrir_windowed,
            hrir_final=hrir_final,
            window=window,
            center_idx=center_idx,
        )

    return equalized


def equalize_old(
    hrir_recorded: ImpulseResponses | Recordings,
    hrir_reference: ImpulseResponses | Recordings,
    n_samp_out: int = 256,
    show: bool = False):

    logging.info("Applying equalization")

    signal_params = hrir_recorded.params["signal"]
    fs = int(hrir_recorded.params["fs"])
    n_samp = int(hrir_recorded.params["n_samp"])

    from_f = signal_params["from_frequency"]
    to_f = signal_params["to_frequency"]

    # ------------------------------------------------------------------
    # 1) Convert recorded HRIRs to pyfar.Signal
    # ------------------------------------------------------------------
    equalized = ImpulseResponses()
    for key, filt in hrir_recorded.items():
        # slab.Filter data: (n_samples, n_channels)
        equalized[key] = pyfar.Signal(filt.data.T, filt.samplerate)

    # ------------------------------------------------------------------
    # 2) Loudspeaker-wise equalization using reference recordings
    # ------------------------------------------------------------------
    # speaker ID = first two chars of key (e.g. "19", "23", ...)
    speaker_ids = list(set(key[:2] for key in equalized.keys()))

    # we’ll remember the center loudspeaker’s reference + inverse for plotting
    center_key_full = "23_0.0_0.0"
    center_ref_signal = None
    center_ref_inv = None

    for spk_id in speaker_ids:
        # find the reference entry for this loudspeaker
        ref_candidates = [k for k in hrir_reference.keys() if k.startswith(spk_id + "_")]
        if not ref_candidates:
            raise KeyError(f"No matching reference for loudspeaker ID '{spk_id}'")
        ref_key = ref_candidates[0]

        reference = pyfar.Signal(hrir_reference[ref_key].data.T, fs)
        # remove DC
        reference.time -= numpy.mean(reference.time, axis=-1, keepdims=True)

        # regularized inverse of reference
        reference_inv = pyfar.dsp.regularized_spectrum_inversion(
            reference,
            frequency_range=(from_f, to_f),
        )  # todo different range (20, 18750)

        # store for plotting if this loudspeaker contains the center key
        rec_keys = [k for k in equalized.keys() if k.startswith(spk_id + "_")]
        if center_key_full in rec_keys:
            center_ref_signal = reference
            center_ref_inv = reference_inv

        # equalize all recordings from this loudspeaker
        for key in rec_keys:
            ir = equalized[key]
            ir.time -= numpy.mean(ir.time, axis=-1, keepdims=True)  # remove DC
            ir_equalized = ir * reference_inv

            # windowing
            n0 = min(int(numpy.argmax(numpy.abs(ir_equalized.time[0]))),
                     int(numpy.argmax(numpy.abs(ir_equalized.time[1]))))
            ir_windowed = pyfar.dsp.time_window(ir_equalized,
                                                  (max(0, n0 - 50), min(n0 + 100, len(ir_equalized.time[0]) - 1)),
                                                  'boxcar', unit="samples",
                                                  crop="window")  # (170,335)
            ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_equalized.n_samples - ir_windowed.n_samples)

            # low freq extrapolation
            ir_low = pyfar.dsp.filter.crossover(
                pyfar.signals.impulse(ir_windowed.n_samples), 4, 400)[0]  # *(10**(-1))
            ir_low.sampling_rate = 48828
            ir_high = pyfar.dsp.filter.crossover(ir_windowed, 4, 400)[1]
            ir_low_delayed = pyfar.dsp.fractional_time_shift(
                ir_low, pyfar.dsp.find_impulse_response_delay(
                    ir_windowed))  # error if too noisy, overwrite function and use exactly the peakpoint?
            ir_extrapolated = ir_low_delayed + ir_high

            # no time shift

            # down sampling
            ir_final = pyfar.dsp.time_window(
                ir_extrapolated, (0, n_samp_out-1), 'boxcar',
                crop='window')

            equalized[key] = ir_final

            if key == center_key_full:
                raw_center = ir
                reference_center = reference
                reference_inv_center = reference_inv
                equalized_center = ir_equalized
                ir_windowed_center = ir_windowed

    #     # ------------------------------------------------------------------
    # 4) Temporal alignment using central loudspeaker
    # ------------------------------------------------------------------
    # hrir_shifted = pyfar.dsp.time_shift(hrir, int(n_samp / 2))
    #
    # # find index of central loudspeaker in keys_list
    # center_key = center_key_full if center_key_full in keys_list else keys_list[0]
    # center_idx = keys_list.index(center_key)
    #
    # center_onset = pyfar.dsp.find_impulse_response_start(
    #     hrir_shifted[center_idx], threshold=15
    # )
    # center_onset = float(numpy.min(center_onset))
    # center_onset_s = center_onset / fs
    #
    # # shift so that center onset is at 1 ms
    # desired_onset_s = 0.001
    # shift_s = desired_onset_s - center_onset_s
    # hrir_aligned = pyfar.dsp.time_shift(hrir_shifted, shift_s, unit="s")

    # ------------------------------------------------------------------
    # 6) Plot equalization stages (optional)
    # ------------------------------------------------------------------
    if show:
        _plot_equalization_pyfar(
            raw_center=raw_center,
            reference=reference_center,
            reference_inv=reference_inv_center,
            equalized_center=equalized_center,
            hrir_windowed=ir_windowed_center,
            )
    # ------------------------------------------------------------------
    # 7) Convert to slab filters and return ImpulseResponses
    # ------------------------------------------------------------------
    equalized.params = {
        "fs": fs,
        "signal": signal_params,
        "n_samp_out": n_samp_out,
        "date": datetime.now().isoformat(),
    }
    for key, filt in equalized.items():
        # pyfar.Signal: (n_channels, n_samples) -> slab.Filter expects (n_samples, n_channels)
        equalized[key] = slab.Filter(data=filt.time.T, samplerate=fs, fir="IR")
    return equalized

def lowfreq_extrapolate(
    irs,
    f_extrap: float = 400.0,
    f_target: float = 150.0,
    head_radius: float | None = .0875,
    show: bool = False,
    probe_index: int | None = None,
):
    """
    Smoothly replace low frequencies with spherical-head magnitude.

    Anchors the magnitude at 0 Hz and `f_target` from a spherical-head model,
    keeps measured magnitudes above `f_extrap`, and linearly interpolates
    across the full band. Phase (and thus ITD) is preserved.

    Parameters
    ----------
    irs : ImpulseResponses
        Binaural IRs in pyfar spherical coords (`domain="sph"`, `top_elev`).
    f_extrap : float
        Frequency (Hz) above which measured magnitudes are trusted.
    f_target : float
        Low-frequency anchor (Hz) taken from the spherical-head model.
    head_radius : float or None
        Spherical-head radius in meters (None → model default).
    show : bool
        If True, plot before/after magnitude for one position.

    Returns
    -------
    ImpulseResponses
        New object with LF-magnitude extrapolated, phase unchanged.

    Notes
    -----
    Interpolates magnitudes per ear per source; uses `domain='freq'`
    when constructing pyfar signals for plotting.
    """
    # ----------------------------------------------------------------------
    # (1) Convert measured IRs -> pyfar.Signal + spherical coordinates
    # ----------------------------------------------------------------------
    hrir_meas, coords, keys, fs = _irs_to_pyfar(irs)

    # Get spherical-head HRTFs for the same coordinates
    shtf = _spherical_head_for(coords, hrir_meas.n_samples, fs, head_radius)

    # ----------------------------------------------------------------------
    # (2) Prepare variables and masks for frequency-domain interpolation
    # ----------------------------------------------------------------------
    freqs = hrir_meas.frequencies           # frequency axis [Hz]
    mask_extrap = freqs >= f_extrap         # region where measurement is trusted

    # Find index closest to f_target (used as second model anchor)
    f_target_idx = hrir_meas.find_nearest_frequency(f_target)
    f_target = freqs[f_target_idx]          # ensure grid alignment

    # Separate magnitude and phase for convenience
    mag_meas = numpy.abs(hrir_meas.freq)
    phase_meas = numpy.angle(hrir_meas.freq)
    mag_sph = numpy.abs(shtf.freq)          # spherical-head magnitude

    # ----------------------------------------------------------------------
    # (3) Build the "anchor" magnitude vectors
    # ----------------------------------------------------------------------
    # - 0 Hz and f_target: take from spherical-head model
    # - f >= f_extrap: take from measurement
    # Everything between will be interpolated later.

    freqs_valid = freqs[mask_extrap]                    # f >= f_extrap
    mag_valid = mag_meas[..., mask_extrap]              # measured magnitudes above f_extrap

    mag0 = mag_sph[..., 0:1]                            # model @ 0 Hz
    mag_ft = mag_sph[..., f_target_idx:f_target_idx+1]  # model @ f_target

    # Concatenate anchor magnitudes for each ear/source:
    # shape: (n_pos, n_ears, 2 + n_valid)
    mag_anchor = numpy.concatenate((mag0, mag_ft, mag_valid), axis=-1)

    # Corresponding anchor frequencies
    freqs_anchor = numpy.concatenate((
        numpy.array([0.0]),
        numpy.array([f_target]),
        freqs_valid,
    ))

    # ----------------------------------------------------------------------
    # (4) Interpolate magnitude across the entire frequency axis
    # ----------------------------------------------------------------------
    mag_interp = numpy.empty_like(hrir_meas.freq)
    for src in range(mag_anchor.shape[0]):       # iterate over all source positions
        for ear in range(mag_anchor.shape[1]):   # iterate over ears (L/R)
            # Linear interpolation in magnitude domain
            mag_interp[src, ear] = numpy.interp(
                freqs,             # full frequency grid
                freqs_anchor,      # anchor frequencies
                mag_anchor[src, ear],  # anchor magnitudes
            )

    # ----------------------------------------------------------------------
    # (5) Combine new magnitudes with original phases
    # ----------------------------------------------------------------------
    # Phase is preserved (no ITD or phase shift change here)
    hrir_meas.freq = mag_interp * numpy.exp(1j * phase_meas)

    # ----------------------------------------------------------------------
    # (6) Optional quick visualization: before vs after (one position)
    # ----------------------------------------------------------------------
    if show:
        idx = 0 if probe_index is None else int(probe_index)

        # Plot frequency responses for left/right ear before and after
        plt.figure()
        ax = pyfar.plot.freq(
            pyfar.Signal(mag_meas[idx] * numpy.exp(1j * phase_meas[idx]), fs, domain='freq'),
            color=[.6, .6, .6],
            label=['before L', 'before R'],
        )
        pyfar.plot.freq(
            pyfar.Signal(hrir_meas.freq[idx], fs, domain='freq'),
            label=['after L', 'after R'],
            ax=ax,
        )
        ax.set_title(f'LF extrapolation (pos {idx})')
        ax.set_xlim(50, fs / 2)
        ax.legend()
        plt.show()

    # ----------------------------------------------------------------------
    # (7) Convert back into ImpulseResponses object for next pipeline step
    # ----------------------------------------------------------------------
    time_data = hrir_meas.time  # (n_pos, n_ears, n_samples)
    out = _pyfar_to_irs(irs, keys, time_data, fs)

    # Bookkeeping / metadata
    out.params.setdefault("processing", {})
    out.params["processing"]["lowfreq_extrapolate"] = {
        "f_extrap": float(f_extrap),
        "f_target": float(f_target),
        "head_radius": float(head_radius) if head_radius is not None else None,
        "date": datetime.now().isoformat(),
    }

    return out

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



# -------------------------------------------------------------------------
# Helpers / Plotting
# -------------------------------------------------------------------------
def _irs_to_pyfar(irs):
    """ImpulseResponses -> (pyfar.Signal, pyfar.Coordinates, keys, fs)"""
    fs = int(irs.params["fs"])
    keys = list(irs.data.keys())
    if not keys:
        raise ValueError("No filters in ImpulseResponses.data")
    filters = [irs.data[k] for k in keys]  # slab.Filter
    data = numpy.stack([f.data.T for f in filters], axis=0)  # (n_pos, n_ears, n_samp)

    sources = irs.get_sources()  # (n_pos, 3): [az, el, r] in deg/m
    coords = pyfar.Coordinates(
        sources[:, 0],  # azimuth in deg
        sources[:, 1],  # elevation in deg
        sources[:, 2],  # radius in m
        domain="sph",
        convention="top_elev",
        unit="deg",
    )
    sig = pyfar.Signal(data, fs)
    return sig, coords, keys, fs

def _pyfar_to_irs(template_irs, keys, time_data, fs):
    """(keys aligned with dimension 0) -> new ImpulseResponses with slab.Filters."""
    out = copy.deepcopy(template_irs)
    for key, sig_time in zip(keys, time_data):  # sig_time: (n_ears, n_samp)
        out.data[key] = slab.Filter(data=sig_time.T, samplerate=fs, fir="IR")
    return out

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

def _plot_processing_pyfar(excitation, ref_inv, recording, hrir_deconvolved, hrir_shifted,
    hrir_aligned, hrir_windowed, hrir_final, window, center_idx: int = 0) -> None:
    """
    Comprehensive overview of the processing pipeline, using pyfar's
    plotting shortcuts (same style as Fabian's notebook).

    Shows, in one figure with subplots:
    - Recorded sweep at center position
    - Excitation + inverse (frequency domain)
    - Deconvolved, shifted, aligned, windowed HRIRs
    - Window overlay
    - Frequency response before vs after final truncation
    """
    import pyfar.plot as pfplot

    # which source index we show in detail
    idx = center_idx

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()

    # 1) Recorded sweep at the ears (time, both channels)
    ax = axes[0]
    pfplot.time(recording[idx], unit="ms", ax=ax, label=["left", "right"])
    ax.set_title(f"Recorded sine sweep (center pos)")
    ax.legend()

    # 2) Excitation in time domain
    ax = axes[1]
    pfplot.time(excitation[0], unit="ms", ax=ax)
    ax.set_title("Excitation sweep (time)")

    # 3) Excitation and inverse in frequency domain
    ax = axes[2]
    pfplot.freq(excitation[0], freq_scale="log", ax=ax, color=[0.6, 0.6, 0.6])
    pfplot.freq(ref_inv[0], freq_scale="log", ax=ax, label=["inverse L", "inverse R"])
    ax.set_title("Excitation & inverse (magnitude)")
    ax.set_xlim(50, excitation.sampling_rate / 2)
    ax.legend(loc="best")

    # 4) Deconvolved HRIR at center position (time, dB)
    ax = axes[3]
    pfplot.time(hrir_deconvolved[idx], dB=True, unit="ms", ax=ax)
    ax.set_title("Deconvolved HRIR (center pos)")

    # 5) Time-shifted HRIRs (all positions)
    ax = axes[4]
    pfplot.time(hrir_shifted, dB=True, unit="ms", ax=ax)
    ax.set_title("Time-shifted HRIRs (all positions)")

    # 6) Aligned HRIR at center (onset at 1 ms)
    ax = axes[5]
    pfplot.time(hrir_aligned[idx], unit="ms", ax=ax)
    ax.axvline(1.0, color="k", linestyle="--",
               label="1 ms: desired onset (center)")
    ax.set_xlim(0, 5)
    ax.legend()
    ax.set_title("Aligned center HRIR")

    # 7) Aligned HRIRs (all positions)
    ax = axes[6]
    pfplot.time(hrir_aligned, unit="ms", ax=ax)
    ax.axvline(1.0, color="k", linestyle="--",
               label="1 ms: desired onset (center)")
    ax.set_xlim(0, 25)
    ax.legend()
    ax.set_title("Aligned HRIRs (all positions)")

    # 8) Windowed HRIRs + window
    ax = axes[7]
    pfplot.time(hrir_windowed, unit="ms", dB=True, ax=ax)
    pfplot.time(window, unit="ms", dB=True,
                color="k", linestyle="--", ax=ax, label="window")
    ax.set_xlim(0, 10)
    ax.set_title("Windowed HRIRs + window")
    ax.legend()

    # 9) Frequency response before vs after final truncation (one position)
    ax = axes[8]
    pfplot.freq(
        hrir_windowed[idx],
        freq_scale="log",
        ax=ax,
        color=[0.6, 0.6, 0.6],
    )
    pfplot.freq(
        hrir_final[idx],
        freq_scale="log",
        ax=ax,
        label=["final L", "final R"],
    )
    ax.set_title(f"HRIRs before/after final truncation (center pos)")
    ax.set_xlim(50, hrir_final.sampling_rate / 2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

def _plot_equalization_pyfar(
    raw_center = None, center_ref = None,
    center_ref_inv = None, hrir_equalized= None, hrir_shifted= None,
    hrir_aligned= None, hrir_windowed= None, hrir_final= None, window= None, center_idx: int = 0) -> None:
    """
    Overview plot for the equalization pipeline, using pyfar's plotting shortcuts.
    Inspired by the example notebook.

    Shows:
    - Raw HRIR (center position)
    - Reference IR and inverse (time + magnitude)
    - Center HRIR before vs after equalization
    - Shifted / aligned / windowed HRIRs
    - Window overlay
    - Frequency response before vs after windowing (center)
    """
    import pyfar.plot as pfplot

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()

    # 1) Raw HRIR at center position (time)
    ax = axes[0]
    pfplot.time(raw_center, unit="ms", ax=ax, label=["left", "right"])
    ax.set_title("Ear pressure (center)")
    ax.legend()

    # ax = axes[1]
    # pfplot.freq(raw_center, ax=ax, label=["left", "right"])
    # ax.set_title("Ear pressure (center)")
    # ax.legend()

    # 2) Reference IR (time)
    ax = axes[1]
    if center_ref is not None:
        pfplot.time(center_ref, unit="ms", ax=ax)
        ax.set_title("Reference pressure (center)")
    else:

        
        ax.set_title("Reference IR (missing)")

    # 3) Reference inverse (magnitude)
    ax = axes[2]
    if center_ref_inv is not None:
        pfplot.freq(center_ref_inv, freq_scale="log", ax=ax, label=["inv L", "inv R"])
        ax.set_title("Inverted reference")
        ax.set_xlim(50, center_ref_inv.sampling_rate / 2)
        ax.legend()
    else:
        ax.set_title("Inverse reference (missing)")

    # 4) Center HRIR before vs after equalization (freq)
    ax = axes[3]
    pfplot.freq(raw_center, ax=ax, color=[0.6, 0.6, 0.6])
    pfplot.freq(hrir_equalized[center_idx], ax=ax, label=["equalized L", "equalized R"])
    ax.set_title("Center HRIR: raw vs equalized")
    ax.legend()

    # 5) Time-shifted HRIRs (center)
    if hrir_shifted is not None:
        ax = axes[4]
        pfplot.time(hrir_shifted[center_idx], dB=False, unit="ms", ax=ax)
        ax.set_title("Equalized center HRIR (time-shifted)")
        ax.set_xlim(0, 75)

    # 6) Aligned center HRIR (1 ms onset)
    if hrir_aligned is not None:
        ax = axes[5]
        pfplot.time(hrir_aligned, unit="ms", ax=ax, linewidth=0.5)
        ax.axvline(1.0, color="k", linestyle="--", label="1 ms onset target")
        ax.set_xlim(0, 25)
        ax.legend()
        ax.set_title("Aligned HRIRs (all positions)")

    # 7) Windowed HRIRs + window
    if hrir_windowed is not None:
        ax = axes[6]
        pfplot.time(hrir_windowed, unit="ms", dB=True, ax=ax)
        if window is not None:
            pfplot.time(window, unit="ms", dB=True, color="k", linestyle="--", ax=ax, label="window")
        ax.set_xlim(0, 10)
        ax.set_title("Windowed HRIRs")
        ax.legend()

    # 8) Frequency response (center) before vs after windowing
    if hrir_aligned is not None:
        ax = axes[7]
        pfplot.freq(
            hrir_aligned[center_idx], freq_scale="log", ax=ax, color=[0.6, 0.6, 0.6]
        )
    if hrir_windowed is not None:
        pfplot.freq(
            hrir_windowed[center_idx],
            freq_scale="log",
            ax=ax,
            label=["windowed L", "windowed R"],
        )
        ax.set_title("Windowed HRIR (center)")
        ax.set_xlim(50, hrir_windowed.sampling_rate / 2)
        ax.legend(loc="lower left")

    # 9) Final cropped IR (center)
    if hrir_final is not None:
        ax = axes[8]
        pfplot.time(hrir_final[center_idx], unit="ms", ax=ax)
        ax.axvline(1.0, color="k", linestyle="--", label="1 ms onset target")
        ax.legend()
        ax.set_title("Final center IR")

    fig.tight_layout()
    plt.show()

def wait_for_button(msg=None):
    if msg:
        logging.info(msg)

    def on_press(key):
        if key == keyboard.Key.enter:
            listener.stop()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def parse_params_file(path: Path | str, filename: str = "params.txt") -> dict:
    """
    Read a params.txt written by `write_params_file` and rebuild a params dict.

    Supports:
    key: value
    nested_dict:
      subkey: subval
    """
    path = Path(path)
    file = path / filename
    params: dict[str, object] = {}

    if not file.exists():
        return params  # no params file → leave dict empty

    def parse_value(s: str):
        s = s.strip()
        # try int, then float, then bool, else leave as string
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    with file.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    current_dict = None
    for line in lines:
        if not line.strip():
            continue
        # top-level line (no leading spaces)
        if not line.startswith(" "):
            if line.endswith(":") and not ":" in line[:-1]:
                # e.g. "signal:" or "Software versions:"
                key = line[:-1].strip()
                current_key = key
                current_dict = {}
                params[key] = current_dict
            else:
                # e.g. "fs: 48828"
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    params[key] = parse_value(val)
                current_dict = None
                current_key = None
        else:
            # indented line: part of current_dict
            if current_dict is not None:
                sub = line.strip()
                if ":" in sub:
                    sk, sv = sub.split(":", 1)
                    current_dict[sk.strip()] = parse_value(sv.strip())
    return params

def _wrap_az_deg_ccw(az):
    """Wrap azimuth(s) to [0, 360) with CCW-positive (pyfar 'sph/top_elev')."""
    az = numpy.asarray(az, dtype=float)
    az = numpy.mod(az, 360.0)
    az[az < 0] += 360.0
    return az

# if __name__ == "__main__":
#     # tiny example call – adjust IDs and reference name
#     logging.basicConfig(level=logging.INFO)
#     hrir = record_hrir(subject_id, reference, samp_rec, n_rec, fs, overwrite, n_samp_out, show)
#     pass
#


"""
hrir.write_sofa(hrtf_dir / 'rec' / subject_id / str(subject_id + '.sofa'))  # write to subject rec folder
hrir.write_sofa(hrtf_dir / 'sofa' / str(subject_id + '.sofa'))  # write to sofa folder
"""