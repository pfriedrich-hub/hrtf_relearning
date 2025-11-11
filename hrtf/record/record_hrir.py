# import matplotlib
# matplotlib.use('tkagg')
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
from hrtf.processing.make.add_interaural import add_itd, add_ild  # from your repo
from hrtf.processing.spherical_head import spherical_head
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)


# -------------------------------------------------------------------------
# Global defaults
# -------------------------------------------------------------------------
subject_id = 'PF'
overwrite = False
reference = 'ref_07.11'
n_samp = 2
n_rec = 5
fs = 48828  # 97656

slab.set_default_samplerate(fs)
data_dir = Path.cwd() / "data"
freefield.set_logger("info")

# -------------------------------------------------------------------------
# High-level: record HRIR for one subject
# -------------------------------------------------------------------------
def record_hrir(
    subject_id: str,
    reference: str,
    n_samp: int = 5,
    n_rec: int = 5,
    fs: int = fs,
    overwrite: bool = False,
    az_range: tuple[float, float] = (-45, 45),
):
    hrtf_dir = Path.cwd() / "data" / "hrtf"
    subject_dir = hrtf_dir / "rec" / subject_id
    ref_dir = hrtf_dir / "rec" / "reference" / reference

    # excitation signal
    params = {"type": "slab.Sound.chirp", "duration": 0.2, "level": 85,
              "from_frequency": 50, "to_frequency": 18e3, "samplerate": fs}
    signal = slab.Sound.chirp(duration=params["duration"], level=params["level"], samplerate=fs,
                              from_frequency=params["from_frequency"], to_frequency=params["to_frequency"])
    signal = signal.ramp(when="both", duration=0.01)  # matches the cos ramp in bi_play_buf.rcx
    signal.params = params

    # 1) in ear recordings
    if (not subject_dir.exists()) or overwrite:
        ear_pressure = Recordings.record_dome(signal, n_samp=n_samp, n_rec=n_rec, fs=fs)
        ear_pressure.params["subject_id"] = subject_id
        ear_pressure.to_wav(subject_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject recordings from wav directory")
        ear_pressure = Recordings.from_wav(subject_dir)

    # 2) reference recordings
    if ref_dir.exists() and not overwrite:
        logging.info("Loading reference recordings")
        reference_pressure = Recordings.from_wav(ref_dir)
    else:
        logging.info("Recording new reference")
        ref_dir.mkdir(exist_ok=True, parents=True)
        reference_pressure = Recordings.record_dome(signal, n_samp=1, n_rec=n_rec, fs=fs)
        reference_pressure.params["subject_id"] = reference
        reference_pressure.to_wav(ref_dir, overwrite=overwrite)

    # plot spectra
    # ear_pressure.plot_spectra()
    # reference_pressure.plot_spectra()

    # 3) Compute TF (deconvolve recordings with inverted signal)
    hrir_recorded = ear_pressure.compute_tf(out_n_samp=256, show=True)
    hrir_reference = reference_pressure.compute_tf(out_n_samp=256, show=True)

    # 4) Equalize (divide ear tf by reference tf)
    hrir_equalized = equalize(hrir_recorded, hrir_reference)


    # # 4) add azimuth sources
    # irs.add_az_sources(az_range=az_range)
    #
    #
    #
    # # 5) build HRTF
    # hrir = irs.to_hrtf(add_itd_flag=True, add_ild_flag=True)
    #
    # # 6) write SOFA
    # slab.HRTF.write_sofa(hrir, subject_dir / f"{subject_id}.sofa")
    #
    # return hrir, reference_recs, recs


# -------------------------------------------------------------------------
# Helper: wait for Enter (optional)
# -------------------------------------------------------------------------
def wait_for_button(msg=None):
    if msg:
        logging.info(msg)

    def on_press(key):
        if key == keyboard.Key.enter:
            listener.stop()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


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
    def get_sources(self, distance: float = 1.2) -> numpy.ndarray:
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
    def record_dome(cls, signal: slab.Sound, n_samp=5, n_rec=5, hp_freq=50, fs=48828):
        """Record across the dome and return a Recordings object with all parameters stored."""
        if freefield.PROCESSORS.mode != "play_birec":
            freefield.initialize("dome", "play_birec")

        speakers_all = freefield.read_speaker_table()
        speakers = cls.get_speakers(speakers_all, azimuth=0, elevation=None)
        if len(speakers) < 2:
            raise RuntimeError("Need at least two speakers to infer vertical resolution.")

        res = abs(speakers[0].elevation - speakers[1].elevation) / n_samp
        min_el = min(spk.elevation for spk in speakers)

        recordings_dict = {}
        filt = slab.Filter.band(kind="hp", frequency=hp_freq, samplerate=fs)
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED

        for n in range(n_samp):
            elevation_step = n * res
            if n_samp > 1:
                freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                                processors=led_speaker.digital_proc)  # illuminate LED
                input(f"Press Enter when head is at {0 + elevation_step}° elevation ...")
            for base_spk in speakers:
                spk = copy.deepcopy(base_spk)
                spk.elevation -= elevation_step
                if spk.elevation >= min_el:
                    logging.info(f"Recording from Speaker {spk.index} at {spk.elevation:.1f}° elevation")
                    key = f"{spk.index}_{spk.azimuth}_{spk.elevation}"
                    rec = cls.record_speaker(spk, signal, n_rec, fs)
                    recordings_dict[key] = filt.apply(rec)
            freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
        params = {
            "fs": fs,
            "n_rec": n_rec,
            "n_samp": n_samp,
            "signal": getattr(signal, "params", {}),
            "highpass frequency": hp_freq,
            "datetime": datetime.now().isoformat(),
        }
        return cls(data=recordings_dict, params=params, signal=signal)

    # --- conversion to IRs --------------------------------------------------
    def compute_tf(self, out_n_samp: int = 256, show: bool = False) -> "ImpulseResponses":
        """
        Convert this set of recordings to impulse responses.

        Steps:
        1. Stack all binaural in-ear recordings into a multi-channel pyfar.Signal
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
            fs_local = int(self.params["fs"])
        else:
            from record_hrir import fs as fs_global
            fs_local = int(fs_global)

        if not self.data:
            raise ValueError("Recordings.compute_tf(): self.data is empty.")

        # ------------------------------------------------------------------
        # 1) Convert recordings -> pyfar.Signal
        # ------------------------------------------------------------------
        ear_arrays = [rec.data.T for rec in self.data.values()]  # (2, n_samples) per rec
        ear_pressure = pyfar.Signal(ear_arrays, fs_local)

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
        if self.params["highpass frequency"]:
            hp_freq = self.params.get('highpass frequency')
            filt = slab.Filter.band(kind="hp", frequency=hp_freq, samplerate=fs)
            sig = filt.apply(sig)

        excitation = pyfar.Signal(sig.data.T, fs_local)  # (2, n_samples)
        ref_inv = pyfar.dsp.regularized_spectrum_inversion(
            excitation,
            (from_f, to_f),
        )

        # ------------------------------------------------------------------
        # 3) Deconvolve: HRIR = Y * X^{-1}
        # ------------------------------------------------------------------
        hrir_deconvolved = ear_pressure * ref_inv

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
        onsets = pyfar.dsp.find_impulse_response_start(hrir_aligned, threshold=15)
        onsets_min = numpy.min(onsets) / hrir_aligned.sampling_rate  # seconds

        times_s = (
            onsets_min - 0.00025,
            onsets_min,
            onsets_min + 0.0048,
            onsets_min + 0.0058,
        )

        hrir_windowed, window = pyfar.dsp.time_window(
            hrir_aligned,
            times_s,
            "hann",
            unit="s",
            crop="end",
            return_window=True,
        )

        times_samples = [0, 10, 246, out_n_samp]
        hrir_final = pyfar.dsp.time_window(
            hrir_windowed,
            times_samples,
            "hann",
            crop="end",
        )

        # ------------------------------------------------------------------
        # 6) Optionally: plot processing steps
        # ------------------------------------------------------------------
        if show:
            self._plot_processing_pyfar(
                excitation=excitation,
                ref_inv=ref_inv,
                ear_pressure=ear_pressure,
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
            "fs": fs_local,
            "signal": self.params.get("signal", {}),
            "n_samp": out_n_samp,
            "date": datetime.now().isoformat(),
        }

        out = copy.deepcopy(self)
        for key, h in zip(out.data.keys(), hrir_final):
            out.data[key] = slab.Filter(
                data=h.time.T,
                samplerate=fs_local,
                fir="IR",
            )

        return ImpulseResponses(data=out.data, params=params)

    @staticmethod
    def _plot_processing_pyfar(
        excitation,
        ref_inv,
        ear_pressure,
        hrir_deconvolved,
        hrir_shifted,
        hrir_aligned,
        hrir_windowed,
        hrir_final,
        window,
        center_idx: int = 0,
    ) -> None:
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
        from matplotlib import pyplot as plt
        import numpy

        # which source index we show in detail
        idx = center_idx

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.ravel()

        # 1) Recorded sweep at the ears (time, both channels)
        ax = axes[0]
        pfplot.time(ear_pressure[idx], unit="ms", ax=ax, label=["left", "right"])
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
        ax.set_xlim(0, 5)
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

    @staticmethod
    def record_speaker(speaker, signal: slab.Sound, n_rec: int, fs: int) -> slab.Binaural:
        recordings = []
        for _ in range(n_rec):
            recordings.append(
                freefield.play_and_record(
                    speaker,
                    signal,
                    equalize=False,
                    recording_samplerate=fs*2,  # due to the rp2 silently running on fs / 2
                )
            )
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
    def plot_spectra(self):
        """
        Rough equivalent of your `plot_dict(..., kind='tf')` for recordings.
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        for key, rec in self.data.items():
            if not isinstance(rec, slab.Binaural):
                continue
            _, _, el = self.parse_key(key)
            rec.spectrum(axis=ax)
            line = ax.lines[-1]
            x, y = line.get_xdata(), line.get_ydata()
            line.set_ydata(y + el)  # offset by elevation
            line.set_label(f"el={el:.1f}°")

        ax.set_xlim([2e3, 18750])
        ax.set_xscale("linear")
        ax.set_ylim(-130, 20)
        ax.set_title("In-ear recordings (spectrum, offset by elevation)")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Level [dB] + elevation offset")
        ax.legend(title="Elevation (°)")
        plt.show()

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

    # --- azimuth extension --------------------------------------------------
    def add_az_sources(self, az_range: tuple[float, float] = (-35, 35)) -> "ImpulseResponses":
        """
        Extend IR set by duplicating each elevation across additional azimuths.
        """
        sources = self.get_sources()
        vertical_res = (sources[:, 1].max() - sources[:, 1].min()) / (len(sources) - 1)
        azimuths = numpy.arange(az_range[0], az_range[1] + 1e-6, vertical_res)

        new_entries: dict[str, slab.Filter] = {}

        for key, filt in self.data.items():
            spk_id, _, el_str = key.split("_")
            for az in azimuths:
                az_str = f"{float(az):.1f}"
                new_key = f"{spk_id}_{az_str}_{el_str}"
                if new_key not in self.data and new_key not in new_entries:
                    new_entries[new_key] = filt  # or copy.deepcopy(filt) if you want independent filters

        self.data.update(new_entries)
        return self

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

    # --- build slab.HRTF ----------------------------------------------------
    def to_hrtf(self, add_itd_flag: bool = True, add_ild_flag: bool = True) -> slab.HRTF:
        """
        Build a slab.HRTF from this IR set and (optionally) add ITD/ILD.
        """
        items_sorted = sorted(
            self.data.items(),
            key=lambda kv: (
                float(kv[0].rsplit("_", 2)[-2]),  # az
                float(kv[0].rsplit("_", 2)[-1]),  # el
            ),)
        data = numpy.array([filt.data for _, filt in items_sorted])
        sources = self.get_sources()
        hrir = slab.HRTF(data=data, sources=sources, samplerate=self.fs, datatype="FIR")
        if add_itd_flag:
            hrir = add_itd(hrir)
        if add_ild_flag:
            hrir = add_ild(hrir)
        return hrir

# -------------------------------------------------------------------------
# Deconvolution / equalization (pyfar pipeline)
# -------------------------------------------------------------------------
def equalize(hrir_recorded: ImpulseResponses | Recordings, hrir_reference: ImpulseResponses | Recordings) -> ImpulseResponses:
    """
    Equalize IRs
    """
    logging.info('Applying equalization')
    signal_params = hrir_recorded.params['signal']
    fs = hrir_recorded.params['fs']
    n_samp = hrir_recorded.params['n_samp']
    # find reference with same speaker index prefix
    equalized = ImpulseResponses()
    for key, filt in hrir_recorded.items():
        equalized[key] = pyfar.Signal(filt.data.T, filt.samplerate)  # convert to pyfar signal

    # convolve IRs with reference (loudspeaker specific recordings)
    speaker_ids =  list(set(key[:2] for key in equalized.keys()))
    for spk_id in speaker_ids:
        # pick reference recording, remove DC and invert its spectrum
        [ref_key] = [k for k in hrir_reference.keys() if spk_id in k[:2]]
        if not ref_key:
            raise KeyError(f"No matching reference for recording key '{key}'")
        reference = pyfar.Signal(hrir_reference[ref_key].data.T, fs)
        reference.time -= numpy.mean(reference.time, axis=1, keepdims=True)
        # Equalize HRIR by convolution with inverted reference signal
        reference_inv = pyfar.dsp.regularized_spectrum_inversion(reference,
                                frequency_range=(signal_params['from_frequency'], signal_params['to_frequency']))
        # equalize IRs recorded from the same speaker
        rec_keys = [k for k in equalized.keys() if spk_id in k[:2]]
        for key in rec_keys:
            ir = equalized[key]
            ir.time -= numpy.mean(ir.time, axis=1, keepdims=True)  # remove DC
            ir_equalized = ir * reference_inv  # convolve and store in equalized dict
            equalized[key] = ir_equalized

    # time align and window (again?)
    hrir = pyfar.Signal(numpy.stack([s.time for s in equalized.values()], axis=0), fs)  # work on a single pyfar object

    # correct group delay / temporal alignment of all HRIRs in the dataset at 1ms (keeps relative timing intact)
    hrir_shifted = pyfar.dsp.time_shift(hrir, int(n_samp / 2))
    center_idx = [i for i, s in enumerate(hrir_reference.keys()) if s == '23_0.0_0.0']
    center_onset = pyfar.dsp.find_impulse_response_start(hrir_shifted[center_idx], threshold=15)  # onsets at the central speaker (0, 0)
    hrir_aligned = pyfar.dsp.time_shift(  # align
        hrir_shifted, -numpy.min(center_onset) / fs + .001, unit='s')

    # window the HRIRs
    onsets = pyfar.dsp.find_impulse_response_start(hrir_aligned, threshold=15)
    onsets_min = numpy.min(onsets) / fs # earliest onset in the HRIR dataset
    times = (onsets_min - .0005,  # start of fade-in
             onsets_min,  # end if fade-in
             onsets_min + .002,  # start of fade_out
             onsets_min + .0025)  # end of_fade_out
    hrir_windowed, window = pyfar.dsp.time_window(
        hrir_aligned, times, 'hann', unit='s', crop='none', return_window=True)

    # convert to slab filters and write to impulseresponses object
    equalized.params = {'fs': fs, 'signal': signal_params, 'n_samp': n_samp, 'date': datetime.now().isoformat()}
    for key, filt in zip(equalized.keys(), hrir_windowed):
        equalized[key] = slab.Filter(data=filt.time, samplerate=fs, fir='IR')
    return equalized


# -------------------------------------------------------------------------
# Compatibility helpers
# -------------------------------------------------------------------------

def recordings2ir(
    recordings_dict: dict[str, slab.Binaural] | Recordings,
    reference_dict: dict[str, slab.Binaural] | Recordings,
) -> dict[str, slab.Filter]:
    """
    Backwards-compatible wrapper for your old function:
    takes plain dicts or `Recordings` and returns a dict of `slab.Filter`.
    """
    if not isinstance(recordings_dict, Recordings):
        recordings = Recordings(data=recordings_dict)
    else:
        recordings = recordings_dict

    if not isinstance(reference_dict, Recordings):
        reference = Recordings(data=reference_dict)
    else:
        reference = reference_dict

    irs = recordings.to_ir(reference)
    return irs.data

def plot_dict(data_dict, kind: str = "tf"):
    """
    Backwards-compatible plotting helper.
    """
    if isinstance(data_dict, ImpulseResponses):
        data_dict.plot(kind=kind)
    elif isinstance(data_dict, Recordings):
        data_dict.plot_spectra()
    else:
        # raw dict fallback
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        for key, item in data_dict.items():
            if isinstance(item, slab.Binaural):
                item.spectrum(axis=ax)
            elif isinstance(item, slab.Filter):
                if kind.upper() == "TF":
                    _ = item.tf(show=True, axis=ax)
                else:
                    ax.plot(item.data)
            line = ax.lines[-1]
            line.set_label(key)
        ax.legend()
        plt.show()

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
    current_key = None

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

    # @staticmethod
    # def _low_freq_extrapolation(ir, shtf):
    #     # Compute spherical head transfer functions using `spherical_head()`
    #     # frequency below which the data is extrapolated
    #     f_extrap = 400
    #     mask_extrap = ir.frequencies >= f_extrap
    #     # frequency below which the HRTF magnitude is assumed to be constant
    #     f_target_idx = ir.find_nearest_frequency(150)
    #     f_target = ir.frequencies[f_target_idx]
    #     # valid frequencies and magnitude values
    #     frequencies = ir.frequencies[mask_extrap]
    #     magnitude = numpy.abs(
    #         ir.freq[..., mask_extrap])
    #     # concatenate target gains and frequencies
    #     magnitude = numpy.concatenate((
    #         numpy.abs(shtf.freq[..., 0, None]),  # 0 Hz
    #         numpy.abs(shtf.freq[..., 1, None]),  # f_target
    #         magnitude), -1)
    #     frequencies = numpy.concatenate(([0], [f_target], frequencies))
    #     # interpolate magnitude
    #     magnitude_interpolated = numpy.empty_like(ir.freq)
    #     for source in range(magnitude.shape[0]):
    #         for ear in range(magnitude.shape[1]):
    #             magnitude_interpolated[source, ear] = numpy.interp(
    #                 ir.frequencies, frequencies, magnitude[source, ear])
    #     # apply new magnitude response
    #     ir_extrapolated = ir.copy()
    #     ir_extrapolated.freq = \
    #         magnitude_interpolated * numpy.exp(1j * numpy.angle(ir.freq))
    #     return ir_extrapolated
    #
if __name__ == "__main__":
    # tiny example call – adjust IDs and reference name
    logging.basicConfig(level=logging.INFO)
    # hrir, ref_recs, recs = record_hrir("MS", "kemar_no_ears", n_samp=1, n_rec=20)
    pass
