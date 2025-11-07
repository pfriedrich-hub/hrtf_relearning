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
import slab
from hrtf.processing.make.add_interaural import add_itd, add_ild  # from your repo
from hrtf.processing.spherical_head import spherical_head
from matplotlib import pyplot as plt

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

    # generate excitation signal
    params = {"type": "slab.Sound.chirp", "duration": 0.2, "level": 85,
              "from_frequency": 50, "to_frequency": 18e3, "samplerate": fs}
    signal = slab.Sound.chirp(duration=params["duration"], level=params["level"], samplerate=fs,
                              from_frequency=params["from_frequency"], to_frequency=params["to_frequency"])
    signal = signal.ramp(when="both", duration=0.01)  # matches the cos ramp in bi_play_buf.rcx
    signal.params = params

    # 1) subject recordings
    if (not subject_dir.exists()) or overwrite:
        ear_pressure = Recordings.record_dome(signal, n_samp=n_samp, n_rec=n_rec, fs=fs)
        ear_pressure.params["subject_id"] = subject_id
        ear_pressure.to_wav(subject_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject recordings from wav directory")
        ear_pressure = Recordings.from_wav(subject_dir / "wav")

    # 2) reference recordings
    if ref_dir.exists() and not overwrite:
        logging.info("Loading reference recordings")
        reference_pressure = Recordings.from_wav(ref_dir / "wav")
    else:
        logging.info("Recording new reference")
        ref_dir.mkdir(exist_ok=True, parents=True)
        reference_pressure = Recordings.record_dome(signal, n_samp=1, n_rec=n_rec, fs=fs)
        reference_pressure.params["subject_id"] = reference
        reference_pressure.to_wav(ref_dir, overwrite=overwrite)
    # optional: plot spectra
    # ear_pressure.plot_spectra()
    # reference_pressure.plot_spectra()

    # 3) convert recordings → IRs
    ear_ir = ear_pressure.compute_tf(out_n_samp=256)
    reference_ir = reference_pressure.compute_tf(out_n_samp=256)

    # 4) equalize
    ir_dir = equalize(ear_ir, reference_ir)  # todo rework pipeline

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
    def __getitem__(self, key: str) -> object:
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
    def record_dome(cls, signal: slab.Sound, n_samp=5, n_rec=20, fs=48828):
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
        filt = slab.Filter.band(kind="hp", frequency=50, samplerate=fs)
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
            "datetime": datetime.now().isoformat(),
        }
        return cls(data=recordings_dict, params=params, signal=signal)

    # --- conversion to IRs --------------------------------------------------
    def compute_tf(self, out_n_samp) -> "ImpulseResponses":
        """
        Convert this set of recordings to impulse responses using a reference set.

        For each key in this Recordings object, find the matching reference entry
        by comparing the first `match_prefix_len` characters (speaker index),
        then compute the equalized IR with `equalize(...)`.
        """
        # --- COMPUTE RAW TF ---- #
        reference = self.signal
        ir_dict: dict[str, pyfar.Signal] = {}
        for key, recording in self.data.items():
            rec = pyfar.Signal(recording.data.T, fs)
            ref_inv = pyfar.dsp.regularized_spectrum_inversion(pyfar.Signal(reference.data.T, fs),
                frequency_range=(self.params['signal']['from_frequency'], self.params['signal']['to_frequency']))
            hrir_deconvolved = (rec * ref_inv)
            ir_dict[key] = hrir_deconvolved

        # --- PROCESS TF ---- #
        onsets = pyfar.dsp.find_impulse_response_start(ir_dict['23_0.0_0.0'])  # onsets at the central speaker (0, 0)
        for key, ir in ir_dict.items():
            # align IRs in time
            ir  = pyfar.dsp.time_shift(ir, -numpy.min(onsets) / ir.sampling_rate + .001, unit='s')
            # window the HRIRs
            onsets = pyfar.dsp.find_impulse_response_start(ir, threshold=20)
            onsets_min = numpy.min(onsets) / ir.sampling_rate  # onset in seconds
            times = (onsets_min - .00025,  # start of fade-in
                     onsets_min,  # end if fade-in
                     onsets_min + .0048,  # start of fade_out
                     onsets_min + .0058)  # end of_fade_out
            ir_windowed, window = pyfar.dsp.time_window(
                ir, times, 'hann', unit='s', crop='end', return_window=True)
            ir_dict[key] = ir_windowed

        ## 6. Low-frequency extrapolation
        # todo get pyfar source grid and compute Spherical Head Transfer Functions - maybe do this after adding all az?
        source_positions = [key[3:] for key in self.keys()] #todo this should be pyfar coordinates
        shtf = spherical_head(source_positions, n_samples=ir_windowed.n_samples,
                              sampling_rate=ir_windowed.sampling_rate)
        for key, ir in ir_dict.items():
            ir_extrapolated = self._low_freq_extrapolation(ir, shtf)
            ir_dict[key] = ir_extrapolated

        ## 7. Far-field extrapolation
        source_positions_far_field = source_positions.copy()
        source_positions_far_field.radius = 100
        shtf_far_field = spherical_head(
            source_positions_far_field,
            n_samples=ir_extrapolated.n_samples,
            sampling_rate=ir_extrapolated.sampling_rate)
        dvf = shtf_far_field / shtf
        for key, ir in ir_dict.items():
            ir_far_field = ir * dvf
            ir_dict[key] = ir_far_field

        ## 8. Window to final length
        for key, ir in ir_dict.items():
            times = [0, 10, out_n_samp-1, out_n_samp]
            hrir_final = pyfar.dsp.time_window(
                hrir_far_field, times, 'hann', crop='end')
            ### END SOLUTION

        slab.Filter(data=hrir_deconvolved.time, samplerate=fs, fir='IR')
        return ImpulseResponses(data=ir_dict, fs=self.fs, meta=self.meta.copy())
    @staticmethod
    def _low_freq_extrapolation(ir, shtf):
        # Compute spherical head transfer functions using `spherical_head()`
        # frequency below which the data is extrapolated
        f_extrap = 400
        mask_extrap = ir.frequencies >= f_extrap
        # frequency below which the HRTF magnitude is assumed to be constant
        f_target_idx = ir.find_nearest_frequency(150)
        f_target = ir.frequencies[f_target_idx]
        # valid frequencies and magnitude values
        frequencies = ir.frequencies[mask_extrap]
        magnitude = numpy.abs(
            ir.freq[..., mask_extrap])
        # concatenate target gains and frequencies
        magnitude = numpy.concatenate((
            numpy.abs(shtf.freq[..., 0, None]),  # 0 Hz
            numpy.abs(shtf.freq[..., 1, None]),  # f_target
            magnitude), -1)
        frequencies = numpy.concatenate(([0], [f_target], frequencies))
        # interpolate magnitude
        magnitude_interpolated = numpy.empty_like(ir.freq)
        for source in range(magnitude.shape[0]):
            for ear in range(magnitude.shape[1]):
                magnitude_interpolated[source, ear] = numpy.interp(
                    ir.frequencies, frequencies, magnitude[source, ear])
        # apply new magnitude response
        ir_extrapolated = ir.copy()
        ir_extrapolated.freq = \
            magnitude_interpolated * numpy.exp(1j * numpy.angle(ir.freq))
        return ir_extrapolated

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

        # 🔍 read params.txt if present
        params = parse_params_file(path)

        # 🔁 try to reconstruct the signal from params["signal"]
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
            ),
        )

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
def equalize(rec_ir: ImpulseResponses | Recordings, ref_ir: ImpulseResponses | Recordings) -> ImpulseResponses:
    """
    Compute a time-limited FIR IR from a recording and its reference.

    This is adapted from your existing `rec2ir.equalize` implementation.
    """
    # find reference with same speaker index prefix
    for key, recording in rec_ir.data.items():
        ref_key = [k for k in ref_ir.data.keys() if k[:2] == key[:2]][0]
        if not ref_key:
            raise KeyError(f"No matching reference for recording key '{key}'")
        reference = ref_ir.data[ref_key]
        fs = recording.samplerate
        if recording.samplerate != reference.samplerate:
            logging.warning("sampling rate mismatch. resampling reference")
            reference = reference.resample(recording.samplerate)

        reference_sig = pyfar.Signal(reference.data.T, fs)
        recording_sig = pyfar.Signal(recording.data.T, fs)

        # remove DC
        recording_sig.time -= numpy.mean(recording_sig.time, axis=1, keepdims=True)
        reference_sig.time -= numpy.mean(reference_sig.time, axis=1, keepdims=True)

        reference_inverted = pyfar.dsp.regularized_spectrum_inversion(
            reference_sig, frequency_range=(20, 18750),)

        ir = recording_sig * reference_inverted

        # use earliest main peak (L/R) as reference
        n0 = min( int(numpy.argmax(numpy.abs(ir.time[0]))), int(numpy.argmax(numpy.abs(ir.time[1]))),)

        # time-window around direct sound
        ir_windowed = pyfar.dsp.time_window(
            ir,
            (max(0, n0 - 50), min(n0 + 100, len(ir.time[0]) - 1)),
            window="boxcar",
            unit="samples",
            crop="window",
        )

        ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir.n_samples - ir_windowed.n_samples)

        # low/high frequency split at 400 Hz
        ir_low = pyfar.dsp.filter.crossover(pyfar.signals.impulse(ir_windowed.n_samples), 4, 400)[0]
        ir_low.sampling_rate = fs
        ir_high = pyfar.dsp.filter.crossover(ir_windowed, 4, 400)[1]

        ir_low_delayed = pyfar.dsp.fractional_time_shift(
            ir_low,
            pyfar.dsp.find_impulse_response_delay(ir_windowed),
        )
        ir_extrapolated = ir_low_delayed + ir_high

        ir_final = pyfar.dsp.time_window(
            ir_extrapolated,
            (0, 255),
            "boxcar",
            crop="window",
        )

    return slab.Filter(data=ir_final.time, samplerate=fs, fir="IR")


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

if __name__ == "__main__":
    # tiny example call – adjust IDs and reference name
    logging.basicConfig(level=logging.INFO)
    # hrir, ref_recs, recs = record_hrir("MS", "kemar_no_ears", n_samp=1, n_rec=20)
    pass
