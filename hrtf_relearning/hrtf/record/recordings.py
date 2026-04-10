# recordings.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging
import copy
from pathlib import Path
from datetime import datetime
import numpy
import slab
import freefield
import soundfile as sf
import pyfar

# ---------------------------------------------------------------------
# Base grid container
# ---------------------------------------------------------------------

class SpeakerGridBase:
    """
    Base container for data defined on a loudspeaker grid.
    Keys: 'idx_az_el' → values (recordings, filters, etc.)
    """

    def __init__(self, data=None, params=None):
        self.data = data or {}
        self.params = params or {}

    # --- dict-like -----------------------------------------------------
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.data.values())[key]
        if isinstance(key, slice):
            return list(self.data.values())[key]
        return self.data[key]

    def __setitem__(self, key, value):
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

    # --- helpers -------------------------------------------------------
    @staticmethod
    def parse_key(key):
        idx, az, el = key.split("_")
        return int(idx), float(az), float(el)

    def get_sources(self, distance=1.4):
        coords = []
        for key in self.data:
            _, az, el = self.parse_key(key)
            coords.append([az, el, distance])
        return numpy.asarray(coords, dtype=float)

    # --- params I/O ----------------------------------------------------
    def write_params_file(self, path, filename="params.txt"):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        with (path / filename).open("w") as f:
            for k, v in self.params.items():
                if isinstance(v, dict):
                    f.write(f"{k}:\n")
                    for sk, sv in v.items():
                        f.write(f"  {sk}: {sv}\n")
                else:
                    f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------
# Recordings (raw binaural sweeps)
# ---------------------------------------------------------------------

class Recordings(SpeakerGridBase):
    """
    Raw in-ear sweep recordings.
    data[key] = list[slab.Binaural]
    """

    def __init__(self, data=None, params=None, signal=None):
        super().__init__(data, params)
        self.signal = signal

    # -------------------- Recording ----------------------------------

    @classmethod
    def record_dome(cls, id=None, azimuth=(-1,1), elevation=(37.5, -37.5),
                    n_directions=3, n_recordings=10, hp_freq=120, fs=48828,
                    equalize=True, key=True, button=False):

        # excitation signal
        sig_params = dict(
            kind="logarithmic",
            duration=0.2,
            level=85,
            from_frequency=120,
            to_frequency=22e3,
            samplerate=fs,
        )
        signal = slab.Sound.chirp(**sig_params)
        signal = signal.ramp(when="both", duration=0.001)

        filt = slab.Filter.band("hp", frequency=hp_freq, samplerate=fs)

        # dome setup
        if freefield.PROCESSORS.mode != "play_birec":
            freefield.initialize("dome", "play_birec")
        speakers = cls._select_speakers(freefield.read_speaker_table(), azimuth, elevation)
        led_bits = ['16', '8', '4']
        # led_bits = ['16', '4']
        res = abs(speakers[0].elevation - speakers[1].elevation) / n_directions
        min_el = min(spk.elevation for spk in speakers)
        data = {}

        for n in range(n_directions):

            elevation_step = n * res
            freefield.write(tag='bitmask', value=led_bits[n], processors='RX81')
            if button:
                print(f"Press Button when head is at {0 + elevation_step}° elevation ...")
                freefield.wait_for_button()
            if key:
                input(f'Press Enter when head is at {0 + elevation_step}° elevation ...')
            for base_spk in speakers:
                [spk] = copy.deepcopy(freefield.pick_speakers(base_spk.index))
                spk.elevation -= elevation_step
                if spk.elevation >= min_el:
                    logging.info(f"Recording from Speaker {spk.index} at {spk.azimuth:.1f}° azimuth"
                                 f" and {spk.elevation:.1f}° elevation")
                    key = f"{spk.index}_{spk.azimuth:.2f}_{spk.elevation:.2f}"
                    recs = cls.record_speaker(spk, signal, n_recordings, fs, equalize)
                    processed = []
                    for r in recs:
                        processed.append(filt.apply(r))
                    data[key] = processed
            freefield.write(tag='bitmask', value=0, processors='RX81')  # turn off LED

        # store parameters
        params = dict(
            id = id,
            fs=fs,
            n_recordings=n_recordings,
            n_directions=n_directions,
            signal=sig_params,
            highpass_frequency=hp_freq,
            equalize_dome=equalize,
            datetime=datetime.now().isoformat(),
        )

        return cls(data=data, params=params, signal=signal)

    @staticmethod
    def record_speaker(speaker, signal, n_recordings, fs, equalize):
        out = []
        for _ in range(n_recordings):
            rec = freefield.play_and_record(
                speaker=speaker,
                sound=signal,
                compensate_delay=True,
                equalize=equalize,
                recording_samplerate=fs,
            )
            out.append(slab.Binaural(rec))
        return out

    @staticmethod
    def _select_speakers(speakers, azimuth=None, elevation=None):
        out = []
        for s in speakers:
            if azimuth is not None:
                lo, hi = min(azimuth), max(azimuth)
                if not (lo <= s.azimuth <= hi):
                    continue
            if elevation is not None:
                lo, hi = min(elevation), max(elevation)
                if not (lo <= s.elevation <= hi):
                    continue
            out.append(s)
        return out

    # -------------------- NPZ I/O -------------------------------------

    def to_npz(self, path, overwrite=False):
        """Save recordings to a single .npz file.

        Array shape: (n_locations, n_recordings, n_channels, n_datapoints).
        Speaker-location keys are stored alongside so the dict can be reconstructed.
        """
        logging.info(f'Writing recordings to .npz: {path}.')
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.write_params_file(path)

        fname = path / "recordings.npz"
        if fname.exists() and not overwrite:
            logging.info(f"{fname} already exists – skipping (use overwrite=True to replace).")
            return

        keys = list(self.data.keys())
        if not keys:
            raise ValueError(
                "Recordings.data is empty – nothing to save. "
                "Check that from_wav() found files in the expected layout."
            )
        sample_rec = self.data[keys[0]][0]
        n_channels  = sample_rec.n_channels
        n_datapoints = sample_rec.n_samples
        samplerate  = sample_rec.samplerate
        n_locations = len(keys)
        n_recordings = max(len(recs) for recs in self.data.values())

        arr = numpy.zeros(
            (n_locations, n_recordings, n_channels, n_datapoints), dtype=numpy.float32
        )
        for i, key in enumerate(keys):
            for j, rec in enumerate(self.data[key]):
                arr[i, j] = rec.data.T  # (n_samples, n_ch) → (n_ch, n_samples)

        numpy.savez(
            fname,
            recordings=arr,
            keys=numpy.array(keys),
            samplerate=numpy.array(samplerate),
        )

    @classmethod
    def from_npz(cls, path):
        """Load recordings from a .npz file saved by to_npz()."""
        path = Path(path)
        params = parse_params_file(path)

        npz = numpy.load(path / "recordings.npz", allow_pickle=False)
        arr        = npz["recordings"]          # (n_locs, n_recs, n_ch, n_samples)
        keys       = npz["keys"].tolist()
        samplerate = int(npz["samplerate"])

        data = {}
        for i, key in enumerate(keys):
            recs = []
            for j in range(arr.shape[1]):
                rec_data = arr[i, j].T          # back to (n_samples, n_ch)
                recs.append(slab.Binaural(rec_data, samplerate))
            data[key] = recs

        return cls(data=data, params=params, signal=_signal_from_params(params))

    @classmethod
    def load(cls, path):
        """Load recordings from *path*, preferring .npz and falling back to .wav."""
        path = Path(path)
        if (path / "recordings.npz").exists():
            logging.info(f"Loading recordings from .npz: {path}")
            return cls.from_npz(path)
        logging.info(f"No .npz found – loading recordings from .wav: {path}")
        return cls.from_wav(path)

    # -------------------- WAV I/O (kept for backward compatibility) ----

    def to_wav(self, path, overwrite=False):
        logging.info(f'Writing recordings to .wav: {path}.')
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.write_params_file(path)

        for key, recs in self.data.items():
            kdir = path / key
            kdir.mkdir(exist_ok=True)
            for i, r in enumerate(recs):
                fname = kdir / f"rec_{i:03d}.wav"
                if fname.exists() and not overwrite:
                    continue
                sf.write(fname, r.data.astype("float32"), r.samplerate, subtype="FLOAT")

    @classmethod
    def from_wav(cls, path):
        path = Path(path)
        params = parse_params_file(path)
        data = {}

        for kdir in sorted(path.iterdir()):
            if not kdir.is_dir():
                continue
            wav_files = sorted(kdir.glob("*.wav"))
            rec_files = sorted(kdir.glob("rec_*.wav"))
            logging.debug(
                f"  {kdir.name}: {len(wav_files)} .wav files total, "
                f"{len(rec_files)} matching rec_*.wav"
            )
            recs = []
            for f in rec_files:
                x, fs = sf.read(f, dtype="float32", always_2d=True)
                recs.append(slab.Binaural(x, fs))
            if recs:
                data[kdir.name] = recs

        if not data:
            # Log what was actually found to help diagnose naming mismatches
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            sample_wavs = []
            for d in path.iterdir():
                if d.is_dir():
                    sample_wavs += [f.name for f in sorted(d.glob("*.wav"))[:3]]
            logging.warning(
                f"from_wav: no data loaded from '{path}'. "
                f"Subdirectories found: {subdirs}. "
                f"Example wav filenames: {sample_wavs}"
            )

        return cls(data=data, params=params, signal=_signal_from_params(params))

    def plot(self, speaker_idx=4):
        plt.figure(figsize=(12, 8))
        fs = self.params["fs"]
        for r in self[speaker_idx]:
            if isinstance(r, slab.Binaural):
                rec_l = pyfar.Signal(r.channel(0).data.T, fs)
                rec_r = pyfar.Signal(r.channel(1).data.T, fs)
            elif isinstance(r, pyfar.Signal):
                rec_l = r[0]
                rec_r = r[1]
            pyfar.plot.time_freq(rec_l, color='red', unit='samples')
            pyfar.plot.time_freq(rec_r, color='blue', unit='samples')
            plt.title(f"Speaker {speaker_idx}")
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_VALID_CHIRP_KINDS = {"linear", "quadratic", "logarithmic", "hyperbolic"}

def _signal_from_params(params):
    """Reconstruct the excitation chirp from a params dict.

    Handles old param files that stored the chirp type as ``type``
    instead of the current ``kind`` key, and silently returns None
    when the stored value is not a valid chirp method (e.g. old files
    that wrote the class name 'slab.Sound.chirp' as the type).
    """
    if "signal" not in params:
        return None
    sp = dict(params["signal"])  # copy – don't mutate the original
    if "type" in sp and "kind" not in sp:
        sp["kind"] = sp.pop("type")
    if sp.get("kind") not in _VALID_CHIRP_KINDS:
        logging.warning(
            f"Unrecognised chirp kind '{sp.get('kind')}' in params.txt – "
            "skipping signal reconstruction."
        )
        return None
    try:
        return slab.Sound.chirp(**sp).ramp(when="both", duration=0.001)
    except Exception as e:
        logging.warning(f"Could not reconstruct signal from params: {e}")
        return None


def parse_params_file(path, filename="params.txt"):
    path = Path(path)
    params = {}
    current = None
    with (path / filename).open() as f:
        for line in f:
            if not line.strip():
                continue
            if not line.startswith(" "):
                if line.endswith(":\n"):
                    key = line[:-2]
                    params[key] = {}
                    current = params[key]
                else:
                    k, v = line.split(":", 1)
                    params[k.strip()] = _parse_value(v.strip())
                    current = None
            else:
                if current is not None:
                    k, v = line.strip().split(":", 1)
                    current[k] = _parse_value(v.strip())
    return params

def _parse_value(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


# ---------------------------------------------------------------------
# Conversion utility
# ---------------------------------------------------------------------

def wav_to_npz(wav_path, npz_path=None, overwrite=False):
    """Convert an existing per-speaker wav folder to a single recordings.npz.

    Parameters
    ----------
    wav_path : path-like
        Folder that was previously written by ``Recordings.to_wav()``.
    npz_path : path-like, optional
        Destination folder for the .npz file.  Defaults to *wav_path*
        (i.e. the .npz lands next to the existing wav sub-folders).
    overwrite : bool
        Passed through to ``Recordings.to_npz()``.

    Returns
    -------
    Recordings
        The loaded object (also saved to disk as a side-effect).
    """
    wav_path = Path(wav_path)
    npz_path = Path(npz_path) if npz_path is not None else wav_path
    logging.info(f"Converting wav folder '{wav_path}' → npz at '{npz_path}'")
    rec = Recordings.from_wav(wav_path)
    rec.to_npz(npz_path, overwrite=overwrite)
    return rec