# recordings.py
import logging
import copy
from pathlib import Path
from datetime import datetime
import numpy
import slab
import freefield
import soundfile as sf

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
    def record_dome(cls, n_directions=5, n_recordings=5, hp_freq=120, fs=48828):

        if freefield.PROCESSORS.mode != "play_birec":
            freefield.initialize("dome", "play_birec")

        # excitation
        sig_params = dict(
            type="slab.Sound.chirp",
            duration=0.2,
            level=85,
            from_frequency=120,
            to_frequency=fs / 2,
            samplerate=fs,
        )
        signal = slab.Sound.chirp(**sig_params)
        signal = signal.ramp(when="both", duration=0.001)
        signal.params = sig_params

        filt = slab.Filter.band("hp", hp_freq, fs)

        speakers = cls._select_speakers(
            freefield.read_speaker_table(), azimuth=0, elevation=(50, -37.5)
        )

        data = {}

        for spk in speakers:
            key = f"{spk.index}_{spk.azimuth}_{spk.elevation}"
            recs = cls.record_speaker(spk, signal, n_recordings, fs * 2)
            processed = []
            for r in recs:
                r.data -= numpy.mean(r.data, axis=0)
                processed.append(filt.apply(r))
            data[key] = processed

        params = dict(
            fs=fs,
            n_recordings=n_recordings,
            n_directions=n_directions,
            signal=sig_params,
            highpass_frequency=hp_freq,
            datetime=datetime.now().isoformat(),
        )

        return cls(data=data, params=params, signal=signal)

    @staticmethod
    def record_speaker(speaker, signal, n_recordings, fs):
        out = []
        for _ in range(n_recordings):
            rec = freefield.play_and_record(
                speaker=speaker,
                sound=signal,
                compensate_delay=True,
                equalize=False,
                recording_samplerate=fs,
            )
            out.append(slab.Binaural(rec))
        return out

    @staticmethod
    def _select_speakers(speakers, azimuth=None, elevation=None):
        out = []
        for s in speakers:
            if azimuth is not None and s.azimuth != azimuth:
                continue
            if elevation is not None:
                lo, hi = min(elevation), max(elevation)
                if not (lo <= s.elevation <= hi):
                    continue
            out.append(s)
        return out

    # -------------------- WAV I/O -------------------------------------

    def to_wav(self, path, overwrite=False):
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

        for kdir in path.iterdir():
            if not kdir.is_dir():
                continue
            recs = []
            for f in sorted(kdir.glob("rec_*.wav")):
                x, fs = sf.read(f, dtype="float32", always_2d=True)
                recs.append(slab.Binaural(x, fs))
            if recs:
                data[kdir.name] = recs

        signal = None
        if "signal" in params:
            sp = params["signal"]
            signal = slab.Sound.chirp(**sp).ramp(when="both", duration=0.001)

        return cls(data=data, params=params, signal=signal)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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
                    current[k] = _parse_value(v)
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
