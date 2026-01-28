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
    def record_dome(cls, id=None, azimuth=0, elevation=(37.5, -37.5),
                    n_directions=2, n_recordings=5, hp_freq=120, fs=48828, equalize=False):

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
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
        res = abs(speakers[0].elevation - speakers[1].elevation) / n_directions
        min_el = min(spk.elevation for spk in speakers)
        data = {}
        for n in range(n_directions):

            elevation_step = n * res
            # freefield.write(tag='bitmask', value=led_speaker.digital_channel,
            #                 processors=led_speaker.digital_proc)  # illuminate LED
            # input(f"Press Enter when head is at {0 + elevation_step}° elevation ...")

            # freefield.play_start_sound()
            print(f"Press Button when head is at {0 + elevation_step}° elevation ...")
            freefield.wait_for_button()

            for base_spk in speakers:
                [spk] = copy.deepcopy(freefield.pick_speakers(base_spk.index))
                spk.elevation -= elevation_step
                if spk.elevation >= min_el:
                    logging.info(f"Recording from Speaker {spk.index} at {spk.azimuth:.1f}° azimuth"
                                 f" and {spk.elevation:.1f}° elevation")
                    key = f"{spk.index}_{spk.azimuth}_{spk.elevation}"
                    recs = cls.record_speaker(spk, signal, n_recordings, fs, equalize)
                    processed = []
                    for r in recs:
                        processed.append(filt.apply(r))
                    data[key] = processed
            freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED

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

    # -------------------- WAV I/O -------------------------------------

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
