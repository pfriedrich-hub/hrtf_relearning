# recordings.py
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
import matplotlib.pyplot as plt
import logging
import copy
from pathlib import Path
from datetime import datetime
import numpy
import slab
import soundfile as sf
import pyfar

# freefield is a RIG-ONLY dependency. A bare `import freefield` here makes every
# consumer of this module hardware-dependent -- including record.processing,
# whose own docstring promises "No I/O. No hardware. No FreeField." -- so the
# cue-editing install (and anything that only wants to re-expand or modify an
# existing SOFA) could not import it at all. It was previously moved into the
# one method that drives the rig and then pulled back out to module level
# ("has to stay here so it is defined across functions"), which reintroduced
# exactly that problem while the comment below still claimed otherwise.
#
# Bind it once, tolerating absence: module-level name for every function that
# needs it, no import cost at call time, and a clear failure at the point of USE
# rather than an ImportError at the point of import.
try:
    import freefield
except ImportError:  # no TDT drivers on this machine -- processing still works
    freefield = None
    logging.info(
        "recordings: freefield not available -- recording disabled, "
        "processing and SOFA editing still work.")


def _require_freefield():
    """Raise a useful error if a rig-only code path is reached without freefield."""
    if freefield is None:
        raise RuntimeError(
            "This function drives the loudspeaker dome and needs the `freefield` "
            "package (and the TDT drivers) installed. It is not available on this "
            "machine -- run it on the rig.")
    return freefield

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
                    equalize_dome=None, key=True, button=False, equalize=None):
        """Record binaural sweeps from the dome.

        `equalize_dome` pre-filters the emitted sweep with each speaker's
        calibration filter (`freefield.play_and_record(equalize=...)`) and is
        recorded verbatim in params.txt.

        NAMING (2026-08-19). The parameter used to be called `equalize` while
        every caller, every variable and params.txt itself called it
        `equalize_dome`. Setting a local `equalize_dome = False` and calling
        this method without passing it therefore left the default TRUE in
        force, silently, with params.txt reporting True -- which is how
        ref_19.08 and ref_19.08_swapped were recorded equalized when they were
        meant not to be. `equalize_dome` is now the parameter name; `equalize`
        still works and warns.

        DEFAULT (2026-08-19). Now **False**, matching what every subject on
        disk was recorded with. A reference recorded with the dome EQ on and a
        subject with it off do not cancel in `equalize()` -- see
        project_dome_eq_mismatch. Whichever you choose, choose the SAME for the
        subject and its reference.
        """
        if equalize is not None:
            if equalize_dome is not None:
                raise TypeError("pass equalize_dome, not both equalize_dome and equalize")
            logging.warning("record_dome: 'equalize' is deprecated, use 'equalize_dome'.")
            equalize_dome = equalize
        if equalize_dome is None:
            equalize_dome = False
        equalize = equalize_dome

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

        _require_freefield()
        if freefield.PROCESSORS.mode != "play_birec":
            freefield.initialize("dome", "play_birec")
        speakers = cls._select_speakers(freefield.read_speaker_table(), azimuth, elevation)
        led_bits = ['1', '4', '16']
        if n_directions > len(led_bits):
            raise ValueError(
                f"record_dome: n_directions={n_directions} but only "
                f"{len(led_bits)} LED bitmasks are defined {led_bits}. Add the "
                "masks for the extra head positions before asking for them.")

        # Interleaving step: the head is tilted by one FRACTION of the dome's
        # own elevation spacing per direction, so `n_directions` passes fill in
        # the gaps between physical speaker rows.
        #
        # This used to read `speakers[0].elevation - speakers[1].elevation`,
        # which assumes the speaker table comes back sorted by elevation AND
        # that its first two entries are adjacent rows -- neither is guaranteed
        # by `read_speaker_table`. Derive it from the sorted unique elevations
        # instead, and refuse to guess when the rows are not evenly spaced.
        unique_el = sorted({round(spk.elevation, 3) for spk in speakers})
        if len(unique_el) < 2:
            raise ValueError(
                "record_dome: need at least two distinct speaker elevations to "
                f"derive the interleaving step, got {unique_el}.")
        steps = numpy.diff(unique_el)
        if not numpy.allclose(steps, steps[0], atol=1e-2):
            raise ValueError(
                "record_dome: speaker elevations are not evenly spaced "
                f"({unique_el}), so a single interleaving step is undefined. "
                "Select a uniform elevation range, or record one row at a time.")
        res = float(abs(steps[0])) / n_directions
        min_el = min(spk.elevation for spk in speakers)
        data = {}

        for n in range(n_directions):

            elevation_step = n * res
            freefield.write(tag='bitmask', value=led_bits[n], processors='RX81')
            if button:
                print(f"Press Button when head is at {elevation_step:.2f}° elevation ...")
                freefield.wait_for_button()
            if key:
                input(f'Press Enter when head is at {elevation_step:.2f}° elevation ...')
            for base_spk in speakers:
                [spk] = copy.deepcopy(freefield.pick_speakers(base_spk.index))
                spk.elevation -= elevation_step
                if spk.elevation >= min_el:
                    logging.info(f"Recording from Speaker {spk.index} at {spk.azimuth:.1f}° azimuth"
                                 f" and {spk.elevation:.1f}° elevation")
                    # NOT `key`: that is the caller's "prompt between directions"
                    # flag. Overwriting it here made `if key:` above truthy from
                    # the second direction on, so key=False still stopped for
                    # Enter. Harmless at n_directions=1, wrong above it.
                    spk_key = f"{spk.index}_{spk.azimuth:.2f}_{spk.elevation:.2f}"
                    recs = cls.record_speaker(spk, signal, n_recordings, fs, equalize)
                    processed = []
                    for r in recs:
                        processed.append(filt.apply(r))
                    data[spk_key] = processed
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

    def to_npz(self, path, overwrite=False, filename="recordings.npz"):
        """Save recordings to a single .npz file.

        Array shape: (n_locations, n_recordings, n_channels, n_datapoints).
        Speaker-location keys are stored alongside so the dict can be reconstructed.

        params.txt is written only once the arrays have actually been stored, so
        the two can never describe different recording sessions. Writing it
        before the overwrite guard used to leave a folder whose params.txt
        announced a new session while recordings.npz still held the old sweeps.

        `filename` lets a second, different set of sweeps live in the SAME
        subject folder rather than in a sibling one -- the head-radius azimuth
        row is stored as `azimuth.npz` + `azimuth_params.txt` next to
        `recordings.npz` + `params.txt`. See `params_filename`.
        """
        logging.info(f'Writing recordings to .npz: {path}.')
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        fname = path / filename
        if fname.exists() and not overwrite:
            logging.warning(
                f"{fname} already exists – NOT saving these recordings "
                f"(use overwrite=True to replace). params.txt is left describing "
                f"the stored arrays."
            )
            return

        keys = list(self.data.keys())
        if not keys:
            raise ValueError(
                "Recordings.data is empty – nothing to save. "
                "Nothing was recorded, or the recording loop never populated it."
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
        self.write_params_file(path, filename=params_filename(filename))

    @classmethod
    def from_npz(cls, path, filename="recordings.npz"):
        """Load recordings from a .npz file saved by to_npz()."""
        path = Path(path)
        params = parse_params_file(path, filename=params_filename(filename))

        npz_file = path / filename
        _warn_if_params_newer_than_arrays(path, npz_file, params)

        npz = numpy.load(npz_file, allow_pickle=False)
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
    def load(cls, path, filename="recordings.npz"):
        """Load recordings from *path*.

        The pre-npz per-speaker .wav layout is gone (removed 2026-08-19); every
        recording folder in the repo carries a .npz. If an archived wav folder
        ever turns up, restore `from_wav`/`wav_to_npz` from git history and
        convert it once rather than reviving the fallback.
        """
        path = Path(path)
        if not (path / filename).exists():
            raise FileNotFoundError(
                f"{path / filename} not found. Recordings are stored as .npz; "
                "there is no .wav fallback any more.")
        logging.info(f"Loading recordings from .npz: {path / filename}")
        return cls.from_npz(path, filename=filename)

    # WAV I/O (to_wav / from_wav) removed 2026-08-19 -- the pre-npz layout is
    # gone and every recording folder carries a .npz. In git history if needed.

    def plot(self, speaker_idx=4):
        """Overlay every repetition recorded at one speaker, both ears.

        INTERACTIVE USE ONLY -- nothing in the pipeline calls this. Kept
        deliberately for eyeballing a set in a console; do not assume it is
        exercised by any run.
        """
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


def params_filename(npz_filename):
    """Name of the params file that belongs to an .npz in the same folder.

    'recordings.npz' -> 'params.txt' (the historical name, unchanged, so every
    existing folder keeps loading). Anything else -> '<stem>_params.txt', e.g.
    'azimuth.npz' -> 'azimuth_params.txt'.
    """
    stem = Path(npz_filename).stem
    return "params.txt" if stem == "recordings" else f"{stem}_params.txt"


def _warn_if_params_newer_than_arrays(path, npz_file, params, tolerance_s=3600):
    """Warn when params.txt describes a later session than the stored arrays.

    A folder written before the to_npz fix can hold a params.txt from a
    re-recording next to the sweeps of the ORIGINAL session, in which case
    everything loaded from here is silently mislabelled.
    """
    stamp = params.get("datetime") if isinstance(params, dict) else None
    if not stamp:
        return
    try:
        params_dt = datetime.fromisoformat(str(stamp))
        npz_dt = datetime.fromtimestamp(npz_file.stat().st_mtime)
    except (ValueError, OSError):
        return
    if (params_dt - npz_dt).total_seconds() > tolerance_s:
        logging.warning(
            f"{path}: params.txt is dated {params_dt.isoformat(timespec='seconds')} but "
            f"recordings.npz was last written {npz_dt.isoformat(timespec='seconds')}. "
            f"The sweeps are probably from an earlier session than params.txt claims – "
            f"do not trust this folder's metadata."
        )


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
