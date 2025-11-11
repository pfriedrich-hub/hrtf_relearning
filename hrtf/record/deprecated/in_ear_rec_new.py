from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional
import slab
import freefield
import logging
import numpy
import copy

@dataclass
class DomeRecordings:
    """Container for binaural recordings across a loudspeaker array."""
    recordings: Dict[str, slab.Binaural] = field(default_factory=dict)
    fs: Optional[int] = None
    meta: Dict[str, object] = field(default_factory=dict)

    # --- Dict-like convenience ------------------------------------------------
    def __getitem__(self, key: str) -> slab.Binaural:
        return self.recordings[key]

    def __setitem__(self, key: str, value: slab.Binaural) -> None:
        self.recordings[key] = value

    def __iter__(self):
        return iter(self.recordings)

    def items(self):
        return self.recordings.items()

    def keys(self):
        return self.recordings.keys()

    def values(self):
        return self.recordings.values()

    def __len__(self) -> int:
        return len(self.recordings)

    # --- IO --------------------------------------------------------------------
    def to_wav(self, path: Path | str) -> None:
        """Write all recordings as WAV files to `path`."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key, recording in self.recordings.items():
            recording.write(path / f"{key}.wav")

    @classmethod
    def from_wav(cls, path: Path | str, fs: Optional[int] = None, meta: Optional[Dict[str, object]] = None):
        """Create DomeRecordings from all .wav files in `path`."""
        path = Path(path)
        recordings = {}
        for file in path.glob("*.wav"):
            recordings[file.stem] = slab.Binaural(file)
        if not recordings:
            raise FileNotFoundError(f"No .wav files found in {path}")
        if fs is None:
            # try to infer samplerate from first recording
            first = next(iter(recordings.values()))
            fs = first.samplerate
        return cls(recordings=recordings, fs=fs, meta=meta or {})

    # --- Helpers for key parsing ----------------------------------------------
    @staticmethod
    def parse_key(key: str) -> Tuple[int, float, float]:
        """
        Parse keys like '27_0.0_-50.0' -> (index, azimuth, elevation).
        """
        parts = key.split("_")
        if len(parts) != 3:
            raise ValueError(f"Cannot parse key '{key}', expected 'idx_az_el'")
        idx = int(parts[0])
        az = float(parts[1])
        el = float(parts[2])
        return idx, az, el

    def select(self,
               azimuth: Optional[float | Tuple[float, float]] = None,
               elevation: Optional[float | Tuple[float, float]] = None) -> "DomeRecordings":
        """
        Return a new DomeRecordings object with a subset of recordings
        matching the given azimuth/elevation criteria (like get_speakers).
        """
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

        subset = {}
        for key, rec in self.recordings.items():
            _, az, el = self.parse_key(key)
            if az_low <= az <= az_high and el_low <= el <= el_high:
                subset[key] = rec

        return DomeRecordings(recordings=subset, fs=self.fs, meta=self.meta.copy())

    # --- Plotting --------------------------------------------------------------
    def plot(self, kind: str = "tf"):
        """
        Plot all recordings similar to your `plot_dict` function.
        `kind` can be 'tf' or 'ir' for Filters; for Binaural we use spectrum().
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        for key, item in self.recordings.items():
            # elevation from key
            try:
                _, _, elevation = self.parse_key(key)
            except Exception:
                elevation = 0.0

            if isinstance(item, slab.Binaural):
                item.spectrum(axis=ax)
                ax.set_ylim(-130, 20)
                # shift last line by elevation
                line = ax.lines[-1]
                x, y = line.get_xdata(), line.get_ydata()
                line.set_ydata(y + elevation)
            elif isinstance(item, slab.Filter):
                if kind.upper() == "TF":
                    _ = item.tf(show=True, axis=ax)
                elif kind.upper() == "IR":
                    ax.plot(item.data)
                line = ax.lines[-1]
                x, y = line.get_xdata(), line.get_ydata()
                line.set_ydata(y + elevation)

            line.set_label(f"el={elevation:.1f}°")

        ax.set_xlim([2e3, 18e3])
        ax.set_xscale("linear")
        ax.legend(title="Elevation (°)")
        ax.set_title(f"Filter dictionary ({kind.upper()})")
        ax.set_xlabel("Frequency" if kind.upper() == "TF" else "Samples")
        ax.set_ylabel("Amplitude (offset by elevation)")
        plt.show()

def record_dome(signal, n_samp=5, n_rec=20, fs=48828):
    """
    Play across the central loudspeaker array and record from binaural in-ear microphones.
    """
    if not freefield.PROCESSORS.mode == 'play_birec':
        freefield.initialize('dome', 'play_birec')

    speakers = get_speakers(speakers=freefield.read_speaker_table(), azimuth=0, elevation=None)
    res = abs(speakers[0].elevation - speakers[1].elevation) / n_samp  # resolution

    recordings_dict = {}

    for n in range(n_samp):
        input('enter')
        elevation_step = n * res
        for speaker in speakers:
            speaker = copy.deepcopy(speaker)
            speaker.elevation = speaker.elevation - elevation_step
            if speaker.elevation >= min([sp.elevation for sp in speakers]):
                logging.info(f'Recording from Speaker index {speaker.index} at {speaker.elevation}° elevation.')
                key = f'{speaker.index}_{speaker.azimuth}_{speaker.elevation}'
                recordings_dict[key] = record_speaker(speaker, signal, n_rec, fs)

    return DomeRecordings(
        recordings=recordings_dict,
        fs=fs,
        meta=dict(n_samp=n_samp, n_rec=n_rec)
    )

def record_reference(signal, n_rec=20, fs=48828):
    """
    Helper function to record the reference with free microphones.
    """
    logging.info('Recording reference')
    reference = record_dome(signal=signal, n_samp=1, n_rec=n_rec, fs=fs)  # adjust signal arg as needed
    return reference  # now a DomeRecordings instance

def record_speaker(speaker, signal, n_rec, fs):
    """
    Record n times from a specified speaker
    Args:
        speaker (Speaker): Speaker to record.
        signal (Signal): Signal to record.
        n_rec (int): Number of recordings per speaker location to average across.
    """
    recordings = []
    for r in range(n_rec):  # record n_rec times and average
        # record with high samplerate for now (bi rec buf rcx is set to 100k fs)
        recordings.append(freefield.play_and_record(speaker, signal, equalize=False, recording_samplerate=fs))# 97656
    return slab.Binaural(numpy.mean(recordings, axis=0))


def get_speakers(speakers, azimuth=None, elevation=None):
    """
    Fetch speaker objects within the provided azimuth/elevation ranges.
    Parameters
    ----------
    azimuth : float | tuple[float, float] | None
        - float → select speakers exactly at this azimuth
        - tuple(min, max) → select speakers in this inclusive azimuth range
        - None → ignore azimuth, return all azimuths
    elevation : float | tuple[float, float] | None
        - float → select speakers exactly at this elevation
        - tuple(min, max) → select speakers in this inclusive elevation range
        - None → ignore elevation, return all elevations
    Returns
    -------
    list[Speaker]
        List of Speaker objects matching the criteria.
    """
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
    for spk in speakers:  # assumes global or passed-in list of Speaker objects
        if az_low <= spk.azimuth <= az_high and el_low <= spk.elevation <= el_high:
            out.append(spk)
    return out

