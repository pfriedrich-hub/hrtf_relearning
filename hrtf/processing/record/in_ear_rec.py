from pynput import keyboard
from pathlib import Path
import slab
import freefield
import logging
import numpy
import copy
data_dir = Path.cwd() / 'data'

def record_dome(n_samp=5, n_rec=20):
    """
    Play cross the central loudspeaker array and record from binaural in-ear microphones.
    Args:
        n_samp (int): Number of different listener orientations to record (increase spatial resolution).
         For this to work, listeners should consecutively orient their head towards n_samp evenly spaced points between
         the center speaker and the neighboring speaker above.
        n_rec (int): Number of recordings per speaker location to average across.
    """
    freefield.initialize('dome', 'play_birec')
    freefield.set_logger('warning')
    signal = slab.Sound(data_dir / 'sounds' / 'log_sweep.wav')  # duration=0.2, level=80, from_frequency=20, to_frequency=fs/2
    speakers = get_speakers(speakers=freefield.read_speaker_table(), azimuth=0, elevation=None)
    res = abs(speakers[0].elevation - speakers[1].elevation) / n_samp  # resolution
    recordings_dict = dict()
    for n in range(n_samp):
        elevation_step = (n) * res  #
        # wait_for_button(msg=f'Rest gaze at +{elevation_step} elevation and press Enter to continue.')
        for speaker in speakers:
            speaker = copy.deepcopy(speaker)
            speaker.elevation = speaker.elevation - elevation_step  # set new elevation
            if speaker.elevation >= min([speaker.elevation for speaker in speakers]):  # lower elevation cut off
                logging.info(f'Recording from Speaker index {speaker.index} at {speaker.elevation}')
                recordings_dict[f'{speaker.index}_{speaker.azimuth}_{speaker.elevation}']=record_speaker(speaker, signal, n_rec)
    return recordings_dict

def record_speaker(speaker, signal, n_rec):
    """
    Record n times from a specified speaker
    Args:
        speaker (Speaker): Speaker to record.
        signal (Signal): Signal to record.
        n_rec (int): Number of recordings per speaker location to average across.
    """
    recordings = []
    for r in range(n_rec):  # record n_rec times and average
        recordings.append(freefield.play_and_record(speaker, signal, equalize=False))  # todo check samplerate and equalize
    return slab.Binaural(numpy.mean(recordings, axis=0))

def recordings2wav(recordings_dict, path):
    """
    Write all recordings in a dict to WAV files.

    Parameters
    ----------
    recordings_dict : dict[float, slab.Binaural] : Dictionary mapping source coords to recordings.
    path : Path | str: Base directory for saving.
    """
    for key, recording in recordings_dict.items():
        recording.write(path / f'{key}.wav')

def wav2recordings(path):
    """
    Create recordings dictionary from wav files.
    """
    recordings_dict = dict()
    for file in path.glob('*.wav'):
        recordings_dict[file.stem] = slab.Binaural(file)
    return recordings_dict

def wait_for_button(msg=None):
    if msg: logging.info(msg)
    def on_press(key):
        if key == keyboard.Key.enter:
            listener.stop()  # stop listening once Enter is pressed
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # block until listener.stop() is called

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

def record_reference(path):
    """
    Helper function to record the reference with free microphones.
    """
    # ref_dir = data_dir / 'hrtf' / 'rec' / 'reference' / fname
    if not path.exists():  # create folder structure
        path.mkdir(exist_ok=True, parents=True)
    reference_dict = record_dome(n_samp=1, n_rec=20)
    recordings2wav(reference_dict, path)
    return reference_dict
