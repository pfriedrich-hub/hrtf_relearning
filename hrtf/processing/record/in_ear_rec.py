import matplotlib
matplotlib.use('tkagg')
from pynput import keyboard
from pathlib import Path
import slab
import freefield
import logging
import numpy
import copy
data_dir = Path.cwd() / 'data'

def record_dome(n_samp=5, n_rec=20, fs=48828):
    """
    Play cross the central loudspeaker array and record from binaural in-ear microphones.
    Args:
        n_samp (int): Number of different listener orientations to record (increase spatial resolution).
         For this to work, listeners should consecutively orient their head towards n_samp evenly spaced points between
         the center speaker and the neighboring speaker above.
        n_rec (int): Number of recordings per speaker location to average across.
    """
    # slab.set_default_samplerate(fs)
    if not freefield.PROCESSORS.mode == 'play_birec':
        freefield.initialize('dome', 'play_birec')
    # signal = slab.Sound(data_dir / 'sounds' / 'log_sweep.wav')  # duration=0.2, level=80, from_frequency=20, to_frequency=fs/2
    signal = slab.Sound.chirp(duration=0.2, level=80, from_frequency=20, to_frequency=48828/2, samplerate=48828)
    speakers = get_speakers(speakers=freefield.read_speaker_table(), azimuth=0, elevation=None)
    res = abs(speakers[0].elevation - speakers[1].elevation) / n_samp  # resolution
    recordings_dict = dict()
    for n in range(n_samp):
        input('enter')
        elevation_step = (n) * res  #
        # wait_for_button(msg=f'Rest gaze at +{elevation_step} elevation and press Enter to continue.')
        for speaker in speakers:
            speaker = copy.deepcopy(speaker)
            speaker.elevation = speaker.elevation - elevation_step  # set new elevation
            if speaker.elevation >= min([speaker.elevation for speaker in speakers]):  # lower elevation cut off
                logging.info(f'Recording from Speaker index {speaker.index} at {speaker.elevation}° elevation.')
                recordings_dict[f'{speaker.index}_{speaker.azimuth}_{speaker.elevation}']\
                    =record_speaker(speaker, signal, n_rec, fs)
    return recordings_dict

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
        if file:
            recordings_dict[file.stem] = slab.Binaural(file)
        else:
            logging.error(f'No .wav files found in {path}.')
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

def record_reference():
    """
    Helper function to record the reference with free microphones.
    """
    logging.info('Recording reference')
    # ref_dir = data_dir / 'hrtf' / 'rec' / 'reference' / fname
    reference_dict = record_dome(n_samp=1, n_rec=20)
    return reference_dict

def plot_dict(data_dict, kind='tf'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ylim = []
    for key, item in data_dict.items():
        # --- Extract elevation (last number after underscores) ---
        try:
            elevation = float(key.split('_')[-1])
        except (ValueError, IndexError):
            elevation = 0.0
        # --- Plot depending on object type ---
        if isinstance(item, slab.Binaural):
            item.spectrum(axis=ax)
            ax.set_ylim(-130, 20)
        elif isinstance(item, slab.Filter):
            if kind.upper() == 'TF':
                _ = item.tf(show=True, axis=ax)
            elif kind.upper() == 'IR':
                ax.plot(item.data)
        ylim.append([item.data.min(), item.data.max()])
        # --- Shift line vertically by elevation ---
        line = ax.lines[-1]
        x, y = line.get_xdata(), line.get_ydata()
        line.set_ydata(y + elevation)  # use elevation as baseline offset
        line.set_label(f"el={elevation:.1f}°")
    # --- Cosmetics ---
    # ylim = numpy.mean(ylim, axis=0)
    ax.set_xlim([2e3, 18e3])
    ax.set_xscale('linear')
    ax.legend(title="Elevation (°)")
    ax.set_title(f"Filter dictionary ({kind.upper()})")
    ax.set_xlabel("Frequency" if kind.upper() == 'TF' else "Samples")
    ax.set_ylabel("Amplitude (offset by elevation)")
    plt.legend()
    plt.show()
