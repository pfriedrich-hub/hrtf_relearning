import freefield
import slab
from pathlib import Path
import numpy

if not freefield.PROCESSORS.mode:
    freefield.initialize('dome', default='play_rec')
freefield.load_equalization(Path.cwd() / 'final_data' / 'calibration' / 'calibration_dome_13.01')

noise = slab.Sound.pinknoise(duration=0.5, level=90)
noise = noise.ramp(when='both', duration=0.01)

# read list of speaker locations
table_file = freefield.DIR / 'final_data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_ids = numpy.loadtxt(table_file, skiprows=1, usecols=(0), delimiter=",", dtype=int)
speaker_ids = numpy.delete(speaker_ids, [19, 23, 27], axis=0)  # remove disconnected speaker from speaker_list

freefield.set_logger('WARNING')
freefield.wait_for_button()
for speaker_id in speaker_ids:
    freefield.set_signal_and_speaker(signal=noise, speaker=speaker_id, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()