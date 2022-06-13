import torch
import freefield
import slab
import numpy as np
from pathlib import Path
from numpy import linalg as la
import time

DIR = freefield.DIR
table_file = DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
SPEAKERS = []
table = np.loadtxt(table_file, skiprows=1, delimiter=",", dtype=str)
for row in table:
    SPEAKERS.append(freefield.Speaker(index=int(row[0]), analog_channel=int(row[1]), analog_proc=row[2],
                            azimuth=float(row[3]), digital_channel=int(row[5]) if row[5] else None,
                            elevation=float(row[4]), digital_proc=row[6] if row[6] else None))
leds=[s for s in SPEAKERS if s.digital_channel is not None]

# calibrate camera
time.sleep(15)
freefield.calibrate_camera(leds, camera_type='flir', pose_method='aruco')



led_speakers = freefield.all_leds()