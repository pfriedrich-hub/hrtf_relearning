import freefield
from pathlib import Path
import numpy as np
DIR = Path.cwd()

# read arduino data
proc_list = [['RP2', 'RP2',  DIR / 'data' / 'rcx' / 'arduino_analog.rcx']]
freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')
while True:
    az = freefield.read(tag='azimuth', processor='RP2', n_samples=1)
    ele = freefield.read(tag='elevation', processor='RP2', n_samples=1)
    az = np.interp(az, [0.55, 2.75], [0, 360])
    ele = np.interp(ele, [0.55, 2.75], [-90, 90])
    print('azimuth: %i, elevation: %i '%(int(az), int(ele)))
