import freefield
import slab
import time
import pathlib
import os
import numpy as np
from matplotlib import pyplot as plt
DIR = pathlib.Path(os.getcwd()) # path for sound and rcx files
sampling_rate=48828

# read arduino data
proc_list = [['RP2', 'RP2',  DIR / 'data' / 'rcx' / 'arduino_analog.rcx']]
freefield.initialize('dome', zbus=True, device=proc_list)

# freefield.write(tag="playbuflen", value=sampling_rate*5, processors="RP2")
# freefield.play(kind='zBusA',proc='RP2')
# time.sleep(5)
# data = freefield.read(tag='azimuth',processor='RP2',n_samples=500)
# plt.figure()
# plt.plot(data)

freefield.set_logger('WARNING')

while True:
    az = freefield.read(tag='azimuth', processor='RP2', n_samples=1)
    ele = freefield.read(tag='elevation', processor='RP2', n_samples=1)
    az = 360-np.interp(az, [0.55, 2.75], [0, 360])
    ele = np.interp(ele, [0.55, 2.75], [-90, 90])
    print('azimuth: %i, elevation: %i '%(int(az), int(ele)))







plt.figure()
spectrum=np.fft.fft(data)
plt.plot(x,spectrum)


# record in ear
proc_list = [['RP2', 'RP2',  DIR / 'data' / 'birec_buf.rcx']]
freefield.initialize('dome', zbus=True, device=proc_list)
sound = slab.Sound.pinknoise(duration=3.0)
sound.level = 90
freefield.load_equalization()

speaker = (0, -50)
speaker = (0, -37.5)
speaker = (0, -25)
speaker = (0, -12.5)
speaker = (0, 0)
speaker = (0, 12.5)
speaker = (0, 25)
speaker = (0, 37.5)
speaker = (0, 50)

recordings = []
recordings.append(freefield.play_and_record(speaker=speaker, sound=sound))


freefield.set_signal_and_speaker(speaker=speaker, signal=sound, equalize=False)
freefield.play()

recording.write('bla.wav')

for i in recordings:
    i.left.spectrum()
    plt.


