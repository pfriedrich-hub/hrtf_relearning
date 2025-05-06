import pathlib

import slab
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import math
import pathlib
sofapath=pathlib.Path.cwd() / 'data' / 'hrtf' / 'sofa'
# needed for button response
slab.psychoacoustics.input_method = 'figure'

# hrtf = slab.HRTF("KU100_HRIR_L2702.sofa")
# hrtf = slab.HRTF("FABIAN.sofa")
hrtf = slab.HRTF(sofapath / "single_notch.sofa")

slab.set_default_samplerate(hrtf.samplerate)

src = np.array(hrtf.cone_sources(0, full_cone=True))
x = hrtf.sources.vertical_polar[src, 0] == 0
sourceidx = src[x]


def set_diff(type, diff):
    fundamental_frequency = 500
    duration = 0.5
    samplerate = hrtf.samplerate
    n_harmonics = 40
    k = 1.0
    a = 0.5
    even_boost = 1.0
    times = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    harmonics = np.zeros_like(times)
    for n in range(1, n_harmonics + 1):
        frequency = fundamental_frequency * n  # Frequenz der n-ten Harmonischen
        if n % 2 == 0:
            amplitude = k / (n ** a) / even_boost  # Amplitude der n-ten Harmonischen
        else:
            amplitude = k / (n ** a)
        harmonics += amplitude * np.sin(2 * np.pi * frequency * times)
    tone = slab.Sound(data=harmonics, samplerate=samplerate)
    tone.level = 80
    tone = tone.ramp(duration=.005)
    spatial_sounds = []
    if type == "timbre":
        spatial_sounds.append(hrtf.apply(sourceidx[int(math.ceil(len(sourceidx)-1)/2)], tone))
        even_boost = diff
        harmonics = np.zeros_like(times)
        for n in range(1, n_harmonics + 1):
            frequency = fundamental_frequency * n
            if n % 2 == 0:
                amplitude = k / (n ** a) / even_boost
            else:
                amplitude = k / (n ** a)
            harmonics += amplitude * np.sin(2 * np.pi * frequency * times)
        tone = slab.Sound(data=harmonics, samplerate=samplerate)
        tone.level = 80
        tone = tone.ramp(duration=.005)
        spatial_sounds.append(hrtf.apply(sourceidx[int(math.ceil(len(sourceidx)-1)/2)], tone))
    if type == "hrtf":
        for i, index in enumerate([sourceidx[0], sourceidx[diff]]):  # evtl null (referenzelevation) ändern
            # diff ist variierender speaker
            spatial_sounds.append(hrtf.apply(index, tone))
    return spatial_sounds


def threshold(type):  # shape = even odd difference factor, speaker = hrtf source location
    if type == "timbre":
        stairs = slab.Staircase(start_val=2, step_sizes=[0.5, 0.05], min_val=1.0)
    if type == "hrtf":
        stairs = slab.Staircase(start_val=6, step_sizes=[2, 1], min_val=0, max_val=len(sourceidx)-1)
    for diff in stairs:
        print(diff)
        stimuli = set_diff(type, diff)
        stimuli[0].play()
        time.sleep(0.5)
        stimuli[1].play()
        with slab.key('Press y for different or n for no difference.') as key:
            response = key.getch()
        if response == 121:  # 121 is the unicode for the "y" key
            stairs.add_response(True)  # initiates calculation of next stimulus value
        else:
            stairs.add_response(False)
    stairs.plot()
    stairs.threshold()
    with open("threshold_timbre_or_hrtf.txt", "a") as f:
        f.write(f"type = {type}, threshold = {stairs.threshold()}, n_elevations in hrtf possible = {hrtf.n_elevations}\n")


threshold("hrtf")
threshold("timbre")


type = "hrtf"
sounds = []
fig, ax = plt.subplots()
for diff in range(0, len(sourceidx)-1):
    sounds.append(set_diff(type, diff))
    sounds[diff][1].spectrum(axis=ax, show=False)
plt.show()

type = "timbre"
sounds = []
fig, ax = plt.subplots()
diff = [1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 4.0, 8.0]
for index, i in enumerate(diff):
    sounds.append(set_diff(type, i))
    sounds[index][1].spectrum(axis=ax, show=False)
plt.show()
