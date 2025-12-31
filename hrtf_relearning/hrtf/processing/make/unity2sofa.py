import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as numpy
import slab
import pyfar
import hrtf_relearning
import re

base = hrtf_relearning.PATH / 'data' / 'hrtf'
pattern = re.compile(r"hrir_az(?P<az>[+-]?\d+\.\d)_el(?P<el>[+-]?\d+\.\d)\.wav")
wav_files = sorted((base / 'rec' / 'universal_hrtf').glob("hrir_az*_el*.wav"))
fs = 48000
dist = 1.0

# I read wav files, write coordinates, align IR onsets and crop to 256 samples
data = []
sources = []
for fname in wav_files:
    ir = pyfar.io.read_audio(fname)
    az = float(pattern.match(fname.name).group('az'))
    el = float(pattern.match(fname.name).group('el'))
    onsets = pyfar.dsp.find_impulse_response_start(ir)
    aligned = pyfar.dsp.time_shift(
        ir, -numpy.min(onsets) / ir.sampling_rate + .001,
        unit='s')
    windowed = pyfar.dsp.time_window(aligned, interval=(0, 255), window='boxcar', crop='window')
    sources.append([az, el, dist])
    data.append(windowed.time)

# write sofa
hrtf = slab.HRTF(data=numpy.array(data), datatype='FIR', samplerate=fs, sources=numpy.array(sources))
hrtf.write_sofa(base / 'sofa' / 'universal.sofa')



"""
Write sources txt for use in unity from numpy array
"""
from pathlib import Path
import numpy
from hrtf_relearning.hrtf.processing.make.spherical_sources import *

import slab
import hrtf_relearning
hrtf_dir = hrtf_relearning.PATH / 'data' / 'hrtf' / 'sofa'

sources = spherical_sources(resolution_deg=5)

out = Path("C:/Users/paulf/UnityProjects/hrtf_relearning/Assets/StreamingAssets/sources.txt")
numpy.savetxt(
    out,
    sources,
    fmt="%.2f %.2f %.2f",
    header="az_deg el_deg radius_m",
    comments=""
)
print(f"Wrote {sources.shape[0]} sources to {out}")


# test
hrtf = slab.HRTF(hrtf_dir / 'universal.sofa')
az_cone = hrtf.cone_sources(plane='horizontal')
el_cone = hrtf.cone_sources(0)

sound = slab.Binaural.pinknoise()
# fig, ax = plt.subplots( figsize=(12, 8))
for src in el_cone:
    # ax.clear()
    # hrtf.plot_ir([src], ear='both', axis=[ax, ax])
    # plt.show()
    print(f'playing from {hrtf.sources.interaural_polar[src]}')
    hrtf.apply(src, sound).play()

# fig, ax = plt.subplots(figsize=(12, 8))
# for src in source_idx:
#     hrtf.plot_ir([src], axis=ax)
#     ax.get_lines()[-1].set_label(f'{hrtf.sources.vertical_polar[src]}')
#     plt.legend()
