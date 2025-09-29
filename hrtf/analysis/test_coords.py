import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import slab
from pathlib import Path

# sofa_name ='single_notch'
sofa_name ='KU100_HRIR_L2702'
# sofa_name ='kemar'

# hrir = slab.HRTF.kemar()
hrir = slab.HRTF(Path.cwd() / 'data' / 'hrtf' / 'sofa' / str(sofa_name + '.sofa'))
azimuth = 15
elevation = 0

src_idx = hrir.get_source_idx(azimuth, elevation, tolerance=0.05)

hrir.sources.vertical_polar[src_idx]
hrir.sources.interaural_polar[src_idx]  # matches azimuth
hrir.plot_sources(src_idx)

filt = hrir[src_idx[0]]
noise_filt = filt.apply(slab.Binaural.pinknoise(samplerate=hrir.samplerate))
itd = - noise_filt._get_itd(0.001) / noise_filt.samplerate
ild = - noise_filt.ild()
# print(slab.Binaural.itd_to_azimuth(itd))  # matches azimuth in kemar and ku100
# print(slab.Binaural.ild_to_azimuth(ild))  # matches azimuth only in kemar


#compare
sofa_name2 ='MRT01'
hrir2 = slab.HRTF(Path.cwd() / 'data' / 'hrtf' / 'sofa' / str(sofa_name2 + '.sofa'))
src_idx2 = hrir2.get_source_idx(azimuth, elevation, tolerance=0.05)
plt.figure()
hrir2.plot_sources(src_idx2)
filt2 = hrir2[src_idx2[0]]
noise_filt1 = filt.apply(slab.Binaural.pinknoise(samplerate=hrir.samplerate))
noise_filt2 = filt2.apply(slab.Binaural.pinknoise(samplerate=hrir2.samplerate))

import time
noise_filt1.play()
time.sleep(noise_filt1.duration)
noise_filt2.play()