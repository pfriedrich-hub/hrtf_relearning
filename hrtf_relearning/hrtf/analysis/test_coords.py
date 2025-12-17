import slab
from pathlib import Path
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

hrir = slab.HRTF(sofa_path / 'KU100.sofa')
hrir = slab.HRTF(sofa_path / 'kemar.sofa')
hrir = slab.HRTF(sofa_path / 'single_notch.sofa')
hrir = slab.HRTF(sofa_path / 'pf.sofa')
hrir = slab.HRTF(sofa_path / 'pf_itd.sofa')



# compute itd / ild of all filters on the horizontal plane
src_idx = hrir.cone_sources(0, True, 'elevation', .01)
# sound = slab.Binaural.pinknoise(samplerate=hrir.samplerate)
sound = slab.Binaural.tone(samplerate=hrir.samplerate, frequency=500, duration=60.0)
_s = sound.itd(30)
sound = slab.Binaural.pinknoise(samplerate=hrir.samplerate)
sound = slab.Binaural.tone(samplerate=hrir.samplerate, frequency=1500, level=65)


# ils = slab.Binaural.make_interaural_level_spectrum(hrir)
for id in src_idx:
    print(f'Source {hrir.sources.vertical_polar[id, 0]}')
    _s = hrir.apply(id, sound)
    _s.play()
    print(f'ITD {-slab.Binaural.itd_to_azimuth(_s.itd() / _s.samplerate)}')
    # print(f'ILD {-slab.Binaural.ild_to_azimuth(_s.ild(), frequency=4000, ils=None)}')
    # print(f'ILD {_s.ild()}')

# azimuths = numpy.arange(-90, 91)
# if not ils:
#     # ils = Binaural.make_interaural_level_spectrum()
# ilds = [numpy.diff(Binaural.azimuth_to_ild(az, frequency=7000, ils=ils)) for az in azimuths]
# ilds = numpy.asarray(ilds).flatten()
# min_ild, max_ild = numpy.min(ilds), numpy.max(ilds)
# if not (min_ild <= ild <= max_ild):
#     raise ValueError(f"ILD value {ild:.2f} dB is outside the interpolation range "
#                      f"({min_ild:.2f} – {max_ild:.2f} dB); extrapolation is not supported.")
# float(numpy.interp(ild, ilds, azimuths))

# play sources at 0 elevation around the listener from slab filter and wav
from pathlib import Path
import slab
sofa_name = Path('KU100.sofa')
sofa_name = Path('single_notch.sofa')
hrtf_path = Path.cwd() / 'data' / 'hrtf'
hrir = slab.HRTF(hrtf_path / 'sofa' / sofa_name)
wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav' / sofa_name.stem / 'IR_data'
sound = slab.Sound.pinknoise(duration=0.5, samplerate=hrir.samplerate)
src_idx = hrir.cone_sources(0, True, 'elevation', .01)
for idx in src_idx:
    source = hrir.sources.vertical_polar[idx]
    filt_wav = slab.Sound.read(wav_path / f'{source[0]}_{source[1]}.wav')
    filt = slab.Filter(data=filt_wav.data, fir='IR')
    print(f'Azimuth: {source[0]}')
    hrir[idx].apply(sound).play()
    # filt.apply(sound).play()

"""
itd = _s.itd() / _s.samplerate
max_lag = 0.001
azimuths = numpy.arange(-90, 91)
itds = [slab.Binaural.azimuth_to_itd(az) for az in azimuths]
if not (-max_lag <= itd <= max_lag):
    raise ValueError(
        f"ITD value {itd:.6f} s is outside the interpolation range "
        f"({-max_lag:.6f} – {max_lag:.6f} s); extrapolation is not supported."
    )
return float(numpy.interp(itd, itds, azimuths))
"""

import slab
from pathlib import Path

hrir_single_notch = slab.HRTF(Path.cwd() / 'single_notch.sofa')
hrir_flat_spectrum = slab.HRTF(Path.cwd() / 'flat_spectrum.sofa')

slab.set_default_samplerate(hrir_single_notch.samplerate)
noise = slab.Sound.pinknoise()

# test single notch hrtf on the horizontal plane
for az in range(-40,40,10):
    source_idx = hrir_single_notch.get_source_idx(azimuth=az, elevation=0)
    filtered_noise = hrir_single_notch.apply(source=source_idx, sound=noise)
    filtered_noise.play()

# test flat hrtf
for az in range(-40, 40, 10):
    source_idx = hrir_single_notch.get_source_idx(azimuth=az, elevation=0)
    filtered_noise = hrir_single_notch.apply(source=source_idx, sound=noise)
    filtered_noise.play()







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