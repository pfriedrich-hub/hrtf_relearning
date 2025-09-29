import slab
from pathlib import Path
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

hrir = slab.HRTF(sofa_path / 'KU100_HRIR_L2702.sofa')
hrir = slab.HRTF(sofa_path / 'kemar.sofa')
hrir = slab.HRTF(sofa_path / 'single_notch.sofa')

# compute itd / ild of all filters on the horizontal plane
src_idx = hrir.cone_sources(0, True, 'elevation', .01)
sound = slab.Binaural.pinknoise(samplerate=hrir.samplerate)
sound = slab.Binaural.tone(samplerate=hrir.samplerate, frequency=4000)


# ils = slab.Binaural.make_interaural_level_spectrum(hrir)
for id in src_idx:
    print(f'Source {hrir.sources.vertical_polar[id, 0]}')
    _s = hrir.apply(id, sound)
    print(f'ITD {-slab.Binaural.itd_to_azimuth(_s.itd() / _s.samplerate)}')
    print(f'ILD {-slab.Binaural.ild_to_azimuth(_s.ild(), frequency=4000, ils=None)}')
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
hrtf_name = 'KU100_HRIR_L2702.sofa'
hrtf_path = Path.cwd() / 'data' / 'hrtf'
hrir = hrtf_path / 'sofa' / hrtf_name
wav_path = Path.cwd() / 'data' / 'hrtf' / 'wav' / 'KU100_HRIR_L2702' / 'IR_data'
sound = slab.Sound.pinknoise(duration=0.1, samplerate=hrir.samplerate)
src_idx = hrir.cone_sources(0, True, 'elevation', .01)
for idx in src_idx:
    source = hrir.sources.vertical_polar[idx]
    filt_wav = slab.Sound.read(wav_path / f'{source[0]}_{source[1]}.wav')
    filt = slab.Filter(data=filt_wav.data, fir='IR')
    print(f'Azimuth: {source[0]}')
    hrir[idx].apply(sound).play()
    filt.apply(sound).play()

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

