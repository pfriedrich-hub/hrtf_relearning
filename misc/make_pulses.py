import slab
slab.set_default_samplerate(44100)
import numpy
from pathlib import Path
soundpath = Path.cwd() / 'data' / 'sounds' / 'pinknoise_pulses'

# with open(soundpath / 'file_list.txt', 'w') as file:
#     file.write('soundfile ')

for pulse_duration in numpy.arange(0.025, 0.501, 0.01):
    print(pulse_duration)
    noise = slab.Sound.pinknoise(duration=pulse_duration, level=75).ramp(when='both', duration=0.01)
    silence = slab.Sound.silence(duration=pulse_duration)
    pulse = slab.Sound.sequence(noise, silence)
    pulse.write(soundpath / ('pulse_%.3f.wav' % pulse_duration))
    # with open(soundpath / 'file_list.txt', 'a') as file:
    #     file.write(f'{soundpath / ("pulse_%.3f.wav" % pulse_duration)}#')


soundpath = Path.cwd() / 'data' / 'sounds'
noise = slab.Sound.pinknoise(duration=30.0, samplerate=44100, level=75).ramp(when='both', duration=0.01)
noise.write(soundpath / 'pinknoise_44100.wav')

# silence = slab.Sound.silence(duration=1.0, samplerate=44100)
# silence.write(soundpath / 'silence_44100.wav')