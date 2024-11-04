import slab
import numpy
import random
import freefield
from pathlib import Path
data_dir = Path.cwd() / 'final_data'
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.

# pinknoise_pulses
# sound = slab.Binaural.pinknoise_pulses(duration=2.0)

# chirp
# sound = slab.Binaural.chirp(duration=0.5, level=70, from_frequency=0, to_frequency=18000)

#  bark
sound = slab.Binaural(data_dir / 'bark.wav')
sound.right.data = sound.left.data
sound = sound.resample(48828)

sound = sound.ramp(duration=0.005, when='both')
sofa_file = 'vk.sofa'
ff = True

def front_back_test(sofa_file, sound, ff):
    if ff:
        freefield.initialize(setup='dome', default="loctest_headphones")
        print('\n.\n.\n.\n.\n.Press GREEN button for sounds from the FRONT \n and RED for sounds from the BACK.')
        freefield.set_logger('warning')
    hrtf = slab.HRTF(data_dir / 'hrtf' / sofa_file)
    # get cones for every azimuth
    azs = numpy.unique(hrtf.sources[hrtf.sources[:, 0] < 90, 0])
    cone_src = []
    for az in azs:
        src = hrtf.cone_sources(az, coords='interaural', full_cone=True)
        cone_src.append(sorted(src))
    # play from random sources on a single random cone and ole_test listeners front back discrimination
    hits = []
    for c in range(4):
        cone = random.choice(cone_src)
        print(numpy.unique(hrtf.sources[cone][:, 0]))
        for s in range(7):
            source_id = random.choice(cone)
            #  print(hrtf.sources[source])
            if numpy.deg2rad(hrtf.sources[source_id][0]) < 1:
                direction = 'front'
            else:
                direction = 'back'
            #  print(direction)
            dir_snd = hrtf.apply(source_id, sound)
            # todo apply loudness equalization
            dir_snd.level = 40
            if ff:
                freefield.write('data_l', dir_snd.left.data, processors='RP2')
                freefield.write('data_r', dir_snd.right.data, processors='RP2')
                freefield.write('playbuflen', dir_snd.n_samples, processors='RP2')
                freefield.play()
                freefield.wait_to_finish_playing()
                response = 0
                while response == 0:
                    response = int(freefield.read(processor='RP2', tag='response'))
                time.sleep(0.5)
            else:
                sound.play()
                response = (input('Press ''f'' for sounds from the FRONT / ''b'' for sounds from the BACK.'))
            if ((response == 'f' or response == 1) and direction == 'front') or \
                    ((response == 'f' or response == 2) and direction == 'back'):
                hits.append(1)
            else:
                hits.append(0)
    return hits.count(1) / len(hits)

if __name__ == "__main__":
    hitrate = front_back_test(sofa_file, sound, ff)
    print('Hitrate: %.2f' % hitrate)

"""
'kemar.sofa'
'QU_KEMAR_anechoic_1m.sofa'
'MRT01.sofa'"""
