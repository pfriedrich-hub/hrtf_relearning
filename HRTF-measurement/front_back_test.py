import slab
import numpy
import random
import freefield
from pathlib import Path
data_dir = Path.cwd() / 'data'
slab.Signal.set_default_samplerate(48828 )  # default samplerate for generating sounds, filters etc.

sound = slab.Binaural.pinknoise(duration=2.0)
def front_back_test(sofa_file='kemar.sofa', sound=sound, ff=False):
    if ff:
        freefield.set_logger('warning')
        freefield.initialize(setup='dome', default="loctest_headphones")
        print('\n.\n.\n.\n.\n.Press GREEN button for sounds from the FRONT \n and RED for sounds from the BACK.')
    hrtf = slab.HRTF(data_dir / 'hrtfs' / sofa_file)
    # get cones for every azimuth
    azs = numpy.unique(hrtf.sources[hrtf.sources[:, 0] < 90, 0])
    cone_src = []
    for az in azs:
        src = hrtf.cone_sources(az, coord_system='interaural', full_cone=True)
        cone_src.append(sorted(src))
    # play from random sources on a single random cone and test listeners front back discrimination
    hits = []
    for c in range(2):
        cone = random.choice(cone_src)
        print(numpy.unique(hrtf.sources[cone][:, 0]))
        for s in range(7):
            source = random.choice(cone)
            #  print(hrtf.sources[source])
            if numpy.deg2rad(hrtf.sources[source][0]) < 1:
                direction = 'front'
            else:
                direction = 'back'
            #  print(direction)
            dir_snd = hrtf.apply(source, sound)
            if ff:
                freefield.write('data_l', dir_snd.left.data, processors='RP2')
                freefield.write('data_r', dir_snd.right.data, processors='RP2')
                freefield.write('playbuflen', dir_snd.n_samples, processors='RP2')
                freefield.play()
                freefield.wait_to_finish_playing()
                response = 0
                while response == 0:
                    response = int(freefield.read(processor='RP2', tag='response'))
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
    hitrate = front_back_test('QU_KEMAR_anechoic_1m.sofa', sound, ff=True)
    print('Hitrate: %.2f' % hitrate)

"""
'kemar.sofa'
'QU_KEMAR_anechoic_1m.sofa'
'MRT01.sofa'"""
