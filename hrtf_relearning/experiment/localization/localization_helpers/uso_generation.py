import math
import numpy
import slab
from pathlib import Path
import random
import hrtf_relearning
from hrtf_relearning.utils import paths
ROOT = Path(hrtf_relearning.__file__).resolve().parent
input_folder = paths.SOUNDS_DIR / 'mitsu_sounds'

BASES = ['dryer', 'particl2', 'spray', 'shaver', 'tear', 'crumple', 'coffmill']


def generate_uso(samplerate, duration=0.225, base=None, n_sounds=5, seed=None,
                 return_params=False):
    """One 225 ms unfamiliar-sound-object composite: a base texture plus
    `n_sounds` impact sounds at staggered onsets.

    NOTE ON `base`. This used to default to `numpy.random.randint(0, 6)`, which
    is a *default argument* and is therefore evaluated once, at import. Every
    token generated in a session then shared one base texture, and index 6
    ('coffmill') was unreachable. Measured across tokens, that collapsed the
    1/6-octave spectral variation to 0.80 dB -- barely twice plain pinknoise
    (0.40 dB) and still ~3x below a typical DTF elevation cue (~3.3 dB), so
    runs made that way are close to a noise test with one fixed colouration.
    `base=None` now draws a fresh base per call; pass an int or a name to pin
    it deliberately (e.g. to run one fixed timbre as its own condition).
    """
    rng = random.Random(seed) if seed is not None else random
    bases = BASES
    if base is None:
        base = rng.randrange(len(bases))
    if isinstance(base, str):
        base = bases.index(base)
    files = ['cherry1', 'cherry2', 'cherry3', 'wood2', 'wood3',
               'bank', 'bowl', 'candybwl', 'colacan', 'metal15', 'metal10', 'metal05', 'trashbox',
               'case1', 'case2', 'case3', 'dice2', 'dice3',
               'bottle1', 'bottle2', 'china3', 'china4',
               'saw2', 'sandpp1', 'sandpp2',
               'sticks',
               'clap1', 'clap2', 'cap1', 'cap2', 'snap', 'cracker',
               'bell2', 'bells3', 'coin2', 'coin3',
               'book1', 'book2',
               'castanet', 'maracas', 'drum',
               'stapler', 'punch']
    sout = slab.Sound.read(input_folder / str(bases[base] + '.wav'))
    base_sr = sout.samplerate
    sout.level += 6
    length = int(base_sr * duration)
    sout = sout.data[:, 0]
    sout = sout[numpy.where((sout > 0.03) == True)[0][0]:numpy.where((sout > 0.03) == True)[0][-1]][1000:length+1000]
    impacts = []
    for i in range(n_sounds):
        name = rng.choice(files)
        s = slab.Sound(input_folder / str(name + '.wav'))
        while not any(numpy.abs(s.data) > 0.1):
            name = rng.choice(files)
            s = slab.Sound(input_folder / str(name + '.wav'))
        impacts.append(name)
        s_sr = s.samplerate
        start, stop = numpy.where(numpy.abs(s.data) > 0.1)[0][0], numpy.where(numpy.abs(s.data) > 5e-3)[0][-1]
        s = s.data[start:stop]
        if base_sr != s_sr:
            print("Error: Samplerates don't match")
        # offset = math.ceil(numpy.random.randint(low=0, high=50, size=1)/100 * length)
        offset = int(length - length / n_sounds * (i + 1))
        s = numpy.append(numpy.zeros(offset), s)
        s = numpy.append(s, numpy.zeros(length))
        s = s[:length]
        # plt.plot(s)  # nice plot
        sout = numpy.sum((sout, s), axis=0)
    sout = slab.Sound(data=sout, samplerate=base_sr)
    sout = sout.ramp(when='both', duration=0.01)
    sout.data = (sout.data / sout.data.max()) - 0.01
    sout = sout.resample(samplerate)
    if not return_params:
        return sout
    return sout, {'kind': 'uso', 'base': bases[base], 'impacts': impacts,
                  'n_sounds': n_sounds, 'seed': None if seed is None else int(seed)}