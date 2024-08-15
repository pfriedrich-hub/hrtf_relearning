import math
import numpy
import slab
from pathlib import Path
import random
fs = 48828
slab.set_default_samplerate(fs)
input_folder = Path.cwd().parent / 'mitsu_sounds' / 'mitsubishi wavs'
output_folder = Path.cwd() / 'final_data' / 'sounds' / 'uso'

def combine_sounds(duration=0.225, base=0, n_sounds=5):
    bases = ['dryer', 'particl2', 'spray', 'shaver', 'tear', 'crumple', 'coffmill']
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
    base_level = sout.level
    length = int(base_sr * duration)
    sout = sout.data[:, 0]
    sout = sout[numpy.where((sout > 0.03) == True)[0][0]:numpy.where((sout > 0.03) == True)[0][-1]][1000:length+1000]
    for i in range(n_sounds):
        f = random.choice(files)
        s = slab.Sound(input_folder / str(f + '.wav'))
        s_sr = s.samplerate
        if any(numpy.abs(s.data) > 0.1):
            start, stop = numpy.where(numpy.abs(s.data) > 0.1)[0][0], numpy.where(numpy.abs(s.data) > 5e-3)[0][-1]
            s = s.data[start:stop]
        else:
            return None
        if base_sr != s_sr:
            print("Error: Samplerates don't match")
        # offset = math.ceil(numpy.random.randint(low=0, high=50, size=1)/100 * length)
        offset = int(length - length / n_sounds * (i + 1))
        s = numpy.append(numpy.zeros(offset), s)
        s = numpy.append(s, numpy.zeros(length))
        s = s[:length]
        # plt.plot(s)  # nice plot
        sout = numpy.sum((sout, s), axis=0)
    sout = slab.Sound(data=sout, samplerate=48000)
    sout = sout.resample(fs)
    sout = sout.ramp(when='both', duration=0.01)
    sout.data = (sout.data / sout.data.max()) - 0.01
    return sout

if __name__ == "__main__":
    n = 0
    while n < 30:
        s = combine_sounds(duration=0.225, base=numpy.random.randint(0, 6), n_sounds=5)
        if s:
            a = 'p'
            while a == "p":
                s.play()
                a = input('Take?')
            if a == "y":
                counter = 1
                file_name = output_folder / str('uso_225ms_' + str(counter) + '_.wav')
                while Path.exists(file_name):
                    file_name = output_folder / str('uso_225ms_' + str(counter) + '_.wav')
                    counter += 1
                s.write(file_name)
            elif a == "s":
                break
            else:
                continue
        else:
            continue
    print("Finished")