from matplotlib import pyplot as plt
import struct
from pathlib import Path
import numpy
import slab
import copy
from array import array
binary_path = Path.cwd() / 'data' / 'hrtf' / 'binary'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
filename = 'MRT01'
hrtf = slab.HRTF(sofa_path / f'{filename}.sofa')


sources = hrtf.sources.vertical_polar

az = 45
ele = 39.6
source_idx, = numpy.where((sources[:, 1]==ele) & (sources[:, 0]==az))
plt.plot(hrtf[source_idx[0]].data)

