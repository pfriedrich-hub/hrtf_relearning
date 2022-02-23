import sofar as sf
import os
import slab
from pathlib import Path
subject = 'paul_hrtf'
filename = subject + '.sofa'
filepath = os.getcwd() + '/data/hrtfs/' + filename

# read with sofar
sofa = sf.read_sofa(os.getcwd() + '/data/hrtfs/' + filename)
# print list of sofa 'spatially oriented format for acoustics' conventions
sofa.verify()
sf.list_conventions()
sofa.list_dimensions
print('load sofa file file, %s recordings'%(sofa.get_dimension('M')))

# read with slab
sofa = slab.HRTF.kemar()
hrtf  = slab.HRTF(data=filepath) # todo fix error




# go into slab hrtf.py
import h5netcdf
import numpy
f = h5netcdf.File(filepath, 'r')
datatype = f.attrs['DataType'].decode('UTF-8')  # get data type

# todo solved by adding custom string var to sofa file
attr = dict(f.variables['Data.SamplingRate'].attrs.items())  # get attributes as dict

# todo: slab uses HRIR sofa type to manipulate sounds with a listeners recorded IR data
#  option 1: write slab function to apply transfer functions (ffts) from HRTF sofa to fft of target sound,
#   return inverse fft of that sound and see if it matches HRIR approach (individualized hrtf - source simulation)

# todo option 2: find smart way to measure IR data and write sofa file in HRIR format

# todo option 3: dont simulate sounds (not my job). instead modify slab.HRTF functions like 'tfs_from_sources'
#  or write your own functions utilizing HRTF sofa format to do the job <-- PRIORITY

if datatype != 'FIR':
    print('something')
    #warnings.warn('Non-FIR data: ' + datatype)
retrn = numpy.array(f.variables['Data.IR'], dtype='float')
