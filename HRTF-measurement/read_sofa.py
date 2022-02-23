import sofar as sf
import os
from pathlib import Path
subject = 'paul_hrtf'
filename = subject + '.sofa'

sofa = sf.read_sofa(os.getcwd() + '/data/hrtfs/' + filename)
# print list of sofa 'spatially oriented format for acoustics' conventions
sofa.verify()
sf.list_conventions()
sofa.list_dimensions
print('load sofa file file, %s recordings'%(sofa.get_dimension('M')))

