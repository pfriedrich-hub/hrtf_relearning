import threading
import slab
from dev.pybinsim.test.binsim import *

hrtf = slab.HRTF.kemar()

t1 = threading.Thread(target=binsim_start, args=('kemar',))
# t2 = threading.Thread(target=filters, args=(hrtf,))

binsim = t1.start()
# t2.start()
binsim.config.configurationDict['loudnessFactor'] = 0.3

while not KeyboardInterrupt:
    print('test')