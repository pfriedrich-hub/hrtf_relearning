import threading
from start_tracker import *
from dev.pybinsim.test.binsim import *

hrtf = slab.HRTF.kemar()

t1 = threading.Thread(target=binsim_start, args=('kemar',))
t2 = threading.Thread(target=filters, args=(hrtf,))

t1.start()
t2.start()

while not KeyboardInterrupt:
    print('test')

