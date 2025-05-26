from win32com.client import Dispatch
from pathlib import Path
import numpy
import slab

def connect_RM1(rcx_path):
    init = numpy.zeros((4))
    RM1 = Dispatch('RPco.X')
    init[0] = RM1.ConnectRM1('USB', 1)
    init[1] = RM1.ClearCOF()
    init[2] = RM1.LoadCOF(rcx_path)
    init[3] = RM1.Run()
    if not all(init):
        print('processor could not be initialized')
    return RM1

def array2RM1(signal, RM1):
    if isinstance(signal, slab.Sound):
        data = signal.data.flatten()
        RM1._oleobj_.InvokeTypes(15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
        'signal', 0, data)