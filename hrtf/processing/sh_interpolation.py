import pyfar as pf # pyfar v0.7.2
import numpy as np

sg = pf.samplings.sph_gaussian(sh_order=7)
azimuth = np.sort(np.unique(np.round(sg.azimuth / np.pi * 180, 1)))
print(azimuth)