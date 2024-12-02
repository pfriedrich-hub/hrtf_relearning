from dev.hrtf.make.make_frontal import make_hrtf
from dev.hrtf.tf2ir import tf2ir
from dev.hrtf.hrtf2rcx import hrtf2binary
# from misc.HRTF_test import HRTF_test
filename = 'hrtf_1'

# create hrtf:
hrtf = make_hrtf(n_azimuths=50, azimuth_range=(0, 50), n_elevations=8, elevation_range=(-40, 40), n_bins=256)

# convert to IR, write to binary file and test
hrir = tf2ir(hrtf)
hrtf2binary(hrtf, filename, add_itd=True)

test = HRTF_test('RM1', target=(0, 0))
test.run()

# inspect
# movie([hrtf], (0,50), (-40,40), interval=200, map='average', kind='waterfall', save='test_hrtf')


