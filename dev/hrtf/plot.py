import slab
from dev.hrtf.movie import movie
from pathlib import Path

# ----- plot aachen database

database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'aachen_database'
hrtf_list = [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]
movie(hrtf_list, azimuth_range=(0,50), elevation_range=(-45,45), interval=500, map='feature_p', kind='image', save='aachen_hrtf')

# ----- plot kemar
#
# hrtf_list = [slab.HRTF.kemar()]
# movie(hrtf_list, azimuth_range=(0,50), elevation_range=(-20,20), map='average', kind='image', save='kemar')

# ------ plot synth HRTF

# hrtf = make_hrtf(n_azimuths=50, azimuth_range=(0, 50), n_elevations=16, elevation_range=(-40, 40), n_bins=256)
# movie([hrtf], (0,50), (-40,40),  interval=100, map='average', kind='waterfall')