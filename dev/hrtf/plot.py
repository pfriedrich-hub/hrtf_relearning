import slab
from dev.hrtf.movie import movie
from pathlib import Path
data_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
# ----- plot feature probability map across database

# database_path =  data_path / 'aachen_database'
# hrtf_list = [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]
# movie(hrtf_list, ear='left', azimuth_range=(-180, 180), elevation_range=(-60, 60), interval=200, map='feature_p',
#       kind='image', save='aachen_hrtf_L_full')

# ----- plot kemar

# hrtf_list = [slab.HRTF(data_path / 'Kemar_HRTF_sofa.sofa')]
# movie(hrtf_list, ear='left', azimuth_range=(-180, 180), elevation_range=(-60,60), interval=20, map='average',
#       kind='image', save='kemar_full')

# ------ plot synth HRTF

# hrtf = make_hrtf(n_azimuths=50, azimuth_range=(0, 50), n_elevations=16, elevation_range=(-40, 40), n_bins=256)
# movie([hrtf], (0,50), (-40,40),  interval=100, map='average', kind='waterfall')