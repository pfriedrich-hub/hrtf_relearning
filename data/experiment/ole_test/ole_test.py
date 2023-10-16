import slab
from pathlib import Path
data_path = path=Path.cwd() / 'data' / 'experiment' / 'ole_test' / 'kemar'
import analysis.hrtf_analysis as analysis

kemar_free = slab.HRTF(path / 'Ears Free' / 'kemar_Ears Free_16.10.sofa')
kemar_speakers = slab.HRTF(path / 'side speakers' / 'kemar_side speakers_16.10.sofa')
kemar_difference = analysis.hrtf_difference(kemar_free, kemar_speakers)

kemar_free.plot_tf(kemar_free.cone_sources(0))
kemar_speakers.plot_tf(kemar_free.cone_sources(0))
kemar_difference.plot_tf(kemar_free.cone_sources(0))

# notes:
# use stimuli from oles experiment: 100 ms white noise bursts (make sure to use the same noise burst twice)
# is the reflection on the orbs direction-dependent?
# first test: variation across elevation in bands where ears typically induce frequency notches

# record again without moving kemar! - maybe just take the speakers away?