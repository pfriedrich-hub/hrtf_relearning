import slab
from pathlib import Path
data_path = path=Path.cwd() / 'data' / 'experiment' / 'ole_test' / 'kemar'
import analysis.hrtf_analysis as analysis
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.processing.hrtf_processing as hrtf_processing
from matplotlib import pyplot as plt
bandwidth = (2000, 16000)

#first test: difference spectrum varies across elevations
kemar_free = slab.HRTF(path / 'Ears Free' / 'kemar_Ears Free_16.10.sofa')
kemar_speakers = slab.HRTF(path / 'side speakers' / 'kemar_side speakers_16.10.sofa')

# test again record twice in a row. result: flat difference spectrum
kemar_free = slab.HRTF(path / 'side speakers' / 'kemar_side speakers_18.10_1.sofa')
kemar_speakers = slab.HRTF(path / 'side speakers' / 'kemar_side speakers_18.10.sofa')

# test again careful not to move kemar in between: difference spectrum still varies across elevations
kemar_free = slab.HRTF(path / 'Ears Free' / 'kemar_Ears Free_18.10.sofa')
kemar_speakers = slab.HRTF(path / 'side speakers' / 'kemar_side speakers_18.10.sofa')

# test without ears:
kemar_free = slab.HRTF(path / 'no ears' / 'kemar_no ears_free_18.10.sofa')
kemar_speakers = slab.HRTF(path / 'no ears' / 'kemar_no ears_side_speakers_18.10.sofa')

# test at 1.4 m: best so far, still gain differences in lower DTFs.
# made sure the speakers are exactly at height of kemars ear canals, moved chair to 1.4m so angles match
kemar_free = slab.HRTF(path / '1.4' / 'kemar_ears_free_18.10.sofa')
kemar_speakers = slab.HRTF(path / '1.4' / 'kemar_side_speakers_18.10.sofa')




# compute difference spectrum
kemar_difference = analysis.hrtf_difference(kemar_free, kemar_speakers)

# baseline for image
kemar_free = hrtf_processing.baseline_hrtf(kemar_free, bandwidth=(2000, 16000))
kemar_speakers = hrtf_processing.baseline_hrtf(kemar_speakers, bandwidth=(2000, 16000))
kemar_difference = hrtf_processing.baseline_hrtf(kemar_difference, bandwidth=(2000, 16000))

# plot
kemar_free.plot_tf(kemar_free.cone_sources(0))
kemar_speakers.plot_tf(kemar_speakers.cone_sources(0))
kemar_difference.plot_tf(kemar_difference.cone_sources(0))

chan = 0
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
hrtf_plot.hrtf_image(kemar_free, bandwidth=(2000, 16000), z_min=-30, z_max=30, cbar=False, axis=axes[0],
                     chan=chan, labels=True)
axes[0].set_title('no side speakers')
hrtf_plot.hrtf_image(kemar_speakers, bandwidth=(2000, 16000), z_min=-30, z_max=30, cbar=False, axis=axes[1],
                     chan=chan, labels=True)
axes[1].set_title('with side speakers')
hrtf_plot.hrtf_image(kemar_difference, bandwidth=(2000, 16000), z_min=-30, z_max=30, cbar=True, axis=axes[2],
                     chan=chan, labels=True)
axes[2].set_title('difference spectrum')
# axes[0].set_ylabel('Elevation (degrees)')
# for ax in axes:
#     ax.set_xlabel('Frequency (kHz)')

# notes:
# use stimuli from oles experiment: 100 ms white noise bursts (make sure to use the same noise burst twice)
# is the reflection on the orbs direction-dependent?
# first test: variation across elevation in bands where ears typically induce frequency notches

# record again without moving kemar! - maybe just take the speakers away?

#---- plot transfer function of calibrated speakers --- #

fig, axis = plt.subplots()
free_mic = slab.HRTF(path / 'free_mic' / 'kemar_free mic_31.10.sofa')
free_mic = hrtf_processing.baseline_hrtf(free_mic, bandwidth=(2000, 16000))
free_mic.plot_tf(free_mic.cone_sources(0), axis=axis)

# fig.savefig(path / 'plots' / 'calibration.svg', format='svg')
