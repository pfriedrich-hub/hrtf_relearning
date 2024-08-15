import old.MSc.analysis.hrtf_analysis as hrtf_an
import old.MSc.analysis.plot.hrtf_plot as hrtf_pl
from pathlib import Path
import numpy
from matplotlib import pyplot as plt
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

# --- load single HRTF  --- #
subject = 'pp'
# subject = random.choice(hrtf_df['subject'].unique())
condition = conditions[0]
hrtf = hrtf_an.load_hrtf(subject, condition, processed=False)

# plot raw
hrtf.plot_tf(hrtf.cone_sources(0))
plt.title(subject)

# load single localization sequence
import slab
from pathlib import Path
file_name = Path.cwd() / 'data/experiment/master/nn/Ears Free/localization_nn_ears_free_10.12'
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_name = file_name)

# ole_test processing
# filter
hrtf = hrtf_an.erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286)
# baseline
hrtf = hrtf_an.baseline_hrtf(hrtf)
# dfe
# hrtf = hrtf.diffuse_field_equalization()

# plot processed
hrtf.plot_tf([0], xlim=(4000,16000))
plt.title(subject)

# vsi and specral strength across bands
# bands = misc.octave_spacing.non_overlapping_bands()[0]
bands = MSc.misc.octave_spacing.overlapping_bands()[0]

fig, axis = plt.subplots(3, 1, figsize=(8, 10))
# image
# hrtf_pl.hrtf_image(hrtf, bandwidth=(numpy.min(bands), numpy.max(bands)), axis=axis[0], z_min=-30, z_max=30, cbar=True)
# axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=-37.5, ymax=37.5, color='black')
# waterfall
hrtf.plot_tf(hrtf.cone_sources(0), axis=axis[0], xlim=(numpy.min(bands), numpy.max(bands)))
axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=axis[0].get_ylim()[0], ymax=axis[0].get_ylim()[1], color='black')
axis[0].set_xticks(numpy.asarray(bands).flatten())
axis[0].set_xticklabels(numpy.asarray(bands).flatten())
hrtf_an.spectral_strength_across_bands(hrtf, bands, show=True, axis=axis[1])
hrtf_an.vsi_across_bands(hrtf, bands, show=True, axis=axis[2])
fig.suptitle(subject)

# vsi and specral strength across bands
# bands = misc.octave_spacing.overlapping_bands()[0]
bands = MSc.misc.octave_spacing.non_overlapping_bands()[0]
hrtf_an.vsi_across_bands(hrtf, bands, show=True)





# spectral strength across bands
hrtf_pl.plot_spectral_strength_across_bands(hrtf, bands=octave_bands)


# ole_test vsi



# ole_test processing
# filter
hrtf = hrtf_an.scepstral_filter_hrtf(hrtf, high_cutoff=1000)
hrtf = hrtf_an.erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286)
# dfe
hrtf = hrtf.diffuse_field_equalization()
# baseline
hrtf = hrtf_an.baseline_hrtf(hrtf)
