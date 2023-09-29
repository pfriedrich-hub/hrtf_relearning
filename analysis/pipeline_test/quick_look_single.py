import analysis.hrtf_analysis as hrtf_an
import analysis.plot.hrtf_plot as hrtf_pl
import analysis.localization_analysis as loc_an
import analysis.plot.localization_plot as loc_pl
import misc.octave_spacing
from pathlib import Path
import random
import numpy
from matplotlib import pyplot as plt
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']


# --- load single HRTF  --- #
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
# load hrtf
subject = 'pp'
# subject = random.choice(hrtf_df['subject'].unique())
condition = conditions[0]
hrtf = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == condition]['hrtf'].values[0]
# plot
hrtf.plot_tf(hrtf.cone_sources(0))
plt.title(subject)


# test processing
# filter
hrtf = hrtf_an.scepstral_filter_hrtf(hrtf, high_cutoff=1000)
hrtf = hrtf_an.erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286)
# dfe
hrtf = hrtf.diffuse_field_equalization()
# baseline
hrtf = hrtf_an.baseline_hrtf(hrtf)

# vsi across bands
octave_bands = misc.octave_spacing.overlapping_bands()[0]
hrtf_pl.plot_vsi_across_bands(hrtf, bands=octave_bands)  # erb binned
hrtf_pl.plot_vsi_across_bands_old(hrtf, bands=octave_bands)  #

# spectral strength across bands
hrtf_pl.plot_spectral_strength_across_bands(hrtf, bands=octave_bands)


# test vsi
