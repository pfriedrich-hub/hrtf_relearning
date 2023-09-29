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
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)

# -- group averages -- #
# adaptation
loc_pl.learning_plot(to_plot='average')
loc_pl.localization_plot(to_plot='average')
# hrtf
hrtf_pl.hrtf_overwiev(hrtf_df, to_plot='average', dfe=True, n_bins=4884)


# choose bands
bands = misc.octave_spacing.non_overlapping_bands()[0]
bands = misc.octave_spacing.overlapping_bands()[0]



# individual hrtf images + cross bands metrics
# use raw HRTF
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter=None, baseline=True, write=False)  # baseline for hrtf image
# use scepstral filtered HRTF
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=True)
# use erb filtered HRTF
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='erb', baseline=True, write=False)  # baseline for hrtf image

# set conditions and bands
condition = conditions[0]
bands = misc.octave_spacing.non_overlapping_bands()[1]

for subject in hrtf_df['subject'].unique():
    fig, axis = plt.subplots(3, 1, figsize=(8,10))
    hrtf = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == condition]['hrtf'].values[0]
    hrtf_pl.hrtf_image(hrtf, bandwidth=(numpy.min(bands), numpy.max(bands)), n_bins=None, axis=axis[0], z_min=-30, z_max=30, cbar=True)
    axis[0].set_xticks(numpy.asarray(bands).flatten())
    axis[0].set_xticklabels(numpy.asarray(bands).flatten())
    axis[0].vlines(numpy.asarray(bands).flatten()[1:-1], ymin=-37.5, ymax=37.5, color='black')
    hrtf_an.spectral_strength_across_bands(hrtf, bands, show=True, axis=axis[1])
    hrtf_an.vsi_across_bands(hrtf, bands, show=True, axis=axis[2])
    fig.suptitle(subject)

# mean vsi / spectral str across bands
hrtf_an.mean_vsi_across_bands(hrtf_df, condition=conditions[0], bands=bands, show=True)
hrtf_an.mean_spectral_strength_across_bands(hrtf_df, condition=conditions[0], bands=bands, show=True)







hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='erb', baseline=True, write=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='scepstral', baseline=True, write=True)


# process all
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='erb', baseline=True, write=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='scepstral', baseline=True, write=True)









# plot all subjects hrtf + vsi
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_pl.subj_hrtf_vsi(hrtf_df, to_plot='all', condition='Ears Free', bands=None)

hrtf_pl.subj_hrtf_vsi_dis(hrtf_df, to_plot='all', conditions=('Ears Free', 'Earmolds Week 1'), bands=None)

