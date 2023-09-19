import analysis.hrtf_analysis as hrtf_an
import analysis.plot.hrtf_plot as hrtf_pl
import analysis.localization_analysis as loc_an
import analysis.plot.localization_plot as loc_pl
import misc.octave_spacing
from pathlib import Path

data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=True)
conditions = hrtf_df['condition'].unique()

# -- group averages -- #
# adaptation
loc_pl.learning_plot(to_plot='average')
loc_pl.localization_plot(to_plot='average')
# hrtf
hrtf_pl.hrtf_overwiev(hrtf_df, to_plot='average', dfe=False, n_bins=4884)
# mean vsi across bands
hrtf_an.mean_vsi_across_bands(hrtf_df, condition=conditions[0], bands=None, n_bins=None, dfe=False, show=True)


# --- DTF processing  --- #
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
# load hrtf
subject = 'jl'
condition = conditions[0]
hrtf = hrtf_df[hrtf_df['subject'] == subject][hrtf_df['condition'] == condition]['hrtf'].values[0]

# plot
hrtf.plot_tf(hrtf.cone_sources(0))

# processing
hrtf = hrtf_an.smoothe_hrtf(hrtf, high_cutoff=1000)

hrtf = hrtf_an.baseline_hrtf(hrtf)

hrtf = hrtf.diffuse_field_equalization()

# vsi across bands
octave_bands = misc.octave_spacing.non_overlapping_bands()
hrtf_pl.plot_vsi_across_bands(hrtf, bands=octave_bands[0])


# all subjects hrtf + vsi
hrtf_pl.subj_hrtf_vsi(hrtf_df, to_plot='all', condition='Ears Free', bands=None)

