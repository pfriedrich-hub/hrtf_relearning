import analysis.hrtf_analysis as hrtf_analysis
import analysis.statistics.stats_df as stats_df
import analysis.plot.plot_spectral_behavior_stats as stats_plot
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.build_dataframe as build_df
import misc.octave_spacing
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
import analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))
hrtf_df = build_df.get_hrtf_df(path, processed=True)
# main_df = stats_df.add_l_r_comparison(main_df, bandwidth=(3700, 12900))

stats_plot.ef_vsi(main_df, 'RMSE ele', axis=None)

# row 1: spectral profile of participants ears
fig, axes = hrtf_plot.hrtf_compare(hrtf_df, axis=None, average_ears=True, hrtf_diff=False, zlim=(-12,8),
                                   width=14, height=5.2)

# row 2: spectral change probability
fig, axes = hrtf_plot.compare_spectral_change_p(main_df, axis=None, bandwidth=(4000, 16000),  width=14, height=5.2)

# row 3: vsi across bands, free ears vsi vs vertical rmse, left right vsi across conditions,
fig, axis = hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=None, ear_idx=[0], figsize=(9, 9))



# row 4: behavior vs free ears vsi

