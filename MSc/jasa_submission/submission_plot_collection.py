import MSc.analysis.build_dataframe as build_df
import MSc.analysis.build_dataframe as get_df
import MSc.analysis.plot.hrtf_plot as hrtf_plot
import MSc.analysis.plot.localization_plot as loc_plot

import MSc.analysis.plot.publication.time_course as time_course
import MSc.analysis.plot.publication.elevation_learning as ele_learning
from MSc.analysis.plot.publication.ef_vsi import ef_vsi
from MSc.analysis.plot.publication.mold_vsi import mold_vsi
from MSc.analysis.plot.publication.acoustic_behavior import acoustic_behavior

from matplotlib import pyplot as plt
from pathlib import Path

data_path = Path.cwd() /'MSc' / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/hrtf relearning submission/Manuscript/figures')
hrtf_df = build_df.get_hrtf_df(data_path, processed=True)
main_df = get_df.main_dataframe(data_path, processed_hrtf=True)

## time course
time_course.plot_time_course(figsize=(13, 5.3))
plt.savefig(plot_path / 'Figure_1_timecourse.svg', format='svg')

# ## acoustic impact of earmold
# hrtf_plot.ear_mod_images(main_df, subject='tk', chan=1, figsize=(11.5, 4))
# plt.savefig(plot_path / 'Figure_3_ear_mod.svg', format='svg')

# # main result: figure box: A - learning curve on elevation gain, B - precision and accuracy
# fig, axes = ele_learning.learning_plot(to_plot='average', path=data_path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(11.5, 6.5))
# plt.savefig(plot_path / 'Figure_4_learning.svg', format='svg')

# evolution of response pattern
# fig, axis = loc_plot.response_evolution(to_plot='average', figsize=(11.5, 6.5))
# plt.savefig(plot_path / 'Figure_5_response_evolution.svg', format='svg')

# free ears - spectral properties and behavior - figure box
# ef_vsi(hrtf_df, main_df, figsize=(7, 9))
# plt.savefig(plot_path / 'Figure_6_ef_vsi.svg', format='svg')

# # spectral image EF and spectral change probability m1 and m2
# hrtf_plot.spectral_overview(main_df, figsize=(11.5, 4))
# plt.savefig(plot_path / 'Figure_7_spectral.svg', format='svg')

# acoustic effects on vsi / vsi dissimilarity figure box
# mold_vsi(main_df, figsize=(8.5, 9.6))
# plt.savefig(plot_path / 'Figure_8_mold_vsi.svg', format='svg')

# acoustic effect on behavioral impact - figure box
# acoustic_behavior(main_df, figsize=(8.5, 3.5))
# plt.savefig(plot_path / 'Figure_9_ac_beh.svg', format='svg')

