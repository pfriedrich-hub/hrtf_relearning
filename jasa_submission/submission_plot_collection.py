import analysis.plot.publication.time_course as time_course
import analysis.plot.publication.elevation_learning as ele_learning
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.plot.localization_plot as loc_plot
from analysis.plot.publication.ef_vsi import ef_vsi
from analysis.plot.publication.mold_vsi import mold_vsi
from analysis.plot.publication.acoustic_behavior import acoustic_behavior
import analysis.build_dataframe as build_df
import analysis.build_dataframe as get_df
from pathlib import Path
from matplotlib import pyplot as plt
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/hrtf relearning submission/Manuscript/figures')
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)


## time course
# time_course.plot_time_course(figsize=(17, 7.5))
# plt.savefig(plot_path / 'Figure_timeline.svg', format='svg')

## acoustic impact of earmold
hrtf_plot.ear_mod_images(main_df, subject='tk', chan=1, figsize=(6.5, 14))
plt.savefig(plot_path / 'Figure_ear_mod.svg', format='svg')


# # main result: figure box: A - learning curve on elevation gain, B - precision and accuracy
# fig, axes = ele_learning.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(17, 8.5))
# plt.savefig(plot_path / 'Figure_learning.svg', format='svg')

# # spectral image EF and spectral change probability m1 and m2
# hrtf_plot.spectral_overview(main_df, figsize=(17, 5.5))
# plt.savefig(plot_path / 'Figure_spectral.svg', format='svg')

# free ears - spectral properties and behavior - figure box
# ef_vsi(hrtf_df, main_df, figsize=(8.5, 10.8))
# plt.savefig(plot_path / 'Figure_ef_vsi.svg', format='svg')

# acoustic effects on vsi / vsi dissimilarity figure box
# mold_vsi(main_df, figsize=(8.5, 9.6))
# plt.savefig(plot_path / 'Figure_mold_vsi.svg', format='svg')

# acoustic effect on behavioral impact - figure box
# acoustic_behavior(main_df, figsize=(12, 5))
# plt.savefig(plot_path / 'Figure_ac_beh.svg', format='svg')

# evolution of response pattern
# plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
# fig, axis = loc_plot.response_evolution(to_plot='average', figsize=(17, 9))
# plt.savefig(plot_path / 'Figure_response_evolution.svg', format='svg')
