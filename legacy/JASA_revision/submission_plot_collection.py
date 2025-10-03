import legacy.MSc.analysis.build_dataframe as build_df
import legacy as get_df
import legacy as ele_learning
from legacy import ef_vsi
from legacy import mold_vsi
from legacy import acoustic_behavior
from legacy import stim_response_plot


from matplotlib import pyplot as plt
from pathlib import Path

data_path = Path.cwd() / 'legacy' / 'MSc' / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/PhD/Jasa/revision/Manuscript/figures')
hrtf_df = build_df.get_hrtf_df(data_path, processed=True)
main_df = build_df.main_dataframe(data_path, processed_hrtf=True)

## time course
# time_course.plot_time_course(figsize=(13, 5.3))
# plt.savefig(plot_path / 'Figure_1_timecourse.svg', format='svg')

# ## acoustic impact of earmold
# hrtf_plot.ear_mod_images(main_df, subject='tk', chan=1, figsize=(11.5, 4))
# plt.savefig(plot_path / 'C.svg', format='svg')

# stimulus response plot
# stim_response_plot(sub_id='vk', figsize=(9.5, 5))
# plt.savefig(plot_path / 'Figure_3_stim_resp.svg', format='svg')

# # # main result: figure box: A - learning curve on elevation gain, B - precision and accuracy
fig, axes = ele_learning.learning_plot(path=data_path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(17.5, 6.5))
plt.savefig(plot_path / 'Figure_4_learning.svg', format='svg')

# evolution of response pattern
# fig, axis = loc_plot.response_evolution(to_plot='average', figsize=(13, 5.3), path=data_path)
# plt.savefig(plot_path / 'Figure_5_response_evolution.svg', format='svg')

# free ears - spectral properties and behavior - figure box
# ef_vsi(hrtf_df, main_df, figsize=(7, 9))
# plt.savefig(plot_path / 'Figure_6_ef_vsi.svg', format='svg')

# # spectral image EF and spectral change probability m1 and m2
# hrtf_plot.spectral_overview(main_df, figsize=(13, 4), return_n=True)
# plt.savefig(plot_path / 'Figure_7_spectral_n.svg', format='svg')

# acoustic effects on vsi / vsi dissimilarity figure box
# mold_vsi(main_df, figsize=(8.5, 9.6))
# plt.savefig(plot_path / 'Figure_8_mold_vsi_modified.svg', format='svg')

# acoustic effect on behavioral impact - figure box
# acoustic_behavior(main_df, figsize=(8.5, 3.5))
# plt.savefig(plot_path / 'Figure_9_ac_beh.svg', format='svg')

# revision plots:

# individual response evolution
# fig, axis = loc_plot.sub_response_evolution(to_plot='ll', figsize=(13, 5.3), path=data_path)
# plt.savefig(plot_path / 'subject_response_evolution.svg', format='svg')

"""
# get subject data  with median SD across molds
import numpy
# SD population mean
mean_sd = numpy.nanmean(numpy.stack((numpy.stack(main_df['M1D5'].to_numpy().flatten())[:, 2],
             numpy.stack(main_df['M2D5'].to_numpy().flatten())[:, 2])))
# SD subject mean
sub_mean_sd = numpy.mean(numpy.stack((numpy.stack(main_df['M1D5'].to_numpy().flatten())[:, 2],
             numpy.stack(main_df['M2D5'].to_numpy().flatten())[:, 2])), axis=0)
# subject no 7: ll
"""