import analysis.localization_analysis as loc_analysis
import analysis.plot.localization_plot as loc_plot
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
import analysis.get_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)

# path = Path.cwd() / 'data' / 'experiment' / 'master'
# w2_exclude = ['cs', 'lm', 'lk']
# loc_df = loc_analysis.get_localization_dataframe(path, w2_exclude)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)



### ---- compare persistence m1 / m2 ---- ###
metric = 'EG'



df = loc_df[~loc_df['subject'].isin(w2_exclude)]
m1_d5 = df[df['condition'] == 'Earmolds Week 1'][df['adaptation day'] == 5][metric]
m1_d10 = df[df['condition'] == 'Earmolds Week 1'][df['adaptation day'] == 6][metric]
m1_persistence = numpy.asarray(m1_d5) - numpy.asarray(m1_d10)
m2_d5 = df[df['condition'] == 'Earmolds Week 2'][df['adaptation day'] == 5][metric]
m2_d10 = df[df['condition'] == 'Earmolds Week 2'][df['adaptation day'] == 6][metric]
m2_persistence = numpy.asarray(m2_d5) - numpy.asarray(m2_d10)

compare_persistence = scipy.stats.wilcoxon(m1_persistence, m2_persistence)  # no sign. diff


### ---- compare first to last day of molds ---- ###
# return dictionary
learning = {'week 1': {}, 'week 2': {}, 'overall': {}}
w1d0 = loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 0]
w1d5 = loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 5]
w2d0 = loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation day'] == 0]
w2d5 = loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation day'] == 5]
# overall
d0 = loc_df[loc_df['adaptation day'] == 0][loc_df['condition'] != 'Ears Free']
d5 = loc_df[loc_df['adaptation day'] == 5][loc_df['condition'] != 'Ears Free']
for measurement in loc_df.columns[6:]:
    # week 1:
    w1_learning = scipy.stats.wilcoxon(w1d0[measurement], w1d5[measurement])
    learning['week 1'][measurement] = w1_learning
    # week 2:
    w2_learning = scipy.stats.wilcoxon(w2d0[measurement], w2d5[measurement])
    learning['week 2'][measurement] = w2_learning
    # # overall
    # overall_learning = scipy.statistics.wilcoxon(d0[measurement], d5[measurement])
    # learning['overall'][measurement] = overall_learning


### ---- divide RMSE on day 0 vs day 5 by initial RMSE increase ---- ###
efd0 = numpy.asarray(loc_df[loc_df['condition'] == 'Ears Free'][loc_df['adaptation day'] == 0]['RMSE ele'])
m1d0 = numpy.asarray(loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 0]['RMSE ele'])
m1d5 = numpy.asarray(loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 5]['RMSE ele'])
w1d0_increase = m1d0 - efd0
m1_reduction = m1d0 - m1d5
reduction_decrease_ratio = m1_reduction / w1d0_increase

w1d0_drop = loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day'] == 0]['RMSE ele'] -\
            loc_df[loc_df['condition'] == 'Ears Free'][loc_df['adaptation day'] == 0]['RMSE ele']


### ---- correlate uso vs noise EG / RMSE ---- ###
uso_ef = loc_df[loc_df['condition'] == 'USO Ears Free']
uso_ef = uso_ef.drop(82)  # remove subj lm
noise_ef = loc_df[loc_df['condition'] == 'Ears Free'][loc_df['adaptation day']==2][loc_df['subject'].isin(uso_ef['subject'])]

uso_m1 = loc_df[loc_df['condition'] == 'USO Earmolds Week 1'][loc_df['subject'].isin(uso_ef['subject'])]
noise_m1 = loc_df[loc_df['condition'] == 'Earmolds Week 1'][loc_df['adaptation day']==5][loc_df['subject'].isin(uso_ef['subject'])]

uso_m2 = loc_df[loc_df['condition'] == 'USO Earmolds Week 2'][loc_df['subject'].isin(uso_ef['subject'])]
noise_m2 = loc_df[loc_df['condition'] == 'Earmolds Week 2'][loc_df['adaptation day']==5][loc_df['subject'].isin(uso_ef['subject'])]

# scatter
metric = 'EG'
plt.figure()
plt.scatter(uso_ef[metric], noise_ef['EG'], label='Ears Free')
plt.scatter(uso_m1[metric], noise_m1['EG'], label='M1')
plt.scatter(uso_m2[metric], noise_m2['EG'], label='M2')
plt.xlabel('uso EG')
plt.ylabel('noise EG')
plt.legend()

# standardize by dividing USO EG by noise EG
std_ef = numpy.asarray(uso_ef[metric]) / numpy.asarray(noise_ef[metric])
std_m1 = numpy.asarray(uso_m1[metric]) / numpy.asarray(noise_m1[metric])
std_m2 = numpy.asarray(uso_m2[metric]) / numpy.asarray(noise_m2[metric])

plt.scatter(std_ef, std_m1)
# spearman correlation?



subj_list = list(loc_df['subject'].unique())
for subj in subj_list:
    loc_plot.learning_plot(to_plot=subj)