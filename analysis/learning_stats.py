import analysis.localization_analysis as localization
from pathlib import Path
import scipy.stats
import pandas
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
from matplotlib import pyplot as plt

path = Path.cwd() / 'data' / 'experiment' / 'master'
loc_dict = localization.get_localization_data(path)
loc_df = localization.get_dataframe(loc_dict)

### ---- compare first to last day of molds ---- ###
# return dictionary
learning = {'week 1': {}, 'week 2': {}, 'overall': {}}
for measurement in loc_df.columns[5:]:
    # week 1:
    w1d0 = loc_df[loc_df['condition'] =='Earmolds Week 1'][loc_df['adaptation_day'] == 0]
    w1d5 = loc_df[loc_df['condition'] =='Earmolds Week 1'][loc_df['adaptation_day'] == 5]
    w1_learning = scipy.stats.wilcoxon(w1d0[measurement], w1d5[measurement])
    learning['week 1'][measurement] = w1_learning
    # week 2:
    w2d0 = loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation_day'] == 0]
    w2d5 = loc_df[loc_df['condition'] =='Earmolds Week 2'][loc_df['adaptation_day'] == 5]
    w2_learning = scipy.stats.wilcoxon(w2d0[measurement], w2d5[measurement])
    learning['week 2'][measurement] = w2_learning
    # overall
    d0 = loc_df[loc_df['adaptation_day'] == 0][loc_df['condition'] != 'Ears Free']
    d5 = loc_df[loc_df['adaptation_day'] == 5][loc_df['condition'] != 'Ears Free']
    overall_learning = scipy.stats.wilcoxon(d0[measurement], d5[measurement])
    learning['overall'][measurement] = overall_learning

### ---- compare persistence m1 / m2 ---- ###





print(loc_df[loc_df['subject']=='lk'])


