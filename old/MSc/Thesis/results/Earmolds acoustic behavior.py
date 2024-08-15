import old.MSc.analysis.statistics.stats_df as stats_df
import old.MSc.analysis.build_dataframe as build_df
from pathlib import Path
import scipy.stats
import pandas
import numpy
import old.MSc.analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
# from methods: filter='erb', bandwidth=(500, 16000), baseline=False, dfe=True
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(path, processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))  # bandwidth of analyses

""" relation between acoustic and behavioral effects after earmold insertion """
# EF vs Mold 1
x = numpy.array([item[1] for item in main_df['M1 drop']])  # RMSE ele
x = numpy.array([item[0] for item in main_df['M1 drop']])  # EG
x = numpy.array([item[2] for item in main_df['M1 drop']])  # SD ele
y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity

# EF vs Mold 2
x = numpy.array([item[0] for item in main_df['M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M2 drop']])  # SD ele
y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity

# Mold 1 vs Mold 2
x = numpy.array([item[0] for item in main_df['M1M2 drop']])  # EG
x = numpy.array([item[1] for item in main_df['M1M2 drop']])  # RMSE ele
x = numpy.array([item[2] for item in main_df['M1M2 drop']])  # SD ele
y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
R, p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')  # VSI dissimilarity

