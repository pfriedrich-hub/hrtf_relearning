import analysis.get_dataframe as get_df
from pathlib import Path
from matplotlib import pyplot as plt
import numpy
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']


""" compare standard localization vs uso localization accuracy """
def uso_boxplot(main_df, measure, axis):
    x = numpy.zeros((3, len(main_df['subject'])))
    x[0] = numpy.array([item[measures.index(measure)] for item in main_df['EF USO dif']])
    x[1] = numpy.array([item[measures.index(measure)] for item in main_df['M1 USO dif']])
    x[2] = numpy.array([item[measures.index(measure)] for item in main_df['M2 USO dif']])
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['M1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M2'])
    axis.set_ylabel(measure)