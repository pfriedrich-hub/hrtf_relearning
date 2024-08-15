import old.MSc.analysis.build_dataframe as get_df
from pathlib import Path
from matplotlib import pyplot as plt
import numpy

main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)

""" compare standard localization vs uso localization accuracy """
def uso_boxplot(main_df, measure, axis=None):
    measures = ['EG', 'RMSE ele', 'SD ele', 'RMSE az', 'SD az']
    ef = numpy.stack((main_df['EFD10']).to_numpy())  # ears free day 10
    efuso = numpy.stack((main_df['EF USO']).to_numpy())
    m1 = numpy.stack((main_df['M1D5']).to_numpy())
    m1uso = numpy.stack((main_df['M1 USO']).to_numpy())
    m2 = numpy.stack((main_df['M2D5']).to_numpy())
    m2uso = numpy.stack((main_df['M2 USO']).to_numpy())
    nan_mask = numpy.where(~numpy.isnan(m2uso[:, 0]))[0]
    x = numpy.zeros((3, len(nan_mask)))
    x[0] = (efuso - ef)[nan_mask, measures.index(measure)]
    x[1] = (m1uso - m1)[nan_mask, measures.index(measure)]
    x[2] = (m2uso - m2)[nan_mask, measures.index(measure)]
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.boxplot(x[0], positions=[0], labels=['EF'])
    axis.boxplot(x[1], positions=[1], labels=['M1'])
    axis.boxplot(x[2], positions=[2], labels=['M2'])
    # axis.set_title(scipy.stats.friedmanchisquare(x[0], x[1], x[2])[1])
    axis.set_ylabel(measure)