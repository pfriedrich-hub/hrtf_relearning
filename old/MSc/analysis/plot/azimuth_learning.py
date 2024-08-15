import old.MSc.analysis.localization_analysis as loc_analysis
import numpy
from matplotlib import pyplot as plt
from pathlib import Path

def learning_plot_azimuth(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master',
                          w2_exclude = ['cs', 'lm', 'lk'], axis=None):
    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """
    localization_dict = loc_analysis.get_localization_dictionary(path=path)
    # localization_dict = deepcopy(localization_dictionary)
    if not to_plot == 'average':
        subjects = [to_plot]
    else:
        subjects = list(localization_dict['Ears Free'].keys())
    localization_dict = loc_analysis.get_localization_data(localization_dict, subjects, w2_exclude)
    days = numpy.arange(12)  # days of measurement
    days[-1] = 16
    # means ears free / mold1 / mold2
    ef = numpy.nanmean(localization_dict['Ears Free']['data'], axis=0)
    m1 = numpy.nanmean(localization_dict['Earmolds Week 1']['data'], axis=0)
    m2 = numpy.nanmean(localization_dict['Earmolds Week 2']['data'], axis=0)
    # optionally delete nan for plt to interpolate points in learning curve
    if len(subjects) == 1:
        if numpy.isnan(ef).any():
            ef = numpy.delete(ef, numpy.where(numpy.isnan(ef))[0][0], numpy.where(numpy.isnan(ef))[1][0])
        if numpy.isnan(m1).any():
            m1 = numpy.delete(m1, numpy.where(numpy.isnan(m1))[0][0] ,numpy.where(numpy.isnan(m1))[1][0])
        if numpy.isnan(m2).any():
            m2 = numpy.delete(m2, numpy.where(numpy.isnan(m2))[0][0] ,numpy.where(numpy.isnan(m2))[1][0])
    # ----- plot ----- #
    labels = ['RMSE', 'SD']
    colors = [0.3, 0.7]
    color = colors[0]
    label = labels[0]
    if not axis:
        fig, axes = plt.subplots(1, 1, figsize=(14, 4))
    for i in [3, 4]:
        if i > 3:
            label = labels[1]
            color = colors[1]
        # EG # week 1
        axes.plot([0, .05], [ef[0, i], m1[0, i]], c=str(color), ls=(0, (5, 10)),
                         lw=0.8)  # day one mold insertion
        axes.plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(color), linewidth=1.5,
                         label=label)  # week 1 learning curve
        axes.plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(color), ls=(0, (5, 10)),
                         lw=0.8)  # week 1 mold removal
        # week 2
        axes.plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(color), ls=(0, (5, 10)),
                         lw=0.8)  # week 2 mold insertion
        axes.plot([5.05, 6, 7, 8, 9, 9.95], m2[:6, i], c=str(color),
                         linewidth=1.5)  # week 2 learning curve
        axes.plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(color), ls=(0, (5, 10)),
                         lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axes.plot([4.95, 10.05], m1[-2:, i], c=str(color), ls=(0, (5, 10)), lw=0.8)
        # mold 2 adaptation persistence
        axes.plot([9.95, 15], m2[-2:, i], c=str(color), ls=(0, (5, 10)), lw=0.8)
        # error bars
        if len(subjects) > 1:
            axes.errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                                 fmt="o", c=str(color), elinewidth=0.8, markersize=5,
                                 fillstyle='none')  # error bar ears free
            axes.errorbar([.05, 1, 2, 3, 4, 4.95, 10.05], m1[:7, i], capsize=3, yerr=localization_dict
                                                                                            ['Earmolds Week 1'][
                                                                                                'SE'][:7, i],
                                 fmt="o", c=str(color), elinewidth=0.8, markersize=5)  # err m1
            axes.errorbar([5.05, 6, 7, 8, 9, 9.95, 15], m2[:7, i], capsize=3,
                                 yerr=localization_dict['Earmolds Week 2']['SE'][:7, i],
                                 fmt="o", c=str(color ), elinewidth=0.8, markersize=5)  # err m2
            # axes.errorbar([6.1,  7,  8,  9, 10, 11.1], m2[:6, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE']
            #                     [:6, i], fmt="o", c=color, elinewidth=0.8, markersize=5)

    # axes.set_yticks(numpy.arange(0, 1.2, 0.2), size=13)
    axes.set_xticks(days, size=13)
    axes.set_ylabel('Azimuth in degrees', size=13)
    axes.set_xlabel('Days', size=13)
    axes.legend()

    for y1 in numpy.linspace(1, 7, 7):
        axes.axhline(y=y1, xmin=0, xmax=20, color='grey', linewidth=.1)

    # fig.text(0.3, 0.8, f'n={len(subjects)}', ha='center', size=10)
    # fig.text(0.5, 0.8, f'n={len(subjects) - len(w2_exclude)}', ha='center', size=10)
    # w1 = ' '.join([str(item) for item in subjects])
    # w2 = subjects
    # for subj in w2_exclude:
    #     if subj in subjects:
    #         w2.remove(subj)
    # w2 = ' '.join([str(item) for item in w2])
    # plt.suptitle(f'w1: {w1}, w2: {w2}')
    plt.show()
    return axes
    #
    # # save as scalable vector graphics
    # fig.savefig(Path.cwd() / 'data' / 'presentation' / 'learning_plot.svg', format='svg')

