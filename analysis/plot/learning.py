import analysis.localization_analysis as loc_analysis
import numpy
import slab
from copy import deepcopy
from matplotlib import pyplot as plt
from pathlib import Path

def learning_plot(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude = ['cs', 'lm', 'lk']):
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
    label = None
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    """ elevation """
    for i, ax_id in enumerate([0, 1, 1]):
        if i >= 1:
            label = labels[i-1]
            color = colors[i-1]
        # EG # week 1
        axes[ax_id].plot([0, .05], [ef[0, i], m1[0, i]], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
        axes[ax_id].plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(color-0.2), linewidth=1.5, label=label)  # week 1 learning curve
        axes[ax_id].plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
        # week 2
        axes[ax_id].plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
        axes[ax_id].plot([5.05,  6,  7,  8, 9, 9.95], m2[:6, i], c=str(color+0.2), linewidth=1.5)  # week 2 learning curve
        axes[ax_id].plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axes[ax_id].plot([4.95, 10.05], m1[-2:, i], c=str(color-0.2), ls=(0, (5, 10)), lw=0.8)
        # mold 2 adaptation persistence
        axes[ax_id].plot([9.95, 15], m2[-2:, i], c=str(color+0.2), ls=(0, (5, 10)), lw=0.8)
        # error bars
        if len(subjects) > 1:
            axes[ax_id].errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                           fmt="o", c=str(color), elinewidth=0.8, markersize=5, fillstyle='none')  # error bar ears free
            axes[ax_id].errorbar([.05, 1, 2, 3, 4, 4.95, 10.05], m1[:7, i], capsize=3, yerr=localization_dict
                            ['Earmolds Week 1']['SE'][:7, i], fmt="o", c=str(color-0.2), elinewidth=0.8, markersize=5)  # err m1
            axes[ax_id].errorbar([5.05,  6,  7,  8, 9, 9.95, 15], m2[:7, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE'][:7, i],
                           fmt="o", c=str(color+0.2), elinewidth=0.8, markersize=5)  # err m2
            # axes[ax_id].errorbar([6.1,  7,  8,  9, 10, 11.1], m2[:6, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE']
            #                     [:6, i], fmt="o", c=color, elinewidth=0.8, markersize=5)
    """ azimuth """
    label = labels[0]
    color = colors[0]
    for i in [3, 4]:
        if i > 3:
            label = labels[1]
            color = colors[1]
        # EG # week 1
        axes[2].plot([0, .05], [ef[0, i], m1[0, i]], c=str(color), ls=(0, (5, 10)),
                     lw=0.8)  # day one mold insertion
        axes[2].plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(color), linewidth=1.5,
                     label=label)  # week 1 learning curve
        axes[2].plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(color), ls=(0, (5, 10)),
                     lw=0.8)  # week 1 mold removal
        # week 2
        axes[2].plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(color), ls=(0, (5, 10)),
                     lw=0.8)  # week 2 mold insertion
        axes[2].plot([5.05, 6, 7, 8, 9, 9.95], m2[:6, i], c=str(color),
                     linewidth=1.5)  # week 2 learning curve
        axes[2].plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(color), ls=(0, (5, 10)),
                     lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axes[2].plot([4.95, 10.05], m1[-2:, i], c=str(color), ls=(0, (5, 10)), lw=0.8)
        # mold 2 adaptation persistence
        axes[2].plot([9.95, 15], m2[-2:, i], c=str(color), ls=(0, (5, 10)), lw=0.8)
        # error bars
        if len(subjects) > 1:
            axes[2].errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                             fmt="o", c=str(color), elinewidth=0.8, markersize=5,
                             fillstyle='none')  # error bar ears free
            axes[2].errorbar([.05, 1, 2, 3, 4, 4.95, 10.05], m1[:7, i], capsize=3, yerr=localization_dict
                                                                                        ['Earmolds Week 1'][
                                                                                            'SE'][:7, i],
                             fmt="o", c=str(color), elinewidth=0.8, markersize=5)  # err m1
            axes[2].errorbar([5.05, 6, 7, 8, 9, 9.95, 15], m2[:7, i], capsize=3,
                             yerr=localization_dict['Earmolds Week 2']['SE'][:7, i],
                             fmt="o", c=str(color), elinewidth=0.8, markersize=5)  # err m2

    axes[0].set_ylim(0, 1.1)
    axes[0].set_yticks(numpy.arange(0, 1.1, 0.2), size=13)
    axes[0].set_yticklabels([int(0) , 0.2, 0.4, 0.6, 0.8, 1. ])
    for y in numpy.arange(0, 1.1, 0.2):
        axes[0].axhline(y, xmin=0, xmax=20, color='grey', linewidth=.1)
    axes[0].set_xticks(days, size=13)
    axes[0].set_xticklabels([])
    axes[0].set_ylabel('Elevation Gain', size=13)

    axes[1].set_ylim(0, 28)
    axes[1].set_yticks(numpy.arange(0, 26, 5), size=13)
    for y in numpy.arange(0, 26, 5):
        axes[1].axhline(y, xmin=0, xmax=20, color='grey', linewidth=.1)
    axes[1].set_xticks(days)
    axes[1].set_xticklabels([])
    axes[1].set_ylabel('Elevation in degrees', size=13)
    axes[1].legend()

    axes[2].set_ylim(0, 8)
    axes[2].set_yticks(numpy.arange(0, 8, 2), size=13)
    for y in numpy.arange(0, 8, 2):
        axes[2].axhline(y, xmin=0, xmax=20, color='grey', linewidth=.1)
    axes[2].set_xticks(days, size=13)
    axes[2].set_ylabel('Azimuth in degrees', size=13)
    axes[2].set_xlabel('Days', size=13)
    axes[2].legend()

    for axis in axes:
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     labelsize=13, width=1.5, length=2)

    plt.show()
    return axes

    # # save as scalable vector graphics
    # fig.savefig(Path.cwd() / 'data' / 'presentation' / 'learning_plot.svg', format='svg')
