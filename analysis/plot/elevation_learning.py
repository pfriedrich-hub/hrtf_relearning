import analysis.localization_analysis as loc_analysis
import numpy
import slab
from copy import deepcopy
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from pathlib import Path
from misc.unit_conversion import cm2in

def learning_plot(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude = ['cs', 'lm', 'lk']
                  , width=30, height=10):
    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """
    labelsize = 8
    localization_dict = loc_analysis.get_localization_dictionary(path=path)
    # localization_dict = deepcopy(localization_dictionary)
    if not to_plot == 'average':
        subjects = [to_plot]
    else:
        subjects = list(localization_dict['Ears Free'].keys())
    localization_dict = loc_analysis.get_localization_data(localization_dict, subjects, w2_exclude)
    days = numpy.arange(12)  # days of measurement
    days[-1] = 15
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
    colors = [0.15, 0.65] # deviation in color of SD adaptation
    w2_dev = 0.15  # deviation in color of w2 adaptation
    color = colors[0] # EG color and label
    label = None
    width = cm2in(width)
    height = cm2in(height)

    fig, axes = plt.subplots(2, 1, figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.05)

    for i, ax_id in enumerate([0, 1, 1]):
        if i >= 1:  # RMSE color and label
            label = labels[i-1]
            color = colors[i-1]

        # week 1
        axes[ax_id].plot([0, .05], [ef[0, i], m1[0, i]], c=str(color-w2_dev), ls=(0, (5, 10)), lw=1)  # day one mold insertion
        axes[ax_id].plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(color-w2_dev), linewidth=0.8, label=label)  # week 1 learning curve
        axes[ax_id].plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(color-w2_dev), ls=(0, (5, 10)), lw=1)  # week 1 mold removal
        # week 2
        axes[ax_id].plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(color+w2_dev), ls=(0, (5, 10)), lw=1)  # week 2 mold insertion
        axes[ax_id].plot([5.05,  6,  7,  8, 9, 9.95], m2[:6, i], c=str(color+w2_dev), linewidth=0.8)  # week 2 learning curve
        axes[ax_id].plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(color+w2_dev), ls=(0, (5, 10)), lw=1)  # week 2 mold removal
        # mold 1 adaptation persistence
        axes[ax_id].plot([4.95, 10.05], m1[-2:, i], c=str(color-w2_dev), ls=(0, (5, 10)), lw=0.8)
        # mold 2 adaptation persistence
        axes[ax_id].plot([9.95, 15], m2[-2:, i], c=str(color+w2_dev), ls=(0, (5, 10)), lw=0.8)
        # error bars
        if len(subjects) > 1:
            axes[ax_id].errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                           fmt="o", c=str(color), elinewidth=0.5, markersize=3, fillstyle='full',
                                 markerfacecolor='white')  # error bar ears free
            axes[ax_id].errorbar([.05, 1, 2, 3, 4, 4.95, 10.05], m1[:7, i], capsize=3, yerr=localization_dict
                            ['Earmolds Week 1']['SE'][:7, i], fmt="o", c=str(color-w2_dev), elinewidth=0.5, markersize=3)  # err m1
            axes[ax_id].errorbar([5.05,  6,  7,  8, 9, 9.95, 15], m2[:7, i], capsize=3,
                                 yerr=localization_dict['Earmolds Week 2']['SE'][:7, i], fmt="o", c=str(color+w2_dev),
                                 elinewidth=0.5, markersize=3)  # err m2
            # axes[ax_id].errorbar([6.1,  7,  8,  9, 10, 11.1], m2[:6, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE']
            #                     [:6, i], fmt="o", c=color, elinewidth=0.5, markersize=3)

    # axes ticks and limits
    axes[0].set_xticks([0,5,10,15], size=labelsize)
    axes[0].set_xticklabels([], size=labelsize)
    axes[0].set_ylim(0.22,1.02)
    axes[0].set_yticks(numpy.arange(0.4, 1.2, 0.2), size=labelsize)
    axes[0].set_ylabel('Elevation Gain', size=labelsize)
    axes[1].set_xticks([0,5,10,15], size=labelsize)
    axes[1].set_xticklabels([0,5,10,15], size=labelsize)
    axes[1].set_ylim(4,26)
    axes[1].set_yticks(numpy.arange(5, 26, 5), size=labelsize)
    axes[1].set_yticklabels(numpy.arange(5, 26, 5))
    axes[1].set_xlabel('Days', size=labelsize)
    axes[1].set_ylabel('Elevation (degrees)', size=labelsize)

    # legend
    legend_elements_1 = [Line2D([0], [0], marker='o', color='black', label='Free ears',
                              markerfacecolor='w', markersize=3),
                       Line2D([0], [0], marker='o', color='black', label='With molds',
                              markerfacecolor='black', markersize=3)]
    axes[0].legend(handles=legend_elements_1, loc='best', frameon=False, handlelength=0, fontsize=str(labelsize))
    axes[1].legend(loc='center right', frameon=False, fontsize=str(labelsize))

    # axes frames
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # axis parameters
    axes[0].tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     labelsize=labelsize, width=1.5, length=2)
    axes[1].tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                        labelsize=labelsize, width=1.5, length=2)
    axes[1].set_ylim(2.5, 26)

    # horizontal lines
    for y1 in numpy.linspace(.1, 1, 10):
        axes[0].axhline(y=y1, xmin=0, xmax=20, color='0.7', linewidth=.1)

    for y2 in numpy.linspace(5, 22.5, 8):
        axes[1].axhline(y=y2, xmin=0, xmax=20, color='0.7', linewidth=.1)

    # subplot labels
    subpl_labels = ['A', 'B']
    for ax_id, ax in enumerate(axes):
        label_x = axes[ax_id].get_position().x0 - 0.06 # move to left
        label_y = axes[ax_id].get_position().y1 - 0.04 # move down # todo should be dynamic
        fig.text(x=label_x, y=label_y, s=subpl_labels[ax_id], size=labelsize+6)
        plt.show()

    return fig, axes


"""
# save as scalable vector graphics
fig, axes = learning_plot('average')
plt.savefig('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures/ele_learning_plot/learning_plot.svg', format='svg')

# change fig size
fig.tight_layout()
dpi = 254   # inch to pixels
# A4
# \pdfpagewidth = 1654pt
# \pdfpageheight = 2339pt % doesnt work
xpx = 4000
ypx = 2000
fig.set_size_inches(xpx / dpi, ypx / dpi)

"""