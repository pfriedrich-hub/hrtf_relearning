import analysis.localization_analysis as loc_analysis
import numpy
import slab
from copy import deepcopy
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from pathlib import Path
from misc.unit_conversion import cm2in
from misc.rcparams import set_rcParams


def learning_plot(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude = ['cs', 'lm', 'lk']
                  , figsize=(17, 9)):

    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """
    set_rcParams()  # plot parameters
    plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
    fig_width = cm2in(figsize[0])
    fig_height = cm2in(figsize[1])
    dpi = 264
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
    labels = ['Elevation Gain', 'RMSE (°)', 'SD (°)']
    w1_color = 0 # EG color and label
    w2_color = 0.5  # deviation in color of w2 adaptation
    markersize = 3
    label = None
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True, dpi=dpi)
    ax0 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1)

    axes = fig.get_axes()
    for i, axis in enumerate(axes):
        # week 1
        axis.plot([0, .05], [ef[0, i], m1[0, i]], c=str(w1_color), ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
        axis.plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(w1_color), label=label, lw=1)  # week 1 learning curve
        axis.plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(w1_color), ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
        # week 2
        # axis.plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(w2_color), ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
        axis.plot([5.05,  6,  7,  8, 9, 9.95], m2[:6, i], c=str(w2_color), lw=1)  # week 2 learning curve
        axis.plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(w2_color), ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axis.plot([4.95, 10.05], m1[-2:, i], c=str(w1_color), ls=(0, (5, 10)), lw=1)
        # mold 2 adaptation persistence
        axis.plot([9.95, 15], m2[-2:, i], c=str(w2_color), ls=(0, (5, 10)), lw=1)#

        # error bars
        axis.errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                       fmt="o", c=str(w1_color), markersize=markersize, fillstyle='full',
                             markerfacecolor='white', markeredgewidth=.5)  # error bar ears free
        # error bars and markers m1 adaptation
        axis.errorbar([.05, 1, 2, 3, 4, 4.95], m1[:6, i], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][:6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)
        # error bars and markers m2 adaptation and persistence
        axis.errorbar([5.05,  6,  7,  8, 9, 9.9, 15], m2[:7, i], capsize=2,
                             yerr=localization_dict['Earmolds Week 2']['SE'][:7, i], fmt="s", c=str(w2_color),
                             markersize=markersize, markeredgewidth=.5)
        # error bars and markers m1 persistence
        axis.errorbar([10.1], m1[6, i], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)  # err m1

        # axes ticks and limits
        axis.set_xticks([0,5,10,15])
        axis.set_xticklabels([0,5,10,15])
        axis.set_ylabel(labels[i])
    axes[1].set_xticklabels([])
    axes[0].set_xlabel('Days')
    axes[2].set_xlabel('Days')
    axes[0].set_ylim(0.22,1.02)
    axes[0].set_yticks(numpy.arange(0.4, 1.2, 0.2))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[-1] = '1'
    axes[0].set_yticklabels(ticklabels)
    axes[1].set_yticks(numpy.arange(10, 26, 5))
    axes[2].set_yticks(numpy.arange(4, 10, 2))
    # axes[1].set_yticks(numpy.linspace(0, 0.0001, 20))
    # axes[2].set_yticks(numpy.linspace(0, 0.0001, 5))

    # annotations
    axes[0].annotate('mold \ninsertion', xy=(.1, .75), xycoords=axes[0].get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('mold \nreplacement', xy=(5.1, .75), xycoords=axes[0].get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('mold \nremoval', xy=(10.1, .75), xycoords=axes[0].get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')

    # horizontal lines
    for y in numpy.linspace(.2, 1, 9):
        axes[0].axhline(y=y, xmin=0, xmax=20, color='0.7', linewidth=.1, zorder=-1)
    for y in numpy.arange(10, 22, 5):
        axes[1].axhline(y=y, xmin=0, xmax=20, color='0.7', linewidth=.1, zorder=-1)
    for y in numpy.arange(4, 9, 2):
        axes[2].axhline(y=y, xmin=0, xmax=20, color='0.7', linewidth=.1, zorder=-1)

    plt.tight_layout(pad=1.08, h_pad=.5, w_pad=None, rect=None)

    # subplot labels
    axes[0].annotate('A', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.1, 1.005))
    axes[1].annotate('B', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005))
    axes[2].annotate('C', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005))

    return fig, axis