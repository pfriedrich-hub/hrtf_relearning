import MSc.analysis.localization_analysis as loc_analysis
import numpy
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from pathlib import Path
from MSc.misc.unit_conversion import cm2in

def learning_plot(to_plot='average', path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude = ['cs', 'lm', 'lk']
                  , figsize=(30, 15)):

    #todo add x axis label adaptation session 1 / session 2, maybe remove dotted lines
    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
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
    colors = [0.2, 0.65] # deviation in color of SD adaptation
    w2_deviation = [0.2, 0.1]  # deviation in color of w2 adaptation
    color = colors[0] # EG color and label
    w2_dev = w2_deviation[0]
    label = None
    fig, axis = plt.subplots(1, 1, figsize=(width, height), layout='constrained')
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.05)
    markersize = 4
    for i, ax_id in enumerate([0]):
        if i >= 1:  # RMSE and SD colors and labels
            label = labels[i-1]
            color = colors[i-1]
            w2_dev = w2_deviation[i-1]
        # week 1
        axis.plot([0, .05], [ef[0, i], m1[0, i]], c=str(color-w2_dev), ls=(0, (5, 10)), lw=0.8)  # day one mold insertion
        axis.plot([.05, 1, 2, 3, 4, 4.95], m1[:6, i], c=str(color-w2_dev), label=label, lw=1)  # week 1 learning curve
        axis.plot([4.95, 5], [m1[5, i], ef[1, i]], c=str(color-w2_dev), ls=(0, (5, 10)), lw=0.8)  # week 1 mold removal
        # week 2
        axis.plot([5, 5.05], [ef[1, i], m2[0, i]], c=str(color+w2_dev), ls=(0, (5, 10)), lw=0.8)  # week 2 mold insertion
        axis.plot([5.05,  6,  7,  8, 9, 9.95], m2[:6, i], c=str(color+w2_dev), lw=1)  # week 2 learning curve
        axis.plot([9.95, 10], [m2[5, i], ef[2, i]], c=str(color+w2_dev), ls=(0, (5, 10)), lw=0.8)  # week 2 mold removal
        # mold 1 adaptation persistence
        axis.plot([4.95, 10.05], m1[-2:, i], c=str(color-w2_dev), ls=(0, (5, 10)), lw=1)
        # mold 2 adaptation persistence
        axis.plot([9.95, 15], m2[-2:, i], c=str(color+w2_dev), ls=(0, (5, 10)), lw=1)
        # error bars
        if len(subjects) > 1:
            axis.errorbar([0, 5, 10], ef[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                           fmt="o", c=str(color), markersize=markersize, fillstyle='full',
                                 markerfacecolor='white', markeredgewidth=.5)  # error bar ears free
            axis.errorbar([.05, 1, 2, 3, 4, 4.95, 10.1], m1[:7, i], capsize=2, yerr=localization_dict
                            ['Earmolds Week 1']['SE'][:7, i], fmt="o", c=str(color-w2_dev), markersize=markersize, markeredgewidth=.5)  # err m1
            axis.errorbar([5.05,  6,  7,  8, 9, 9.9, 15], m2[:7, i], capsize=2,
                                 yerr=localization_dict['Earmolds Week 2']['SE'][:7, i], fmt="s", c=str(color+w2_dev),
                                 markersize=markersize, markeredgewidth=.5)  # err m2
            # axis.errorbar([6.1,  7,  8,  9, 10, 11.1], m2[:6, i], capsize=3, yerr=localization_dict['Earmolds Week 2']['SE']
            #                     [:6, i], fmt="o", c=color, elinewidth=0.5, markersize=3)
    # axes ticks and limits
    axis.set_xticks([0,5,10,15])
    axis.set_xticklabels([0,5,10,15])
    axis.set_ylim(0.22,1.02)
    axis.set_yticks(numpy.arange(0.4, 1.2, 0.2))
    axis.set_ylabel('Elevation Gain')
    axis.set_xlabel('Days')

    # annotations
    axis.annotate('mold \ninsertion', xy=(.1, .75), xycoords=axis.get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')

    axis.annotate('mold \nreplacement', xy=(5.1, .75), xycoords=axis.get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axis.annotate('mold \nremoval', xy=(10.1, .75), xycoords=axis.get_xaxis_transform(),
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')

    # axis 1 legend
    legend_elements_1 = [Line2D([0], [0], marker='o', color='black', label='Free ears',
                              markerfacecolor='w', markersize=markersize, markeredgewidth=.5),
                       Line2D([0], [0], marker='o', color=str(colors[0] - w2_deviation[0]), label='Molds 1',
                              markerfacecolor=str(colors[0] - w2_deviation[0]), markersize=markersize, markeredgewidth=.5),
                         Line2D([0], [0], marker='s', color=str(colors[0] + w2_deviation[0]), label='Molds 2',
                                markerfacecolor=str(colors[0] + w2_deviation[0]), markersize=markersize, markeredgewidth=.5)]
    l1 = axis.legend(handles=legend_elements_1, loc='best', frameon=False, handlelength=0, fontsize=plt.rcParams.get('axes.labelsize'))
    c_list = ['black', str(colors[0] - w2_deviation[0]), str(colors[0] + w2_deviation[0])]
    for i, text in enumerate(l1.get_texts()):
        text.set_color(c_list[i])


    # horizontal lines
    for y1 in numpy.linspace(.1, 1, 10):
        axis.axhline(y=y1, xmin=0, xmax=20, color='0.7', linewidth=.1, zorder=-1)

    # remove spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)

    return fig, axis
