import legacy as loc_analysis
import numpy
from matplotlib import pyplot as plt
from legacy import cm2in
from legacy import set_rcParams


def learning_plot(to_plot, path, w2_exclude, figsize):

    """
    Plot single subject stimulus response pattern
    and draw SD RMS and EG indications
    """
    set_rcParams()  # plot parameters
    plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
    fig_width = cm2in(figsize[0])
    fig_height = cm2in(figsize[1])
    dpi = 264
    fs = 8  # label fontsize
    markersize = 2
    lw = .7
    params = {'font.family':'Helvetica', 'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
              'boxplot.capprops.linewidth': lw, 'lines.linewidth': lw,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2, 'axes.linewidth': lw}
    plt.rcParams.update(params)

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
    labels = ['Azimuth Gain', 'RMSE (°)', 'SD (°)']
    w1_color = 0 # EG color and label
    w2_color = 0.6  # deviation in color of w2 adaptation

    label = None
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True, dpi=dpi) #  gridspec_kw = {'width_ratios': [1, 1]}

    ax0 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1)

    axes = fig.get_axes()
    for i, axis in enumerate(axes):
        marker_dist = .15
        if i > 0:      # adjust distance between final markers for left and right plots separately
            marker_dist = .3
        else:  # dotted lines
            # axis.plot([0, .05], [ef[0, i+3], m1[0, i+3]], c=str(w1_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # day one mold insertion
            # axis.plot([4.95, 5], [m1[5, i+3], ef[1, i+3]], c=str(w1_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # week 1 mold removal
            # axis.plot([9.95, 10], [m2[5, i+3], ef[2, i+3]], c=str(w2_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # week 2 mold removal
            # mold 1 adaptation persistence
            axis.plot([4.95, 10.05], m1[-2:, i+3], c=str(w1_color), ls=(0, (5, 10)), lw=lw)
            # mold 2 adaptation persistence
            axis.plot([9.95, 15], m2[-2:, i+3], c=str(w2_color), ls=(0, (5, 10)), lw=lw)  #

        # week 1
        axis.plot([marker_dist, 1, 2, 3, 4, 5-marker_dist], m1[:6, i+3], c=str(w1_color), label=label, lw=lw)  # week 1 learning curve
        # week 2
        axis.plot([5+marker_dist,  6,  7,  8, 9, 10-marker_dist], m2[:6, i+3], c=str(w2_color), lw=lw)  # week 2 learning curve

        # error bars
        axis.errorbar([0, 5, 10], ef[:3, i+3], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i+3],
                       fmt="o", c=str(w1_color), markersize=markersize, fillstyle='full',
                             markerfacecolor='white', markeredgewidth=.5)  # error bar ears free
        # error bars and markers m1 adaptation
        axis.errorbar([marker_dist, 1, 2, 3, 4, 5-marker_dist], m1[:6, i+3], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][:6, i+3], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)
        # error bars and markers m2 adaptation and persistence
        axis.errorbar([5+marker_dist,  6,  7,  8, 9, 10-marker_dist, 15], m2[:7, i+3], capsize=2,
                             yerr=localization_dict['Earmolds Week 2']['SE'][:7, i+3], fmt="s", c=str(w2_color),
                             markersize=markersize, markeredgewidth=.5)
        # error bars and markers m1 persistence
        axis.errorbar([10+marker_dist], m1[6, i+3], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][6, i+3], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)  # err m1

        # axes ticks and limits
        axis.set_xticks([0,5,10,15])
        axis.set_xticklabels([0,5,10,15])
        axis.set_ylabel(labels[i])
    axes[1].set_xticklabels([])
    axes[0].set_xlabel('Days')
    axes[2].set_xlabel('Days')

    axes[0].set_ylim(0, 1.2)
    axes[0].set_yticks(numpy.arange(0, 1.3, 0.2))
    # kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False, linewidth=lw)
    # axes[0].plot((1+0.005, 1+0.01), (-.03, +.03), **kwargs)
    # kwargs.update(transform=axes[1].transAxes)  # switch to the right axis
    # axes[1].plot((-0.01*11, -0.005*11), (-.03, +.03), **kwargs)
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axes[0].set_yticklabels(ticklabels)


    axes[1].set_yticks(numpy.arange(0, 9, 2))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    axes[2].set_yticks(numpy.arange(0, 4, 1))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    # axes[1].set_yticks(numpy.linspace(0, 0.0001, 20))
    # axes[2].set_yticks(numpy.linspace(0, 0.0001, 5))

    # annotations
    axes[0].annotate('insertion', xy=(0, .7), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('replace-\nment', xy=(5, .7), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('removal', xy=(10, .7), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    # horizontal lines
    for y in numpy.linspace(.2, 1.2, 6):
        axes[0].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    for y in numpy.arange(2, 9, 2):
        axes[1].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    for y in numpy.arange(1, 4, 1):
        axes[2].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)

    plt.tight_layout(pad=1.08, h_pad=.5, w_pad=None, rect=None)

    # subplot labels
    axes[0].annotate('A', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.1, 1.005), fontsize=fs)
    axes[1].annotate('B', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)
    axes[2].annotate('C', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)

    return fig, axis