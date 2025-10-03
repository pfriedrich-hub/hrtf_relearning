import legacy.MSc.analysis.localization_analysis as loc_analysis
import numpy
from matplotlib import pyplot as plt
from legacy.MSc.misc.unit_conversion import cm2in
from legacy.MSc.misc.rcparams import set_rcParams
from pathlib import Path
path=Path.cwd() / 'legacy' / 'MSc' / 'data' / 'experiment' / 'master'
w2_exclude = ['cs', 'lm', 'lk']
figsize=(17.5, 6.5)

def learning_plot(path, w2_exclude, figsize):
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

    # get data
    localization_dict = loc_analysis.get_localization_dictionary(path=path)
    subjects = list(localization_dict['Ears Free'].keys())
    localization_dict = loc_analysis.get_localization_data(localization_dict, subjects, w2_exclude)
    ef = localization_dict['Ears Free']['data']
    m1 = localization_dict['Earmolds Week 1']['data']
    m2 = localization_dict['Earmolds Week 2']['data']

    # delete nan for plt to interpolate points in learning curve
    # if numpy.isnan(ef).any():
    #     ef = numpy.delete(ef, numpy.where(numpy.isnan(ef))[0][0], numpy.where(numpy.isnan(ef))[1][0])
    #
    #     for subj_idx in numpy.unique(numpy.where(numpy.isnan(ef))[0]):
    #         for day_idx in numpy.unique(numpy.where(numpy.isnan(ef[subj_idx]))[0]):
    #             ef[subj_idx, day_idx]
    #     numpy.delete(ef, numpy.where(numpy.isnan(ef)))
    #
    # if numpy.isnan(m1).any():
    #     m1 = numpy.delete(m1, numpy.where(numpy.isnan(m1))[0][0] ,numpy.where(numpy.isnan(m1))[1][0])
    # if numpy.isnan(m2).any():
    #     m2 = numpy.delete(m2, numpy.where(numpy.isnan(m2))[0][0] ,numpy.where(numpy.isnan(m2))[1][0])

    # means ears free / mold1 / mold2
    ef_mean = numpy.nanmean(ef, axis=0)
    m1_mean = numpy.nanmean(m1, axis=0)
    m2_mean = numpy.nanmean(m2, axis=0)

    # ----- plot ----- #
    days = numpy.arange(12)  # days of measurement
    days[-1] = 15
    labels = [ 'Az Gain', 'RMSE (deg)', 'SD (deg)']
    w1_color = 0 # EG color and label
    w2_color = 0.6  # deviation in color of w2 adaptation
    indiv_color = 0.9
    label = None
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True, dpi=dpi) #  gridspec_kw = {'width_ratios': [1, 1]}
    axes.remove()
    ax0 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1)

    axes = fig.get_axes()
    for i, axis in enumerate(axes):
        i += 3  # get azimuth data
        marker_dist = .15
        if i > 2:      # adjust distance between final markers for left and right plots separately
            marker_dist = .3
        else:
            # individual trajectories
            # week 1
            for subject_data in m1[:, :, i]:
                nan = numpy.isnan(subject_data)
                if nan.any():  # interpolate NaN
                    subject_data[nan] = numpy.interp(numpy.where(nan)[0], numpy.where(~nan)[0], subject_data[~nan])
                axis.plot([marker_dist, 1, 2, 3, 4, 5 - marker_dist], subject_data[:6], c=str(indiv_color), lw=.5)
            # week 2
            for subject_data in m2[:, :, i]:
                nan = numpy.isnan(subject_data)
                if nan.any():  # interpolate NaN
                    subject_data[nan] = numpy.interp(numpy.where(nan)[0], numpy.where(~nan)[0], subject_data[~nan])
                axis.plot([5 + marker_dist, 6, 7, 8, 9, 10 - marker_dist], subject_data[:6], c=str(indiv_color), lw=.5)

            # dotted lines
            # axis.plot([0, .05], [ef_mean[0, i], m1_mean[0, i]], c=str(w1_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # day one mold insertion
            # axis.plot([4.95, 5], [m1_mean[5, i], ef_mean[1, i]], c=str(w1_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # week 1 mold removal
            # axis.plot([9.95, 10], [m2_mean[5, i], ef_mean[2, i]], c=str(w2_color), ls=(0, (5, 10)),
            #           lw=lw - .2)  # week 2 mold removal
            # mold 1 adaptation persistence
            axis.plot([4.95, 10.05], m1_mean[-2:, i], c=str(w1_color), ls=(0, (5, 10)), lw=lw)
            # mold 2 adaptation persistence
            axis.plot([9.95, 15], m2_mean[-2:, i], c=str(w2_color), ls=(0, (5, 10)), lw=lw)  #

        # week 1
        axis.plot([marker_dist, 1, 2, 3, 4, 5-marker_dist], m1_mean[:6, i], c=str(w1_color), label=label, lw=lw)  # week 1 learning curve
        # week 2
        axis.plot([5+marker_dist,  6,  7,  8, 9, 10-marker_dist], m2_mean[:6, i], c=str(w2_color), lw=lw)  # week 2 learning curve

        # error bars
        axis.errorbar([0, 5, 10], ef_mean[:3, i], capsize=3, yerr=localization_dict['Ears Free']['SE'][:3, i],
                       fmt="o", c=str(w1_color), markersize=markersize, fillstyle='full',
                             markerfacecolor='white', markeredgewidth=.5)  # error bar ears free
        # error bars and markers m1_mean adaptation
        axis.errorbar([marker_dist, 1, 2, 3, 4, 5-marker_dist], m1_mean[:6, i], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][:6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)
        # error bars and markers m2_mean adaptation and persistence
        axis.errorbar([5+marker_dist,  6,  7,  8, 9, 10-marker_dist, 15], m2_mean[:7, i], capsize=2,
                             yerr=localization_dict['Earmolds Week 2']['SE'][:7, i], fmt="s", c=str(w2_color),
                             markersize=markersize, markeredgewidth=.5)
        # error bars and markers m1_mean persistence
        axis.errorbar([10+marker_dist], m1_mean[6, i], capsize=2, yerr=localization_dict
                        ['Earmolds Week 1']['SE'][6, i], fmt="o", c=str(w1_color), markersize=markersize, markeredgewidth=.5)  # err m1_mean


        # axes ticks and limits
        # axis.set_xticks([0,5,10,15])
        # axis.set_xticklabels([0,5,10,15])
        axis.set_ylabel(labels[i-3])
    axes[1].set_xticklabels([])
    axes[0].set_xlabel('Days')
    axes[2].set_xlabel('Days')

    # axes[0].set_ylim(0, 1.02)
    # axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))

    # kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False, linewidth=lw)
    # axes[0].plot((1+0.005, 1+0.01), (-.03, +.03), **kwargs)
    # kwargs.update(transform=axes[1].transAxes)  # switch to the right axis
    # axes[1].plot((-0.01*11, -0.005*11), (-.03, +.03), **kwargs)

    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axes[0].set_yticklabels(ticklabels)


    # axes[1].set_yticks(numpy.arange(0, 26, 5))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    # axes[2].set_yticks(numpy.arange(0, 10, 2))
    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = '0'

    # axes[1].set_yticks(numpy.linspace(0, 0.0001, 20))
    # axes[2].set_yticks(numpy.linspace(0, 0.0001, 5))

    # annotations
    axes[0].annotate('insertion', xy=(0, .08), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('replacement', xy=(5, .08), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    axes[0].annotate('removal', xy=(10, .5), xycoords=axes[0].get_xaxis_transform(), fontsize=fs,
                xytext=(0, 0), textcoords="offset points", ha="left", va="center", rotation='horizontal')
    # horizontal lines
    # for y in numpy.linspace(.1, 1, 9):
    #     axes[0].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    # for y in numpy.arange(5, 22, 5):
    #     axes[1].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)
    # for y in numpy.arange(2, 9, 2):
    #     axes[2].axhline(y=y, xmin=0, xmax=20, color='0.9', linewidth=.5, zorder=-1)

    plt.tight_layout(pad=1.08, h_pad=.5, w_pad=None, rect=None)

    # subplot labels
    axes[0].annotate('A', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.1, 1.005), fontsize=fs)
    axes[1].annotate('B', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)
    axes[2].annotate('C', c='k', weight='bold', xycoords='axes fraction',
                xy=(-.3, 1.005), fontsize=fs)

    return fig, axis