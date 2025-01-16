import matplotlib
matplotlib.use('tkagg')
import legacy.MSc.analysis.localization_analysis as loc_analysis
import numpy
import scipy
from matplotlib import pyplot as plt
from pathlib import Path
from legacy.MSc.misc.unit_conversion import cm2in
from legacy.MSc.misc.rcparams import set_rcParams

def stim_response_plot(sub_id='vk', figsize=(14, 7), conditions = ['Ears Free', 'Earmolds Week 1'],
                       path=Path.cwd() / 'legacy' / 'Msc' / 'data' / 'experiment' / 'master'):

    """
    Plot m1 m2 adaptation curves throughout the experiment
    localization_dictionary (dict): dictionary of localization
    to_plot (string) can be 'average' (cross subject) or subject-id for single subject learning curve
    """

    set_rcParams()  # plot parameters
    fig_width = cm2in(figsize[0])
    fig_height = cm2in(figsize[1])
    dpi = 264
    fs = 8  # label fontsize
    lw = .7
    plt.rcParams.update({'font.family':'Helvetica', 'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
              'boxplot.capprops.linewidth': lw, 'lines.linewidth': lw,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2, 'axes.linewidth': lw, 'axes.spines.right': False, 'axes.spines.top': False})

    localization_dict = loc_analysis.get_localization_dictionary(path=path)

    # description:
    # stimulus response plot of vertical localization A) with free ears (B) after mold insertion
    # black whiskers indicate target elevations, grey circles indicate individual responses
    # Dotted lines result from linear regression between target and response coordinates.
    # The slopes define elevation gain.
    # grey lines show mean distance of individual responses to target locations (indicated by black dots)
    # response errors were quantified by the mean distance of individual responses from target locations
    # grey bars indicate standard deviations of responses for each target
    # response variability was quantified by the mean sd across targets

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                             layout='constrained', dpi=dpi, sharey=True)

    for axis, condition in zip(axes, conditions):
        data = numpy.asarray(list(localization_dict[condition][sub_id].values())[0].data)
        data = numpy.squeeze(data, axis=1)
        elevation_ticks = numpy.unique(data[:, 1, 1])

        # gain
        elevation_gain, n = scipy.stats.linregress(data[:, 1, 1], data[:, 0, 1])[:2]
        x = numpy.array((-38, 38))
        y = elevation_gain * x + n
        axis.plot(x, y, c='0', linewidth=.5, linestyle='--', label='EG')
        # boxes
        width = 3
        for idx, target in enumerate(elevation_ticks):
            boxdata = data[numpy.where(data[:, 1, 1] == target), 0, 1][0]
            mean = numpy.mean(boxdata)
            sd = numpy.std(boxdata)
            # RMSE
            axis.plot([target, target], [mean, target], c='0.6', linestyle='-', linewidth=lw, label='RMSE')
            # mean
            axis.hlines(mean, target-width/2, target+width/2, colors='0.6', )
            # target
            axis.hlines(target, target-width/2, target+width/2, colors='0', )
            # SD
            axis.bar([target], height=2*sd, bottom=mean-sd, color='0.9',
                                  width=width, linewidth=lw)

        # individual responses
        axis.scatter(data[:, 1, 1], data[:, 0, 1], s=5, linewidth=.5, edgecolor = '.75', facecolor = 'none')

        elevation_gain, n = scipy.stats.linregress(data[:, 1, 1], data[:, 0, 1])[:2]
        x = numpy.array((-38, 38))
        y = elevation_gain * x + n
        axis.plot(x, y, c='0', linewidth=.5, linestyle='--', label='EG')

        # ticks and labels
        axis.set_xticks(elevation_ticks)
        axis.set_yticks(elevation_ticks)
        axis.set_xlabel('Target elevations (deg)')
        labels = axis.get_xticklabels()
        labels[1].set_text('-25')
        labels[3].set_text('0')
        labels[5].set_text('25')
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)

    # legenda
    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[:2]
    labels = labels[:2]
    handles.append(matplotlib.patches.Rectangle((1,1), width=0, height=0, color='0.9', ))
    labels.append('SD')
    axes[0].legend(handles=handles, labels=labels, frameon=False, loc='upper left', handleheight=0.25,
                labelspacing=0, alignment='left', fontsize=8, bbox_to_anchor=((-0, 0.95)))

    axes[0].set_ylabel('Response elevations (deg)')
    axes[0].annotate('A', (.02, 0.95), c='k', weight='bold', xycoords='axes fraction')
    axes[1].annotate('B', (.02, 0.95), c='k', weight='bold', xycoords='axes fraction')
