from matplotlib import pyplot as plt
from misc import rcparams
import numpy
from misc.unit_conversion import cm2in


def plot_time_course(figsize=(17, 7)):
    rcparams.set_rcParams_timeline()
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    dpi = 264
    # dpi=None
    fig, axes = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': [11, 1]}
        ,constrained_layout=True, dpi=dpi)

    # arrow
    axes[1].set_ylim(0,2)
    axes[1].plot(1.1, 0, lw=0, marker=">", ms=7, color="k",
            transform=axes[1].get_yaxis_transform(), clip_on=False)

    for axis in axes:
        axis.set_yticks([])
        axis.tick_params(axis='x', direction="in", bottom=True, top=False, left=False, right=False,
                         width=1.5, length=4)

    x1 = numpy.arange(11)  # days of measurement
    x2 = [15]
    axes[0].set_xticks(x1, size=13)
    axes[0].set_xlim(-.5, 10.5)
    axes[1].set_xticks(x2, size=13)
    axes[1].set_xlim(14.5, 15.5)

    # // break in x axis
    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False, linewidth=1)
    axes[0].plot((1+0.005, 1+0.01), (-.03, +.03), **kwargs)
    kwargs.update(transform=axes[1].transAxes)  # switch to the right axis
    axes[1].plot((-0.01*11, -0.005*11), (-.03, +.03), **kwargs)

    # textbox props
    bbox_props = dict(boxstyle='round, pad=.5', facecolor='none', edgecolor='black', alpha=0.5)
    # mold_boxprops = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr_day0 = '\n'.join((r'$\bullet\,$' + 'Localization test \n   with free ears',
                              r'$\bullet\,$' + 'Insertion of the\n   first earmolds',))

    textstr_day5 = '\n'.join((r'$\bullet\,$' + 'Removal of the\n   first earmolds',
                              r'$\bullet\,$' + 'Localization test\n   with free ears',
                              r'$\bullet\,$' + 'Insertion of the\n   second Earmolds'))

    textstr_day10 = '\n'.join((r'$\bullet\,$' + 'Removal of the\n   second earmolds',
                               r'$\bullet\,$' + 'Brief re-insertion of\n   the first earmolds\n   & localization test',
                               r'$\bullet\,$' + 'Localization test\n   with free ears'))

    textstr_day15 = 'Re-insertion of\nthe second earmolds\n& localization test'

    textbox1_h = .48
    textbox2_h = .25
    # annotate text
    axes[0].annotate(textstr_day0, xy=(0, 0.05), xytext=(-1.15, textbox1_h),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=8, ha='left', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-', lw=1, color='k'))

    axes[0].annotate('Daily training and\nlocalization tests\nwith the first earmolds', xy=(2.5, 0.13), xytext=(2.5, textbox2_h),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=8, ha='center', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-[, widthB=9, lengthB=1', lw=1, color='k'))

    axes[0].annotate(textstr_day5, xy=(5, 0.05), xytext=(3.8, textbox1_h),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=8, ha='left', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-', lw=1, color='k'))

    axes[0].annotate('Daily training and\nlocalization tests with\nthe second earmolds', xy=(7.5, 0.13), xytext=(7.5, textbox2_h),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=8, ha='center', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-[, widthB=9, lengthB=1', lw=1, color='k'))

    axes[0].annotate(textstr_day10, xy=(10, 0.05), xytext=(8.65, textbox1_h),
                     xycoords=axes[0].get_xaxis_transform(), fontsize=8, ha='left', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-', lw=1, color='k'))

    axes[1].annotate(textstr_day15, xy=(15, 0.05), xytext=(15, textbox2_h),
                     xycoords=axes[1].get_xaxis_transform(), fontsize=8, ha='center', va='bottom',
                    bbox=bbox_props, arrowprops=dict(arrowstyle='-', lw=1, color='k'))
    # arrow
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=-2.5, rect=None)
    fig.text(0.5, 0.02, 'Days into the experiment', va='center',
             ha='center', fontsize=plt.rcParams['axes.labelsize'])

    return fig, axes