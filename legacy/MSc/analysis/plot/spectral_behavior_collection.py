import numpy
import scipy
import math
from matplotlib import pyplot as plt

import legacy as stats_df
from legacy import cm2in

measures = ['EG', 'vertical RMSE', 'vertical SD', 'horizontal RMSE', 'horizontal SD']
""" plot """

""" EF vsi """
def vsi_ef_l_r_pub(main_df, axis=None, figsize=(12, 8)):
    x = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2, labelsize=8)
    fig = axis.get_figure()
    # plot data
    axis.scatter(x, y, marker='.', c='grey', label='Free ears', s=8)
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.5)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0) # doesnt work
    axis.set_yticks(ticks)
    axis.set_xlim(0, 1.1)
    axis.set_ylim(0, 1.1)
    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])

    # plot line
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array((axis.get_xlim()))
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    axis.set_aspect('equal')
    return fig, axis


def ef_vsi_pub(main_df, axis=None, measure='vertical RMSE', figsize=(9, 7)):
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2, labelsize=8)
    fig = axis.get_figure()
    # x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0'].drop(1)])  # remove outlier
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])  # remove outlier
    # y = main_df['EF VSI'].drop(1).to_numpy(dtype='float16')
    y = main_df['EF VSI'].to_numpy(dtype='float16')

    axis.scatter(x, y, marker='.', color='grey', s=8)
    axis.set_ylabel('VSI')
    axis.set_xlabel('RMSE (deg)')

    # axes ticks and limits Y
    yticks = numpy.arange(0.2, 1.1, 0.2)
    axis.set_yticks(yticks)
    axis.set_ylim(0.1, 1)
    yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    yticklabels[-1] = '1'
    axis.set_yticklabels(yticklabels)
    # axes ticks and limits X
    axis.set_xlim(5.5, 14)
    axis.set_xticks([7.5, 10, 12.5])
    xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    xticklabels[1] = '10'
    axis.set_xticklabels(xticklabels)

    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # plot regression line
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])  # dont remove outlier
    y = main_df['EF VSI'].to_numpy(dtype='float16')
    slope, intercept = scipy.stats.linregress(x, y, alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    return fig, axis


"""" mold vsi """

def vsi_m_l_r_pub(main_df, axis=None, figsize=(12, 8)):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    # plot
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)
    fig = axis.get_figure()
    # plot data
    c_list = ['0.8', '#2668B3', '#CF1F48']  # triadic color scheme
    # plt.style.use('grayscale')
    c_list = ['0', '0', '0.5']  # greyscale color scheme
    axis.scatter(x[0], y[0],fc='none', edgecolors=c_list[0], linewidth=.5, c=c_list[0], label='Free ears', s=5)
    axis.scatter(x[1], y[1], marker='o', c=c_list[1], label='Mold 1', s=5)
    axis.scatter(x[2], y[2], marker='s', c=c_list[2], label='Mold 2', s=5)
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.4)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0)
    axis.set_yticks(ticks)
    axis.set_xlim(0, 1.1)
    axis.set_ylim(0, 1.1)
    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])

    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    # plot line
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    for i in range(len(x)):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array((axis.get_xlim()))
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.5, c=c_list[i], zorder=-1)
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # legend
    l = axis.legend(frameon=False, loc='lower left', markerscale=1, labelspacing=0, alignment='left',
                    fontsize=8, bbox_to_anchor=((-0, 0.8)))
    for i, text in enumerate(l.get_texts()):
        text.set_color(c_list[i])
    axis.set_aspect('equal')
    return fig, axis

def boxplot_vsi_pub(main_df, axis=None):
    """
    VSI across conditions
    """
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    ef_l = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    ef_r = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((ef_l, ef_r))
    m1_l = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1_r = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((m1_l, m1_r))
    m2_l = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m2_r = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m2_l, m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI')
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)
    axis.set_ylabel('VSI')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['Free\nears'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['Mold 1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['Mold 2'])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axes
    ticks = numpy.arange(0, 1.2, 0.5)
    y_ticks = axis.set_yticks(ticks)
    ticklabels = ['0', '0.5', '1']
    axis.set_yticklabels(ticklabels)
    # axis.ticks()[0]
    # axis.set_yticks(axis.get_yticks()[1:])
    # y_ticks[0]._apply_params(width=0)

    # axis.set_yticks([0, 0.5, 1])
    # yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    # yticklabels[-1] = '1'
    # yticklabels[0] = '0'
    # axis.set_yticklabels(yticklabels)
    # axis.set_yticks(axis.get_yticks()[1:])

def scatter_perm_vsi_dis_pub(main_df, bandwidth, axis=None):
    """
    compute VSI dissimilarity between every possible pair of participants
    and compare it with the VSI dissimilarities / spectral difference between free and M1 / M2 DTFs
    of each participant.
    """
    vsi_dis = stats_df.ef_vsi_dis_perm(main_df, bandwidth)
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)
    axis.set_xlabel('Left Ear\nVSI Dissimilarity')
    axis.set_ylabel('Right Ear\nVSI Dissimilarity')
    # c_list = ['0.8', '#2668B3', '#CF1F48', '#F7D724']  # triadic color scheme
    c_list = ['0.8', '0', '0.5']  # greyscale color scheme

    axis.scatter(vsi_dis[:, 0], vsi_dis[:, 1], fc='none', edgecolors=c_list[0], linewidth=.5, label='Free vs. Free', s=5)
    axis.scatter(x[0], y[0], marker='o', c=c_list[1], label='Free vs. Mold 1', s=5)
    axis.scatter(x[1], y[1], marker='s', c=c_list[2], label='Free vs. Mold 2', s=5)
    # axis.scatter(x[2], y[2], marker='.', c=c_list[3], label='Molds1 vs. Molds2', s=8)
    axis.set_ylim(0, 1.2)
    axis.set_xlim(0, 1.2)
    # legend
    l = axis.legend(frameon=False, loc='lower left', markerscale=1, labelspacing=0,
                    fontsize=8, bbox_to_anchor=(-0.2, 0.85), alignment='left')
    for i, text in enumerate(l.get_texts()):
        text.set_color(c_list[i])
    # axes and ticks
    ticks = numpy.arange(0, 1.3, 0.4)
    x_ticks = axis.set_xticks(ticks)
    y_ticks = axis.set_yticks(ticks)
    ticklabels = ['0', '0.4', '0.8', '1.2']
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])
    # yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    # xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    # xticklabels[0] = '0'
    # xticklabels[-2] = '1'
    # yticklabels[-2] = '1'
    # y_ticks = axis.set_yticks(axis.get_yticks())
    # x_ticks = axis.set_xticks(axis.get_xticks())
    x_ticks[0]._apply_params(width=0)
    y_ticks[0]._apply_params(width=0)
    axis.set_aspect('equal')

def triangular_vsi_dis_pub(main_df, axis=None):
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    efm1_l = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm1_r = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((efm1_l, efm1_r))
    efm2_l = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm2_r = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((efm2_l, efm2_r))
    m1m2_l = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1m2_r = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m1m2_l, m1m2_r))

    # mean vsi dissimilarities
    efm1 = numpy.nanmean(x, axis=1)[0]
    efm2 = numpy.nanmean(x, axis=1)[1]
    m1m2 = numpy.nanmean(x, axis=1)[2]
    data = numpy.array((efm1, efm2, m1m2))
    sem_data = scipy.stats.sem(x, axis=1, nan_policy='omit').data

    p_ef = (0, 0)
    p_m1 = (efm1, 0)
    # Calculate cosine of angle A
    cos_A = (efm1 ** 2 + efm2 ** 2 - m1m2 ** 2) / (2 * efm1 * efm2)
    # Calculate the angle A in radians
    A = math.acos(cos_A)
    # Calculate coordinates of P3
    p_m2 = (efm2 * cos_A, efm2 * math.sin(A))
    points = numpy.array([p_ef, p_m1, p_m2])
    # points[1] = points[1][::-1] # rotate triangle
    # points[0,0] = -points[0,0] # mirror triangle
    if not axis:
        fig, axis = plt.subplots(1, 1)
        # axis.set_title('VSI Dissimilarity')
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)

    scipy.spatial.distance.euclidean(p_m1, p_m2)

    axis.plot(points[:,0], points[:, 1], color='0.8', lw=0.5, linestyle='--', zorder=1)
    axis.plot((points[0, 0], points[2, 0]), (points[0, 1], points[2, 1]), color='0.8', lw=0.5, linestyle='--', zorder=1)

    colors = ['0', '0', '0.5']  # greyscale color scheme
    axis.scatter(p_ef[0], p_ef[1], fc='none', edgecolors=colors[0], linewidth=.5, label='Free Ears', s=5, zorder=2)
    axis.scatter(p_m1[0], p_m1[1],  marker='o', c=colors[1], label='Mold 1', s=5, zorder=2)
    axis.scatter(p_m2[0], p_m2[1],  marker='s', c=colors[2], label='Mold 2', s=5, zorder=2)

    plt.annotate('{0:.2f}'.format(data[0]) +  '\n' + u'\u00b1' + ' ' + ('%.2f' % sem_data[0]).lstrip('0'),
                 (points[0, 0] + (.02), points[0, 1] + .02), fontsize=8)
    plt.annotate('{0:.2f}'.format(data[1]) +  '\n' + u'\u00b1' + ' ' + ('%.2f' % sem_data[1]).lstrip('0'),
                 (points[1, 0] + (.02), points[1, 1] + .02), fontsize=8)
    plt.annotate('{0:.2f}'.format(data[2]) +  '\n' + u'\u00b1' + ' ' + ('%.2f' % sem_data[2]).lstrip('0'),
                 (points[2, 0] + (.02), points[2, 1] + .02), fontsize=8)



    # legend
    l = axis.legend(frameon=False, loc='lower left', markerscale=1, labelspacing=0,
                    fontsize=8, bbox_to_anchor=(-0.2, .8), alignment='left')
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # # axes
    axis.set_xlim(-.1, .7)
    axis.set_ylim(-.2, .6)
    axis.set_yticks([])
    axis.set_xticks([])

    axis.set_aspect('equal')


def boxplot_vsi_dis_pub(main_df, axis=None):
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    efm1_l = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm1_r = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((efm1_l, efm1_r))
    efm2_l = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm2_r = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((efm2_l, efm2_r))
    m1m2_l = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1m2_r = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m1m2_l, m1m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI Dissimilarity')
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)
    axis.set_ylabel('VSI Dissimilarity')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['Free vs\nMolds1'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[2], labels=['Free vs\nMolds2'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[1], labels=['Molds1 vs\nMolds2'])
    plt.xticks(rotation=90)
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axes
    axis.set_yticks(numpy.arange(0.2, 1.5, 0.4))
    yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    yticklabels[2] = '1'
    axis.set_yticklabels(yticklabels)
    xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.yticks(rotation=0)











""" thesis: VSI dissimilarity EF / M1M2 vs EF / EF """
def th_scatter_perm_vsi_dis(main_df, bandwidth, axis=None):
    """
    compute VSI dissimilarity between every possible pair of participants
    and compare it with the VSI dissimilarities / spectral difference between free and M1 / M2 DTFs
    of each participant.
    """
    vsi_dis = stats_df.ef_vsi_dis_perm(main_df, bandwidth)
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.set_xlabel('Left Ear VSI Dissimilarity')
    axis.set_ylabel('Right Ear VSI Dissimilarity')
    c_list = ['0.8', '#2668B3', '#CF1F48', '#F7D724']  # triadic color scheme
    axis.scatter(vsi_dis[:, 0], vsi_dis[:, 1], marker='.', color=c_list[0], label='Free vs. Free', s=15)
    axis.scatter(x[0], y[0], marker='.', c=c_list[1], label='Free vs. Molds1', s=15)
    axis.scatter(x[1], y[1], marker='.', c=c_list[2], label='Free vs. Molds2', s=15)
    axis.scatter(x[2], y[2], marker='.', c=c_list[3], label='Molds1 vs. Molds2', s=15)
    axis.set_ylim(0, 1.2)
    axis.set_xlim(0, 1.2)
    # legend
    l = axis.legend(frameon=False, loc='lower left', markerscale=0, labelspacing=0,
                    fontsize=8, bbox_to_anchor=(-0.2, 0.8), alignment='left')
    for i, text in enumerate(l.get_texts()):
        text.set_color(c_list[i])
    # axes and ticks
    ticks = numpy.arange(0, 1.3, 0.4)
    x_ticks = axis.set_xticks(ticks)
    y_ticks = axis.set_yticks(ticks)
    ticklabels = ['0', '0.4', '0.8', '1.2']
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])
    # yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    # xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    # xticklabels[0] = '0'
    # xticklabels[-2] = '1'
    # yticklabels[-2] = '1'
    # y_ticks = axis.set_yticks(axis.get_yticks())
    # x_ticks = axis.set_xticks(axis.get_xticks())
    x_ticks[0]._apply_params(width=0)
    y_ticks[0]._apply_params(width=0)
    axis.set_aspect('equal')

""" thesis: vsi overview """
def th_vsi_m_l_r(main_df, axis=None, figsize=(12, 8)):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    # plot
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    # plot data
    c_list = ['0.8', '#2668B3', '#CF1F48']  # triadic color scheme
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Free ears', s=15)
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='Molds 1', s=15)
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='Molds 2', s=15)
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.2)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0)  # doesnt work
    axis.set_yticks(ticks)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])

    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    # plot line
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    for i in range(len(x)):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array((axis.get_xlim()))
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.5, c=c_list[i], zorder=-1)
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # legend
    l = axis.legend(frameon=False, loc='lower left', markerscale=0, labelspacing=0, alignment='left',
                    fontsize=8, bbox_to_anchor=((-0.2, 0.8)))
    for i, text in enumerate(l.get_texts()):
        text.set_color(c_list[i])
    axis.set_aspect('equal')
    return fig, axis

def th_vsi_ef_l_r(main_df, axis=None, figsize=(12, 8)):
    x = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
    axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                     width=0.5, length=2)
    fig = axis.get_figure()
    # plot data
    axis.scatter(x, y, marker='.', c='grey', label='Free ears', s=15)
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.5)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0) # doesnt work
    axis.set_yticks(ticks)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])

    # plot line
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array((axis.get_xlim()))
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    axis.set_aspect('equal')
    return fig, axis

def th_boxplot_vsi(main_df, axis=None):
    """
    VSI across conditions
    """
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    ef_l = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    ef_r = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((ef_l, ef_r))
    m1_l = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1_r = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((m1_l, m1_r))
    m2_l = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m2_r = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m2_l, m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI')
    axis.set_ylabel('VSI')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['Free\nears'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['Molds 1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['Molds 2'])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axes
    axis.set_ylim(0, 1)
    ticks = numpy.arange(0, 1.1, 0.2)
    y_ticks = axis.set_yticks(ticks)
    y_ticks[0]._apply_params(width=0)  # doesnt work
    ticklabels = [item.get_text() for item in axis.get_yticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_yticklabels(ticklabels)
    xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels, size=8)

def th_boxplot_vsi_dis(main_df, axis=None):
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    efm1_l = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm1_r = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((efm1_l, efm1_r))
    efm2_l = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm2_r = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((efm2_l, efm2_r))
    m1m2_l = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1m2_r = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m1m2_l, m1m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI Dissimilarity')
    axis.set_ylabel('VSI Dissimilarity')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['Free vs\nMolds1'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[2], labels=['Free vs\nMolds2'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[1], labels=['Molds1 vs\nMolds2'])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axes
    axis.set_yticks(numpy.arange(0.2, 1.5, 0.4))
    yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    yticklabels[2] = '1'
    axis.set_yticklabels(yticklabels)
    xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels, size=8)


def th_d5dr_vsi_dis_m1m2(main_df, axis=None, measure='vertical RMSE', figsize=(10, 7)):
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 drop']])
    y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    axis.scatter(x, y, marker='.', color='grey', s=15)
    axis.set_ylabel('VSI dissimilarity')
    axis.set_xlabel('RMSE (°)')
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axis.set_title('Mold 1 vs Mold 2')
    # axes and ticks
    xticks = numpy.arange(0,21,5)
    x_ticks = axis.set_xticks(xticks)
    yticks = numpy.arange(0, 1.3, 0.4)
    y_ticks = axis.set_yticks(yticks)
    ticklabels = ['0', '0.4', '0.8', '1.2']
    axis.set_yticklabels(ticklabels)
    axis.set_yticks(axis.get_yticks()[1:])
    # y_ticks[0]._apply_params(width=0)
    return fig, axis


def th_d5dr_vsi_dis(main_df, axis=None, measure='vertical RMSE', figsize=(10, 7)):
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 drop']])
    y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    axis.scatter(x, y, marker='.', color='grey', s=15)
    axis.set_ylabel('VSI Dissimilarity')
    axis.set_xlabel('RMSE (deg)')
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axis.set_title('Free vs Mold 2')
    xticks = numpy.arange(10,26,5)
    x_ticks = axis.set_xticks(xticks)
    return fig, axis

def th_d0dr_vsi_dis(main_df, axis=None, measure='vertical RMSE', figsize=(10, 7)):
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 drop']])
    y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    axis.scatter(x, y, marker='.', color='grey', s=15)
    axis.set_ylabel('VSI Dissimilarity')
    axis.set_xlabel('RMSE (°)')
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # axis.set_title('Free vs Mold 1')
    xticks = numpy.arange(10,26,5)
    x_ticks = axis.set_xticks(xticks)
    return fig, axis

""" thesis: ears free vs vertical rmse """
def ef_vsi_th(main_df, axis=None, measure='vertical RMSE', figsize=(9, 7)):
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    # x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0'].drop(1)])  # remove outlier
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])  # remove outlier
    # y = main_df['EF VSI'].drop(1).to_numpy(dtype='float16')
    y = main_df['EF VSI'].to_numpy(dtype='float16')

    axis.scatter(x, y, marker='.', color='grey', s=15)
    axis.set_ylabel('VSI')
    axis.set_xlabel(measure)

    # axes ticks and limits Y
    yticks = numpy.arange(0.2, 1.1, 0.2)
    axis.set_yticks(yticks)
    axis.set_ylim(0.1, 1)
    yticklabels = [item.get_text() for item in axis.get_yticklabels()]
    yticklabels[-1] = '1'
    axis.set_yticklabels(yticklabels)
    # axes ticks and limits X
    axis.set_xlim(5.5, 14)
    axis.set_xticks([7.5, 10, 12.5])
    xticklabels = [item.get_text() for item in axis.get_xticklabels()]
    xticklabels[1] = '10'
    axis.set_xticklabels(xticklabels)

    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # plot regression line
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])  # dont remove outlier
    y = main_df['EF VSI'].to_numpy(dtype='float16')
    slope, intercept = scipy.stats.linregress(x, y, alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0.8', zorder=-1)
    return fig, axis

def th_vsi_l_r(main_df, axis=None, figsize=(12, 10)):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    # plot
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    # plot data
    c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Free ears')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='Molds 1')
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='Molds 2')
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.2)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_yticks(axis.get_yticks()[1:])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.8)
    # string format
    ticklabels = [item.get_text() for item in axis.get_xticklabels()]
    ticklabels[0] = '0'
    ticklabels[-1] = '1'
    axis.set_xticklabels(ticklabels)
    # plot line
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    for i in range(3):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array((axis.get_xlim()))
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.5, c=c_list[i], zorder=-1)
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # legend
    axis.legend(frameon=False, loc='upper left')
    # axis.set_aspect('equal')
    return fig, axis


""" Ears Free baseline """


def ef_vsi(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
    # ears free performance / vsi
    # x = numpy.array([item[measures.index(measure)] for item in main_df['EF avg']])
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])
    y = main_df['EF VSI'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    # axis.set_title('Ears Free')
    axis.set_ylabel('VSI')
    axis.set_xlabel(measure)
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    slope, intercept = scipy.stats.linregress(x, y, alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')


def ef_spstr(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
    # ears free performance / spectral strength
    # x = numpy.array([item[measures.index(measure)] for item in main_df['EF avg']])
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])
    y = main_df['EF spectral strength'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 100)
    axis.set_ylabel('spectral strength')
    # axis.set_title('Ears Free')
    axis.set_xlabel(measure)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    slope, intercept = scipy.stats.linregress(x, y, alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


""" M1 effect """


def d0dr_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M1 d0')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 drop']])
    y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d0dr_w_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('weighted VSI dissimilarity')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M1 d0')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 drop']])
    y = main_df['EF M1 weighted VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d0dr_sp_dif(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('Ears free vs M1 d0')
        axis.set_xlabel(measure)
    # d1 drop / spectral difference
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 drop']])
    y = main_df['EF M1 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d0dr_pcw_dist(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M1 d0')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 drop']])
    y = main_df['EF M1 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5ga_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_title('ears free vs M1 d5')
        axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 gain']])
    y = main_df['EF M1 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5ga_w_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('weighted VSI dissimilarity')
        axis.set_title('ears free vs M1 d5')
        axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 gain']])
    y = main_df['EF M1 weighted VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5ga_sp_dif(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('ears free vs M1 d5')
        axis.set_xlabel(measure)
    # d1 drop / spectral difference
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 gain']])
    y = main_df['EF M1 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5ga_pcw_dist(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M1 d0')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1 gain']])
    y = main_df['EF M1 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


""" M2 effect """


def d5dr_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_title('Ears free vs M2 d0')
        axis.set_xlabel(measure)
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 drop']])
    y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5dr_sp_dif(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('Ears free vs M2 d0')
        axis.set_xlabel(measure)
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 drop']])
    y = main_df['EF M2 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5dr_pcw_dist(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M2 d0')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 drop']])
    y = main_df['EF M2 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_vsi_dis(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_title('Ears free vs M2 d5')
        axis.set_xlabel(measure)
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 gain']])
    y = main_df['EF M2 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_sp_dif(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('Ears free vs M2 d5')
        axis.set_xlabel(measure)
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 gain']])
    y = main_df['EF M2 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_pcw_dist(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('Ears free vs M2 d5')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M2 gain']])
    y = main_df['EF M2 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


# m1 vs m2
def d5dr_vsi_dis_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_title('M1/M2 drop vs M1/M2 difference')
        axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 drop']])
    y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5dr_sp_dif_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('M1/M2 drop vs M1/M2 difference')
        axis.set_xlabel(measure)
    # d1 drop / spectral difference
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 drop']])
    y = main_df['M1 M2 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d5dr_pcw_dist_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('M1/M2 drop vs M1/M2 difference')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 drop']])
    y = main_df['M1 M2 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_vsi_dis_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('VSI dissimilarity')
        axis.set_title('M1/M2 gain vs M1/M2 difference')
        axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 gain']])
    y = main_df['M1 M2 VSI dissimilarity'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0.1, 1.2)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_sp_dif_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('spectral difference')
        axis.set_title('M1/M2 gain vs M1/M2 difference')
        axis.set_xlabel(measure)
    # d1 drop / spectral difference
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 gain']])
    y = main_df['M1 M2 spectral difference'].to_numpy(dtype='float16', na_value=numpy.NaN)
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(0, 120)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


def d10ga_pcw_dist_m1m2(main_df, measure, axis=None):
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_ylabel('PCW distance')
        axis.set_xlabel(measure)
        axis.set_title('M1/M2 gain vs M1/M2 difference')
    # d1 drop / vsi dissimilarity
    x = numpy.array([item[measures.index(measure)] for item in main_df['M1M2 gain']])
    y = main_df['M1 M2 PCW dist'].to_numpy(dtype='float16')
    p_val = scipy.stats.spearmanr(x, y, nan_policy='omit')[1]
    axis.scatter(x, y, marker='.', color='0')
    axis.set_ylim(20, 100)
    axis.annotate('p=%.4f' % (p_val), (.4, .9), xycoords='axes fraction')
    for i, s in enumerate(list(main_df['subject'].unique())):
        axis.annotate(s, (x[i], y[i]))
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.4, c='0')


"""# ----  compare spectral features between ears ----- #"""


def vsi_l_r(main_df, axis=None, show=True):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if show:
        if not axis:
            fig, axis = plt.subplots(1, 1)
            axis.set_title('VSI')
        axis.set_xlabel('Left Ear VSI')
        axis.set_ylabel('Right Ear VSI')
        c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
        axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Ears Free')
        axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='M1')
        axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M2')
        axis.legend()
        axis.set_ylim(0, 1.2)
        axis.set_xlim(0, 1.2)
        mask = ~numpy.isnan(x) & ~numpy.isnan(y)
        for i in range(3):
            slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
            x_vals = numpy.array(axis.get_xlim())
            ys = slope * x_vals + intercept
            axis.plot(x_vals, ys, lw=0.4, c=c_list[i])
    vsi_l = x
    vsi_r = y
    return vsi_l, vsi_r


def sp_str_l_r(main_df, axis=None, show=True):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if show:
        if not axis:
            fig, axis = plt.subplots(1, 1)
            axis.set_title('spectral strength')
        axis.set_xlabel('Left Ear spectral strength')
        axis.set_ylabel('Right Ear spectral strength')
        c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
        axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Ears Free')
        axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='M1')
        axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M2')
        axis.legend()
        axis.set_ylim(0, 150)
        axis.set_xlim(0, 150)
        mask = ~numpy.isnan(x) & ~numpy.isnan(y)
        for i in range(3):
            slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
            x_vals = numpy.array(axis.get_xlim())
            ys = slope * x_vals + intercept
            axis.plot(x_vals, ys, lw=0.4, c=c_list[i])
    sp_st_l = x
    sp_st_r = y
    return sp_st_l, sp_st_r


def scatter_vsi_dis_l_r(main_df, axis=None):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('left and right ear VSI Dissimilarity')
    axis.set_xlabel('Lef Ear VSI Dissimilarity')
    axis.set_ylabel('Right Ear VSI Dissimilarity')
    c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='EF / M1')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='EF / M2')
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M1 / M2')
    axis.legend()
    axis.set_ylim(0, 1.2)
    axis.set_xlim(0, 1.2)
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    for i in range(3):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array(axis.get_xlim())
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.4, c=c_list[i])


def scatter_sp_dif_l_r(main_df, axis=None):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('left and right ear spectral difference')
    axis.set_xlabel('Left Ear spectral difference')
    axis.set_ylabel('Right Ear spectral difference')
    c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='EF / M1')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='EF / M2')
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M1 / M2')
    axis.legend()
    axis.set_ylim(0, 120)
    axis.set_xlim(0, 120)
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    for i in range(3):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array(axis.get_xlim())
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.4, c=c_list[i])


def boxplot_vsi_dis(main_df, axis=None):
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    efm1_l = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm1_r = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((efm1_l, efm1_r))
    efm2_l = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm2_r = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((efm2_l, efm2_r))
    m1m2_l = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1m2_r = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m1m2_l, m1m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI Dissimilarity')
    axis.set_ylabel('VSI Dissimilarity')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF / M1'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['EF / M2'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M1 / M2'])
    # wilcoxon signed rank test - dependent non-parametric
    scipy.stats.wilcoxon(x[0], x[1], nan_policy='omit')
    # mann-whitney U - independent non-parametric
    scipy.stats.mannwhitneyu(x[0], x[1], nan_policy='omit')


def boxplot_sp_dif(main_df, axis=None):
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    efm1_l = main_df['EF M1 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm1_r = main_df['EF M1 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((efm1_l, efm1_r))
    efm2_l = main_df['EF M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    efm2_r = main_df['EF M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((efm2_l, efm2_r))
    m1m2_l = main_df['M1 M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1m2_r = main_df['M1 M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m1m2_l, m1m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.set_ylabel('spectral difference')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF / M1'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['EF / M2'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M1 / M2'])
    # wilcoxon signed rank test - dependent non-parametric
    scipy.stats.wilcoxon(x[0], x[1], nan_policy='omit')
    # mann-whitney U - independent non-parametric
    scipy.stats.mannwhitneyu(x[0], x[1], nan_policy='omit')


def scatter_perm_vsi_dis(main_df, bandwidth, axis=None):
    """
    compute VSI dissimilarity between every possible pair of participants
    and compare it with the VSI dissimilarities / spectral difference between free and M1 / M2 DTFs
    of each participant.
    """
    vsi_dis = stats_df.ef_vsi_dis_perm(main_df, bandwidth)
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 VSI dissimilarity l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 VSI dissimilarity r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.set_xlabel('Left Ear VSI Dissimilarity')
    axis.set_ylabel('Right Ear VSI Dissimilarity')
    axis.set_title('left and right ear VSI Dissimilarity')
    c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
    axis.scatter(vsi_dis[:, 0], vsi_dis[:, 1], marker='.', color='0.5', label='EF / EF')
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='EF / M1')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='EF / M2')
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M1 / M2')
    axis.legend()
    axis.set_ylim(0, 1.2)
    axis.set_xlim(0, 1.2)
    # mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    # for i in range(3):
    #     slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
    #     x_vals = numpy.array(axis.get_xlim())
    #     ys = slope * x_vals + intercept
    #     axis.plot(x_vals, ys, lw=0.4, c=c_list[i])


def scatter_perm_sp_dif(main_df, bandwidth, axis=None):
    """
    compute spectral difference between every possible pair of participants
    and compare it with the VSI dissimilarities / spectral difference between free and M1 / M2 DTFs
    of each participant.
    """
    sp_dif = stats_df.ef_sp_dif_perm(main_df, bandwidth)
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF M1 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['EF M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M1 M2 spectral difference l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF M1 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['EF M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M1 M2 spectral difference r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    if not axis:
        fig, axis = plt.subplots(1, 1)
    axis.set_xlabel('Left Ear spectral difference')
    axis.set_ylabel('Right Ear spectral difference')
    axis.set_title('left and right ear spectral difference')
    c_list = ['#2668B3', '#F7D724', '#CF1F48']  # triadic color scheme
    axis.scatter(sp_dif[:, 0], sp_dif[:, 1], marker='.', color='0.5', label='EF / EF')
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='EF / M1')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='EF / M2')
    axis.scatter(x[2], y[2], marker='.', c=c_list[2], label='M1 / M2')
    axis.legend()
    axis.set_ylim(0, 150)
    axis.set_xlim(0, 150)
    # mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    # for i in range(3):
    #     slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
    #     x_vals = numpy.array(axis.get_xlim())
    #     ys = slope * x_vals + intercept
    #     axis.plot(x_vals, ys, lw=0.4, c=c_list[i])


def boxplot_vsi(main_df, axis=None):
    """
    VSI across conditions
    """
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    ef_l = main_df['EF VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    ef_r = main_df['EF VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((ef_l, ef_r))
    m1_l = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1_r = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((m1_l, m1_r))
    m2_l = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m2_r = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m2_l, m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('VSI')
    axis.set_ylabel('VSI')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['M1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M2'])
    # wilcoxon signed rank test - dependent non-parametric
    scipy.stats.wilcoxon(x[0], x[1], nan_policy='omit')
    # mann-whitney U - independent non-parametric
    scipy.stats.mannwhitneyu(x[0], x[1], nan_policy='omit')


def boxplot_sp_str(main_df, axis=None):
    """
    spectral strength across conditions
    """
    x = numpy.zeros((3, len(main_df['subject']) * 2))
    ef_l = main_df['EF spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    ef_r = main_df['EF spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[0] = numpy.hstack((ef_l, ef_r))
    m1_l = main_df['M1 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m1_r = main_df['M1 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = numpy.hstack((m1_l, m1_r))
    m2_l = main_df['M2 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    m2_r = main_df['M2 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = numpy.hstack((m2_l, m2_r))
    if not axis:
        fig, axis = plt.subplots(1, 1)
        axis.set_title('spectral strength')
    axis.set_ylabel('spectral strength')
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['M1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M2'])
    # wilcoxon signed rank test - dependent non-parametric
    scipy.stats.wilcoxon(x[0], x[1], nan_policy='omit')
    # mann-whitney U - independent non-parametric
    scipy.stats.mannwhitneyu(x[0], x[1], nan_policy='omit')
