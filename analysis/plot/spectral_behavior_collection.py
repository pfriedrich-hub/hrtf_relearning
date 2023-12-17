import numpy
import scipy
from matplotlib import pyplot as plt
import analysis.statistics.stats_df as stats_df
from misc.unit_conversion import cm2in

measures = ['EG', 'vertical RMSE', 'vertical SD', 'horizontal RMSE', 'horizontal SD']


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
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['M1'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M2'])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)


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
    axis.boxplot(x[0][~numpy.isnan(x[0])], positions=[0], labels=['EF / M1'])
    axis.boxplot(x[1][~numpy.isnan(x[1])], positions=[1], labels=['EF / M2'])
    axis.boxplot(x[2][~numpy.isnan(x[2])], positions=[2], labels=['M1 / M2'])
    # spines
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)


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
    axis.scatter(x, y, marker='.', color='grey')
    axis.set_ylabel('VSI dissimilarity')
    axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0')
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    axis.set_title('Mold 1 vs Mold 2')
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
    axis.scatter(x, y, marker='.', color='grey')
    axis.set_ylabel('VSI dissimilarity')
    axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0')
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    axis.set_title('Ears free vs Mold 2')
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
    axis.scatter(x, y, marker='.', color='grey')
    axis.set_ylabel('VSI dissimilarity')
    axis.set_xlabel(measure)
    # d1 drop / vsi dissimilarity
    mask = ~numpy.isnan(x) & ~numpy.isnan(y)
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0')
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    axis.set_title('Ears free vs Mold 1')
    return fig, axis


""" thesis: ears free vs vertical rmse """
def ef_vsi_th(main_df, axis=None, measure='RMSE ele', figsize=(10, 7)):
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0'].drop(1)])  # remove outlier
    y = main_df['EF VSI'].drop(1).to_numpy(dtype='float16')
    axis.scatter(x, y, marker='.', color='grey', s=15)
    axis.set_ylabel('VSI')
    axis.set_xlabel(measure)
    x = numpy.array([item[measures.index(measure)] for item in main_df['EFD0']])  # remove outlier
    y = main_df['EF VSI'].to_numpy(dtype='float16')
    slope, intercept = scipy.stats.linregress(x, y, alternative='two-sided')[:2]
    x_vals = numpy.array(axis.get_xlim())
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='0')
    axis.set_xlim(x_vals[0], x_vals[1])
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
    return fig, axis


""" thesis: vsi overview """
def th_vsi_m_l_r(main_df, axis=None, figsize=(12, 8)):
    x, y = numpy.zeros((2, len(main_df['subject']))), numpy.zeros((2, len(main_df['subject'])))
    x[0] = main_df['M1 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M2 VSI l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['M1 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M2 VSI r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    # plot
    width = cm2in(figsize[0])
    height = cm2in(figsize[1])
    if not axis:
        fig, axis = plt.subplots(figsize=(width, height))
        axis.tick_params(axis='both', direction="in", bottom=True, top=False, left=True, right=False,
                         width=0.5, length=2)
    fig = axis.get_figure()
    # plot data
    c_list = ['#2668B3', '#CF1F48']  # triadic color scheme
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Molds 1')
    axis.scatter(x[1], y[1], marker='.', c=c_list[1], label='Molds 2')
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.2)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0)
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
    for i in range(len(x)):
        slope, intercept = scipy.stats.linregress(x[i][mask[i]], y[i][mask[i]], alternative='two-sided')[:2]
        x_vals = numpy.array((axis.get_xlim()))
        ys = slope * x_vals + intercept
        axis.plot(x_vals, ys, lw=0.5, c=c_list[i])
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
    l = axis.legend(frameon=False, loc='upper right', markerscale=0, labelspacing=0,
                    fontsize=plt.rcParams.get('axes.labelsize'))
    for i, text in enumerate(l.get_texts()):
        text.set_color(c_list[i])

    # axis.set_aspect('equal')
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
    axis.scatter(x, y, marker='.', c='grey', label='Ears Free', s=15)
    # axes ticks and limits
    ticks = numpy.arange(0, 1.1, 0.2)
    x_ticks = axis.set_xticks(ticks)
    x_ticks[0]._apply_params(width=0) # doesnt work
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
    slope, intercept = scipy.stats.linregress(x[mask], y[mask], alternative='two-sided')[:2]
    x_vals = numpy.array((axis.get_xlim()))
    ys = slope * x_vals + intercept
    axis.plot(x_vals, ys, lw=0.7, c='k')
    # axis labels
    axis.set_xlabel('Left Ear VSI')
    axis.set_ylabel('Right Ear VSI')
    # disable spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # set plot spines lw
    for loc, spine in axis.spines.items():
        spine.set_lw(0.5)
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
    axis.scatter(x[0], y[0], marker='.', c=c_list[0], label='Ears Free')
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
        axis.plot(x_vals, ys, lw=0.5, c=c_list[i])
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


def sp_str_l_r(main_df, axis=None):
    x, y = numpy.zeros((3, len(main_df['subject']))), numpy.zeros((3, len(main_df['subject'])))
    x[0] = main_df['EF spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[1] = main_df['M1 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    x[2] = main_df['M2 spectral strength l'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[0] = main_df['EF spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[1] = main_df['M1 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
    y[2] = main_df['M2 spectral strength r'].to_numpy(dtype='float16', na_value=numpy.NaN)
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
