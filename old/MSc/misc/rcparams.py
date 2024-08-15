from matplotlib import pyplot as plt

def set_rcParams():
    params = {'axes.spines.right': True, 'axes.spines.top': True,
              'axes.spines.left': True, 'axes.spines.bottom': True,
              'axes.titlesize': 10,
              'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.labelsize': 10,
              'boxplot.boxprops.linewidth': 0.5, 'boxplot.meanprops.color': 'k',
              'boxplot.meanprops.linewidth': 0.5, 'boxplot.whiskerprops.linewidth': 0.5,
              'boxplot.capprops.linewidth': 0.5, 'lines.linewidth': .5,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2, 'axes.linewidth': .5}
    plt.rcParams.update(params)

def set_rcParams_poster():
    textsize = 25
    params = {'axes.spines.right': True, 'axes.spines.top': True, 'axes.titlesize': textsize,
              'xtick.labelsize': textsize, 'ytick.labelsize': textsize, 'axes.labelsize': textsize,
              'lines.linewidth': 2,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2}
    plt.rcParams.update(params)

def set_rcParams_timeline():
    textsize = 10
    params = {'axes.spines.right': False, 'axes.spines.top': False, 'axes.spines.left': False,
              'axes.linewidth': 1,
              'axes.titlesize': textsize, 'xtick.labelsize': textsize,
              'ytick.labelsize': textsize, 'axes.labelsize': textsize,
              'xtick.direction': 'in', 'xtick.major.size': 5}
    plt.rcParams.update(params)
