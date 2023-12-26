from matplotlib import pyplot as plt

def set_rcParams():
    params = {'axes.spines.right': True, 'axes.spines.top': True, 'axes.titlesize': 10,
              'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.labelsize': 10,
              'boxplot.boxprops.linewidth': 0.5, 'boxplot.meanprops.color': 'k',
              'boxplot.meanprops.linewidth': 0.5, 'boxplot.whiskerprops.linewidth': 0.5,
              'boxplot.capprops.linewidth': 0.5, 'lines.linewidth': .5,
              'ytick.direction': 'in', 'xtick.direction': 'in', 'ytick.major.size': 2,
              'xtick.major.size': 2}
    plt.rcParams.update(params)
