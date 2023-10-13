import analysis.create_dataframe as create_df
import misc.octave_spacing
from analysis.plot import stats_plot_collection as stats_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from matplotlib import pyplot as plt
main_df = create_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')


# bandwidth = (5700, 8000)
# bandwidth = (5700, 11300) # 2015, clearer relation between spectral features in this band and behavior
bandwidth = (3700, 12900) # 1999, 3700 may include spectral variance due to low freq artifacts



"""  --- compare spectral features left and right ear  ---  """

main_df = create_df.add_l_r_comparison(main_df, bandwidth)

