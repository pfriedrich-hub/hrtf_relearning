import scipy
import slab
from pathlib import Path
data_dir = Path.cwd() / 'data'
import pandas as pd
from matplotlib import pyplot as plt

subj_id = '001'
sequence = slab.Trialsequence(conditions=47, n_reps=1)
sequence.load_pickle(file_name=data_dir/data_dir / 'localization_data' / subj_id)
[data] = sequence.data

data = pd.DataFrame(data)
data['pose']


scipy.stats.linregress(x,y)