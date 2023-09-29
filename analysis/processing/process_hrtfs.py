import analysis.hrtf_analysis as hrtf_an
from pathlib import Path
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

# process all hrtfs with specified parameters
hrtf_df = hrtf_an.get_hrtf_df(path=data_path, processed=False)
hrtf_df = hrtf_an.process_hrtfs(hrtf_df, filter='scepstral', baseline=True, write=True)

