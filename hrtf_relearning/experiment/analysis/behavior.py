from hrtf_relearning.experiment.Subject import Subject
from analysis.localization import *
from pathlib import Path
data_dir = Path.cwd() / 'data'
id = 'pf_test'
subject = Subject(id)

sequence = subject.localization['rws_KU100_HRIR_L2702_loc_01.10.09:16']
plot_localization(sequence, report_stats=['elevation', 'azimuth'], filepath=data_dir / 'results' / 'plot' / subject.id)

