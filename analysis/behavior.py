from experiment.Subject import Subject
from analysis.localization import *
from pathlib import Path
data_dir = Path.cwd() / 'data'
id = 'PF'
subject = Subject(id)  #todo make this work on MacOS

sequence = subject.localization['PF_single_notch_loc_03.10_15.44']
plot_localization(sequence, report_stats=['elevation', 'azimuth'], filepath=data_dir / 'results' / 'plot' / subject.id)

