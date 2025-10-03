from experiment.Subject import Subject
from analysis.localization import *

id = 'rws'
subject = Subject(id)  #todo make this work on MacOS

sequence = subject.localization['rws_KU100_HRIR_L2702_loc_01.10.09:16']
plot_localization(sequence, report_stats=['elevation', 'azimuth'], filepath=None)

