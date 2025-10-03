import old.MSc.analysis.processing.hrtf_processing as hrtf_processing
import old.MSc.analysis.build_dataframe as build_df
from pathlib import Path

path = Path.cwd() / 'data' / 'experiment' / 'master'

""" HRTF processing """
main_df = build_df.get_subject_df(path)
hrtf_df = build_df.get_hrtf_df(path=path, processed=False)
hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', bandwidth=(500, 16000),
                                        baseline=True, dfe=True, write=True)
main_df = build_df.add_hrtf_data(main_df, hrtf_df)

"""
For DTF processing: see hrtf_processing.py,
for acoustic metrics used in the thesis: see hrtf.py,
for behavioral metrics used in the analyses: see localization.py.
"""