import legacy.MSc.analysis.localization_analysis as loc_analysis
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import numpy
import slab
from pathlib import Path

def main_dataframe(path=Path.cwd() / 'legacy' / 'MSc' /  'data' / 'experiment' / 'master', processed_hrtf=True):
    main_df = get_subject_df(path)  # get subject df
    # add localization data
    localization_df = get_localization_dataframe(path)
    main_df = add_localization_data(main_df, localization_df)
    # add hrtf data
    if processed_hrtf:
        hrtf_df = get_hrtf_df(path, processed=True)
        # hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', bandwidth=(4000, 16000),
        #                                         baseline=False, dfe=True, write=False)
    elif not processed_hrtf:
        hrtf_df = get_hrtf_df(path, processed=False)
    main_df = add_hrtf_data(main_df, hrtf_df)

    # statistics
    # bandwidth=(4000, 16000)
    # main_df = stats_df.add_hrtf_stats(main_df, bandwidth)
    # main_df = stats_df.add_l_r_comparison(main_df, bandwidth)
    # main_df, _ = stats_df.add_pca_coords(main_df, path, q=10, bandwidth=bandwidth)
    # main_df.to_csv('/Users/paulfriedrich/Desktop/hrtf_relearning/data/main_df.csv')
    return main_df

def get_subject_df(path):
    subjects = list([path.name for path in path.iterdir()])
    subject_df = pandas.DataFrame({'subject': subjects})
    return subject_df

def add_localization_data(main_df, localization_df):
    main_df['EFD0'], main_df['M1D0'], main_df['M1D5'], main_df['M1 drop'], main_df['M1 gain'], \
    main_df['EFD5'], main_df['M2D0'], main_df['M2D5'], main_df['M2 drop'], main_df['M2 gain'], \
    main_df['M1M2 drop'], main_df['M1M2 gain'], main_df['EF avg'], main_df['EF USO'], main_df['M1 USO'],\
    main_df['M2 USO'], main_df['EFD10'], main_df['M1D10'], main_df['M2D10'] = \
        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
    for sub_id, row in main_df.iterrows():
        subj_loc = localization_df[localization_df['subject'] == row['subject']]
        efd0 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 0][
            subj_loc.columns[6:]].values[0]
        efd5 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 5][
            subj_loc.columns[6:]].values[0]
        efd10 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 10][
            subj_loc.columns[6:]].values[0]
        try:
            efuso = subj_loc[subj_loc['condition'] == 'USO Ears Free'][subj_loc['day'] == 10][
                subj_loc.columns[6:]].values[0]
        except (ValueError, IndexError):
            efuso = numpy.array([numpy.nan] * 6)
        efavg = numpy.nanmean((efd0, efd5, efd10), axis=0)
        row['EFD0'] = efd0
        row['EFD5'] = efd5
        row['EFD10'] = efd10
        row['EF avg'] = efavg
        row['EF USO'] = efuso
        try:
            m1d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m1d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            m1d10 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 6][
                subj_loc.columns[6:]].values[0]
            d0drop = m1d0 - efd0
            d5gain = m1d5 - m1d0
        except (ValueError, IndexError):
            m1d0 = m1d5 = m1d10 = d0drop = d5gain = numpy.array([numpy.nan] * 6)
        try:
            m1uso = subj_loc[subj_loc['condition'] == 'USO Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
        except (ValueError, IndexError):
            m1uso = numpy.array([numpy.nan] * 6)
        row['M1D0'] = m1d0
        row['M1D5'] = m1d5
        row['M1D10'] = m1d10
        row['M1 drop'] = d0drop
        row['M1 gain'] = d5gain
        row['M1 USO'] = m1uso
        try:
            m2d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m2d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            m2d10 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 6][
                subj_loc.columns[6:]].values[0]
            d5drop = m2d0 - efd5
            d10gain = m2d5 - m2d0
        except (ValueError, IndexError):
            m2d0 = m2d5 = m2d10 = d5drop = d10gain = numpy.array([numpy.nan] * 6)
        try:
            m2uso = subj_loc[subj_loc['condition'] == 'USO Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
        except (ValueError, IndexError):
            m2uso = numpy.array([numpy.nan] * 6)
        try:
            d5m1m2drop = m2d0 - m1d5
            d5m1m2gain = m2d5 - m1d5
        except (ValueError, IndexError):
            d5m1m2drop = d5m1m2gain = numpy.array(numpy.array([numpy.nan] * 6))
        row['M2D0'] = m2d0
        row['M2D5'] = m2d5
        row['M2D10'] = m2d10
        row['M2 drop'] = d5drop
        row['M2 gain'] = d10gain
        row['M1M2 drop'] = d5m1m2drop
        row['M1M2 gain'] = d5m1m2gain
        row['M2 USO'] = m2uso
        main_df.loc[sub_id] = row
    return main_df

def add_hrtf_data(main_df, hrtf_df):
    # get hrtf and behavior data
    main_df['EF hrtf'], main_df['M1 hrtf'], main_df['M2 hrtf'] = '', '', ''
    for subject in main_df['subject']:
        # subject data
        subj_hrtfs = hrtf_df[hrtf_df['subject'] == subject]
        main_df['EF hrtf'][main_df['subject'] == subject]\
            = subj_hrtfs[subj_hrtfs['condition'] == 'Ears Free']['hrtf'].item()
        try:
            main_df['M1 hrtf'][main_df['subject'] == subject]\
                = subj_hrtfs[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf'].item()
        except (ValueError, IndexError):
            main_df['M1 hrtf'][main_df['subject'] == subject] = None
        try:
            main_df['M2 hrtf'][main_df['subject'] == subject]\
                = subj_hrtfs[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf'].item()
        except (ValueError, IndexError):
            main_df['M2 hrtf'][main_df['subject'] == subject] = None
    return main_df

def get_hrtf_df(path=Path.cwd() / 'data' / 'experiment' / 'master', processed=True, exclude=[]):
    subject_paths = list(path.iterdir())
    hrtf_df = pandas.DataFrame({'subject': [], 'filename': [], 'condition': [], 'hrtf': []})
    conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
    for subject_path in subject_paths:
        subject = subject_path.name
        if subject not in exclude:
            for condition in conditions:
                if processed:
                    condition_path = subject_path / condition / 'processed_hrtf'
                else:
                    condition_path = subject_path / condition
                if condition_path.exists():
                    for file_name in sorted(list(condition_path.iterdir())):
                        if file_name.is_file() and file_name.suffix == '.sofa':
                            hrtf = slab.HRTF(file_name)
                            new_row = [subject, file_name.name, condition, hrtf]
                            hrtf_df.loc[len(hrtf_df)] = new_row
    # hrtf_df.to_csv('/Users/paulfriedrich/projects/hrtf_relearning/data/experiment/data.csv')
    return hrtf_df

def get_localization_dataframe(path=Path.cwd() / 'data' / 'experiment' / 'master', w2_exclude=['cs', 'lm', 'lk']):
    localization_dict = loc_analysis.get_localization_dictionary(path=path)
    localization_data = pandas.DataFrame({'subject': [], 'filename': [], 'sequence': [], 'condition': [], 'day': [],
                                          'adaptation day': [], 'EG': [], 'RMSE ele': [], 'SD ele': [],
                                          'Az gain': [], 'RMSE az': [], 'SD az': []})
    subjects = list(localization_dict['Ears Free'].keys())
    test_order = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 1', 'Earmolds Week 1', 'Earmolds Week 1',
                  'Earmolds Week 1', 'Earmolds Week 1', 'Ears Free', 'Earmolds Week 2', 'Earmolds Week 2',
                  'Earmolds Week 2', 'Earmolds Week 2', 'Earmolds Week 2', 'Earmolds Week 2', 'Earmolds Week 1',
                  'Ears Free',  'Earmolds Week 2']
    total_days = [0, 0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10, 10, 10, 15]
    adaptation_days = [0, 0, 1, 2, 3, 4, 5, 1, 0, 1, 2, 3, 4, 5, 6, 2, 6]
    for subject in subjects:
        for idx, condition in enumerate(test_order):
            if not (subject in w2_exclude and condition) == 'Earmolds Week 2':
                file_name = list(localization_dict[condition][subject].keys())[adaptation_days[idx]]
                sequence = localization_dict[condition][subject][file_name]
                new_row = [subject, file_name, sequence, condition, total_days[idx], adaptation_days[idx]]
                new_row.extend(list(loc_analysis.localization_accuracy(sequence, show=False)))
                localization_data.loc[len(localization_data)] = new_row
                if adaptation_days[idx] == 5 or (adaptation_days[idx] == 2 and condition == 'Ears Free'): # add uso
                    file_name = [name for name in list(localization_dict[condition][subject].keys())
                                 if name.startswith('uso')]
                    if file_name:
                        sequence = localization_dict[condition][subject][file_name[0]]
                        new_row = [subject, file_name[0], sequence, 'USO ' + condition, total_days[idx], adaptation_days[idx]]
                        new_row.extend(list(loc_analysis.localization_accuracy(sequence, show=False)))
                        localization_data.loc[len(localization_data)] = new_row
    # localization_data.to_csv('/Users/paulfriedrich/projects/hrtf_relearning/data/experiment/data.csv')
    return localization_data
