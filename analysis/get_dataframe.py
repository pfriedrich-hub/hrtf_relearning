import analysis.localization_analysis as loc_analysis
import analysis.processing.hrtf_processing as hrtf_processing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import numpy
from pathlib import Path

def main_dataframe(path=Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True):
    subjects = list([path.name for path in path.iterdir()])
    main_df = pandas.DataFrame({'subject': subjects})
    # primary data
    main_df = add_localization_data(main_df, path)
    main_df = add_hrtf_data(main_df, processed_hrtf, path)
    # statistics
    # bandwidth=(4000, 16000)
    # main_df = stats_df.add_hrtf_stats(main_df, bandwidth)
    # main_df = stats_df.add_l_r_comparison(main_df, bandwidth)
    # main_df, _ = stats_df.add_pca_coords(main_df, path, q=10, bandwidth=bandwidth)
    # main_df.to_csv('/Users/paulfriedrich/Desktop/hrtf_relearning/data/main_df.csv')
    return main_df

def add_localization_data(main_df, path):
    localization_df = loc_analysis.get_localization_dataframe(path)
    main_df['EFD0'], main_df['M1D0'], main_df['M1D5'], main_df['M1 drop'], main_df['M1 gain'], \
    main_df['EFD5'], main_df['M2D0'], main_df['M2D5'], main_df['M2 drop'], main_df['M2 gain'], \
    main_df['M1M2 drop'], main_df['M1M2 gain'], main_df['EF avg'], main_df['EF USO dif'], main_df['M1 USO dif'],\
    main_df['M2 USO dif'] = '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
    for sub_id, row in main_df.iterrows():
        subj_loc = localization_df[localization_df['subject'] == row['subject']]
        efd0 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 0][
            subj_loc.columns[6:]].values[0]
        efd5 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 5][
            subj_loc.columns[6:]].values[0]
        efd10 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['day'] == 10][
            subj_loc.columns[6:]].values[0]
        try:
            efd10uso = subj_loc[subj_loc['condition'] == 'USO Ears Free'][subj_loc['day'] == 10][
                subj_loc.columns[6:]].values[0]
            efusodif = efd10uso - efd10
        except (ValueError, IndexError):
            efusodif = [numpy.nan] * 5
        efavg = numpy.nanmean((efd0, efd5, efd10), axis=0)
        row['EFD0'] = efd0
        row['EFD5'] = efd5
        # row['EFD10'] = efd10
        row['EF avg'] = efavg
        row['EF USO dif'] = efusodif
        try:
            m1d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m1d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            d0drop = m1d0 - efd0
            d5gain = m1d5 - efd0
        except (ValueError, IndexError):
            m1d0 = m1d5 = d0drop = d5gain = m1d5uso = [numpy.nan] * 5
        try:
            m1d5uso = subj_loc[subj_loc['condition'] == 'USO Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            m1usodif = m1d5uso - m1d5
        except (ValueError, IndexError):
            m1usodif = [numpy.nan] * 5
        row['M1D0'] = m1d0
        row['M1D5'] = m1d5
        row['M1 drop'] = d0drop
        row['M1 gain'] = d5gain
        row['M1 USO dif'] = m1usodif
        try:
            m2d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m2d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            d5drop = m2d0 - efd5
            d10gain = m2d5 - efd5
        except (ValueError, IndexError):
            m2d0 = m2d5 = d5drop = d10gain = [numpy.nan] * 5
        try:
            m2d5uso = subj_loc[subj_loc['condition'] == 'USO Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            m2usodif = m2d5uso - m2d5
        except (ValueError, IndexError):
            m2usodif = [numpy.nan] * 5
        try:
            d5m1m2drop = m2d0 - m1d5
            d5m1m2gain = m2d5 - m1d5
        except (ValueError, IndexError):
            d5m1m2drop = d5m1m2gain = [numpy.nan] * 5
        row['M2D0'] = m2d0
        row['M2D5'] = m2d5
        row['M2 drop'] = d5drop
        row['M2 gain'] = d10gain
        row['M1M2 drop'] = d5m1m2drop
        row['M1M2 gain'] = d5m1m2gain
        row['M2 USO dif'] = m2usodif
    return main_df

def add_hrtf_data(main_df, processed, path):
    hrtf_df = hrtf_processing.get_hrtf_df(path, processed=False)
    if processed:
        hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', bandwidth=(4000, 16000),
                                                baseline=False, dfe=True, write=False)
    # get hrtfs and behavior data
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
