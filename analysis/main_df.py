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
    main_df['EFD5'], main_df['M2D0'], main_df['M2D5'], main_df['M2 drop'], main_df['M2 gain'] \
        = '', '', '', '', '', '', '', '', '', ''
    for subject in main_df['subject']:
        subj_loc = localization_df[localization_df['subject'] == subject]
        efd0 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['adaptation day'] == 0][
            subj_loc.columns[6:]].values
        efd5 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['adaptation day'] == 1][
            subj_loc.columns[6:]].values
        main_df['EFD0'][main_df['subject'] == subject] = efd0
        main_df['EFD5'][main_df['subject'] == subject] = efd5
        try:
            m1d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values
            m1d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values
            d0drop = m1d0 - efd0
            d5gain = m1d5 - efd0
        except (ValueError, IndexError):
            m1d0 = m1d5 = d0drop = d5gain = [numpy.nan] * 5
        main_df['M1D0'][main_df['subject'] == subject] = m1d0
        main_df['M1D5'][main_df['subject'] == subject] = m1d5
        main_df['M1 drop'][main_df['subject'] == subject] = d0drop
        main_df['M1 gain'][main_df['subject'] == subject] = d5gain
        try:
            m2d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values
            m2d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values
            d5drop = m2d0 - efd5
            d10gain = m2d5 - efd5
        except (ValueError, IndexError):
            m2d0 = m2d5 = d5drop = d10gain = [numpy.nan] * 5
        main_df['M2D0'][main_df['subject'] == subject] = m2d0
        main_df['M2D5'][main_df['subject'] == subject] = m2d5
        main_df['M2 drop'][main_df['subject'] == subject] = d5drop
        main_df['M2 gain'][main_df['subject'] == subject] = d10gain
    return main_df

def add_hrtf_data(main_df, processed, path):
    hrtf_df = hrtf_processing.get_hrtf_df(path, processed=processed)
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
