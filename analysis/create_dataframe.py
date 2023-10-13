import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import pandas
import numpy
from pathlib import Path

def main_dataframe(path=Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True):
    localization_dataframe = loc_analysis.get_localization_dataframe(path)
    hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=processed_hrtf)
    # get hrtfs and behavior data
    # main_df = pandas.DataFrame({'subject': [],
    #                                'EF hrtf': [], 'EFD0': [], 'EFD5': [],
    #                                'M1 hrtf': [], 'M1D0': [], 'M1D5': [],
    #                                'M1 drop': [], 'M1 gain': [],
    #                                'M2 hrtf': [], 'M2D0': [], 'M2D5': [],
    #                                'M2 drop': [], 'M2 gain': []})
    main_df = pandas.DataFrame({'subject': [],
                                   'EF hrtf': [], 'EFD0': [],
                                   'M1 hrtf': [],
                                   'M1 drop': [], 'M1 gain': [],
                                   'M2 hrtf': [],
                                   'M2 drop': [], 'M2 gain': []})
    for subject in hrtf_dataframe['subject'].unique():
        # subject data
        subj_hrtfs = hrtf_dataframe[hrtf_dataframe['subject'] == subject]
        subj_loc = localization_dataframe[localization_dataframe['subject'] == subject]
        # ears free
        ef = subj_hrtfs[subj_hrtfs['condition'] == 'Ears Free']['hrtf'].item()
        efd0 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['adaptation day'] == 0][
            subj_loc.columns[6:]].values[0]
        efd5 = subj_loc[subj_loc['condition'] == 'Ears Free'][subj_loc['adaptation day'] == 1][
            subj_loc.columns[6:]].values[0]
        # m1
        try:
            m1 = subj_hrtfs[hrtf_dataframe['condition'] == 'Earmolds Week 1']['hrtf'].item()
        except (ValueError, IndexError):
            m1 = None
        try:
            m1d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m1d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 1'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            d0drop = m1d0 - efd0
            d5gain = m1d5 - efd0
        except (ValueError, IndexError):
            m1d0 = m1d5 = d0drop = d5gain = [numpy.nan] * 5
        # m2
        try:
            m2 = subj_hrtfs[hrtf_dataframe['condition'] == 'Earmolds Week 2']['hrtf'].item()
        except (ValueError, IndexError):
            m2 = None
        try:
            m2d0 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 0][
                subj_loc.columns[6:]].values[0]
            m2d5 = subj_loc[subj_loc['condition'] == 'Earmolds Week 2'][subj_loc['adaptation day'] == 5][
                subj_loc.columns[6:]].values[0]
            d5drop = m2d0 - efd5
            d10gain = m2d5 - efd5
        except (ValueError, IndexError):
            m2d0 = m2d5 = d5drop = d10gain = [numpy.nan] * 5
        # new_row = [subject,
        #            ef, efd0, efd5,
        #            m1, m1d0, m1d5,
        #            d0drop, d5gain,
        #            m2, m2d0, m2d5,
        #            d5drop, d10gain]
        new_row = [subject,
                   ef, efd0,
                   m1,
                   d0drop, d5gain,
                   m2,
                   d5drop, d10gain]
        # main_df.to_csv('/Users/paulfriedrich/Desktop/hrtf_relearning/data/main_df.csv')
        main_df.loc[len(main_df)] = new_row
    return main_df

def add_hrtf_stats(main_df, bandwidth):
    main_df['EF VSI'] = ''
    main_df['EF spectral strength'] = ''
    main_df['EF M1 VSI dissimilarity'] = ''
    main_df['EF M1 spectral difference'] = ''
    main_df['EF M2 VSI dissimilarity'] = ''
    main_df['EF M2 spectral difference'] = ''
    main_df['M1 M2 VSI dissimilarity'] = ''
    main_df['M1 M2 spectral difference'] = ''
    for subject_id, row in main_df.iterrows():
        # hrtf_stats.loc[subject_id]['EFD0'] = hrtf_stats.loc[subject_id]['EFD0'][measure_idx]
        hrtf_ef = main_df.iloc[subject_id]['EF hrtf']
        hrtf_m1 = main_df.iloc[subject_id]['M1 hrtf']
        hrtf_m2 = main_df.iloc[subject_id]['M2 hrtf']
        main_df.loc[subject_id]['EF VSI'] = hrtf_analysis.vsi(hrtf_ef, bandwidth)
        main_df.loc[subject_id]['EF spectral strength'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth)
        if hrtf_m1:
            main_df.loc[subject_id]['EF M1 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1, bandwidth)
            main_df.loc[subject_id]['EF M1 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth)
        if hrtf_m2:
            main_df.loc[subject_id]['EF M2 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2, bandwidth)
            main_df.loc[subject_id]['EF M2 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2, bandwidth)
        if (hrtf_m2 and hrtf_m1):
            main_df.loc[subject_id]['M1 M2 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2, bandwidth)
            main_df.loc[subject_id]['M1 M2 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2, bandwidth)
    main_df = main_df.replace(r'^\s*$', None, regex=True)
    return main_df

def add_l_r_comparison(main_df, bandwidth):
    # vsi / spectral strength left and right free and molds
    main_df['EF VSI r'] = ''
    main_df['EF VSI l'] = ''
    main_df['M1 VSI r'] = ''
    main_df['M1 VSI l'] = ''
    main_df['M2 VSI r'] = ''
    main_df['M2 VSI l'] = ''
    main_df['EF spectral strength r'] = ''
    main_df['EF spectral strength l'] = ''
    main_df['M1 spectral strength r'] = ''
    main_df['M1 spectral strength l'] = ''
    main_df['M2 spectral strength r'] = ''
    main_df['M2 spectral strength l'] = ''
    # vsi dissimilarity / spectral difference left right m1 m2
    main_df['EF M1 VSI dissimilarity r'] = ''
    main_df['EF M1 VSI dissimilarity l'] = ''
    main_df['EF M2 VSI dissimilarity r'] = ''
    main_df['EF M2 VSI dissimilarity l'] = ''
    main_df['M1 M2 VSI dissimilarity r'] = ''
    main_df['M1 M2 VSI dissimilarity l'] = ''
    main_df['EF M1 spectral difference r'] = ''
    main_df['EF M1 spectral difference l'] = ''
    main_df['EF M2 spectral difference r'] = ''
    main_df['EF M2 spectral difference l'] = ''
    main_df['M1 M2 spectral difference r'] = ''
    main_df['M1 M2 spectral difference l'] = ''
    for subject_id, row in main_df.iterrows():
        # vsi / spectral strength left and right free and molds
        hrtf_ef = main_df.iloc[subject_id]['EF hrtf']
        hrtf_m1 = main_df.iloc[subject_id]['M1 hrtf']
        hrtf_m2 = main_df.iloc[subject_id]['M2 hrtf']
        main_df.loc[subject_id]['EF VSI l'] = hrtf_analysis.vsi(hrtf_ef, bandwidth, ear_idx=[0])
        main_df.loc[subject_id]['EF VSI r'] = hrtf_analysis.vsi(hrtf_ef, bandwidth, ear_idx=[1])
        main_df.loc[subject_id]['EF spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth, ear='right')
        main_df.loc[subject_id]['EF spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth, ear='left')
        if hrtf_m1:
            main_df.loc[subject_id]['M1 VSI l'] = hrtf_analysis.vsi(hrtf_m1, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['M1 VSI r'] = hrtf_analysis.vsi(hrtf_m1, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['M1 spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_m1, bandwidth, ear='right')
            main_df.loc[subject_id]['M1 spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_m1, bandwidth, ear='left')
        if hrtf_m2:
            main_df.loc[subject_id]['M2 VSI l'] = hrtf_analysis.vsi(hrtf_m2, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['M2 VSI r'] = hrtf_analysis.vsi(hrtf_m2, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['M2 spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_m2, bandwidth, ear='right')
            main_df.loc[subject_id]['M2 spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_m2, bandwidth, ear='left')
        # vsi dissimilarity / spectral difference left right m1 m2
        if hrtf_m1:
            main_df.loc[subject_id]['EF M1 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['EF M1 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['EF M1 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth, ear='left')
            main_df.loc[subject_id]['EF M1 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth, ear='right')
        if hrtf_m2:
            main_df.loc[subject_id]['EF M2 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['EF M2 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['EF M2 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2, bandwidth, ear='left')
            main_df.loc[subject_id]['EF M2 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2, bandwidth, ear='right')
        if (hrtf_m1 and hrtf_m2):
            main_df.loc[subject_id]['M1 M2 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['M1 M2 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['M1 M2 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2, bandwidth, ear='left')
            main_df.loc[subject_id]['M1 M2 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2, bandwidth, ear='right')
    main_df = main_df.replace(r'^\s*$', None, regex=True)
    return main_df