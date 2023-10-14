import analysis.hrtf_analysis as hrtf_analysis
import analysis.processing.hrtf_processing as hrtf_processing
import analysis.hrtf_analysis as hrtf_analysis
import numpy

def add_pca_coords(main_df, path, q=10, bandwidth=(4000, 16000), return_pca=False):
    hrtf_df = hrtf_processing.get_hrtf_df(path, processed=False)
    hrtf_df, pca = hrtf_analysis.hrtf_pca_space(hrtf_df, q, bandwidth)
    main_df['EF W'] = ''
    main_df['M1 W'] = ''
    main_df['M2 W'] = ''
    if return_pca:
        main_df['EF binned'] = ''
        main_df['M1 binned'] = ''
        main_df['M2 binned'] = ''
    for subject in main_df['subject']:
        subject_data = hrtf_df[hrtf_df['subject']==subject]
        if main_df[main_df['subject']==subject]['EF hrtf'].values:
            main_df['EF W'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Ears Free']['pc weights'].values
            if return_pca:
                ef_binned = (subject_data[hrtf_df['condition'] == 'Ears Free']['hrtf binned'].values)
                main_df['EF binned'][main_df['subject'] == subject] = ef_binned
        if main_df[main_df['subject'] == subject]['M1 hrtf'].values:
            main_df['M1 W'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Earmolds Week 1']['pc weights'].values
            if return_pca:
                m1_binned = subject_data[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf binned'].values
                main_df['M1 binned'][main_df['subject'] == subject] = m1_binned
        if main_df[main_df['subject'] == subject]['M2 hrtf'].values:
            main_df['M2 W'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Earmolds Week 2']['pc weights'].values
            if return_pca:
                m2_binned = subject_data[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf binned'].values
                main_df['M2 binned'][main_df['subject'] == subject] = m2_binned
    if return_pca:
        return main_df, pca
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

def ef_vsi_dis_perm(main_df, bandwidth):
    subj_ids = numpy.arange(0, len(main_df['subject']))
    comparison_idx = [(a, b) for idx, a in enumerate(subj_ids) for b in subj_ids[idx + 1:]]
    vsi_dis = []
    for comparison_id in comparison_idx:
        hrtf_ef_1 = main_df.iloc[comparison_id[0]]['EF hrtf']
        hrtf_ef_2 = main_df.iloc[comparison_id[1]]['EF hrtf']
        vsi_dis_l = hrtf_analysis.vsi_dissimilarity(hrtf_ef_1, hrtf_ef_2, bandwidth, ear_idx=[0])
        vsi_dis_r = hrtf_analysis.vsi_dissimilarity(hrtf_ef_1, hrtf_ef_2, bandwidth, ear_idx=[1])
        vsi_dis.append([vsi_dis_l, vsi_dis_r])
    return numpy.array(vsi_dis)

def ef_sp_dif_perm(main_df, bandwidth):
    subj_ids = numpy.arange(0, len(main_df['subject']))
    comparison_idx = [(a, b) for idx, a in enumerate(subj_ids) for b in subj_ids[idx + 1:]]
    sp_dif = []
    for comparison_id in comparison_idx:
        hrtf_ef_1 = main_df.iloc[comparison_id[0]]['EF hrtf']
        hrtf_ef_2 = main_df.iloc[comparison_id[1]]['EF hrtf']
        vsi_dis_l = hrtf_analysis.spectral_difference(hrtf_ef_1, hrtf_ef_2, bandwidth, ear='left')
        vsi_dis_r = hrtf_analysis.spectral_difference(hrtf_ef_1, hrtf_ef_2, bandwidth, ear='right')
        sp_dif.append([vsi_dis_l, vsi_dis_r])
    return numpy.array(sp_dif)

def spectral_change_p(main_df, threshold=5):
    spectral_changes_efm1 = []
    spectral_changes_efm2 = []
    spectral_changes_m1m2 = []
    m1_sub = 0
    m2_sub = 0
    m1m2_sub = 0
    for subject_id, row in main_df.iterrows():
        # vsi / spectral strength left and right free and molds
        hrtf_ef = main_df.iloc[subject_id]['EF hrtf']
        hrtf_m1 = main_df.iloc[subject_id]['M1 hrtf']
        hrtf_m2 = main_df.iloc[subject_id]['M2 hrtf']
        if hrtf_m1:
            m1_sub += 1
            ef_m1_dif = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1)
            data = ef_m1_dif.tfs_from_sources(ef_m1_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_efm1.append(numpy.abs(data) > threshold)
        if hrtf_m2:
            m2_sub += 1
            ef_m2_dif = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m2)
            data = ef_m2_dif.tfs_from_sources(ef_m2_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_efm2.append(numpy.abs(data) > threshold)
        if (hrtf_m1 and hrtf_m2):
            m1m2_sub += 1
            m1_m2_dif = hrtf_analysis.hrtf_difference(hrtf_m1, hrtf_m2)
            data = m1_m2_dif.tfs_from_sources(m1_m2_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_m1m2.append(numpy.abs(data) > threshold)
    efm1_p = numpy.mean(numpy.sum(numpy.array(spectral_changes_efm1), axis=0) / m1_sub, axis=2)
    efm2_p = numpy.mean(numpy.sum(spectral_changes_efm2, axis=0) / m2_sub, axis=2)
    m1m2_p = numpy.mean(numpy.sum(spectral_changes_m1m2, axis=0) / m1m2_sub, axis=2)
    return numpy.array((efm1_p, efm2_p, m1m2_p))

