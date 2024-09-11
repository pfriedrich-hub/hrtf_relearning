import old.MSc.analysis.processing.hrtf_processing as hrtf_processing
import old.MSc.analysis.hrtf_analysis as hrtf_analysis
import numpy
from pathlib import Path
path = Path.cwd() / 'data' / 'experiment' / 'master'

def add_hrtf_stats(main_df, bandwidth, vsi_dis_bw=None):  # , bands=None):
    if not vsi_dis_bw:
        vsi_dis_bw = bandwidth
    main_df['EF VSI'] = ''
    main_df['EF spectral strength'] = ''
    main_df['EF VSI r'] = ''
    main_df['EF VSI l'] = ''
    main_df['EF spectral strength r'] = ''
    main_df['EF spectral strength l'] = ''

    main_df['M1 VSI'] = ''
    main_df['M1 spectral strength'] = ''
    main_df['M1 VSI r'] = ''
    main_df['M1 VSI l'] = ''
    main_df['M1 spectral strength r'] = ''
    main_df['M1 spectral strength l'] = ''

    main_df['M2 VSI'] = ''
    main_df['M2 spectral strength'] = ''
    main_df['M2 VSI r'] = ''
    main_df['M2 VSI l'] = ''
    main_df['M2 spectral strength r'] = ''
    main_df['M2 spectral strength l'] = ''

    main_df['EF M1 VSI dissimilarity'] = ''
    main_df['EF M1 spectral difference'] = ''
    main_df['EF M1 weighted VSI dissimilarity'] = ''
    main_df['EF M1 VSI dissimilarity r'] = ''
    main_df['EF M1 VSI dissimilarity l'] = ''
    main_df['EF M1 spectral difference r'] = ''
    main_df['EF M1 spectral difference l'] = ''

    main_df['EF M2 VSI dissimilarity'] = ''
    main_df['EF M2 spectral difference'] = ''
    main_df['EF M2 weighted VSI dissimilarity'] = ''
    main_df['EF M2 VSI dissimilarity r'] = ''
    main_df['EF M2 VSI dissimilarity l'] = ''
    main_df['EF M2 spectral difference r'] = ''
    main_df['EF M2 spectral difference l'] = ''

    main_df['M1 M2 VSI dissimilarity'] = ''
    main_df['M1 M2 weighted VSI dissimilarity'] = ''
    main_df['M1 M2 spectral difference'] = ''
    main_df['M1 M2 spectral difference r'] = ''
    main_df['M1 M2 spectral difference l'] = ''
    main_df['M1 M2 VSI dissimilarity r'] = ''
    main_df['M1 M2 VSI dissimilarity l'] = ''

    for subject_id, row in main_df.iterrows():
        # hrtf_stats.loc[subject_id]['EFD0'] = hrtf_stats.loc[subject_id]['EFD0'][measure_idx]
        hrtf_ef = main_df.iloc[subject_id]['EF hrtf']
        hrtf_m1 = main_df.iloc[subject_id]['M1 hrtf']
        hrtf_m2 = main_df.iloc[subject_id]['M2 hrtf']
        main_df.loc[subject_id]['EF VSI'] = hrtf_analysis.vsi(hrtf_ef, bandwidth, ear_idx=[0, 1], average=True)
        main_df.loc[subject_id]['EF spectral strength'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth)

        main_df.loc[subject_id]['EF VSI l'] = hrtf_analysis.vsi(hrtf_ef, bandwidth, ear_idx=[0])
        main_df.loc[subject_id]['EF VSI r'] = hrtf_analysis.vsi(hrtf_ef, bandwidth, ear_idx=[1])
        main_df.loc[subject_id]['EF spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth,
                                                                                            ear='right')
        main_df.loc[subject_id]['EF spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_ef, bandwidth,
                                                                                            ear='left')
        if hrtf_m1:
            main_df.loc[subject_id]['EF M1 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1, vsi_dis_bw)
            main_df.loc[subject_id]['EF M1 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, vsi_dis_bw)
            # main_df.loc[subject_id]['EF M1 weighted VSI dissimilarity'] = \
            #     hrtf.weighted_vsi_dissimilarity(hrtf_ef, hrtf_m1, bands)
            main_df.loc[subject_id]['M1 VSI'] = hrtf_analysis.vsi(hrtf_m1, bandwidth, ear_idx=[0, 1], average=True)
            main_df.loc[subject_id]['M1 spectral strength'] = hrtf_analysis.spectral_strength(hrtf_m1, bandwidth)

            main_df.loc[subject_id]['M1 VSI l'] = hrtf_analysis.vsi(hrtf_m1, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['M1 VSI r'] = hrtf_analysis.vsi(hrtf_m1, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['M1 spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_m1, bandwidth, ear='right')
            main_df.loc[subject_id]['M1 spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_m1, bandwidth, ear='left')
            main_df.loc[subject_id]['EF M1 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1,
                                                                                                   vsi_dis_bw,
                                                                                                   ear_idx=[0])
            main_df.loc[subject_id]['EF M1 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m1,
                                                                                                   vsi_dis_bw,
                                                                                                   ear_idx=[1])
            main_df.loc[subject_id]['EF M1 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='left')
            main_df.loc[subject_id]['EF M1 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='right')

        if hrtf_m2:

            main_df.loc[subject_id]['M2 VSI'] = hrtf_analysis.vsi(hrtf_m2, bandwidth, ear_idx=[0, 1], average=True)
            main_df.loc[subject_id]['M2 spectral strength'] = hrtf_analysis.spectral_strength(hrtf_m2, bandwidth)
            main_df.loc[subject_id]['M2 VSI l'] = hrtf_analysis.vsi(hrtf_m2, bandwidth, ear_idx=[0])
            main_df.loc[subject_id]['M2 VSI r'] = hrtf_analysis.vsi(hrtf_m2, bandwidth, ear_idx=[1])
            main_df.loc[subject_id]['M2 spectral strength r'] = hrtf_analysis.spectral_strength(hrtf_m2, bandwidth,
                                                                                                ear='right')
            main_df.loc[subject_id]['M2 spectral strength l'] = hrtf_analysis.spectral_strength(hrtf_m2, bandwidth,
                                                                                                ear='left')

            main_df.loc[subject_id]['EF M2 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2, vsi_dis_bw)
            main_df.loc[subject_id]['EF M2 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2, vsi_dis_bw)
            # main_df.loc[subject_id]['EF M2 weighted VSI dissimilarity'] = \
            #     hrtf.weighted_vsi_dissimilarity(hrtf_ef, hrtf_m2, bands)
            main_df.loc[subject_id]['EF M2 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2,
                                                                                                   vsi_dis_bw,
                                                                                                   ear_idx=[0])
            main_df.loc[subject_id]['EF M2 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_ef, hrtf_m2,
                                                                                                   vsi_dis_bw,
                                                                                                   ear_idx=[1])
            main_df.loc[subject_id]['EF M2 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='left')
            main_df.loc[subject_id]['EF M2 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m2,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='right')

        if (hrtf_m2 and hrtf_m1):
            main_df.loc[subject_id]['M1 M2 VSI dissimilarity'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2, vsi_dis_bw)
            main_df.loc[subject_id]['M1 M2 spectral difference'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2, vsi_dis_bw)
            # main_df.loc[subject_id]['M1 M2 weighted VSI dissimilarity'] = \
            #     hrtf.weighted_vsi_dissimilarity(hrtf_m1, hrtf_m2, bands)

            main_df.loc[subject_id]['M1 M2 VSI dissimilarity l'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2,
                                                                                                   vsi_dis_bw, ear_idx=[0])
            main_df.loc[subject_id]['M1 M2 VSI dissimilarity r'] = hrtf_analysis.vsi_dissimilarity(hrtf_m1, hrtf_m2,
                                                                                                   vsi_dis_bw, ear_idx=[1])
            main_df.loc[subject_id]['M1 M2 spectral difference l'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='left')
            main_df.loc[subject_id]['M1 M2 spectral difference r'] = hrtf_analysis.spectral_difference(hrtf_m1, hrtf_m2,
                                                                                                       vsi_dis_bw,
                                                                                                       ear='right')
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

def spectral_change_p(main_df, threshold=None, bandwidth=(4000,16000)):
    spectral_changes_efm1 = []
    spectral_changes_efm2 = []
    spectral_changes_m1m2 = []
    thresholds = []
    m1_sub = 0
    m2_sub = 0
    m1m2_sub = 0
    for subject_id, row in main_df.iterrows():
        # vsi / spectral strength left and right free and molds
        hrtf_ef = main_df.iloc[subject_id]['EF hrtf']
        hrtf_m1 = main_df.iloc[subject_id]['M1 hrtf']
        hrtf_m2 = main_df.iloc[subject_id]['M2 hrtf']
        if not threshold:
            # get mean of RMS differences across all combinations of DTFs measured with free ears (Trapeau, Schönwiesner 2015)
            n_sources = hrtf_ef.n_sources
            diff = numpy.zeros((2, n_sources, n_sources))
            for i in range(n_sources):  # decreasing elevation
                for j in range(n_sources):  # increasing elevation
                    wi, hi = hrtf_ef[i].tf(show=False)
                    _, hj = hrtf_ef[j].tf(show=False)
                    hi = hi[numpy.logical_and(wi > bandwidth[0], wi < bandwidth[1])]
                    hj = hj[numpy.logical_and(wi > bandwidth[0], wi < bandwidth[1])]
                    diff[:, i, j] = numpy.sqrt(numpy.mean((hi - hj) ** 2))
                    # hi = numpy.sqrt(numpy.mean(hi**2))
                    # hj = numpy.sqrt(numpy.mean(hj**2))
                    # diff[:, i, j] = numpy.abs(hi-hj)
            # diff = numpy.mean(diff, axis=0)
            # mask = numpy.ones(diff.shape, dtype=bool)  # remove main diagonal - doesnt help much
            # mask[numpy.diag_indices(7)] = False
            # diff = diff[mask]
            thresh = numpy.mean(diff)
            thresholds.append(thresh)
        else:
            thresh = threshold
        if hrtf_m1:
            m1_sub += 1
            ef_m1_dif = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1)
            data = ef_m1_dif.tfs_from_sources(ef_m1_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_efm1.append(numpy.abs(data) > thresh)
        if hrtf_m2:
            m2_sub += 1
            ef_m2_dif = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m2)
            data = ef_m2_dif.tfs_from_sources(ef_m2_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_efm2.append(numpy.abs(data) > thresh)
        if (hrtf_m1 and hrtf_m2):
            m1m2_sub += 1
            m1_m2_dif = hrtf_analysis.hrtf_difference(hrtf_m1, hrtf_m2)
            data = m1_m2_dif.tfs_from_sources(m1_m2_dif.cone_sources(0), ear='both', n_bins=None)
            spectral_changes_m1m2.append(numpy.abs(data) > thresh)
    efm1_p = numpy.mean(numpy.sum(numpy.array(spectral_changes_efm1), axis=0) / m1_sub, axis=2)
    efm2_p = numpy.mean(numpy.sum(spectral_changes_efm2, axis=0) / m2_sub, axis=2)
    m1m2_p = numpy.mean(numpy.sum(spectral_changes_m1m2, axis=0) / m1m2_sub, axis=2)
    if not threshold:
        return numpy.array((efm1_p, efm2_p, m1m2_p)), thresholds
    else:
        return numpy.array((efm1_p, efm2_p, m1m2_p)), None


def add_pca_coords(main_df, path, q=10, bandwidth=(4000, 16000), return_components=False):
    hrtf_df = hrtf_processing.get_hrtf_df(path, processed=False)
    hrtf_df, components = hrtf_analysis.hrtf_pca_space(hrtf_df, q, bandwidth)
    main_df['EF PCW'] = ''
    main_df['M1 PCW'] = ''
    main_df['M2 PCW'] = ''
    if return_components:
        main_df['EF binned'] = ''
        main_df['M1 binned'] = ''
        main_df['M2 binned'] = ''
    for subject in main_df['subject']:
        subject_data = hrtf_df[hrtf_df['subject']==subject]
        if main_df[main_df['subject']==subject]['EF hrtf'].values:
            main_df['EF PCW'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Ears Free']['pc weights'].values
            if return_components:
                ef_binned = (subject_data[hrtf_df['condition'] == 'Ears Free']['hrtf binned'].values)
                main_df['EF binned'][main_df['subject'] == subject] = ef_binned
        if main_df[main_df['subject'] == subject]['M1 hrtf'].values:
            main_df['M1 PCW'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Earmolds Week 1']['pc weights'].values
            if return_components:
                m1_binned = subject_data[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf binned'].values
                main_df['M1 binned'][main_df['subject'] == subject] = m1_binned
        if main_df[main_df['subject'] == subject]['M2 hrtf'].values:
            main_df['M2 PCW'][main_df['subject'] == subject]\
                = subject_data[hrtf_df['condition']=='Earmolds Week 2']['pc weights'].values
            if return_components:
                m2_binned = subject_data[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf binned'].values
                main_df['M2 binned'][main_df['subject'] == subject] = m2_binned
    if return_components:
        return main_df, components
    return main_df

def add_pca_stats(main_df, path, q=10, bandwidth=(4000, 16000)):
    main_df = add_pca_coords(main_df, path, q, bandwidth)
    main_df['EF M1 PCW dist'] = ''
    main_df['EF M2 PCW dist'] = ''
    main_df['M1 M2 PCW dist'] = ''
    for subject_id, row in main_df.iterrows():
        ef_weights, m1_weights, m2_weights = row['EF PCW'], row['M1 PCW'], row['M2 PCW']
        # try ear shape difference quantification by pca space distance
        # idea I: difference quantified by mean euclidean distance between DTF pairs of same direction
        try:
            efm1_l_pcdist = numpy.mean([numpy.linalg.norm(ef_weights[0][dtf] - m1_weights[0][dtf]) for dtf in range(7)])
            efm1_r_pcdist = numpy.mean([numpy.linalg.norm(ef_weights[1][dtf] - m1_weights[1][dtf]) for dtf in range(7)])
            efm1_pcdist = (efm1_l_pcdist + efm1_r_pcdist) / 2
        except IndexError:
            efm1_pcdist = None
        try:
            efm2_l_pcdist = numpy.mean([numpy.linalg.norm(ef_weights[0][dtf] - m2_weights[0][dtf]) for dtf in range(7)])
            efm2_r_pcdist = numpy.mean([numpy.linalg.norm(ef_weights[1][dtf] - m2_weights[1][dtf]) for dtf in range(7)])
            efm2_pcdist = (efm2_l_pcdist + efm2_r_pcdist) / 2
        except IndexError:
            efm2_pcdist = None
        try:
            m1m2_l_pcdist = numpy.mean([numpy.linalg.norm(m1_weights[0][dtf] - m2_weights[0][dtf]) for dtf in range(7)])
            m1m2_r_pcdist = numpy.mean([numpy.linalg.norm(m1_weights[1][dtf] - m2_weights[1][dtf]) for dtf in range(7)])
            m1m2_pcdist = (m1m2_l_pcdist + m1m2_r_pcdist) / 2
        except IndexError:
            m1m2_pcdist = None
        row['EF M1 PCW dist'] = efm1_pcdist
        row['EF M2 PCW dist'] = efm2_pcdist
        row['M1 M2 PCW dist'] = m1m2_pcdist
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