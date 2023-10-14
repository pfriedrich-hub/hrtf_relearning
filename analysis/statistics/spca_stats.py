import numpy
import analysis.main_df
import analysis.statistics.stats_df as stats_df
from pathlib import Path
from matplotlib import pyplot as plt
path=Path.cwd() / 'data' / 'experiment' / 'master'
q=10
bandwidth = (4000, 16000)
main_df = analysis.main_df.main_dataframe(path, processed_hrtf=True)

main_df = stats_df.add_pca_coords(main_df, path, q, bandwidth)

# compare pca reconstructions of DTFs to ERB filtered to original:
main_df, pca = stats_df.add_pca_coords(main_df, path, q, bandwidth, return_pca=True)
idx = numpy.random.randint(0, len(main_df['subject']), 10)
conditions = ['EF', 'M1', 'M2']
basis_functions = pca.components_
for i in idx:
    condition = numpy.random.choice(conditions)
    elevation_idx = numpy.random.randint(0, 6)
    ear_idx = numpy.random.randint(0, 2)
    # binned HRTF
    DTF_erb_bin = main_df.iloc[i][condition + ' binned'][ear_idx][elevation_idx]
    DTF_erb_bin = 20 * numpy.log10(DTF_erb_bin)
    # original DTF
    DTF = main_df.iloc[i][condition + ' hrtf'].tfs_from_sources(sources=[elevation_idx],
            n_bins=None, ear='both')[0, :, ear_idx]
    dtf_freqs = main_df.iloc[i][condition + ' hrtf'][0].frequencies
    DTF = DTF[numpy.logical_and(dtf_freqs > bandwidth[0], dtf_freqs < bandwidth[1])]
    dtf_freqs = dtf_freqs[numpy.logical_and(dtf_freqs > bandwidth[0], dtf_freqs < bandwidth[1])]
    dtf_freq_bins = numpy.linspace(bandwidth[0], bandwidth[1], len(DTF_erb_bin))
    DTF_interp = numpy.interp(dtf_freq_bins, dtf_freqs, DTF)
    # reconstruction of DTF by basis functions and weights
    DTF_weights = main_df.iloc[i][condition + ' W'][ear_idx][elevation_idx]
    cc = []
    for j in range(10):
        cc.append(DTF_weights[i] * pca.components_[i])  # matches tf_orig
    cc = numpy.sum(cc, axis=0)
    plt.figure()
    plt.plot(DTF_interp)
    plt.plot(DTF_erb_bin)
    plt.plot(cc)
