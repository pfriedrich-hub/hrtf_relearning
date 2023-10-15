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

# todo next steps:
#  I get weighted difference of pc weights across DTFs of each HRTF to quantify its vertical spectral information
#  (depending on amount of explained vars by corresponding components)
#  and see if EF weighted diff and EF performance correlate
#  II get center of mass in the pca space (mean(x1, x2, .. , x6), mean(y1, y2, .. ,y6), ... q dimensions
#  compute euclidean distances between EFM1 EFM2 and most importantly M1M2 and see if correlates with behavior (drop, gain etc)



















# compare pca reconstructions of DTFs to ERB filtered to original:
main_df, components = stats_df.add_pca_coords(main_df, path, q, bandwidth, return_components=True)
idx = numpy.random.randint(0, len(main_df['subject']), 10)
conditions = ['EF', 'M1', 'M2']
for i in idx:
    condition = numpy.random.choice(conditions)
    elevation_idx = numpy.random.randint(0, 6)
    ear_idx = numpy.random.randint(0, 2)
    # binned HRTF
    try:
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
        # """ weights are stored within subject as Ears x Elevations x Weights """
        cc = []
        for j in range(10):
            cc.append(DTF_weights[j] * components[j])  # matches tf_orig
        cc = numpy.sum(cc, axis=0)
        plt.figure()
        plt.plot(DTF_interp)
        plt.plot(DTF_erb_bin)
        plt.plot(cc)
    except IndexError:
        continue