from old.MSc.analysis.plot import spectral_behavior_collection as stats_plot
import old.MSc.analysis.statistics.stats_df as stats_df
import old.MSc.analysis.build_dataframe as get_df
from matplotlib import pyplot as plt
from pathlib import Path

path=Path.cwd() / 'data' / 'experiment' / 'master'
main_df = get_df.main_dataframe(path, processed_hrtf=True)

q = 10  # n principal components
bandwidth = (4000, 16000)
# bandwidth = (5700, 8000)
# bandwidth = (5700, 11300) # 2015, clearer relation between spectral features in this band and behavior
# bandwidth = (3700, 12900) # 1999, 3700 may include spectral variance due to low freq artifacts

"""  --- ole_test pc space distance between conditions correlation with behavior ---  """

main_df = stats_df.add_pca_stats(main_df, path, q, bandwidth)

# m1 vs ears free
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.d0dr_pcw_dist(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d0dr_pcw_dist(main_df, 'EG', axis=axes[1, 0])
stats_plot.d5ga_pcw_dist(main_df, 'RMSE ele', axis=axes[0, 1])
stats_plot.d5ga_pcw_dist(main_df, 'EG', axis=axes[1, 1])
axes[0, 0].set_xlabel('RMSE')
axes[1, 0].set_xlabel('Elevation Gain')
axes[0, 1].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('PCW distance')
axes[1, 0].set_ylabel('PCW distance')
fig.text(.22, .92, 'Ears Free / M1 difference day 0', fontsize=12)
fig.text(.61, .92, 'Ears Free / M1 difference day 5', fontsize=12)

for axis in axes:
    for ax in axis:
        ax.set_ylim(0, .2)

# m2 vs ears free
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.d5dr_pcw_dist(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d5dr_pcw_dist(main_df, 'EG', axis=axes[1, 0])
stats_plot.d10ga_pcw_dist(main_df, 'RMSE ele', axis=axes[0, 1])
stats_plot.d10ga_pcw_dist(main_df, 'EG', axis=axes[1, 1])
axes[0, 0].set_xlabel('RMSE')
axes[1, 0].set_xlabel('Elevation Gain')
axes[0, 1].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('PCW distance')
axes[1, 0].set_ylabel('PCW distance')
fig.text(.22, .92, 'Ears Free / M2 difference day 0', fontsize=12)
fig.text(.61, .92, 'Ears Free / M2 difference day 5', fontsize=12)


# m1m2 pca distance vs m1m2 localization accuracy
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.d5dr_pcw_dist_m1m2(main_df, 'RMSE ele', axis=axes[0, 0])
stats_plot.d5dr_pcw_dist_m1m2(main_df, 'EG', axis=axes[1, 0])
stats_plot.d10ga_pcw_dist_m1m2(main_df, 'RMSE ele', axis=axes[0, 1])
stats_plot.d10ga_pcw_dist_m1m2(main_df, 'EG', axis=axes[1, 1])
axes[0, 0].set_xlabel('RMSE')
axes[1, 0].set_xlabel('Elevation Gain')
axes[0, 1].set_xlabel('RMSE')
axes[1, 1].set_xlabel('Elevation Gain')
axes[0, 0].set_ylabel('PCW distance')
axes[1, 0].set_ylabel('PCW distance')
fig.text(.22, .92, 'M1/M2 drop vs M1/M2 difference', fontsize=12)
fig.text(.61, .92, 'M1/M2 gain vs M1/M2 difference', fontsize=12)



# todo next steps:
#  I get weighted difference of pc weights across DTFs of each HRTF to quantify its vertical spectral information
#  (depending on amount of explained vars by corresponding components)
#  and see if EF weighted diff and EF performance correlate
#  II get center of mass in the pca space (mean(x1, x2, .. , x6), mean(y1, y2, .. ,y6), ... q dimensions
#  compute euclidean distances between EFM1 EFM2 and most importantly M1M2 and see if correlates with behavior (drop, gain etc)
#  III extra: see if centering DTFs in PCA improves correlation with behavior - maybe implement for hrtf processing
#  check if 1st component changes significantly with centered dtfs to see whether centering increases information in pca space

"""
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
        # weights are stored within subject as Ears x Elevations x Weights 
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

"""