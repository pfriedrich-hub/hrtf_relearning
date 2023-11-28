import analysis.processing.hrtf_processing as hrtf_processing
from pathlib import Path
data_path = path=Path.cwd() / 'data' / 'experiment'
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']

# process all hrtfs with specified parameters
hrtf_df = hrtf_processing.get_hrtf_df(path=data_path / 'master', processed=False)
hrtf_df = hrtf_processing.process_hrtfs(hrtf_df, filter='erb', bandwidth=(4000, 16000),
                                        baseline=False, dfe=True, write=True)


# create average across DTFs
import slab
import numpy
from pathlib import Path
path=Path.cwd() / 'data' / 'experiment'
kemar = slab.HRTF(path / 'kemar_kemar_360_19.10.sofa')
for i in range(0, kemar.n_sources, 10):
    # kemar.plot_tf(numpy.arange(i, i+10), ear='left')  # some are noisy, low battery ? check again
    kemar.plot_tf(numpy.arange(i, i+10), ear='left')
mean_tf = []
for tf in kemar.data:
    mean_tf.append([tf.data[:, 0], tf.data[:, 1]])

mean_tf = numpy.mean(mean_tf, axis=0)
mean_tf = numpy.mean(mean_tf, axis=0)
mean_tf = slab.Filter(mean_tf, fir=False, samplerate=kemar.samplerate)

mean_tf.tf()  # looks good

fs = mean_tf.samplerate
data = mean_tf.data
freqs = mean_tf.frequencies

t_fs = hrtf_df.iloc[0]['hrtf'][0].samplerate
t_freqs = hrtf_df.iloc[0]['hrtf'][0].frequencies

data_interp = numpy.interp(t_freqs, freqs, data[:, 0])
t_mean_tf = slab.Filter(data=data_interp, samplerate=t_fs, fir=False)

t_mean_tf.save(path / 'kemar_dfa.npy')


# compare with participants average
dfa_kemar = slab.Filter.load(Path.cwd() / 'data' / 'experiment' / 'kemar_dfa.npy')
dfa_subj = slab.Filter.load(Path.cwd() / 'data' / 'experiment' / 'average_TF.npy')




