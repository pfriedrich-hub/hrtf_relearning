import slab
from pathlib import Path
import numpy
import copy
import scipy
import pandas
pandas.set_option('display.max_rows', 1000, 'display.max_columns', 1000, 'display.width', 1000)

# todo: fix problem of resolution downscaling with frequency for vsi:
#  for example kemar vsi with 0.1s looks better than for 0.05 seconds (higher frequencies have to
#  few samples here to be represented correctly in vsi) - temporary fix: increase samplerate to 96 kHz

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
                for file_name in sorted(list(condition_path.iterdir())):
                    if file_name.is_file() and file_name.suffix == '.sofa':
                        hrtf = slab.HRTF(file_name)
                        new_row = [subject, file_name.name, condition, hrtf]
                        hrtf_df.loc[len(hrtf_df)] = new_row
    # hrtf_df.to_csv('/Users/paulfriedrich/projects/hrtf_relearning/data/experiment/data.csv')
    return hrtf_df

# ----- HRTF processing ----- #
def process_hrtfs(hrtf_dataframe, filter='erb', bandwidth=(4000, 16000), baseline=True, dfe=False, write=False):
    path = Path.cwd() / 'data' / 'experiment' / 'master'
    for index, row in hrtf_dataframe.iterrows():
        if write:
            processed_path = Path(path / row['subject'] / row['condition'] / 'processed_hrtf')
            if not processed_path.exists():
                processed_path.mkdir()
        hrtf = copy.deepcopy(row['hrtf'])
        if dfe:
            # hrtf = hrtf.diffuse_field_equalization()  # not on the subject level
            dfa = slab.Filter.load(Path.cwd() / 'data' / 'experiment' / 'average_TF.npy')
            hrtf = hrtf.diffuse_field_equalization(dfa)  # divide by mean across all measured transfer functions (n=42)
        if filter == 'scepstral':
            hrtf = scepstral_filter_hrtf(hrtf, high_cutoff=1500)
        elif filter == 'erb':
            hrtf = erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=bandwidth[0], high_cutoff=bandwidth[1], bandwidth=0.0286,
                                   pass_bands=True, return_bins=False)
        if baseline:
            hrtf = baseline_hrtf(hrtf, bandwidth=bandwidth)  # baseline should be done after smoothing / dfe
        if write:
            hrtf.write_sofa(filename=processed_path / str('proccessed_' + row['filename']))
        print('processed ' + row['filename'])
        hrtf_dataframe.hrtf[index] = hrtf
    return hrtf_dataframe

def baseline_hrtf(hrtf, bandwidth=(3000, 16000)): #todo doesnt work yet (increases corrleation)
    "Center transfer functions around 0"
    hrtf_out = copy.deepcopy(hrtf)
    sources = hrtf_out.cone_sources(0)
    frequencies = hrtf[0].frequencies
    tf_data = hrtf_out.tfs_from_sources(sources, n_bins=len(frequencies), ear='both')
    in_range = tf_data[:, numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])]
    tf_data -= numpy.mean(numpy.mean(in_range, axis=1), axis=0)  # subtract mean for left and right ear separately
    # tf_data[:, numpy.logical_or(frequencies < bandwidth[0], frequencies > bandwidth[1])] = 0
    tf_data = 10 ** (tf_data / 20)
    for idx, source in enumerate(sources):
        hrtf_out[source].data = tf_data[idx]
    return hrtf_out

def erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286,
                    pass_bands=True, return_bins=False):
    """
    smoothe a transfer function by applying an erb-spaced triangular filterbank:
    compute a weighted sum of the energy in a range of FFT bins to get a number which can be interpreted
    as the energy measured at the output of a band-pass filter of a given center frequency/width
    """
    hrtf_out = copy.deepcopy(hrtf)
    hrtf_freqs = hrtf[0].frequencies
    n_freqs = hrtf[0].n_frequencies
    center_freqs_erb, oct_bandwidth, erb_spacing = slab.Filter._center_freqs(
        low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, pass_bands=pass_bands)
    tf_freqs_erb = slab.Filter._freq2erb(hrtf_freqs)
    center_freqs = slab.Filter._erb2freq(center_freqs_erb)
    n_bins = len(center_freqs_erb)
    windows = numpy.zeros((n_freqs, n_bins))
    dtf_binned = numpy.zeros((n_bins))
    hrtf_binned = numpy.zeros((hrtf_out.n_elevations, n_bins, 2))
    for dtf_idx, dtf in enumerate(hrtf_out):
        for chan_idx in range(dtf.n_channels):
            for bin_id in range(n_bins):
                l = center_freqs_erb[bin_id] - erb_spacing
                h = center_freqs_erb[bin_id] + erb_spacing
                window_size = ((tf_freqs_erb > l) & (tf_freqs_erb < h)).sum()  # width of the triangular window
                if kind == 'triangular':
                    window = scipy.signal.windows.triang(window_size, sym=True) #todo peak vals not always 1
                elif kind == 'cosine':
                    window = scipy.signal.windows.cosine(window_size, sym=True)
                windows[(tf_freqs_erb > l) & (tf_freqs_erb < h), bin_id] = window
                weighted_sum = numpy.sum(dtf.data[:, chan_idx] * windows[:, bin_id]) / window_size  # normalize by window size
                dtf_binned[bin_id] = weighted_sum
            hrtf_binned[dtf_idx, :, chan_idx] = dtf_binned
            hrtf_out[dtf_idx].data[:, chan_idx] = numpy.interp(hrtf_freqs, center_freqs, dtf_binned) # interpolate
    if return_bins:
        return hrtf_binned, center_freqs, hrtf_out
    else:
        return hrtf_out

def scepstral_filter_hrtf(hrtf, high_cutoff=1500):
    hrtf_out = copy.deepcopy(hrtf)
    filt = slab.Filter.band(kind='lp', frequency=high_cutoff, samplerate=hrtf.samplerate,
                            length=hrtf[0].n_samples, fir=True)
    for tf in hrtf_out:
        tf_data = 20 * numpy.log10(tf.data)
        to_filter = slab.Sound(tf_data, samplerate=tf.samplerate)
        filtered = filt.apply(to_filter)
        tf_data = 10 ** (filtered.data/20)
        tf.data = tf_data
    return hrtf_out

def average_hrtf(hrtf_list):
    list = copy.deepcopy(hrtf_list)
    tf_data = numpy.zeros((hrtf_list[0].n_sources, len(hrtf_list), hrtf_list[0][0].n_samples, 2))
    for hrtf_idx, hrtf in enumerate(hrtf_list):
        for src_idx, tf in enumerate(hrtf.data):
            tf_data[src_idx, hrtf_idx] = tf.data
    tf_data = numpy.mean(tf_data, axis=1)
    # dtf = copy.deepcopy(hrtf)
    for src_idx, tf_data in enumerate(tf_data):
        hrtf[src_idx].data = tf_data
    return hrtf

def write_average_tf(hrtf_df, path=Path.cwd() / 'data' / 'experiment'):
    hrtf_list = []
    for df_id, row in hrtf_df.iterrows():
        hrtf_list.append(copy.deepcopy(hrtf_df.iloc[df_id]['hrtf']))
    tf_data = numpy.zeros((hrtf_list[0].n_sources, len(hrtf_list), hrtf_list[0][0].n_samples, 2))
    for hrtf_idx, hrtf in enumerate(hrtf_list):
        for src_idx, tf in enumerate(hrtf.data):
            tf_data[src_idx, hrtf_idx] = tf.data
    tf_data = numpy.mean(tf_data, axis=1)
    # dtf = copy.deepcopy(hrtf)
    for src_idx, tf_data in enumerate(tf_data):
        hrtf[src_idx].data = tf_data
    mean_tf = []
    for tf in hrtf.data:
        for chan_idx in range(tf.n_channels):
            mean_tf.append(tf.data[:, chan_idx])
    mean_tf = numpy.mean(mean_tf, axis=0)
    mean_tf = slab.Filter(mean_tf, fir=False, samplerate=hrtf.samplerate)
    mean_tf.save(path / 'average_TF.npy')


