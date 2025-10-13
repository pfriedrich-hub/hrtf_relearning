import slab
import numpy
from pathlib import Path
import copy

def get_hrtf_list(database_name='aachen_database'):
    database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / database_name
    return [slab.HRTF(sofa_path) for sofa_path in list(database_path.glob('*.sofa'))]
#
# def make_avg_hrtf():
#     """
#     Construct a new HRTF from a list of existing HRTFs.
#     Computes the probability of peaks and notches for each source and frequency bin across HRTFs in the list.
#
#     Args:
#         database_name (string): Name the folder containing the HRTF database.
#                                 HRTFs in the folder must have uniform source space and filters.
#         threshold (tuple): upper and lower threshold in dB for feature detection
#     """
#     hrtf_list = get_hrtf_list()
#     hrtf = hrtf_list[0]
#     w, _ = hrtf[0].tf(show=False)
#     in_freq = w > 4000
#     data = []
#     thr = []
#     for i, hrtf in enumerate(hrtf_list):
#         tfs = hrtf.tfs_from_sources(sources=range(len(hrtf.sources.vertical_polar)), n_bins=None, ear='both')
#         tfs = tfs[:, in_freq]
#         thr.append((numpy.percentile(tfs, 25), numpy.percentile(tfs, 75)))
#         print(numpy.percentile(tfs, 25), numpy.percentile(tfs, 75))
#         data.append(hrtf.tfs_from_sources(sources=range(len(hrtf.sources.vertical_polar)), n_bins=None, ear='both'))
#
#     data = numpy.asarray(data)
#     threshold = numpy.percentile(data, 25), numpy.percentile(data, 75)
#     #todo find a better measure than thresholding,
#     # for example take the jnd spectral notch width and depth in dB
#     print(f"Thresholds; lower: {threshold[0]}, upper: {threshold[1]}")
#     # compute probability map of spectral peaks and notches across frequencies and sources
#         print(f"Retrieving TF data from database: {i / len(hrtf_list) * 100:.2f} %")
#         dtfs = []
#         for filter in hrtf:
#             w, h = filter.tf(n_bins=None, show=False)
#             dtfs.append(h)
#         # thresh = [numpy.percentile(tfs, 25), numpy.percentile(tfs, 75)]  # local threshold
#         data.append(dtfs)
#     del hrtf_list
#     data = numpy.array(data)
#     freq_mask = numpy.where(w > 4000)  # disregard torso shadow for cue threshold estimation
#     if threshold is None:
#         threshold = numpy.percentile(data[:, :, freq_mask], 25), numpy.percentile(data[:, :, freq_mask], 75)  # global threshold
#         print(f"Thresholds; lower: {threshold[0]}, upper: {threshold[1]}")
#     notch_map = (data < threshold[0]) * -1 # binary map of notches across hrtfs, sources, frequencies
#     peak_map = (data > threshold[1]) # binary map of peaks across hrtfs, sources, frequencies
#     p_map = numpy.mean(notch_map + peak_map, axis=0)
#     # rescale to dB
#
#
#     hrtf = copy.deepcopy(hrtf_list[0])  # new HRTF
#     for i, tf in enumerate(p_map):  # reconstruct filters from feature probability
#         tf = (tf + 1) * 0.5 * (threshold[1] - threshold[0]) + threshold[0]  # rescale probabilities between thresholds
#         hrtf[i].data = slab.Filter(data=tf, samplerate=hrtf_list[0].samplerate, fir='TF')
#     # return numpy.sum(numpy.array(feature_maps), axis=0) / len(feature_maps), frequencies[freq_idx]  # average
#     feature_map = numpy.sum(notch_map + peak_map, axis=0) / len(data)
#     # rescale probabilities to thresholds
#     dtfs = (feature_map + 1) * 0.5 * (threshold[1] - threshold[0]) + threshold[0]
#     for i, tf in enumerate(dtfs):  # reconstruct filters from feature probability
#         hrtf[i].data = slab.Filter(data=tf, samplerate=hrtf.samplerate, fir='TF')
#
import numpy
import slab
from pathlib import Path
import copy

# ---------- helpers ----------
def moving_average(x, k=5, axis=-1):
    """Simple moving average along a chosen axis."""
    if k <= 1:
        return x
    pad = k // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad, pad)
    xpad = numpy.pad(x, pad_width, mode='edge')
    kernel = numpy.ones(k) / k
    # move axis to last for apply_along_axis
    xpad_last = numpy.moveaxis(xpad, axis, -1)
    y_last = numpy.apply_along_axis(lambda v: numpy.convolve(v, kernel, mode='valid'), -1, xpad_last)
    return numpy.moveaxis(y_last, -1, axis)

def slope_on_logfreq(logmag, freqs_hz, smooth_len=7, axis=-2):
    """
    d(logmag)/d(log2 f) along the frequency axis.
    logmag: array (..., F, ...) with freq along 'axis'
    returns same shape as logmag.
    """
    nu = numpy.log2(numpy.maximum(freqs_hz, 1.0))  # [F]
    # smooth along frequency
    smoothed = moving_average(logmag, k=smooth_len, axis=axis)
    # numpy.gradient needs freq axis contiguous; move, grad, move back
    x = numpy.moveaxis(smoothed, axis, -1)  # (..., F)
    # gradient w.r.t. nu (non-uniform spacing aware)
    g = numpy.gradient(x, nu, axis=-1, edge_order=2)
    return numpy.moveaxis(g, -1, axis)

# ---------- I/O ----------
def get_hrtf_list(database_name='aachen_database'):
    database_path = Path.cwd() / 'data' / 'hrtf' / 'sofa' / database_name
    sofa_files = sorted(database_path.glob('*.sofa'))
    return [slab.HRTF(sofa_path) for sofa_path in sofa_files]

# ---------- main ----------
def make_avg_hrtf(database_name='aachen_database',
                  fmin_hz=4000,
                  smooth_len=7,
                  slope_thresh_db_per_oct=0.6):
    """
    Build an averaged HRTF using slope-based 'probability' maps.

    Steps:
      1) Load TFs in dB: shape (N, D, F, E).
      2) DTF per subject & ear: subtract subject's across-direction mean (per freq, per ear).
      3) Compute slope dB / octave on log2(f).
      4) Sign maps with dead-zone threshold: +1 rising, -1 falling, 0 flat.
      5) Average signs across subjects -> p_map in [-1, 1] for each (D, F, E).
      6) Rescale p_map by a robust contrast (in dB) -> representative DTF.
      7) Write TFs back into a new HRTF object.

    Returns:
      new_hrtf: slab.HRTF
      debug: dict with p_map, freqs_band, and parameters
    """
    hrtf_list = get_hrtf_list(database_name)
    if not hrtf_list:
        raise RuntimeError("No SOFA files found in the given database folder.")

    # frequency axis from first filter
    w_ref, _ = hrtf_list[0][0].tf(show=False)             # [F]
    F = w_ref.size
    freq_mask = w_ref > fmin_hz
    F_band = int(freq_mask.sum())

    # ----- 1) Load all TFs into (N, D, F, E) -----
    all_tf = []  # in dB
    print("Loading TFs...")
    for i, hrtf in enumerate(hrtf_list, 1):
        tfs = hrtf.tfs_from_sources(
            sources=range(len(hrtf.sources.vertical_polar)),
            n_bins=None,
            ear='both'   # -> (D, F, E=2)
        )
        # sanity check last dim is ears
        if tfs.ndim != 3 or tfs.shape[-1] != 2:
            raise ValueError(f"Expected (D,F,2) from tfs_from_sources, got {tfs.shape}")
        all_tf.append(tfs)
        print(f"  {i}/{len(hrtf_list)} loaded")

    all_tf = numpy.asarray(all_tf)                         # (N, D, F, E)
    N, D, F_chk, E = all_tf.shape
    assert F_chk == F and E == 2, f"Unexpected shape {all_tf.shape}"

    # ----- 2) DTF per subject & ear -----
    # subtract across-direction mean for each subject, freq, and ear
    subj_mean = all_tf.mean(axis=1, keepdims=True)        # (N,1,F,E)
    dtf = all_tf - subj_mean                               # (N,D,F,E)

    # analysis band
    dtf_band = dtf[:, :, freq_mask, :]                    # (N,D,Fb,E)
    freqs_band = w_ref[freq_mask]                         # (Fb,)

    # ----- 3) slopes on log-frequency (per ear independently) -----
    # frequency axis is -2 (the F dim) for dtf_band
    slopes = slope_on_logfreq(dtf_band, freqs_band, smooth_len=smooth_len, axis=-2)  # (N,D,Fb,E)

    # ----- 4) sign maps with dead-zone -----
    signs = numpy.zeros_like(slopes, dtype=numpy.int8)    # (N,D,Fb,E)
    signs[slopes >  slope_thresh_db_per_oct] =  1
    signs[slopes < -slope_thresh_db_per_oct] = -1

    # ----- 5) average across subjects -> p_map (D,Fb,E) in [-1,1] -----
    p_map = signs.mean(axis=0)                            # (D,Fb,E)

    # ----- 6) rescale p_map to a DTF (in dB) with robust contrast -----
    # use robust range of |DTF| within the band across all N,D,E
    abs_dtf = numpy.abs(dtf_band.reshape(N*D*E, F_band))
    low, high = numpy.percentile(abs_dtf, [60, 95])
    contrast_db = (low + high) / 2.0                      # single scalar; tune as needed

    dtf_recon_band = p_map * contrast_db                  # (D,Fb,E) in dB

    # stitch back to full band (zeros outside analysis band)
    dtf_recon_full = numpy.zeros((D, F, E), dtype=float)  # (D,F,E)
    dtf_recon_full[:, freq_mask, :] = dtf_recon_band

    # Optional: enforce zero-mean across directions per ear (DTF property)
    dtf_recon_full -= dtf_recon_full.mean(axis=0, keepdims=True)

    # ----- 7) write into a new HRTF object -----
    new_hrtf = copy.deepcopy(hrtf_list[0])
    # For each filter (direction×ear-pair), assign TF in dB with both ears in last dim
    for i in range(D):
        # expected shape for slab.Filter fir='TF': (F,) or (F,2). We'll pass (F,2).
        tf_db = dtf_recon_full[i]                         # (F,2)
        new_hrtf[i].data = slab.Filter(data=tf_db, samplerate=new_hrtf.samplerate, fir='TF')

    debug = dict(
        p_map=p_map,                      # (D,Fb,E)
        freqs_band=freqs_band,            # (Fb,)
        params=dict(
            fmin_hz=fmin_hz,
            smooth_len=smooth_len,
            slope_thresh_db_per_oct=slope_thresh_db_per_oct,
            contrast_db=float(contrast_db),
        ),
        shapes=dict(N=N, D=D, F=F, E=E, F_band=F_band)
    )
    return new_hrtf, debug
