"""
plot_compare.py — before/after QC figures shared by the manipulations in this
package.

``plot`` shows a native and a modified HRTF side by side on a common colour
scale; ``plot_split_qc`` checks the coarse/fine split that the manipulations
rely on. Neither knows anything about a particular manipulation, so both
shift_spectral_detail.py and synth_spectral_features.py use them.
"""

import numpy
import matplotlib
import matplotlib.ticker
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude


def _build_tf_image(hrtf, sourceidx, ear, n_bins, xlim, floor_db=-25):
    """
    Build the image array used by plot's 'image' mode, using tfs_from_sources to
    obtain the raw dB data.

    Returns
    -------
    freqs      : 1-D array, frequency axis trimmed to xlim[1]
    elevations : 1-D array, elevation for each source
    img        : 2-D array (n_freq_bins, n_sources), clipped at floor_db
    """
    chan = {'left': 0, 'right': 1}[ear]
    n_b = n_bins if n_bins is not None else hrtf[sourceidx[0]].n_taps
    # tfs_from_sources returns (n_sources, n_bins, 1) — squeeze to (n_sources, n_bins)
    tfs = hrtf.tfs_from_sources(sourceidx, n_bins=n_b, ear=ear)
    img = numpy.clip(tfs.squeeze(-1).T, floor_db, None)   # (n_bins, n_sources)
    freqs, _ = hrtf[sourceidx[0]].tf(channels=chan, n_bins=n_bins, show=False)
    elevations = hrtf.sources.vertical_polar[sourceidx, 1]
    mask = freqs <= xlim[1]
    return freqs[mask], elevations, img[mask, :]


def plot(hrtf, hrtf_modified, kind='image', ear='left', n_bins=None, xlim=(1000, 18000),
         vsi_orig=None, vsi_mod=None, vsi_dis=None, vsi_bw=None):
    """Native vs modified median-plane transfer function, side by side."""
    sources = hrtf.cone_sources(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if kind == 'image':
        # Build raw dB image data for both HRTFs via tfs_from_sources
        freqs, elevations, img_orig = _build_tf_image(hrtf,          sources, ear, n_bins, xlim)
        _,     _,          img_mod  = _build_tf_image(hrtf_modified, sources, ear, n_bins, xlim)

        # Joint colorbar limits across both images
        vmin   = float(min(img_orig.min(), img_mod.min()))
        vmax   = float(max(img_orig.max(), img_mod.max()))
        levels = numpy.linspace(vmin, vmax, 21)

        ct = None
        for ax, img, title, vsi_val in zip(
                axes,
                [img_orig, img_mod],
                ['original', 'modified'],
                [vsi_orig,  vsi_mod]):
            ct = ax.contourf(freqs, elevations, img.T, cmap='hot', levels=levels)
            title_full = title + (f'   (VSI = {vsi_val:.3f})' if vsi_val is not None else '')
            ax.set_title(title_full)
            ax.set(xlabel='Frequency [kHz]', ylabel='Elevation [°]', xlim=xlim)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x / 1000))))
            ax.autoscale(tight=True)
            ax.tick_params('both', length=2, pad=2)

        # Single shared colorbar to the right of both subplots, matplotlib-managed
        # so it stays aligned to the axes height (steals space from both). Don't
        # call tight_layout here -- it would conflict with the managed colorbar.
        cbar_ticks = numpy.arange(numpy.ceil(vmin / 6.0) * 6.0, vmax, 6.0)
        cbar = fig.colorbar(ct, ax=list(axes), shrink=0.9, pad=0.02, ticks=cbar_ticks)
        cbar.set_label('Magnitude [dB]')

        # VSI dissimilarity as a footer below both plots
        if vsi_dis is not None:
            bw_str = (f'{vsi_bw[0]/1000:.1f}–{vsi_bw[1]/1000:.1f} kHz'
                      if vsi_bw is not None else '')
            fig.text(0.5, 0.01,
                     f'VSI dissimilarity = {vsi_dis:.3f}   ({bw_str}, Trapeau et al. 2016)',
                     ha='center', va='bottom', fontsize=9)

    else:
        # waterfall / surface: fall back to plot_tf (no shared scale needed)
        hrtf.plot_tf(         sources, kind=kind, axis=axes[0], ear=ear, xlim=xlim, show=False)
        hrtf_modified.plot_tf(sources, kind=kind, axis=axes[1], ear=ear, xlim=xlim, show=False)
        axes[0].set_title('original')
        axes[1].set_title('modified')
        plt.tight_layout()

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def plot_split_qc(hrtf, envelope_n_keep, ear='right', xlim=(2000, 18000), band=None):
    """QC for the coarse/fine split the manipulations depend on.

    For each median-plane elevation, overlay the full log-magnitude (thin grey)
    and the truncated-cosine envelope (thick red) that the manipulation holds
    fixed. The envelope should be smooth AND roughly elevation-invariant: if the
    red curves still track elevation, the split is freezing a cue that also
    carries elevation information (a cue conflict), not cleanly separating macro
    shape from the fine structure being manipulated. Curves are mean-removed and
    stacked.
    """
    chan = {'left': 0, 'right': 1}[ear]
    sources = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[sources, 1]
    order = numpy.argsort(elevations)
    sources_sorted = numpy.array(sources)[order]

    fig, ax = plt.subplots(figsize=(6, 8))
    offset = 25.0  # dB between stacked elevations
    for row, idx in enumerate(sources_sorted):
        ir = numpy.asarray(hrtf[idx].data, dtype=float)
        n = ir.shape[0]
        freqs = numpy.fft.rfftfreq(n, d=1.0 / hrtf[idx].samplerate)
        mag = numpy.abs(numpy.fft.rfft(ir, axis=0))
        env = smooth_magnitude(mag, n_keep=envelope_n_keep)
        eps = numpy.finfo(float).tiny
        full_db = 20.0 * numpy.log10(numpy.maximum(mag[:, chan], eps))
        env_db = 20.0 * numpy.log10(numpy.maximum(env[:, chan], eps))
        m = (freqs >= xlim[0]) & (freqs <= xlim[1])
        y0 = row * offset
        ax.plot(freqs[m], full_db[m] - full_db[m].mean() + y0, lw=0.6, color='0.6')
        ax.plot(freqs[m], env_db[m] - env_db[m].mean() + y0, lw=1.8, color='C3')
        ax.text(xlim[1], y0, f'{elevations[order][row]:.0f}°', va='center', fontsize=7)

    if band is not None:
        ax.axvspan(band[0], band[1], color='C0', alpha=0.08, lw=0)
    ax.set(xlabel='Frequency [Hz]', ylabel='elevation (stacked, mean-removed dB)',
           xlim=xlim, title=f'split QC: envelope (M={envelope_n_keep}) vs full — {ear} ear')
    ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return fig
