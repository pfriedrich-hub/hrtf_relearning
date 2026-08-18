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
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
from matplotlib import pyplot as plt

from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude


def _band_spread_db(hrtf, sourceidx, ear, band):
    """Spread of the spectrum across directions inside ``band``, in dB.

    Read from the impulse responses directly rather than from the image built
    below. ``_build_tf_image`` defaults to ``n_bins = n_taps``, i.e. it asks
    slab for 512 spectral points out of a 257-point rfft, and that
    interpolation invents ~0.5 dB of direction-dependent variation where there
    is none — an elevation-averaged ear that is provably flat reads 0.56 dB
    through it. Fine for a picture, not fine for a number that is supposed to
    match qc_midline.

    Each direction's own mean level is removed first: that part is the intended
    ILD, not a cue.
    """
    channel = {'left': 0, 'right': 1}[ear]
    spectra = []
    for index in sourceidx:
        ir = numpy.asarray(hrtf[index].data[:, channel], dtype=float)
        spectra.append(20.0 * numpy.log10(numpy.maximum(
            numpy.abs(numpy.fft.rfft(ir)), 1e-30)))
    spectra = numpy.asarray(spectra)
    freqs = numpy.fft.rfftfreq(hrtf[sourceidx[0]].data.shape[0],
                               d=1.0 / float(hrtf[sourceidx[0]].samplerate))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float('nan')
    spectra = spectra - spectra.mean(axis=1, keepdims=True)
    return float(numpy.mean(numpy.std(spectra[:, in_band], axis=0)))


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


def plot_ears(hrtf, hrtf_modified, n_bins=None, xlim=(1000, 18000),
              vsi_dis=None, vsi_bw=None, band=None, suptitle=None, show=True):
    """2x2 before/after image: rows are the two ears, columns original/modified.

    The standard QC figure for any manipulation that touches both ears
    differently — a donor composite on one ear with an envelope or flat other
    ear looks fine in a single-ear plot and only shows its asymmetry here.
    All four panels share one colour scale so interaural level differences stay
    visible; ``band`` (low, high) shades the scoring band if given.

    Rendered as an image (pcolormesh) rather than filled contours: contour
    bands quantise the spectrum into 20 steps, which is enough to hide a notch
    that has moved by less than one step and to invent edges where the surface
    is smooth. That matters here — what the manipulation does to the fine
    structure IS the figure's subject. Each panel is annotated with the spread
    of its own spectrum across elevation inside ``band``, which is the number
    the monaural reduction has to drive to zero on the processed ear.
    """
    sources = hrtf.cone_sources(0)
    images = {}
    for ear in ('left', 'right'):
        freqs, elevations, original = _build_tf_image(hrtf, sources, ear, n_bins, xlim)
        _, _, modified = _build_tf_image(hrtf_modified, sources, ear, n_bins, xlim)
        images[ear] = (freqs, elevations, original, modified)

    flat = [img for _, _, a, b in images.values() for img in (a, b)]
    vmin = float(min(i.min() for i in flat))
    vmax = float(max(i.max() for i in flat))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    mesh = None
    for row, ear in enumerate(('left', 'right')):
        freqs, elevations, original, modified = images[ear]
        for column, (img, label) in enumerate(((original, 'original'),
                                               (modified, 'modified'))):
            axis = axes[row, column]
            mesh = axis.pcolormesh(freqs, elevations, img.T, cmap='magma',
                                   vmin=vmin, vmax=vmax, shading='gouraud',
                                   rasterized=True)
            if band is not None:
                axis.axvspan(band[0], band[1], color='#00c8ff', alpha=0.10, lw=0)
                for edge in band:
                    axis.axvline(edge, color='#00c8ff', lw=0.9, ls=':')
                # spread across elevation inside the band = the cue this ear
                # still carries. Measured off the IRs, not off the image above
                # -- see _band_spread_db for why.
                source_of = hrtf if label == 'original' else hrtf_modified
                spread = _band_spread_db(source_of, sources, ear, band)
                axis.text(0.02, 0.04, f'{spread:.2f} dB', transform=axis.transAxes,
                          fontsize=8, va='bottom', color='#1b1b1f',
                          bbox=dict(fc='white', ec='#bbbbbb', lw=0.6, pad=2))
            axis.set_title(f'{ear} ear — {label}', fontsize=10)
            axis.set(xlim=xlim)
            axis.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x / 1000))))
            if row == 1:
                axis.set_xlabel('Frequency [kHz]')
            if column == 0:
                axis.set_ylabel('Elevation [°]')
            axis.tick_params('both', length=2, pad=2)

    cbar = fig.colorbar(mesh, ax=list(axes.ravel()), shrink=0.9, pad=0.02)
    cbar.set_label('Magnitude [dB]')
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    if vsi_dis is not None:
        bandwidth = (f'{vsi_bw[0] / 1000:.1f}–{vsi_bw[1] / 1000:.1f} kHz'
                     if vsi_bw is not None else '')
        fig.text(0.5, 0.01,
                 f'VSI dissimilarity = {vsi_dis:.3f}   ({bandwidth}, Trapeau et al. 2016)',
                 ha='center', va='bottom', fontsize=9)
    # `show=False` returns the figure without displaying it, so a caller can
    # save it as provenance without three windows opening during donor staging.
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return fig


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
