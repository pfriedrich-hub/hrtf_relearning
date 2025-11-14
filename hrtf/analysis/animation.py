import numpy
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from hrtf.processing.average import hrtf_average
from hrtf.analysis.feature_p import feature_p

# -------------------------------------------------------------------------
# Globals used by the animation callbacks (kept for drop-in compatibility)
# -------------------------------------------------------------------------
data = None
fig = None
ax = None
frequencies = None
azimuths = None
elevations = None
settings = None
cbar_axis = None
cbar_levels = None
cbar_ticks = None
_elev_to_row = None


def _normalize_hrtf_input(hrtf):
    """
    Normalize hrtf input to a non-empty list of HRTF objects.

    Accepts:
      - a single HRTF object (duck-typed on 'get_source_idx' and 'sources')
      - a list/tuple/numpy array of HRTF objects
    """
    # Single HRTF object
    if hasattr(hrtf, "get_source_idx") and hasattr(hrtf, "sources"):
        return [hrtf]

    # Sequence of HRTFs
    if isinstance(hrtf, (list, tuple, numpy.ndarray)):
        hrtf_list = list(hrtf)
        if len(hrtf_list) == 0:
            raise ValueError("hrtf list is empty.")
        if not (hasattr(hrtf_list[0], "get_source_idx") and hasattr(hrtf_list[0], "sources")):
            raise TypeError(
                "Elements of hrtf list do not look like HRTF objects "
                "(missing 'get_source_idx' or 'sources')."
            )
        return hrtf_list

    raise TypeError(
        "hrtf must be a single HRTF object or a list/tuple/array of HRTF objects."
    )


def hrtf_animation(
    hrtf,
    azimuth_range=(-180, 180),
    elevation_range=(-60, 60),
    ear="left",
    interval=100,
    map="average",
    kind="image",
    filename=None,
    write=None,
    show=True,
    figsize=(5, 5),
):
    """
    Create an animation of HRTF features across azimuths.

    Parameters are meant to be drop-in compatible with your original script.

    write:
        - None / False: do not save
        - True: auto -> try ffmpeg (.mp4), fallback to Pillow (.gif)
        - 'ffmpeg': force FFMpegWriter (.mp4)
        - 'pillow': force PillowWriter (.gif)
    """

    global data, fig, ax, frequencies, azimuths, elevations, settings
    global cbar_axis, cbar_levels, cbar_ticks, _elev_to_row

    settings = {"map": map, "kind": kind}

    # ------------------------------------------------------------------
    # Input normalization and basic setup
    # ------------------------------------------------------------------
    hrtf_list = _normalize_hrtf_input(hrtf)
    ref_hrtf = hrtf_list[0]

    # Get all sources in requested azimuth/elevation ranges
    source_idx_all = ref_hrtf.get_source_idx(
        azimuth=azimuth_range,
        elevation=elevation_range,
    )
    if source_idx_all is None or len(source_idx_all) == 0:
        raise ValueError(
            f"No sources found in azimuth_range={azimuth_range}, "
            f"elevation_range={elevation_range}."
        )

    source_idx_all = numpy.array(source_idx_all, dtype=int)
    vp_all = ref_hrtf.sources.vertical_polar[source_idx_all]

    # Unique azimuths and elevations present in this region
    azimuths = numpy.unique(vp_all[:, 0])
    elevations = numpy.unique(vp_all[:, 1])

    if azimuths.size == 0 or elevations.size == 0:
        raise ValueError("No valid azimuths or elevations found for the given ranges.")

    # Map from elevation value -> row index in our global elevation grid
    _elev_to_row = {float(e): i for i, e in enumerate(elevations)}

    # Frequency band used for analysis
    bandwidth = (1000, 18000)

    # ------------------------------------------------------------------
    # Compute data for each azimuth (one frame per azimuth)
    # ------------------------------------------------------------------
    data_frames = []
    used_azimuths = []

    for az in azimuths:

        # All sources for this azimuth within the selected elevation range

        src_idx = ref_hrtf.cone_sources(az)
        if src_idx is None or len(src_idx) == 0:
            print(f"  Warning: no sources at azimuth {az}, skipping.")
            continue

        print(f"Azimuth: {az}: n_elevations: {len(src_idx)}: ")

        src_idx = numpy.array(src_idx, dtype=int)
        src_vp = ref_hrtf.sources.vertical_polar[src_idx]

        # Sort by elevation
        sort_idx = numpy.argsort(src_vp[:, 1])
        src_idx = src_idx[sort_idx]
        elev_local = src_vp[sort_idx, 1]

        # --------------------------------------------------------------
        # Compute map for this azimuth
        # --------------------------------------------------------------
        if settings["map"] == "feature_p":
            feature_map_local, freqs_local = feature_p(
                hrtf_list, src_idx, thresholds=None, bandwidth=bandwidth, ear=ear
            )
            # (n_local_elev, n_freq)
            feature_map_local = feature_map_local.reshape(feature_map_local.shape[:2])
            frequencies = freqs_local  # same for all azimuths

        elif settings["map"] == "average":
            # Frequencies from reference HRTF
            freqs_all = ref_hrtf[0].tf(show=False)[0]
            freq_idx = numpy.logical_and(
                freqs_all >= bandwidth[0], freqs_all <= bandwidth[1]
            )
            if not numpy.any(freq_idx):
                raise ValueError(
                    f"No frequency bins within bandwidth {bandwidth}."
                )
            frequencies = freqs_all[freq_idx]

            # Average across HRTFs
            avg_hrtf = hrtf_average(hrtf_list)
            tf_map = avg_hrtf.tfs_from_sources(src_idx, n_bins=None, ear=ear)
            feature_map_local = tf_map.reshape(tf_map.shape[:2])[:, freq_idx]

        else:
            raise ValueError(
                f"Unknown map type '{settings['map']}'. Use 'feature_p' or 'average'."
            )

        # --------------------------------------------------------------
        # Align this azimuth's data onto the global elevation grid
        # --------------------------------------------------------------
        n_elev_global = len(elevations)
        n_freq = feature_map_local.shape[1]
        frame_data = numpy.full((n_elev_global, n_freq), numpy.nan, dtype=float)

        for row_local, elev_val in enumerate(elev_local):
            key = float(elev_val)
            if key not in _elev_to_row:
                continue  # Shouldn't happen, but be safe
            row_global = _elev_to_row[key]
            frame_data[row_global, :] = feature_map_local[row_local, :]

        data_frames.append(frame_data)
        used_azimuths.append(az)

    if len(data_frames) == 0:
        raise RuntimeError(
            "No frames could be generated (no valid sources for any azimuth)."
        )

    # Final data and azimuths (skip azimuths that had no sources)
    data = data_frames
    azimuths = numpy.array(used_azimuths)

    # ------------------------------------------------------------------
    # Set up figure, axes, and colorbar
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    if filename is not None:
        ax.set_title(str(filename))
    else:
        ax.set_title(f"Azimuth: {azimuths[0]}")

    if settings["kind"] == "image":
        # Robust global color limits across all frames
        frame_mins = [numpy.nanmin(f) for f in data]
        frame_maxs = [numpy.nanmax(f) for f in data]
        z_min = float(numpy.floor(numpy.nanmin(frame_mins)))
        z_max = float(numpy.ceil(numpy.nanmax(frame_maxs)))

        if numpy.isclose(z_min, z_max):
            z_min -= 0.5
            z_max += 0.5

        cbar_levels = numpy.linspace(z_min, z_max, 50)
        cbar_ticks = numpy.arange(z_min, z_max, 6)[1:]

        # Create a separate axis for the colorbar
        cax_pos = list(ax.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[2] = cax_pos[2] * 0.06  # cbar width relative to axis width
        cax_pos[0] = 0.91
        cbar_axis = fig.add_axes(cax_pos)

        im0 = ax.contourf(frequencies, elevations, data[0], levels=cbar_levels)
        cbar = fig.colorbar(im0, cbar_axis, orientation="vertical", ticks=cbar_ticks)

        if settings["map"] == "feature_p":
            cbar_axis.set_title("p")
        elif settings["map"] == "average":
            cbar_axis.set_title("dB")

    # ------------------------------------------------------------------
    # Create animation
    # ------------------------------------------------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(data),
        interval=interval,
        blit=False,
    )

    # ------------------------------------------------------------------
    # Save video if requested
    # ------------------------------------------------------------------
    if write:
        if filename is None:
            raise ValueError("filename must be provided when 'write' is truthy.")

        out_dir = Path.cwd() / "data" / "img" / "animations"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Interpret 'write' argument
        if isinstance(write, str):
            mode = write.lower()
        else:
            mode = "auto"  # write=True, None/False -> no saving

        fps = max(1, int(1000 / interval))

        # Helper: try saving with ffmpeg, optionally fall back to pillow
        def _save_with_ffmpeg(path):
            writer_ffmpeg = animation.FFMpegWriter(fps=fps)
            ani.save(path, writer=writer_ffmpeg)
            print(f"Saved animation to: {path}")

        def _save_with_pillow(path):
            writer_pillow = animation.PillowWriter(fps=fps)
            ani.save(path, writer=writer_pillow)
            print(f"Saved animation to: {path}")

        if mode in ("auto", "ffmpeg"):
            mp4_path = out_dir / f"{filename}.mp4"
            try:
                _save_with_ffmpeg(mp4_path)
            except FileNotFoundError as e:
                if mode == "ffmpeg":
                    raise RuntimeError(
                        "FFmpeg writer requested (write='ffmpeg'), "
                        "but the 'ffmpeg' executable was not found on your system. "
                        "Install ffmpeg or use write='pillow' to save as GIF."
                    ) from e
                # auto fallback
                print(
                    "FFmpeg not found, falling back to PillowWriter and saving as GIF."
                )
                gif_path = out_dir / f"{filename}.gif"
                _save_with_pillow(gif_path)

        elif mode == "pillow":
            gif_path = out_dir / f"{filename}.gif"
            _save_with_pillow(gif_path)

        else:
            raise ValueError(
                f"Unknown value for 'write': {write!r}. "
                "Use True/'auto', 'ffmpeg', or 'pillow'."
            )

    # ------------------------------------------------------------------
    # Show or close figure
    # ------------------------------------------------------------------
    if show:
        plt.show()
    else:
        plt.close(fig)


def init():
    """
    Initialization function for FuncAnimation.
    Draw the first frame.
    """
    im = plot(data=data[0])
    return (im,)


def animate(i):
    """
    Animation function called sequentially by FuncAnimation.
    """
    im = plot(data=data[i])
    ax.set_title(f"Azimuth: {azimuths[i]}")
    return (im,)


def plot(data):
    """
    Helper to draw a single frame.

    For kind == 'image': contour plot of elevation vs frequency.
    For kind == 'waterfall': stacked spectra with elevation encoded in vertical offset.
    """
    # Clear main axis but leave colorbar axis untouched
    fig.axes[0].clear()

    if settings["kind"] == "image":
        im = ax.contourf(frequencies, elevations, data, levels=cbar_levels)

    elif settings["kind"] == "waterfall":
        linesep = 20.0
        vlines = numpy.arange(0, len(elevations)) * linesep

        for idx, spectrum in enumerate(data):
            # skip rows that are all NaN (no source at this elevation for this azimuth)
            if numpy.all(numpy.isnan(spectrum)):
                continue
            im = ax.plot(
                frequencies,
                spectrum + vlines[idx],
                linewidth=0.75,
                color="0.0",
                alpha=0.7,
            )

        ticks = vlines[::2]  # every second elevation
        labels = elevations[::2].astype(int)
        ax.set(yticks=ticks, yticklabels=labels)
        ax.grid(visible=True, axis="y", which="both", linewidth=0.25)

        # dB scale bar
        ax.plot(
            [frequencies[0] + 500, frequencies[0] + 500],
            [vlines[-1] + 10, vlines[-1] + 10 + linesep],
            linewidth=1,
            color="0.0",
            alpha=0.9,
        )
        ax.text(
            x=frequencies[0] + 600,
            y=vlines[-1] + 10 + linesep / 2,
            s=f"{int(linesep)} dB",
            va="center",
            ha="left",
            fontsize=6,
            alpha=0.7,
        )

        # im is a list of Line2D; FuncAnimation only needs some artist
        im = im[0] if isinstance(im, list) else im

    else:
        raise ValueError(
            f"Unknown kind '{settings['kind']}'. Use 'image' or 'waterfall'."
        )

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Elevation (degrees)")
    return im
