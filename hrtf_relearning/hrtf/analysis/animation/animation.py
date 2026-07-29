import numpy
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import slab
from hrtf_relearning.utils import paths

hrtf_dir = paths.SOFA_DIR / "GS"

# ---------------------------------------------------------------------------
# Parameters – edit here
# ---------------------------------------------------------------------------
hrtf_id         = "GS"  # HRTF file stem (loads <hrtf_id>.sofa)
                                # NB kemar.sofa / kemar_pir.sofa hold only the midline arc
                                # (7 sources, all at azimuth 0) - nothing to animate there
azimuth_range   = (0, 360)     # (min, max) in degrees; None = all
elevation_range = (-40, 40)     # (min, max) in degrees; None = all
ear             = "both"        # "left", "right", or "both"
kind            = "image"       # "image" (contourf) or "waterfall"
bandwidth       = (200, 18000) # frequency range in Hz
sampling_mode   = "interpolate" # "measured", "nearest", or "interpolate"
azimuth_step    = 2.5           # grid step in degrees (nearest/interpolate)
elevation_step  = 1.0           # grid step in degrees (nearest/interpolate)
n_bins          = 512           # frequency bins (nearest/interpolate); None = native
interval        = 100           # frame interval in ms
dpi             = 200           # output resolution when writing mp4/gif
write           = True          # False | True/'ffmpeg' (mp4) | 'pillow' (gif)
show            = True          # display the animation window
# ---------------------------------------------------------------------------


def _sorted_unique(values):
    return numpy.unique(numpy.asarray(values))


def _wrap_azimuth(values):
    values = numpy.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def _get_selected_source_idx(hrtf, azimuth_range=None, elevation_range=None):
    """
    Select raw measured sources by vertical-polar azimuth/elevation ranges.
    This does NOT use hrtf.get_source_idx because that routes through cone_sources,
    which is cone-of-confusion based rather than simple vertical-polar meridian selection.
    """
    vp = hrtf.sources.vertical_polar.astype(float)

    az = _wrap_azimuth(vp[:, 0])
    el = vp[:, 1]

    keep = numpy.ones(len(vp), dtype=bool)

    if azimuth_range is not None:
        az_min, az_max = float(azimuth_range[0]), float(azimuth_range[1])
        keep &= (az >= az_min) & (az <= az_max)

    if elevation_range is not None:
        el_min, el_max = float(elevation_range[0]), float(elevation_range[1])
        keep &= (el >= el_min) & (el <= el_max)

    source_idx = numpy.where(keep)[0]
    if source_idx.size == 0:
        raise ValueError(
            f"No sources found for azimuth_range={azimuth_range}, "
            f"elevation_range={elevation_range}."
        )
    return source_idx.astype(int)


def _make_target_grid(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    azimuth_step=5.0,
    elevation_step=5.0,
):
    vp = hrtf.sources.vertical_polar.astype(float)

    az_all = _wrap_azimuth(vp[:, 0])
    el_all = vp[:, 1]

    if azimuth_range is None:
        az_min, az_max = float(numpy.min(az_all)), float(numpy.max(az_all))
    else:
        az_min, az_max = float(azimuth_range[0]), float(azimuth_range[1])

    if elevation_range is None:
        el_min, el_max = float(numpy.min(el_all)), float(numpy.max(el_all))
    else:
        el_min, el_max = float(elevation_range[0]), float(elevation_range[1])

    azimuths = numpy.arange(az_min, az_max + azimuth_step / 2, azimuth_step, dtype=float)
    elevations = numpy.arange(el_min, el_max + elevation_step / 2, elevation_step, dtype=float)

    if azimuths.size == 0 or elevations.size == 0:
        raise ValueError("Target grid is empty.")

    return azimuths, elevations


def _get_all_tfs(hrtf, n_bins=None):
    """
    dB magnitude spectra of every source, both ears.

    Returns
    -------
    frequencies : (n_bins,) array
    tfs : (n_sources, n_bins, 2) array in dB
    """
    all_idx = numpy.arange(hrtf.n_sources, dtype=int)
    tfs = hrtf.tfs_from_sources(all_idx, n_bins=n_bins, ear="both")
    frequencies, _ = hrtf[0].tf(channels=0, n_bins=n_bins, show=False)
    return numpy.asarray(frequencies, dtype=float).squeeze(), numpy.asarray(tfs, dtype=float)


def _bandpass_freq_axis(frequencies, bandwidth):
    freq_idx = numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])
    if not numpy.any(freq_idx):
        raise ValueError(f"No frequency bins found inside bandwidth={bandwidth}.")
    return frequencies[freq_idx], freq_idx


def _frames_from_grid(grid, azimuths, ear):
    """grid shape: (n_azimuths, n_elevations, n_frequencies, 2)."""
    frames = []
    for i, azimuth in enumerate(azimuths):
        frame = {"azimuth": float(azimuth)}
        if ear in ("left", "both"):
            frame["left"] = grid[i, :, :, 0]
        if ear in ("right", "both"):
            frame["right"] = grid[i, :, :, 1]
        frames.append(frame)
    return frames


def _prepare_frames_measured(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    bandwidth=(1000, 18000),
    n_bins=None,
):
    """
    Build one frame per measured vertical-polar azimuth using only actually measured sources.

    Best for datasets recorded in vertical cones.
    For more irregular sphere sampling, rows may be missing for some azimuths.
    """
    source_idx_all = _get_selected_source_idx(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
    )
    vp = hrtf.sources.vertical_polar[source_idx_all].astype(float)

    azimuths_raw_all = _sorted_unique(vp[:, 0])
    elevations = _sorted_unique(vp[:, 1])

    wrapped_to_raw = {}
    for az_raw in azimuths_raw_all:
        az_wrapped = float(_wrap_azimuth([az_raw])[0])
        if az_wrapped not in wrapped_to_raw:
            wrapped_to_raw[az_wrapped] = float(az_raw)

    azimuths = numpy.array(sorted(wrapped_to_raw.keys()), dtype=float)
    azimuths_raw = numpy.array([wrapped_to_raw[az] for az in azimuths], dtype=float)

    if azimuths.size == 0 or elevations.size == 0:
        raise ValueError("No valid azimuths or elevations found in selected source range.")

    frequencies_full, tfs = _get_all_tfs(hrtf, n_bins=n_bins)
    frequencies, freq_idx = _bandpass_freq_axis(frequencies_full, bandwidth)
    tfs = tfs[:, freq_idx, :]

    elev_to_row = {float(elevation): row for row, elevation in enumerate(elevations)}
    frames = []

    for azimuth_raw, azimuth_wrapped in zip(azimuths_raw, azimuths):
        keep_az = numpy.isclose(vp[:, 0], azimuth_raw, atol=1e-6)
        source_idx = source_idx_all[keep_az]
        src_vp = vp[keep_az]

        if len(source_idx) == 0:
            continue

        order = numpy.argsort(src_vp[:, 1])
        source_idx = source_idx[order]
        elev_local = src_vp[order, 1]

        frame = {
            "azimuth": float(azimuth_wrapped),
            "azimuth_raw": float(azimuth_raw),
        }

        for ear_name, chan in (("left", 0), ("right", 1)):
            if ear not in (ear_name, "both"):
                continue
            grid = numpy.full((len(elevations), len(frequencies)), numpy.nan, dtype=float)
            for row_local, elevation in enumerate(elev_local):
                grid[elev_to_row[float(elevation)], :] = tfs[source_idx[row_local], :, chan]
            frame[ear_name] = grid

        frames.append(frame)

    if not frames:
        raise RuntimeError("No animation frames could be created.")

    return frames, azimuths, elevations, frequencies


def _prepare_frames_nearest(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    bandwidth=(1000, 18000),
    azimuth_step=5.0,
    elevation_step=5.0,
    n_bins=None,
):
    """
    Build one frame per azimuth on a fixed grid.
    Fill each grid point with the nearest measured source (in az/el degrees).
    """
    source_idx_subset = _get_selected_source_idx(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
    )

    azimuths, elevations = _make_target_grid(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        azimuth_step=azimuth_step,
        elevation_step=elevation_step,
    )

    frequencies_full, tfs = _get_all_tfs(hrtf, n_bins=n_bins)
    frequencies, freq_idx = _bandpass_freq_axis(frequencies_full, bandwidth)
    tfs = tfs[:, freq_idx, :]

    vp = hrtf.sources.vertical_polar[source_idx_subset].astype(float)
    points = numpy.column_stack([_wrap_azimuth(vp[:, 0]), vp[:, 1]])
    tree = scipy.spatial.cKDTree(points)

    az_grid, el_grid = numpy.meshgrid(azimuths, elevations, indexing="ij")
    targets = numpy.column_stack([az_grid.ravel(), el_grid.ravel()])
    _, nearest = tree.query(targets)
    nearest_idx = source_idx_subset[nearest]

    grid = tfs[nearest_idx].reshape(len(azimuths), len(elevations), len(frequencies), 2)
    frames = _frames_from_grid(grid, azimuths, ear)
    return frames, azimuths, elevations, frequencies


def _prepare_frames_interpolated(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    bandwidth=(1000, 18000),
    azimuth_step=5.0,
    elevation_step=5.0,
    n_bins=None,
):
    """
    Build one frame per azimuth on a fixed az/el grid using barycentric interpolation
    of the dB magnitude spectra (same principle as slab.HRTF.interpolate with
    method='barycentric', but vectorized: one Delaunay triangulation of all source
    positions, then scipy find_simplex + barycentric weights for the whole grid).

    Smoothest option; most robust for irregular / uniform sphere sampling.
    """
    azimuths, elevations = _make_target_grid(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        azimuth_step=azimuth_step,
        elevation_step=elevation_step,
    )

    frequencies_full, tfs = _get_all_tfs(hrtf, n_bins=n_bins)
    frequencies, freq_idx = _bandpass_freq_axis(frequencies_full, bandwidth)
    tfs = tfs[:, freq_idx, :]

    # triangulate all source positions in (azimuth, elevation) space;
    # replicate at +-360 deg azimuth so interpolation is seamless across the wrap
    vp = hrtf.sources.vertical_polar.astype(float)
    points = numpy.column_stack([_wrap_azimuth(vp[:, 0]), vp[:, 1]])
    points_ext = numpy.vstack([
        points,
        points + numpy.array([360.0, 0.0]),
        points - numpy.array([360.0, 0.0]),
    ])
    tfs_ext = numpy.vstack([tfs, tfs, tfs])
    tri = scipy.spatial.Delaunay(points_ext)

    az_grid, el_grid = numpy.meshgrid(azimuths, elevations, indexing="ij")
    targets = numpy.column_stack([az_grid.ravel(), el_grid.ravel()])

    simplex = tri.find_simplex(targets)
    inside = simplex >= 0

    interpolated = numpy.empty((len(targets), len(frequencies), 2), dtype=float)

    if numpy.any(inside):
        transform = tri.transform[simplex[inside]]
        offset = targets[inside] - transform[:, 2]
        bary = numpy.einsum("nij,nj->ni", transform[:, :2, :], offset)
        weights = numpy.column_stack([bary, 1.0 - bary.sum(axis=1)])
        vertices = tri.simplices[simplex[inside]]
        # weighted average of dB spectra of the 3 enclosing sources
        interpolated[inside] = numpy.einsum("nk,nkfc->nfc", weights, tfs_ext[vertices])

    if numpy.any(~inside):
        # outside the convex hull of measured sources: fall back to nearest source
        tree = scipy.spatial.cKDTree(points_ext)
        _, nearest = tree.query(targets[~inside])
        interpolated[~inside] = tfs_ext[nearest]

    grid = interpolated.reshape(len(azimuths), len(elevations), len(frequencies), 2)
    frames = _frames_from_grid(grid, azimuths, ear)
    return frames, azimuths, elevations, frequencies


def _prepare_frames(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    bandwidth=(1000, 18000),
    sampling_mode="interpolate",
    azimuth_step=5.0,
    elevation_step=5.0,
    n_bins=None,
):
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")
    if sampling_mode == "measured":
        return _prepare_frames_measured(
            hrtf=hrtf,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            ear=ear,
            bandwidth=bandwidth,
            n_bins=n_bins,
        )
    if sampling_mode == "nearest":
        return _prepare_frames_nearest(
            hrtf=hrtf,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            ear=ear,
            bandwidth=bandwidth,
            azimuth_step=azimuth_step,
            elevation_step=elevation_step,
            n_bins=n_bins,
        )
    if sampling_mode == "interpolate":
        return _prepare_frames_interpolated(
            hrtf=hrtf,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            ear=ear,
            bandwidth=bandwidth,
            azimuth_step=azimuth_step,
            elevation_step=elevation_step,
            n_bins=n_bins,
        )
    raise ValueError("sampling_mode must be 'measured', 'nearest', or 'interpolate'.")


def _compute_color_limits(frames, ear):
    values = []

    for frame in frames:
        if ear in ("left", "both"):
            values.append(frame["left"][numpy.isfinite(frame["left"])])
        if ear in ("right", "both"):
            values.append(frame["right"][numpy.isfinite(frame["right"])])

    values = [v for v in values if v.size > 0]
    if not values:
        raise RuntimeError("Could not determine color limits because all values are NaN.")

    all_values = numpy.concatenate(values)
    z_min = float(numpy.floor(numpy.nanmin(all_values)))
    z_max = float(numpy.ceil(numpy.nanmax(all_values)))

    if numpy.isclose(z_min, z_max):
        z_min -= 0.5
        z_max += 0.5

    return z_min, z_max


def _update_contourf(ax, contour, frequencies, elevations, data, levels, vmin=None, vmax=None):
    contour.remove()
    contour = ax.contourf(
        frequencies,
        elevations,
        data,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
    )
    return contour


def _plot_waterfall(ax, frequencies, elevations, data, line_separation=20.0):
    ax.clear()
    offsets = numpy.arange(len(elevations)) * line_separation
    artists = []

    for row, spectrum in enumerate(data):
        if numpy.all(numpy.isnan(spectrum)):
            continue
        line = ax.plot(
            frequencies,
            spectrum + offsets[row],
            linewidth=0.75,
            color="0.0",
            alpha=0.7,
        )
        artists.extend(line)

    tick_idx = numpy.arange(0, len(elevations), 2)
    ax.set_yticks(offsets[tick_idx])
    ax.set_yticklabels(elevations[tick_idx].astype(int))
    ax.grid(visible=True, axis="y", which="both", linewidth=0.25)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Elevation (deg)")

    if len(offsets) > 0:
        x0 = frequencies[0] + 0.03 * (frequencies[-1] - frequencies[0])
        y0 = offsets[-1] + 10
        ax.plot([x0, x0], [y0, y0 + line_separation], linewidth=1, color="0.0", alpha=0.9)
        ax.text(
            x0 + 0.01 * (frequencies[-1] - frequencies[0]),
            y0 + line_separation / 2,
            f"{int(line_separation)} dB",
            va="center",
            ha="left",
            fontsize=7,
            alpha=0.8,
        )

    return artists


def hrtf_animation(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    kind="image",
    interval=100,
    bandwidth=(1000, 18000),
    sampling_mode="interpolate",
    azimuth_step=5.0,
    elevation_step=5.0,
    n_bins=None,
    filename=None,
    write=False,
    show=True,
    figsize=(8, 6),
    dpi=200,
):
    """
    Animate vertical DTF magnitude slices across azimuth.

    Parameters
    ----------
    hrtf
        HRTF object whose stored TFs are already DTF magnitudes.
    azimuth_range : tuple | float | None
    elevation_range : tuple | float | None
    ear : {'left', 'right', 'both'}
    kind : {'image', 'waterfall'}
    interval : int
        Frame interval in ms.
    bandwidth : tuple
        Frequency range in Hz.
    sampling_mode : {'measured', 'nearest', 'interpolate'}
        measured:
            use actually recorded sources at each azimuth slice
        nearest:
            fixed grid, nearest measured source at each grid point
        interpolate:
            fixed grid, barycentric interpolation of dB spectra (smoothest)
    azimuth_step : float
        Used for nearest/interpolate grid.
    elevation_step : float
        Used for nearest/interpolate grid.
    n_bins : int | None
        Number of frequency bins (spectral resolution). None uses the native
        resolution of the impulse responses, which can look blocky; 512 is smooth.
    filename : str | None
        Base filename for saving.
    write : bool | str
        False: do not save
        True / 'auto' / 'ffmpeg': save mp4
        'pillow': save gif
    show : bool
    figsize : tuple
    dpi : int
        Output resolution when writing to file.
    """
    if kind not in ("image", "waterfall"):
        raise ValueError("kind must be 'image' or 'waterfall'.")
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")

    source_azimuths = numpy.unique(
        numpy.round(_wrap_azimuth(hrtf.sources.vertical_polar[:, 0].astype(float)), 3)
    )
    if source_azimuths.size < 2:
        raise ValueError(
            f"HRTF contains only a single source azimuth ({source_azimuths[0]:g} deg) - "
            "there is no azimuth variation to animate; every frame would be identical. "
            "Use a recording with multiple azimuths (e.g. kemar_test.sofa, FABIAN.sofa)."
        )

    frames, azimuths, elevations, frequencies = _prepare_frames(
        hrtf=hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        ear=ear,
        bandwidth=bandwidth,
        sampling_mode=sampling_mode,
        azimuth_step=azimuth_step,
        elevation_step=elevation_step,
        n_bins=n_bins,
    )

    if ear == "both":
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(figsize[0] * 1.8, figsize[1]),
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if kind == "image":
        vmin, vmax = _compute_color_limits(frames, ear)
        levels = numpy.linspace(vmin, vmax, 120)

        if ear == "both":
            contour_left = ax_left.contourf(
                frequencies,
                elevations,
                frames[0]["left"],
                levels=levels,
                vmin=vmin,
                vmax=vmax,
            )
            contour_right = ax_right.contourf(
                frequencies,
                elevations,
                frames[0]["right"],
                levels=levels,
                vmin=vmin,
                vmax=vmax,
            )

            ax_left.set_xlabel("Frequency (Hz)")
            ax_left.set_ylabel("Elevation (deg)")
            ax_right.set_xlabel("Frequency (Hz)")
            ax_right.set_ylabel("Elevation (deg)")
            ax_left.set_title(f"Left ear")
            ax_right.set_title(f"Right ear")
            fig.suptitle(f"azimuth {frames[0]['azimuth']:g}°")

            cbar = fig.colorbar(contour_left, ax=[ax_left, ax_right], shrink=0.95)
            cbar.set_label("DTF magnitude (dB)")

            def animate(frame_idx):
                nonlocal contour_left, contour_right

                frame = frames[frame_idx]
                azimuth = frame["azimuth"]

                contour_left = _update_contourf(
                    ax_left,
                    contour_left,
                    frequencies,
                    elevations,
                    frame["left"],
                    levels,
                    vmin,
                    vmax,
                )
                contour_right = _update_contourf(
                    ax_right,
                    contour_right,
                    frequencies,
                    elevations,
                    frame["right"],
                    levels,
                    vmin,
                    vmax,
                )


                ax_left.set_title(f"Left ear")
                ax_right.set_title(f"Right ear")
                fig.suptitle(f"azimuth {azimuth:g}°")
                return []

        else:
            first_data = frames[0]["left"] if ear == "left" else frames[0]["right"]
            contour = ax.contourf(
                frequencies,
                elevations,
                first_data,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Elevation (deg)")
            ax.set_title(f"Azimuth {frames[0]['azimuth']:g}°")

            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("DTF magnitude (dB)")

            def animate(frame_idx):
                nonlocal contour

                frame = frames[frame_idx]
                azimuth = frame["azimuth"]
                data = frame["left"] if ear == "left" else frame["right"]

                contour = _update_contourf(
                    ax,
                    contour,
                    frequencies,
                    elevations,
                    data,
                    levels,
                    vmin,
                    vmax,
                )
                ax.set_title(f"Azimuth {azimuth:g}°")
                return []

    else:
        def animate(frame_idx):
            frame = frames[frame_idx]
            azimuth = frame["azimuth"]

            if ear == "both":
                artists = []
                artists.extend(_plot_waterfall(ax_left, frequencies, elevations, frame["left"]))
                artists.extend(_plot_waterfall(ax_right, frequencies, elevations, frame["right"]))
                ax_left.set_title(f"Left ear – azimuth {azimuth:g}°")
                ax_right.set_title(f"Right ear – azimuth {azimuth:g}°")
                return artists

            if ear == "left":
                artists = _plot_waterfall(ax, frequencies, elevations, frame["left"])
            else:
                artists = _plot_waterfall(ax, frequencies, elevations, frame["right"])
            ax.set_title(f"Azimuth {azimuth:g}°")
            return artists

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=interval,
        blit=False,
    )

    if write:
        if filename is None:
            raise ValueError("filename must be provided when write is enabled.")

        out_dir = paths.ANI_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        mode = "auto" if write is True else str(write).lower()
        fps = max(1, int(round(1000 / interval)))

        if mode in ("auto", "ffmpeg"):
            out_path = out_dir / f"{filename}.mp4"
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
            ani.save(out_path, writer=writer, dpi=dpi)
            print(f"Saved animation to: {out_path}")
        elif mode == "pillow":
            out_path = out_dir / f"{filename}.gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(out_path, writer=writer, dpi=dpi)
            print(f"Saved animation to: {out_path}")
        else:
            raise ValueError("write must be False, True, 'ffmpeg', or 'pillow'.")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani, frames


def main(hrtf_id):
    hrtf = slab.HRTF(hrtf_dir / f"{hrtf_id}.sofa")

    ani, frames = hrtf_animation(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        ear=ear,
        kind=kind,
        bandwidth=bandwidth,
        sampling_mode=sampling_mode,
        azimuth_step=azimuth_step,
        elevation_step=elevation_step,
        n_bins=n_bins,
        interval=interval,
        filename=f"{hrtf_id}_animation",
        write=write,
        show=show,
        dpi=dpi,
    )
    return ani, frames


if __name__ == "__main__":
    ani, frames = main(hrtf_id)
