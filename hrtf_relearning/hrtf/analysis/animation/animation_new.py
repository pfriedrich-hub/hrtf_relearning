import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import slab
from pathlib import Path

from hrtf_relearning import PATH

hrtf_dir = PATH / "data" / "hrtf" / "sofa"

# ---------------------------------------------------------------------------
# Parameters – edit here
# ---------------------------------------------------------------------------
hrtf_id         = "universal"   # HRTF file stem (loads <hrtf_id>.sofa)

azimuth_range   = (-50, 50)     # (min, max) in degrees; None = all
elevation_range = (-40, 40)     # (min, max) in degrees; None = all
ear             = "both"        # "left", "right", or "both"
kind            = "image"   # "image" (contourf) or "waterfall"
bandwidth       = (1000, 18000) # frequency range in Hz
sampling_mode   = "nearest" # "measured", "nearest", or "interpolate"
azimuth_step    = 5.0           # grid step in degrees (nearest/interpolate)
elevation_step  = 2.0           # grid step in degrees (nearest/interpolate)
n_bins          = None          # frequency resolution for interpolation; None = default
interval        = 220           # frame interval in ms
write           = True          # False | True/'ffmpeg' (mp4) | 'pillow' (gif)
show            = True          # display the animation window
# ---------------------------------------------------------------------------


def _sorted_unique(values):
    return numpy.unique(numpy.asarray(values))


def _wrap_azimuth(values):
    values = numpy.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def _get_frequency_axis_from_hrtf(hrtf, bandwidth):
    frequencies = hrtf[0].tf(show=False)[0]
    freq_idx = numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])
    if not numpy.any(freq_idx):
        raise ValueError(f"No frequency bins found inside bandwidth={bandwidth}.")
    return frequencies[freq_idx], freq_idx


def _tf_from_filter(filt, ear="left", n_bins=None):
    if ear == "left":
        chan = 0
    elif ear == "right":
        chan = 1
    else:
        raise ValueError("ear must be 'left' or 'right'.")

    frequencies, tf = filt.tf(channels=chan, n_bins=n_bins, show=False)
    tf = numpy.asarray(tf).squeeze()
    return frequencies, tf


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


def _nearest_source_idx(hrtf, azimuth, elevation, source_idx_subset=None):
    vp = hrtf.sources.vertical_polar.astype(float)

    if source_idx_subset is None:
        source_idx_subset = numpy.arange(hrtf.n_sources, dtype=int)
    else:
        source_idx_subset = numpy.asarray(source_idx_subset, dtype=int)

    vp_sub = vp[source_idx_subset]
    az = _wrap_azimuth(vp_sub[:, 0])
    el = vp_sub[:, 1]

    daz = _wrap_azimuth(az - float(azimuth))
    delv = el - float(elevation)

    dist = numpy.sqrt(daz**2 + delv**2)
    return int(source_idx_subset[numpy.argmin(dist)])


def _prepare_frames_measured(
    hrtf,
    azimuth_range=None,
    elevation_range=None,
    ear="left",
    bandwidth=(1000, 18000),
):
    """
    Build one frame per measured vertical-polar azimuth using only actually measured sources.

    Best for datasets recorded in vertical cones.
    For more irregular sphere sampling, rows may be missing for some azimuths.
    """
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")

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

    frequencies, freq_idx = _get_frequency_axis_from_hrtf(hrtf, bandwidth)
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
        src_vp = src_vp[order]
        elev_local = src_vp[:, 1]

        frame = {
            "azimuth": float(azimuth_wrapped),
            "azimuth_raw": float(azimuth_raw),
        }

        if ear in ("left", "both"):
            data_left = hrtf.tfs_from_sources(source_idx, ear="left", n_bins=None)
            data_left = data_left.reshape(data_left.shape[:2])[:, freq_idx]

            grid_left = numpy.full((len(elevations), len(frequencies)), numpy.nan, dtype=float)
            for row_local, elevation in enumerate(elev_local):
                row = elev_to_row[float(elevation)]
                grid_left[row, :] = data_left[row_local, :]
            frame["left"] = grid_left

        if ear in ("right", "both"):
            data_right = hrtf.tfs_from_sources(source_idx, ear="right", n_bins=None)
            data_right = data_right.reshape(data_right.shape[:2])[:, freq_idx]

            grid_right = numpy.full((len(elevations), len(frequencies)), numpy.nan, dtype=float)
            for row_local, elevation in enumerate(elev_local):
                row = elev_to_row[float(elevation)]
                grid_right[row, :] = data_right[row_local, :]
            frame["right"] = grid_right

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
):
    """
    Build one frame per azimuth on a fixed grid.
    Fill each grid point with the nearest measured source.
    """
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")

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

    frequencies, freq_idx = _get_frequency_axis_from_hrtf(hrtf, bandwidth)

    frames = []

    for azimuth in azimuths:
        frame = {"azimuth": float(azimuth)}

        nearest_indices = [
            _nearest_source_idx(
                hrtf,
                azimuth=azimuth,
                elevation=elevation,
                source_idx_subset=source_idx_subset,
            )
            for elevation in elevations
        ]

        nearest_indices = numpy.asarray(nearest_indices, dtype=int)

        if ear in ("left", "both"):
            data_left = hrtf.tfs_from_sources(nearest_indices, ear="left", n_bins=None)
            data_left = data_left.reshape(data_left.shape[:2])[:, freq_idx]
            frame["left"] = data_left.astype(float)

        if ear in ("right", "both"):
            data_right = hrtf.tfs_from_sources(nearest_indices, ear="right", n_bins=None)
            data_right = data_right.reshape(data_right.shape[:2])[:, freq_idx]
            frame["right"] = data_right.astype(float)

        frames.append(frame)

    if not frames:
        raise RuntimeError("No animation frames could be created.")

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
    Build one frame per azimuth on a fixed elevation grid using HRTF interpolation.

    Most robust option for irregular / uniform sphere sampling.
    """
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")

    azimuths, elevations = _make_target_grid(
        hrtf,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        azimuth_step=azimuth_step,
        elevation_step=elevation_step,
    )

    test_filt = hrtf.interpolate(
        azimuth=float(azimuths[0]),
        elevation=float(elevations[0]),
    )

    test_ear = "left" if ear in ("left", "both") else "right"
    frequencies_full, _ = _tf_from_filter(test_filt, ear=test_ear, n_bins=n_bins)

    freq_idx = numpy.logical_and(
        frequencies_full >= bandwidth[0],
        frequencies_full <= bandwidth[1],
    )
    if not numpy.any(freq_idx):
        raise ValueError(f"No frequency bins found inside bandwidth={bandwidth}.")
    frequencies = frequencies_full[freq_idx]

    frames = []

    for azimuth in azimuths:
        frame = {"azimuth": float(azimuth)}

        if ear in ("left", "both"):
            grid_left = numpy.zeros((len(elevations), len(frequencies)), dtype=float)
            for row, elevation in enumerate(elevations):
                filt = hrtf.interpolate(
                    azimuth=float(azimuth),
                    elevation=float(elevation),
                )
                _, tf_left = _tf_from_filter(filt, ear="left", n_bins=n_bins)
                grid_left[row, :] = tf_left[freq_idx]
            frame["left"] = grid_left

        if ear in ("right", "both"):
            grid_right = numpy.zeros((len(elevations), len(frequencies)), dtype=float)
            for row, elevation in enumerate(elevations):
                filt = hrtf.interpolate(
                    azimuth=float(azimuth),
                    elevation=float(elevation),
                )
                _, tf_right = _tf_from_filter(filt, ear="right", n_bins=n_bins)
                grid_right[row, :] = tf_right[freq_idx]
            frame["right"] = grid_right

        frames.append(frame)

    if not frames:
        raise RuntimeError("No animation frames could be created.")

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
    if sampling_mode == "measured":
        return _prepare_frames_measured(
            hrtf=hrtf,
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            ear=ear,
            bandwidth=bandwidth,
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
            fixed grid, interpolate HRTF at each grid point
    azimuth_step : float
        Used for nearest/interpolate grid.
    elevation_step : float
        Used for nearest/interpolate grid.
    n_bins : int | None
        Frequency resolution for interpolation mode.
    filename : str | None
        Base filename for saving.
    write : bool | str
        False: do not save
        True / 'auto' / 'ffmpeg': save mp4
        'pillow': save gif
    show : bool
    figsize : tuple
    """
    if kind not in ("image", "waterfall"):
        raise ValueError("kind must be 'image' or 'waterfall'.")
    if ear not in ("left", "right", "both"):
        raise ValueError("ear must be 'left', 'right', or 'both'.")

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
            ax_left.set_title(f"Left ear – azimuth {frames[0]['azimuth']:g}°")
            ax_right.set_title(f"Right ear – azimuth {frames[0]['azimuth']:g}°")

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

                ax_left.set_title(f"Left ear – azimuth {azimuth:g}°")
                ax_right.set_title(f"Right ear – azimuth {azimuth:g}°")
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

        out_dir = PATH / "data" / "img" / "animations"
        out_dir.mkdir(parents=True, exist_ok=True)

        mode = "auto" if write is True else str(write).lower()
        fps = max(1, int(round(1000 / interval)))

        if mode in ("auto", "ffmpeg"):
            out_path = out_dir / f"{filename}.mp4"
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(out_path, writer=writer)
            print(f"Saved animation to: {out_path}")
        elif mode == "pillow":
            out_path = out_dir / f"{filename}.gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(out_path, writer=writer)
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
    )
    return ani, frames


if __name__ == "__main__":
    ani, frames = main(hrtf_id)