import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy
import scipy
import logging

def localization_accuracy(sequence):
    if sequence.this_n == -1 or sequence.n_remaining == len(sequence.data) or not sequence.data:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]

    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    azimuth_gain, n = scipy.stats.linregress(targets[:, 0], responses[:, 0])[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    az_rmse, ele_rmse = rmse[0], rmse[1]
    variability = compute_sector_precision(targets, responses, sequence.sector_centers, sequence.settings['sector_size'])
    az_sd, ele_sd = variability[0], variability[1]
    return elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd


def compute_sector_precision(targets, responses, sector_centers, sector_size):
    """
    Estimates response precision per sector by aligning targets and measuring
    the spread (std) of aligned responses, then averaging over sectors.

    Parameters:
    - targets: Nx2 array (azimuth, elevation)
    - responses: Nx2 array (azimuth, elevation)
    - sector_centers: Mx2 array of sector center coordinates
    - sector_size: tuple (az_size, el_size)

    Returns:
    - per_sector_std: list of (azimuth_std, elevation_std) for each sector
    - mean_std: tuple of (mean_azimuth_std, mean_elevation_std)
    """
    az_size, el_size = sector_size
    per_sector_std = []
    for center in sector_centers:
        # Define bounds of the current sector
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2
        # Get indices of targets in this sector
        in_sector = numpy.where((targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max))[0]
        if len(in_sector) >= 2:
            # Shift targets and responses so that all targets align at origin
            response_shift = responses[in_sector] - targets[in_sector]
            az_std = numpy.std(response_shift[:, 0])
            el_std = numpy.std(response_shift[:, 1])
            per_sector_std.append((az_std, el_std))
    # Compute mean std across sectors
    if per_sector_std:
        per_sector_std = numpy.array(per_sector_std)
        mean_std = tuple(numpy.mean(per_sector_std, axis=0))
    else:
        mean_std = (numpy.nan, numpy.nan)
    return mean_std


def _wrap_diff_deg(a, b):
    """Smallest signed difference a-b on a 360° circle, result in [-180, 180)."""
    return (numpy.asarray(a) - numpy.asarray(b) + 180.0) % 360.0 - 180.0


def target_p(sequence, show=False, axis=None):
    """
    Compute per-sector error and target probabilities from a localization run.

    Returns
    -------
    response_errors : (N_sectors, 4) array
        columns: [sector_center_az, sector_center_el, polar_error, probability]
    """
    if not sequence:
        logging.debug('No sequence found')
        return None
    if not hasattr(sequence, "settings"):
        raise AttributeError("sequence must have a 'settings' dict.")
    if not hasattr(sequence, "sector_centers"):
        raise AttributeError("sequence must have 'sector_centers'.")

    settings = sequence.settings
    az_size, el_size = settings['sector_size']
    half_az, half_el = az_size / 2.0, el_size / 2.0
    centers = numpy.asarray(sequence.sector_centers, dtype=float)  # (N,2)
    # --- unpack data (targets = 2nd row, responses = 1st row) ---
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    # --- assign each target to exactly one sector (nearest center within box) ---
    # Compute deltas of each target to every center
    d_az = _wrap_diff_deg(targets[:, None, 0], centers[None, :, 0])  # (T,N)
    d_el = targets[:, None, 1] - centers[None, :, 1]                # (T,N)
    # inside rectangular box?
    inside = (numpy.abs(d_az) <= half_az) & (numpy.abs(d_el) <= half_el)  # (T,N)
    # If multiple sectors match (edge cases), pick the nearest; if none match,
    # pick the nearest anyway (prevents drops due to rounding).
    rect_dist = numpy.stack([numpy.abs(d_az) / half_az, numpy.abs(d_el) / half_el], axis=-1)  # (T,N,2)
    rect_dist = numpy.linalg.norm(rect_dist, axis=-1)  # normalized rectangular distance (T,N)
    # Prefer valid-inside sectors; if none, fall back to absolute nearest
    big = 1e6
    choice_inside = numpy.where(inside, rect_dist, big)
    idx_inside = numpy.argmin(choice_inside, axis=1)  # (T,)
    none_inside = ~inside.any(axis=1)
    if numpy.any(none_inside):
        # fall back to nearest by rect_dist for those
        idx_fallback = numpy.argmin(rect_dist[none_inside], axis=1)
        idx_inside[none_inside] = idx_fallback
    # Now we have, for each trial, the chosen sector index
    T = targets.shape[0]
    N = centers.shape[0]
    # --- aggregate per-sector errors ---
    response_errors = numpy.zeros((N, 3), dtype=float)
    # Prepare lists of trial indices per sector
    trials_per_sector = [[] for _ in range(N)]
    for t in range(T):
        s = int(idx_inside[t])
        trials_per_sector[s].append(t)
    # Compute RMSE in az/el, then use polar (el) RMSE as your training metric
    for s, center in enumerate(centers):
        idxs = trials_per_sector[s]
        if len(idxs) == 0:
            rmse_el = 0.0
        else:
            tgt_s = targets[idxs]        # (#,2)
            rsp_s = responses[idxs]      # (#,2)
            # Polar (elevation) error only, per your definition
            el_err = tgt_s[:, 1] - rsp_s[:, 1]
            rmse_el = float(numpy.sqrt(numpy.mean(el_err ** 2)))
        response_errors[s, :] = [center[0], center[1], rmse_el]
    # --- probabilities proportional to polar error (handle all-zero safely) ---
    pe = response_errors[:, 2]
    total = float(pe.sum())
    if total <= 0:
        probs = numpy.full_like(pe, 1.0 / len(pe))
    else:
        probs = pe / total
    response_errors = numpy.column_stack([response_errors, probs])  # (N,4)
    # --- optional heatmap ---
    if show:
        # make regular grids of unique az/el from centers
        az_vals = numpy.unique(centers[:, 0])
        el_vals = numpy.unique(centers[:, 1])
        # map each sector to its grid cell
        P = numpy.zeros((len(el_vals), len(az_vals)))
        for row in response_errors:
            az, el, _, p = row
            xi = numpy.where(az_vals == az)[0][0]
            yi = numpy.where(el_vals == el)[0][0]
            P[yi, xi] = p
        if axis is None:
            fig, axis = plt.subplots(figsize=(7, 5))
        mesh = axis.pcolormesh(az_vals, el_vals, P, shading='auto')
        cbar = plt.colorbar(mesh, ax=axis)
        cbar.set_label('Probability')
        # ticks aligned to sector edges
        axis.set_xlabel('Azimuth (°)')
        axis.set_ylabel('Elevation (°)')
        axis.set_xticks(az_vals)
        axis.set_yticks(el_vals)
        axis.grid(True, linestyle='--', linewidth=0.4)
        # optional: draw sector boxes lightly
        for (caz, cel) in centers:
            axis.add_patch(
                plt.Rectangle((caz - half_az, cel - half_el),
                              az_size, el_size, fill=False, linestyle='--', linewidth=0.6, alpha=0.7)
            )
        axis.set_title('Per-sector training probability (from polar RMSE)')
    return response_errors


def plot_localization(sequence, report_stats=['elevation', 'azimuth'], filepath=None):
    """
    Plots representative mean responses by aligning targets,
    connects them in a grid, and shows trimmed sector center lines only across actual field.
    """
    if sequence.this_n == -1 or sequence.n_remaining == len(sequence.data) or not sequence.data:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    sector_centers = sequence.sector_centers
    az_size, el_size = sequence.settings['sector_size']
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(sequence)

    mean_responses = []
    center_grid = {}
    # get targets and responses in each sector
    for center in sector_centers:
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2
        in_sector = numpy.where((targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max))[0]
        if len(in_sector) == 0:
            continue
        # for each sector, calculate mean vector across target-response pairs #todo test this
        response_shift = responses[in_sector] - targets[in_sector]
        mean_shift = numpy.mean(response_shift, axis=0)
        representative_response = center + mean_shift
        mean_responses.append(representative_response)
        center_grid[tuple(center)] = representative_response

    mean_responses = numpy.array(mean_responses)

    # Axis setup
    az_vals = sorted(set([c[0] for c in sector_centers]))
    el_vals = sorted(set([c[1] for c in sector_centers]))
    az_min = min(az_vals) - az_size - 5
    az_max = max(az_vals) + az_size + 5
    el_min = min(el_vals) - el_size - 5
    el_max = max(el_vals) + el_size + 5

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(az_min, az_max)
    ax.set_ylim(el_min, el_max)
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    title = sequence.name
    if 'elevation' in report_stats:
        title += f"\nEG: {eg:.2f}, RMSE: {ele_rmse:.2f}, SD: {ele_sd:.2f}"
    if 'azimuth' in report_stats:
        title += f"\nAG: {ag:.2f}, az RMSE: {az_rmse:.2f}, az SD: {az_sd:.2f}"
    ax.set_title(title)
    ax.grid(False)

    # Draw trimmed sector center lines
    for x in az_vals:
        ax.plot([x, x], [min(el_vals), max(el_vals)], color='gray', linestyle='-', linewidth=1)
    for y in el_vals:
        ax.plot([min(az_vals), max(az_vals)], [y, y], color='gray', linestyle='-', linewidth=1)

    # Plot mean responses
    ax.plot(mean_responses[:, 0], mean_responses[:, 1], 'ko', markersize=6)

    # Connect mean responses in grid layout
    sector_lookup = {tuple(sc): center_grid[tuple(sc)] for sc in sector_centers if tuple(sc) in center_grid}
    for el in el_vals:
        row = [sector_lookup[(az, el)] for az in az_vals if (az, el) in sector_lookup]
        if len(row) > 1:
            ax.plot([p[0] for p in row], [p[1] for p in row], 'k-', linewidth=2)
    for az in az_vals:
        col = [sector_lookup[(az, el)] for el in el_vals if (az, el) in sector_lookup]
        if len(col) > 1:
            ax.plot([p[0] for p in col], [p[1] for p in col], 'k-', linewidth=2)
    plt.tight_layout()
    if filepath:
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / f'{sequence.name}.png')

    plt.show()