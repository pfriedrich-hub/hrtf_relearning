import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import slab
import numpy
import random
import logging


def make_sequence_from_sources(settings, hrir_sources=None):
    """
    Picks random targets from the HRIR source space within sectors,
    while ensuring a minimum distance between successive sectors.

    Parameters:
    - azimuth_range: tuple (min_azimuth, max_azimuth)
    - elevation_range: tuple (min_elevation, max_elevation)
    - sector_size: tuple (azimuth_size, elevation_size) in degrees
    - min_distance: minimum distance between successive sectors in sequence
    - targets_per_sector: number of points per sector
    - hrir_sources: list of HRIR sources (hrir.sources.vertical_polar)

    Returns:
        slab.Trialsequence
    """
    azimuth_range = settings['azimuth_range']
    elevation_range = settings['elevation_range']
    sector_size = settings['sector_size']  # (azimuth_size, elevation_size)
    targets_per_sector = settings['targets_per_sector']
    min_distance = settings['min_distance']
    azimuth_size, elevation_size = sector_size
    num_azimuth_sectors = (azimuth_range[1] - azimuth_range[0]) // azimuth_size

    if elevation_range == (0,0):  # only place targets on the horizontal plane
        sector_centers = [
            (azimuth_range[0] + (i + 0.5) * azimuth_size, 0)
            for i in range(num_azimuth_sectors)
        ]
        random.shuffle(sector_centers)
        num_sectors = num_azimuth_sectors
    else:
        # Compute sector centers
        num_elevation_sectors = (elevation_range[1] - elevation_range[0]) // elevation_size
        num_sectors = num_azimuth_sectors * num_elevation_sectors
        sector_centers = [
            (azimuth_range[0] + (i + 0.5) * azimuth_size, elevation_range[0] + (j + 0.5) * elevation_size)
            for i in range(num_azimuth_sectors) for j in range(num_elevation_sectors)
        ]
        random.shuffle(sector_centers)

    # Select sectors ensuring minimum distance constraint
    selected_sectors = []
    remaining_sectors = sector_centers[:]
    while len(selected_sectors) < num_sectors:
        if not selected_sectors:
            selected_sectors.append(remaining_sectors.pop(0))
        else:
            last_sector = selected_sectors[-1]
            valid_sectors = [
                sec for sec in remaining_sectors
                if numpy.linalg.norm(numpy.array(sec) - numpy.array(last_sector)) >= min_distance
            ]
            if valid_sectors:
                selected_sector = valid_sectors.pop(0)
                selected_sectors.append(selected_sector)
                remaining_sectors.remove(selected_sector)
            else:
                logging.error('Can not create target sequence with given settings. '
                              'Check min distance and target range.')
                break  # Stop if no valid sector is found

    # pick random targets from the hrir sources within each selected sector
    src_az = numpy.asarray(hrir_sources[:, 0], dtype=float)
    src_el = numpy.asarray(hrir_sources[:, 1], dtype=float)

    def _wrap_diff(a, b):
        """Smallest signed difference a-b on a 360° circle, result in [-180, 180)."""
        d = (a - b + 180.0) % 360.0 - 180.0
        return d

    def sources_in_sector(center_az, center_el):
        """
        Return indices of sources that fall inside the az/el rectangular sector
        centered at (center_az, center_el) with half-sizes azimuth_size/2, elevation_size/2.
        Azimuth is treated circularly (0..360). Elevation is linear.
        """
        # Azimuth within bounds (circular)
        d_az = numpy.abs(_wrap_diff(src_az, center_az))
        az_ok = d_az <= (azimuth_size / 2.0)
        # Elevation within bounds (linear)
        el_min = center_el - (elevation_size / 2.0)
        el_max = center_el + (elevation_size / 2.0)
        el_ok = (src_el >= el_min) & (src_el <= el_max)
        return numpy.nonzero(az_ok & el_ok)[0]

    # Build candidate lists per sector (indices into the sources grid)
    sector_candidates = []
    for (caz, cel) in selected_sectors:
        idx = sources_in_sector(caz, cel)
        if idx.size < targets_per_sector:
            logging.error(
                f'Not enough sources in sector centered at (az={caz:.2f}, el={cel:.2f}). '
                f'Required {targets_per_sector}, found {idx.size}. '
                'Consider enlarging sector_size, adjusting ranges, or lowering targets_per_sector.'
            )
            raise ValueError('Insufficient sources in one or more sectors for requested targets_per_sector.')
        # shuffle for randomness
        idx = numpy.random.permutation(idx).tolist()
        sector_candidates.append(idx)

    # Global-unique sampling across sectors (round-robin)
    num_sectors = len(selected_sectors)
    T = targets_per_sector
    used = set()
    sector_picks = [[] for _ in range(num_sectors)]

    for t in range(T):  # pass t: give each sector 1 pick
        for s in range(num_sectors):
            cand = sector_candidates[s]
            # advance until we find an unused index or run out
            while cand and cand[-1] in used:
                cand.pop()
            if not cand:
                # No unused left for this sector → cannot satisfy global uniqueness
                caz, cel = selected_sectors[s]
                logging.error(
                    f'Global-unique sampling failed for sector (az={caz:.2f}, el={cel:.2f}) '
                    f'at pass {t + 1}/{T}. Try increasing sector_size or lowering targets_per_sector.'
                )
                raise ValueError('Not enough unique grid points across overlapping sectors.')
            pick = cand.pop()  # take one unused
            used.add(pick)
            sector_picks[s].append(pick)

    # Interleave to preserve min-distance between consecutive sectors
    chosen_indices = []
    for t in range(T):
        for s in range(num_sectors):
            chosen_indices.append(sector_picks[s][t])

    # Build (az, el) points from unique picks
    points = numpy.column_stack([src_az[chosen_indices], src_el[chosen_indices]])
    points = numpy.round(points, 2)  # only keep 2nd decimal
    points[:, 0] = (points[:, 0] + 180) % 360 - 180  # wrap to (-180,180)
    sequence = slab.Trialsequence(points)
    sequence.sector_centers = sector_centers
    sequence.settings = settings  # ← store all parameters here
    return sequence

def make_sequence(settings):
    """
    Generates uniformly random points within sectors while ensuring a minimum distance between successive sectors.

    Parameters:
    - azimuth_range: tuple (min_azimuth, max_azimuth)
    - elevation_range: tuple (min_elevation, max_elevation)
    - sector_size: tuple (azimuth_size, elevation_size) in degrees
    - min_distance: minimum distance between successive sectors in sequence
    - targets_per_sector: number of points per sector

    Returns:
    - points: List of (azimuth, elevation) tuples
    - selected_sectors: List of selected sector centers
    """
    azimuth_range = settings['azimuth_range']
    elevation_range = settings['elevation_range']
    sector_size = settings['sector_size']  # (azimuth_size, elevation_size)
    targets_per_sector = settings['targets_per_sector']
    min_distance = settings['min_distance']
    azimuth_size, elevation_size = sector_size
    num_azimuth_sectors = (azimuth_range[1] - azimuth_range[0]) // azimuth_size

    if elevation_range == (0,0):  # only plce targets on the horizontal plane
        sector_centers = [
            (azimuth_range[0] + (i + 0.5) * azimuth_size, 0)
            for i in range(num_azimuth_sectors)
        ]
        random.shuffle(sector_centers)
        num_sectors = num_azimuth_sectors
    else:
        # Compute sector centers
        num_elevation_sectors = (elevation_range[1] - elevation_range[0]) // elevation_size
        num_sectors = num_azimuth_sectors * num_elevation_sectors
        sector_centers = [
            (azimuth_range[0] + (i + 0.5) * azimuth_size, elevation_range[0] + (j + 0.5) * elevation_size)
            for i in range(num_azimuth_sectors) for j in range(num_elevation_sectors)
        ]
        random.shuffle(sector_centers)

    # Select sectors ensuring minimum distance constraint
    selected_sectors = []
    remaining_sectors = sector_centers[:]
    while len(selected_sectors) < num_sectors:
        if not selected_sectors:
            selected_sectors.append(remaining_sectors.pop(0))
        else:
            last_sector = selected_sectors[-1]
            valid_sectors = [
                sec for sec in remaining_sectors
                if numpy.linalg.norm(numpy.array(sec) - numpy.array(last_sector)) >= min_distance
            ]
            if valid_sectors:
                selected_sector = valid_sectors.pop(0)
                selected_sectors.append(selected_sector)
                remaining_sectors.remove(selected_sector)
            else:
                logging.error('Can not create target sequence with given settings. '
                              'Check min distance and target range.')
                break  # Stop if no valid sector is found


    # Generate random points within each selected sector
    points = numpy.float16([
        (
            numpy.random.uniform(sector[0] - azimuth_size / 2, sector[0] + azimuth_size / 2),
            numpy.random.uniform(sector[1] - elevation_size / 2, sector[1] + elevation_size / 2)
        )
        for _ in range(targets_per_sector) for sector in selected_sectors
    ])
    points = numpy.round(points, 2)
    sequence = slab.Trialsequence(points)
    sequence.sector_centers = sector_centers
    sequence.sector_size = sector_size
    return sequence

def plot_sequence_targets(sequence, title="Recorded targets over sectors"):
    """
    Plot target coordinates from sequence.data over the sector grid.
    Uses sequence.settings to retrieve az/el ranges and sector size.
    """
    if not hasattr(sequence, "settings"):
        raise AttributeError("sequence must have a 'settings' attribute (dict).")
    if not hasattr(sequence, "sector_centers"):
        raise AttributeError("sequence must have 'sector_centers' attribute.")

    settings = sequence.settings
    sector_centers = list(sequence.sector_centers)
    az_size, el_size = settings['sector_size']
    az_range = settings['azimuth_range']
    el_range = settings['elevation_range']

    # --- extract target coordinates (2nd row) from sequence.data ---
    points = []
    for entry in sequence.data:
        arr = entry
        while isinstance(arr, (list, tuple)) and len(arr) == 1:
            arr = arr[0]
        arr = numpy.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Unexpected shape in sequence.data entry: {arr.shape}")
        points.append(arr[1])  # target row
    points = numpy.asarray(points, dtype=float)
    points[:, 0] = (points[:, 0] + 180) % 360 - 180  # wrap azimuth

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(az_range)
    ax.set_ylim(el_range)
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(title)

    # sector rectangles
    for (caz, cel) in sector_centers:
        rect = matplotlib.patches.Rectangle(
            (caz - az_size / 2.0, cel - el_size / 2.0),
            az_size, el_size,
            fill=False, linestyle="--", linewidth=1.0
        )
        ax.add_patch(rect)

    # grid
    az_ticks = numpy.arange(az_range[0], az_range[1] + az_size, az_size)
    el_ticks = numpy.arange(el_range[0], el_range[1] + el_size, el_size)
    ax.set_xticks(az_ticks)
    ax.set_yticks(el_ticks)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # scatter
    ax.scatter(points[:, 0], points[:, 1], s=25, color="red", label="Targets")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Example usage
# azimuth_range = (-50, 50)
# elevation_range = (-40, 40)
# sector_size = (10, 10)  # (azimuth_size, elevation_size)
# min_distance = 30
# targets_per_sector = 3
#
# _, points, selected_sectors = make_sequence(
#     azimuth_range, elevation_range, sector_size, min_distance, targets_per_sector
# )
# plot_random_points(points, selected_sectors, azimuth_range, elevation_range, sector_size)

# import matplotlib
# matplotlib.use("Qt5Agg")
# # matplotlib.use("TkAgg")