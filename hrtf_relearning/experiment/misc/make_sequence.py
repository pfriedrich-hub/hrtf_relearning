import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import slab
import numpy
import random
import logging

def make_sequence(settings, hrir_sources):
    if settings['kind'] == 'standard':
        return std_targets(settings, hrir_sources)  # play 3 times from each source in the sequence
    elif settings['kind'] == 'sectors':
        return sector_targets(settings, hrir_sources)


def sector_targets(settings, hrir_sources=None):
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
    num_azimuth_sectors = int(azimuth_range[1] - azimuth_range[0]) // azimuth_size

    if elevation_range == (0,0):  # only place targets on the horizontal plane
        sector_centers = [
            (azimuth_range[0] + (i + 0.5) * azimuth_size, 0)
            for i in range(num_azimuth_sectors)
        ]
        random.shuffle(sector_centers)
        num_sectors = num_azimuth_sectors
    else:
        # Compute sector centers
        num_elevation_sectors = int((elevation_range[1] - elevation_range[0]) // elevation_size)
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
        if idx.size < targets_per_sector and not settings['replace']:
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

            if settings['replace']:
                # Mit Zurücklegen: wir ignorieren "used" und ziehen einfach zufällig aus cand
                if not cand:
                    caz, cel = selected_sectors[s]
                    logging.error(
                        f'No candidate sources in sector (az={caz:.2f}, el={cel:.2f}) '
                        f'for pass {t + 1}/{T} with replace=True.'
                    )
                    raise ValueError('Empty candidate list for sector with replace=True.')
                pick = int(numpy.random.choice(cand))
                sector_picks[s].append(pick)
            else:
                # Ohne Zurücklegen: sicherstellen, dass wir keinen Index doppelt nehmen
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
    sequence.settings = settings  # ← store all parameters here
    sequence.settings['sector_centers'] = sector_centers
    return sequence

def std_targets(settings, hrir_sources, max_tries=1000):
    """
    Default sequence:
    - use every HRIR source `n_reps` times
    - random order
    - enforce min_distance between successive targets
    - supports azimuth_range and elevation_range
    - fills missing settings like azimuth_range, elevation_range, sector_size, sector_centers
    """

    repeats = settings['targets_per_speaker']
    min_distance = settings['min_distance']

    az_range = settings.get('azimuth_range', None)
    el_range = settings.get('elevation_range', None)

    if hrir_sources is None:
        raise ValueError("hrir_sources must be provided.")

    src = numpy.asarray(hrir_sources, dtype=float)
    if src.ndim != 2 or src.shape[1] < 2:
        raise ValueError("hrir_sources must be of shape (N, 2) -> [az, el].")

    # initial extraction
    src_az = src[:, 0].astype(float)
    src_el = src[:, 1].astype(float)

    # -------------------------------------------
    # Apply azimuth/elevation limits
    # -------------------------------------------
    mask = numpy.ones(len(src), dtype=bool)

    if az_range is not None:
        mask &= (src_az >= az_range[0]) & (src_az <= az_range[1])

    if el_range is not None:
        mask &= (src_el >= el_range[0]) & (src_el <= el_range[1])

    src_az = src_az[mask]
    src_el = src_el[mask]

    if len(src_az) == 0:
        raise ValueError(
            "No HRIR sources match the specified azimuth/elevation ranges "
            f"(az_range={az_range}, el_range={el_range})."
        )

    # If ranges were not given, infer them from the filtered sources
    if az_range is None:
        az_range = (float(src_az.min()), float(src_az.max()))
    if el_range is None:
        el_range = (float(src_el.min()), float(src_el.max()))

    n_sources = len(src_az)

    def _wrap_diff(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return d

    # distance matrix
    az1 = src_az[:, None]
    az2 = src_az[None, :]
    d_az = _wrap_diff(az1, az2)
    d_el = src_el[:, None] - src_el[None, :]
    dist = numpy.sqrt(d_az ** 2 + d_el ** 2)

    total_len = n_sources * repeats

    for attempt in range(max_tries):
        remaining = numpy.full(n_sources, repeats, dtype=int)
        seq_indices = []

        first = numpy.random.randint(0, n_sources)
        seq_indices.append(first)
        remaining[first] -= 1

        success = True
        for _ in range(1, total_len):
            last = seq_indices[-1]
            candidates = numpy.where(
                (remaining > 0) & (dist[last] >= min_distance)
            )[0]

            if candidates.size == 0:
                success = False
                break

            nxt = numpy.random.choice(candidates)
            seq_indices.append(nxt)
            remaining[nxt] -= 1

        if success:
            seq_indices = numpy.asarray(seq_indices, dtype=int)
            points = numpy.column_stack([src_az[seq_indices], src_el[seq_indices]])
            points[:, 0] = (points[:, 0] + 180) % 360 - 180
            points = numpy.round(points, 2)

            # --------------------------------------------------
            # Infer a "sector_size" from the HRIR grid spacing
            # --------------------------------------------------
            sector_size_setting = settings.get("sector_size", None)

            if sector_size_setting is not None:
                az_step, el_step = sector_size_setting
            else:
                unique_az = numpy.unique(numpy.round(src_az, 4))
                unique_el = numpy.unique(numpy.round(src_el, 4))

                if unique_az.size > 1:
                    az_diffs = numpy.diff(numpy.sort(unique_az))
                    az_step = float(numpy.min(az_diffs))
                else:
                    az_step = float(az_range[1] - az_range[0]) or 10.0  # fallback

                if unique_el.size > 1:
                    el_diffs = numpy.diff(numpy.sort(unique_el))
                    el_step = float(numpy.min(el_diffs))
                else:
                    el_step = float(el_range[1] - el_range[0]) or 10.0  # fallback

            sector_size = (az_step, el_step)

            # --------------------------------------------------
            # Build a virtual sector grid (centers)
            # --------------------------------------------------
            az_span = az_range[1] - az_range[0]
            el_span = el_range[1] - el_range[0]

            num_az = max(1, int(round(az_span / az_step)))
            num_el = max(1, int(round(el_span / el_step)))

            sector_centers = [
                (
                    az_range[0] + (i + 0.5) * az_step,
                    el_range[0] + (j + 0.5) * el_step,
                )
                for i in range(num_az)
                for j in range(num_el)
            ]

            sequence = slab.Trialsequence(points)

            # Start from existing settings, then fill/overwrite what we know
            settings_out = dict(settings)
            settings_out.update({
                "kind": "standard",
                "mode": "all_sources_repeated",
                "repeats": repeats,
                "min_distance": min_distance,
                "azimuth_range": az_range,
                "elevation_range": el_range,
                "sector_size": sector_size,
                "sector_centers": sector_centers,
            })

            sequence.settings = settings_out

            return sequence

    raise RuntimeError(
        f"Could not construct sequence with min_distance={min_distance}° "
        f"after {max_tries} attempts."
    )

def plot_sequence_targets(sequence, title="Recorded targets over sectors"):
    """
    Plot target coordinates from sequence over the sector grid (if available).

    - For settings['kind'] == 'sectors':
        draws sector rectangles + targets.
    - For settings['kind'] == 'standard':
        just plots the targets, using az/el ranges and (optional) sector_size
        for ticks/grid.
    """
    if not hasattr(sequence, "settings"):
        raise AttributeError("sequence must have a 'settings' attribute (dict).")

    settings = sequence.settings
    kind = settings.get("kind", "sectors")

    az_range = settings.get("azimuth_range", None)
    el_range = settings.get("elevation_range", None)
    sector_size = settings.get("sector_size", None)

    # Extract target coordinates
    points = numpy.asarray(sequence.conditions, dtype=float)
    points[:, 0] = (points[:, 0] + 180) % 360 - 180  # wrap azimuth

    # Infer ranges if missing
    if az_range is None:
        az_range = (float(points[:, 0].min()), float(points[:, 0].max()))
    if el_range is None:
        el_range = (float(points[:, 1].min()), float(points[:, 1].max()))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(az_range)
    ax.set_ylim(el_range)
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(title)

    # Draw sector rectangles if we actually have sector_centers
    if kind == "sectors" and hasattr(sequence, "sector_centers") and sequence.sector_centers is not None:
        az_size, el_size = sector_size
        sector_centers = list(sequence.sector_centers)

        for (caz, cel) in sector_centers:
            rect = matplotlib.patches.Rectangle(
                (caz - az_size / 2.0, cel - el_size / 2.0),
                az_size, el_size,
                fill=False, linestyle="--", linewidth=1.0
            )
            ax.add_patch(rect)

        az_ticks = numpy.arange(az_range[0], az_range[1] + az_size, az_size)
        el_ticks = numpy.arange(el_range[0], el_range[1] + el_size, el_size)
    else:
        # standard mode or no sector_centers: simpler grid
        if sector_size is not None:
            az_step, el_step = sector_size
        else:
            az_step, el_step = 10.0, 10.0
        az_ticks = numpy.arange(az_range[0], az_range[1] + az_step, az_step)
        el_ticks = numpy.arange(el_range[0], el_range[1] + el_step, el_step)

    ax.set_xticks(az_ticks)
    ax.set_yticks(el_ticks)
    ax.grid(True, linestyle="--", linewidth=0.5)

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