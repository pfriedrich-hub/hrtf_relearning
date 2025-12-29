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

def az_el_distance(p, q):
    """Euclidean distance in az/el with circular azimuth."""
    daz = (p[0] - q[0] + 180) % 360 - 180
    del_ = p[1] - q[1]
    return numpy.hypot(daz, del_)

def sector_targets(settings, hrir_sources):
    """
    Generate a localization trial sequence by sampling targets from spatial
    sectors and enforcing a minimum distance between consecutive targets.

    The azimuth–elevation space is divided into rectangular sectors. A fixed
    number of HRIR source positions is sampled from each sector, after which
    all sampled targets are globally reordered such that the angular distance
    between successive targets is at least `min_distance`.

    The minimum-distance constraint is applied at the level of the *actual
    target positions*, not at the sector level.

    Parameters
    ----------
    settings : dict
        Required keys:
        - 'azimuth_range' : (min_az, max_az) in degrees
        - 'elevation_range' : (min_el, max_el) in degrees
        - 'sector_size' : (az_size, el_size) in degrees
        - 'targets_per_sector' : int
        - 'min_distance' : float, minimum allowed distance between consecutive targets
        - 'replace' : bool, sample HRIR positions with or without replacement

        Additional keys are stored but not interpreted by this function.

    hrir_sources : array-like, shape (N, 2)
        Available HRIR source positions as (azimuth, elevation) in degrees.
        Azimuth is treated circularly, elevation linearly.

    Returns
    -------
    slab.Trialsequence
        Trial sequence containing ordered (azimuth, elevation) targets with a
        fixed order preserving the minimum-distance constraint. All parameters
        used to generate the sequence are stored in `sequence.settings`.

    Notes
    -----
    - Distances are computed as Euclidean distance in azimuth–elevation space
      with circular azimuth wrapping.
    - Randomness depends on NumPy's RNG; set `numpy.random.seed(...)` for
      reproducibility.
    - An error is raised if the constraints cannot be satisfied.
    """

    az_range = settings['azimuth_range']
    el_range = settings['elevation_range']
    az_size, el_size = settings['sector_size']
    n_per_sector = settings['targets_per_sector']
    min_dist = settings['min_distance']
    replace = settings['replace']

    src_az = numpy.asarray(hrir_sources[:, 0], float)
    src_el = numpy.asarray(hrir_sources[:, 1], float)

    # --- build sector centers ---
    az_centers = numpy.arange(
        az_range[0] + az_size / 2, az_range[1], az_size
    )
    el_centers = (
        [0] if el_range == (0, 0)
        else numpy.arange(el_range[0] + el_size / 2, el_range[1], el_size)
    )
    sector_centers = [(az, el) for az in az_centers for el in el_centers]

    # --- candidate sources per sector ---
    def sources_in_sector(caz, cel):
        daz = numpy.abs((src_az - caz + 180) % 360 - 180)
        az_ok = daz <= az_size / 2
        el_ok = (src_el >= cel - el_size / 2) & (src_el <= cel + el_size / 2)
        return numpy.where(az_ok & el_ok)[0]

    sector_samples = []
    used = set()

    for caz, cel in sector_centers:
        idx = sources_in_sector(caz, cel)
        if len(idx) < n_per_sector and not replace:
            raise ValueError(f"Not enough sources in sector ({caz},{cel})")

        picks = []
        for _ in range(n_per_sector):
            cand = idx if replace else [i for i in idx if i not in used]
            if not cand:
                raise ValueError("Global uniqueness violated")
            p = int(numpy.random.choice(cand))
            picks.append(p)
            used.add(p)
        sector_samples.extend(picks)

    # --- build point list ---
    points = numpy.column_stack([src_az[sector_samples], src_el[sector_samples]])

    # --- constrained ordering (greedy with restart) ---
    def order_with_min_distance(points, max_tries=500):
        N = len(points)
        for _ in range(max_tries):
            remaining = list(range(N))
            order = [remaining.pop(numpy.random.randint(len(remaining)))]
            while remaining:
                last = order[-1]
                valid = [
                    i for i in remaining
                    if az_el_distance(points[last], points[i]) >= min_dist
                ]
                if not valid:
                    break
                nxt = numpy.random.choice(valid)
                order.append(nxt)
                remaining.remove(nxt)
            if len(order) == N:
                return order
        raise RuntimeError("Could not satisfy min_distance constraint")

    order = order_with_min_distance(points)
    points = points[order]

    # --- formatting ---
    points = numpy.round(points, 2)
    points[:, 0] = (points[:, 0] + 180) % 360 - 180
    seq = slab.Trialsequence(points)
    seq.trials = numpy.arange(1, len(points) + 1)
    seq.settings = settings
    seq.settings['sector_centers'] = sector_centers
    return seq

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