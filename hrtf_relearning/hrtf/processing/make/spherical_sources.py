import numpy
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy
import matplotlib.pyplot as plt

import numpy

import numpy

import numpy

def spherical_sources(
    resolution_deg: float,
    distance: float = 1.0,
) -> numpy.ndarray:
    """
    Generate a near-uniform spherical source grid using a Fibonacci sphere,
    returned as [azimuth_deg, elevation_deg, distance], sorted by:
        1) azimuth ascending (0..360, CCW)
        2) elevation descending (+90..-90)

    Parameters
    ----------
    resolution_deg : float
        Approximate angular spacing between neighboring sources (degrees).
    distance : float
        Source distance (same for all sources).

    Returns
    -------
    sources : (N, 3) numpy.ndarray
        Columns: [azimuth_deg, elevation_deg, distance]
    """
    # --- estimate number of sources ---
    delta = numpy.deg2rad(resolution_deg)
    N = int(numpy.ceil(4.0 * numpy.pi / (delta * delta)))
    N = max(N, 1)

    # --- Fibonacci (golden-angle) sphere ---
    golden_angle = numpy.pi * (3.0 - numpy.sqrt(5.0))
    i = numpy.arange(N, dtype=float) + 0.5  # avoid poles

    z = 1.0 - 2.0 * i / N
    r_xy = numpy.sqrt(numpy.maximum(0.0, 1.0 - z * z))
    phi = i * golden_angle

    x = r_xy * numpy.cos(phi)
    y = r_xy * numpy.sin(phi)

    # --- Cartesian → azimuth / elevation ---
    az = numpy.rad2deg(numpy.arctan2(y, x)) % 360.0
    el = numpy.rad2deg(numpy.arcsin(z))
    r = numpy.full_like(az, distance)

    # --- round angles ---
    az = numpy.round(az, decimals=1)
    el = numpy.round(el, decimals=1)

    sources = numpy.stack([az, el, r], axis=1)

    # --- sort: azimuth ↑, elevation ↓ ---
    sort_idx = numpy.lexsort((-el, az))
    sources = sources[sort_idx]

    return sources

def plot_sources(sources: numpy.ndarray):
    """
    3D scatter plot of vertical-polar sources.

    Parameters
    ----------
    sources : (N, 3) numpy.ndarray
        [az_deg, el_deg, r]
    """
    az = numpy.deg2rad(sources[:, 0])
    el = numpy.deg2rad(sources[:, 1])
    r = sources[:, 2]

    # vertical polar → Cartesian
    x = r * numpy.cos(el) * numpy.cos(az)
    y = r * numpy.cos(el) * numpy.sin(az)
    z = r * numpy.sin(el)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, s=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Uniform spherical source grid")

    plt.tight_layout()
    plt.show()


# sources = spherical_sources(resolution_deg=5)
# plot_sources(sources)