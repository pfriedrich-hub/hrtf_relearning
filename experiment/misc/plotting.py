from pathlib import Path
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy
import slab

def distance_to_interval(distance, az_range, ele_range, target_size):
    max_interval = 250  # max interval duration in ms
    min_interval = 50  # min interval duration before entering target window
    steepness = 30  # controls how the interval duration decreases when approaching the target window
    max_distance = numpy.linalg.norm(numpy.subtract([0, 0], [az_range[0], ele_range[0]]))  # max possible distance
    # Normalize distance: [target_size → max_distance] → [0 → 1]
    norm_dist = (distance - target_size) / (max_distance - target_size)
    norm_dist = numpy.clip(norm_dist, 0, 1)
    # Logarithmic interpolation between min_interval and max_interval
    scale = numpy.log1p(steepness * norm_dist) / numpy.log1p(steepness)
    interval = (min_interval + (max_interval - min_interval) * scale).astype(int)
    return int(interval) / 1000  # convert to seconds