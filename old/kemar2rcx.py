import slab
import numpy
from pathlib import Path
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'
hrtf = slab.HRTF.kemar()
hrtf = slab.HRTF(sofa_path / 'MRT01.sofa')

sources = hrtf.sources.vertical_polar
elevations = numpy.asarray(sorted(list(set(hrtf.sources.vertical_polar[:, 1]))))

# find azimuths for which there are positions at each elevation up to 60°
az_positions = []
for ele in elevations: #[elevations <= 90]:
    az_positions.append(sources[sources[:,1]==ele, 0])

from functools import reduce
common_az_positions = reduce(numpy.intersect1d, (az_positions))


out = []
for az in azimuths:
    for ele in elevations:  # for each elevation, find the source closest to the reference y
        az_range = [az-1, az+1]
        idx, = numpy.where(
            numpy.logical_and((numpy.round(hrtf.sources.vertical_polar[:, 1]) == ele),
            ((hrtf.sources.vertical_polar[:, 0] >= az_range[0]) &
             (hrtf.sources.vertical_polar[:, 0] <= az_range[1]))
                              ))
        out.extend(idx)
out = numpy.unique(out)

hrtf.plot_sources(out)


# get all sources that lie on the same azimuth
import numpy
cone = 45
cone = numpy.sin(numpy.deg2rad(cone))
elevations = hrtf.elevations()
_cartesian = hrtf.sources.cartesian / 1.4  # get cartesian coordinates on the unit sphere
out = []
for ele in elevations:  # for each elevation, find the source closest to the reference y
    subidx, = numpy.where(numpy.round(hrtf.sources.vertical_polar[:, 1]) == ele)
    if subidx.size != 0:  # check whether sources exist
        cmin = numpy.min(numpy.abs(_cartesian[subidx, 1] - cone).astype('float16'))
        if cmin < 0.05:  # only include elevation where the closest source is less than 5 cm away
            idx, = numpy.where((numpy.round(hrtf.sources.vertical_polar[:, 1]) == ele) & (
                    numpy.abs(_cartesian[:, 1] - cone).astype('float16') == cmin))  # avoid rounding error
            out.append(idx[0])
            if full_cone and len(idx) > 1:
                out.append(idx[1])