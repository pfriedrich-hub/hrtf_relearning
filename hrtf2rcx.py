import struct
from pathlib import Path
import numpy
import copy
from array import array
data_path = Path.cwd() / 'data' / 'hrtf'

# -- convert kemar sofa to binary file to be used with HRTFCoef in RPvdsEx --- #

# todo filter HRTF set to get a uniform number of sources on each elevation (interpolate if neccessary)

import slab
kemar = slab.HRTF.kemar()
header = numpy.zeros(12)
sources = kemar.sources.vertical_polar
mean_az_dist = []  #todo cannot be uniform if number is constant! ask tdt support (mismatch with the picture)
n_az = []  # number of azimuths per elevation (cannot be uniform if distance is constant)
for ele in kemar.elevations():
    mean_az_dist.append(numpy.nanmean(numpy.diff(sources[sources[:,1]==ele, 0])))
    n_az.append(len(sources[sources[:,1]==ele, 0]))
header[0] = int(kemar.n_sources + 1)  # I Number of filter positions in RAM buffer. (number of Azimuths * Number of elevations)+1.
header[1] = int((kemar[0].n_taps * 2) + 2)  # II Number of taps (coefficients) including Interaural delay (ITD) delay (x2) per filter. e.g. 31 tap filter =31 x 2 + 2 (delay values)=64
header[2] = int(kemar[0].n_taps + 1)  # III Number of taps including the delay. e.g. 31 tap filter= 31 taps + delay value
header[3] = - 180  # IV Minimum Azimuth value in degrees (e.g. -165)
header[4] = 180  # V Maximum Azimuth value in degrees (e.g. 180)
header[5] = numpy.nanmean(mean_az_dist)  # VI Inverse of the Position separation of Az in degrees, defined as 1.0/(AZ separation) e.g. 15 degrees between channel would =0.066666.
header[6] = max(n_az)  # VII % Number of Az positions at each elevation
header[7] = min(kemar.elevations())  # VIII Minimum Elevation value in degrees.
header[8] = max(kemar.elevations())  # IX Maximum Elevation value in degrees (Must include a value for 90).
header[9] = 1 / numpy.mean(numpy.diff(kemar.elevations()))  # X Inverse of the Position separation of Elevation in degrees, defined as 1.0/(EL separation). e.g. 30 degrees between EL would be 1/30=.0333.
header[10] = kemar.n_elevations  # XI Number of elevation positions for each Azimuth+1. The additional value is for the filter at 90 degrees. In cases where there will be no filter at 90 degrees elevation it is still necessary to include a dummy filter.
header[11] = 1e6 / kemar[0].samplerate  # XII Filter sampling period in microseconds. Calculated as the inverse of the sampling rate * 1,000,000.

# writing the header(see RPvdsEx help)
with open(data_path / 'kemar_FIRcoefs.f32', 'wb') as output_file:

    # header
    array('i', header[0:3].astype('int32')).tofile(output_file)
    array('f', header[3:6].astype('float32')).tofile(output_file)
    array('i', [header[6].astype('int32')]).tofile(output_file)
    array('f', header[7:10].astype('float32')).tofile(output_file)
    array('i', [header[10].astype('int32')]).tofile(output_file)
    array('f', [header[11].astype('float32')]).tofile(output_file)

    # Filter Coefficients
    elevations = kemar.elevations()
    elevations.sort(reverse=True)  # sort elevations in Descending Order
    for elevation in elevations:
        ele_sources = sources[numpy.where(sources[:, 1] == elevation)[0]]  # get sources at each elevation
        azimuths = ele_sources[:, 0]
        azimuths.sort()  # sort azimuths in ascending order
        for azimuth in azimuths:
            source_idx = numpy.where(numpy.logical_and(sources[:, 0] == azimuth, sources[:, 1] == elevation))[0]
            dtfs = kemar.tfs_from_sources(source_idx, ear='both', n_bins=None)[0]  # extract transfer functions
            dtfs = numpy.vstack((dtfs, numpy.zeros(2)))  # add group delays
            array('f', dtfs[:, 0].astype('float32')).tofile(output_file)  # write left ear filter
            array('f', dtfs[:, 1].astype('float32')).tofile(output_file)  # write right ear filter

output_file.close()



file = open(data_path / 'kemar_FIRcoefs.f32', "rb")
data = file.read()

struct.unpack('f', data)


# using struct
# with open(data_path / 'kemar_FIRcoefs.f32', "wb") as f:
#     header = struct.pack('f', v1, v2, ...)
#     b = struct.pack('f', header[0])
#     f.write(b)

number = list(file.read(-1))


# --- read and write f32 files ---- #
# write f32
f32_number = float(9)
with open(data_path / 'test.f32', "wb") as f:
    b = struct.pack('f', f32_number)
    f.write(b)

# read f32
# python read method
file = open(data_path / 'Jac2_32.f32', "rb")
data = file.read()
number = list(file.read(-1))
# numpy
data = numpy.fromfile(data_path / 'Jac2_32.f32', dtype=float, count=-1, sep='', offset=0)



