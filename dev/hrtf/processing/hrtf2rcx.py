from pathlib import Path
import numpy
import slab
from array import array
from dev.hrtf.processing.tf2ir import tf2ir
fs = 48828
binary_path = Path.cwd() / 'data' / 'hrtf' / 'binary'
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

def hrtf2binary(hrtf, filename, n_bins=None, add_itd=False, add_ild=False):
    """
    Convert HRIR sofa to binary file to be used with HRTFCoef in RPvdsEx.
    """
    if hrtf.datatype not in ['TF', 'FIR']:
        raise ValueError('Unknown datatype.')
    if hrtf.datatype == 'TF':
        hrtf = tf2ir(hrtf)
    sources = hrtf.sources.vertical_polar
    elevations = numpy.asarray(sorted(list(set(sources[:, 1]))))
    azimuths = numpy.asarray(sorted(list(set(sources[:, 0]))))

    # ascertain equal resolution in each axis
    if not numpy.all(numpy.isclose(numpy.diff(azimuths), numpy.diff(azimuths[0:2]), atol=.1)):
        print('warning: non-uniform azimuth resolution')
    az_res = 1 / numpy.mean(numpy.diff(azimuths))
    if not numpy.all(numpy.isclose(numpy.diff(elevations), numpy.diff(elevations[0:2]), atol=1e-01)):
        print('warning: non-uniform elevation resolution')
    ele_res = 1 / numpy.mean(numpy.diff(elevations))

    # ascertain equal number of points in each axis
    n_az_ref = len(sources[sources[:, 1] == elevations[0]])
    for ele in elevations:
        n_az = len(sources[sources[:, 1] == ele])
        if not n_az == n_az_ref:
            print(f'warning: varying number of azimuth sources across elevations (at {ele}° elevation).')
    n_ele_ref = len(sources[sources[:, 0] == azimuths[0]])
    for az in azimuths:
        n_ele = len(sources[sources[:, 0] == az])
        if not n_ele == n_ele_ref:
            print(f'warning: varying number of elevation sources at {az}° Azimuth.')

    if n_bins is None:
        n_bins = hrtf[0].n_taps
    else:
        print(f'interpolating IR to {n_bins} bins.')

    # create header array
    header = numpy.zeros(12)
    header[0] = int(hrtf.n_sources + 1)  # I Number of filter positions in RAM buffer. (number of Azimuths * Number of elevations)+1.
    header[1] = int((n_bins * 2) + 2)  # II Number of taps (coefficients) including Interaural delay (ITD) delay (x2) per filter. e.g. 31 tap filter =31 x 2 + 2 (delay values)=64
    header[2] = int(n_bins + 1)  # III Number of taps including the delay. e.g. 31 tap filter= 31 taps + delay value
    header[3] = min(azimuths)  # IV Minimum Azimuth value in degrees (e.g. -165)
    header[4] = max(azimuths)  # V Maximum Azimuth value in degrees (e.g. 180)
    header[5] = az_res  # VI Inverse of the Position separation of Az in degrees, defined as 1.0/(AZ separation) e.g. 15 degrees between channel would =0.066666.
    header[6] = n_az_ref  # VII % Number of Az positions at each elevation
    header[7] = min(elevations)  # VIII Minimum Elevation value in degrees.
    header[8] = max(elevations)  # IX Maximum Elevation value in degrees (Must include a value for 90).
    header[9] = ele_res  # X Inverse of the Position separation of Elevation in degrees, defined as 1.0/(EL separation). e.g. 30 degrees between EL would be 1/30=.0333.
    header[10] = n_ele_ref  # XI Number of elevation positions for each Azimuth+1. The additional value is for the filter at 90 degrees. In cases where there will be no filter at 90 degrees elevation it is still necessary to include a dummy filter.
    header[11] = 1e6 / hrtf.samplerate  # XII Filter sampling period in microseconds. Calculated as the inverse of the sampling rate * 1,000,000.

    # write header and filter coefficients to binary file (see RPvdsEx help)
    with open(binary_path / f'{filename}.f32', 'wb') as output_file:
        # write header
        array('i', header[0:3].astype('int32')).tofile(output_file)
        array('f', header[3:6].astype('float32')).tofile(output_file)
        array('i', [header[6].astype('int32')]).tofile(output_file)
        array('f', header[7:10].astype('float32')).tofile(output_file)
        array('i', [header[10].astype('int32')]).tofile(output_file)
        array('f', [header[11].astype('float32')]).tofile(output_file)

        # write Filter Coefficients
        elevations[::-1].sort()  # sort elevations in Descending Order
        # elevations[::1].sort()  # hrtfs from aachen have up and down reversed?
        for elevation in elevations:
            ele_sources = sources[numpy.where(sources[:, 1] == elevation)[0]]  # get sources at each elevation
            azimuths = ele_sources[:, 0]
            azimuths[::-1].sort()  # sort azimuths in descending order
            # azimuths[::1].sort()  # hrtfs from aachen have clockwise decreasing az values?
            for azimuth in azimuths:
                source_idx, = numpy.where(numpy.logical_and(sources[:, 0] == azimuth, sources[:, 1] == elevation))[0]
                if not n_bins == hrtf[source_idx].n_taps:  # interpolate bins if necessary
                    t = numpy.linspace(0, hrtf[source_idx].duration, hrtf[source_idx].n_taps)
                    t_interp = numpy.linspace(0, t[-1], n_bins)
                    fir_coefs = numpy.zeros((n_bins, 2))
                    for idx in range(2):
                        fir_coefs[:, idx] = numpy.interp(t_interp, t, hrtf[source_idx].data[:, idx])
                else:
                    fir_coefs = hrtf[source_idx].data

                if add_itd:
                    itd = slab.Binaural.azimuth_to_itd(azimuth, head_radius=11)  # head radius in cm
                    if itd >= 0:  # add left delay
                        delay = numpy.array((int(itd / 2 * hrtf.samplerate), 0))
                    elif itd < 0:  # add right delay
                        delay = numpy.array((0, int(- itd / 2 * hrtf.samplerate)))
                    fir_coefs = numpy.vstack((fir_coefs, delay))  # add group delays
                else:
                    fir_coefs = numpy.vstack((fir_coefs, numpy.zeros(2)))  # add zero group delays
                array('f', fir_coefs[:, 0].astype('float32')).tofile(output_file)  # write left ear filter
                array('f', fir_coefs[:, 1].astype('float32')).tofile(output_file)  # write right ear filter
    output_file.close()


# todo add ild: use mean level difference above 1.5 khz? -
#  i dont want to impose spectral cues other than the ones chosen by design


"""
file = open(data_path / 'hrtf_FIRcoefs.f32', "rb")
data = file.read()

struct.unpack('f', data)


# using struct
# with open(data_path / 'hrtf_FIRcoefs.f32', "wb") as f:
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



"""