import pyfar
import numpy
import warnings

def spherical_head(
    coordinates, head=pyfar.Coordinates(0, [.0875, -.0875], 0),
    head_position='origin', reference_distance='coordinates',
    n_max=100, n_samples=256, sampling_rate=44100, speed_of_sound=343):
    r"""
    Generate spherical head transfer functions (SHTFs).

    Generates free field SHTFs following [#]_ and returns them as a
    :py:class:`pyfar.Signal` object. Following the HRTF definition [#]_,
    the SHTFs are defined as the ratio
dsvxy
    :math:`\mathrm{SHTF}(\mathbf{x}_\text{ear}, \mathbf{x}_\text{source})=\frac{P_\text{sphere}(\mathbf{x}_\text{ear}, \mathbf{x}_\text{source})}{P_\text{ref}(\mathbf{x}_\text{ref})}`

    with :math:`P_\text{sphere}` being the complex sound pressure on the
    surface of the sphere, :math:`P_\text{ref}` the reference sound pressure at
    the origin of coordinates, and :math:`\mathbf{x}` the positions of the ear
    (on the surface of a sphere), the free field point source, and the point
    source used for referencing. Often
    :math:`\mathbf{x}_\text{source}=\mathbf{x}_\text{ref}` is used. The
    spherical head views in positive x direction, the left ear has a positive y
    coordinate, and the right ear a negative y coordinate.

    .. note ::

        The SHTFs are shifted in time to assure that the earliest possible peak
        in the impulse responses occurs at approximately 20 samples for the
        case of a contralateral source position. This works well if
        :math:`\mathbf{x}_\text{source}=\mathbf{x}_\text{ref}`. Other cases
        might require an additional shift to force causality.

    .. note ::

        Because the 0 Hz value can not be computed, a low frequency with a
        wavelength equal to a hundred times the spherical head diameter is
        used. This yields 19.6 Hz for the default values.

    Parameters
    ----------
    coordinates : pyfar.Coordinates
        The coordinates of the point sources :math:`\mathbf{x}_\text{source}`
        for which the SHTFs are computed. Must have a `cdim` (channel
        dimension) of 1.
    head : sparhpy.SamplingSphere
        The positions of the left ear defined by the first point and the
        position of the right ear defined by the second point. Must be
        of cshape of ``(2, )``. Both points must have the same radius, which
        defines the radius of the spherical head. The default uses ear
        positions at azimuth angles of :math:`\pm90^\circ` on the equator of
        the spherical head and a radius of ``0.0875`` meter.
    head_position : str, pyfar.Coordinates, optional
        Defines the position of the head center with respect to the global
        coordinate system in which `coordinates` are given.

        ``'origin'``
            The head center is at the origin of coordinates.
        ``'interaural center'``
            The center of the head is moved to make sure that the interaural
            center, that is, the mid point between the two ears, is at the
            origin of coordinates.
        coordinate
            A pyfar Coordinates object with a single point that defines the
            center of the head.

        The default is ``'origin'``
    reference_distance : str, float, array like, optional
        Defines the distance of the reference point source from the origin
        of coordinates.

        ``'coordinates'``
            Takes the distance from `coordinates` (see above).
        float or array like
            The reference distance in meter.

        The default is ``'coordinates'``.
    n_max : int, optional
        The spherical harmonic order up to which the SHTF is approximated.
         The default is ``100``.
    n_samples : int, optional
        The length of the spherical head impulse responses in samples.
        The default is ``1024``
    sampling_rate : float, optional
        The sampling rate in Hz. The default is 44100.
    speed_of_sound : int, optional
        The speed of sound in meter per second. The default is ``343``.

    Returns
    -------
    shtf : pyfar.Signal
        The spherical head transfer function as a pyfar Signal of cshape
        (M, 2), where M is the number of points given in `coordinates`.
        ``shtf[:, 0]`` contains the left ear data and ``shtf[:, 1]`` the right
        ear data.

    References
    ----------
    [#] Duda, R. O., and Martens, W. L. (1998). “Range dependence of the
        response of a spherical head model,” J. Acoust. Soc. Am., 104,
        3048–3058.
    """

    # input checks ------------------------------------------------------------
    if not isinstance(coordinates, pyfar.Coordinates) or\
            coordinates.cdim != 1:
        raise TypeError("coordinates must be a pyfar Coordinates object with"
                        "a cdim of 1")
    if not isinstance(head, pyfar.Coordinates) or\
            head.cshape != (2, ) or  head.csize != 2:
        raise TypeError("head must be a spharpy SamplingSphere object with"
                        "a cshape of (2, )")

    # pre-process input -------------------------------------------------------
    # compute the center of the spherical head
    if head_position == 'origin':
        head_center = numpy.array([0, 0, 0])
    elif head_position == 'interaural center':
        head_center = numpy.mean(coordinates.cartesian, axis=0)
    elif isinstance(head_position, pyfar.Coordinates):
        if head_position.csize != 1:
            raise ValueError("head_position must contain a single point")
        head_center = head_position.cartesian.flatten()
    else:
        raise ValueError("head_position must be 'origin', 'interaural center'"
                         "or a pyfar Coordinates object")

    # per definition the sphere is in the origin of coordinates. To realize
    # an off-center spherical head model, the coordinates are translated
    coordinates.cartesian -= head_center

    # the spherical head model from Duda and Martens takes only a single angle,
    # `theta`due to the symmetry of the sphere. This angle is the great cicle
    # distance between the ear positions and the source positions

    # prepare angles to be of the required shape
    # source angles must be of shape [1, number of sources]
    source_elevation = coordinates.elevation[None, :]
    source_azimuth = coordinates.azimuth[None, :]
    # ear angles must be of the shape [number of ears, 1]
    ear_elevation = head.elevation[:, None]
    ear_azimuth = head.azimuth[:, None]
    # great circle distance of shape (number of ear, number of sources)
    theta = numpy.arccos(
        numpy.sin(source_elevation) * numpy.sin(ear_elevation) +
        numpy.cos(source_elevation) * numpy.cos(ear_elevation) *
        numpy.cos(source_azimuth - ear_azimuth))

    # get unique combinations of theta and radii to avoid computing the same
    # SHTF more than once

    # all source positions of shape (number of sources, 2) defined by great
    # circle distance and radius in last dimension
    positions = numpy.vstack((theta.flatten(),
                           numpy.tile(coordinates.radius, 2))).T
    positions_unique, idx_to_unique, idx_from_unique = numpy.unique(
        positions, return_index=True, return_inverse=True, axis=0)

    frequencies = pyfar.dsp.fft.rfftfreq(n_samples, sampling_rate)

    # 0 Hz can not be computed with the below. Take a small frequency instead
    # to get an approximate value.
    frequencies_compute = frequencies.copy()
    # This takes the frequency with wave length of hundred time the spherical
    # head diameter (19.6 Hz for the default values)
    small_frequency = speed_of_sound / (200 * head.radius[0])
    frequencies_compute[0] = min(small_frequency, frequencies_compute[1])

    if reference_distance == 'coordinates':
        reference_distance = numpy.tile(coordinates.radius, 2)
        reference_distance = reference_distance[idx_to_unique]
    elif isinstance(reference_distance, (int, float)) and \
            reference_distance > 0:
        reference_distance = numpy.ones_like(idx_to_unique) * reference_distance
    elif isinstance(reference_distance, (list, numpy.ndarray)):
        reference_distance = numpy.tile(reference_distance.flatten(), 2)
        reference_distance = reference_distance[idx_to_unique]
    else:
        raise ValueError(
            "reference_distance must be 'coordinates', and int or float > 0")

    # get SHTFs ---------------------------------------------------------------
    # compute spectrum
    shtf = sound_pressure_on_sphere(
        float(head.radius[0]), positions_unique[:, 0], positions_unique[:, 1],
        reference_distance, frequencies_compute, speed_of_sound, n_max)

    # bring back to original shape
    # (reverse unique operation and flattening across the two ears)
    freq = shtf.freq[idx_from_unique]
    freq_left = freq[:coordinates.csize]
    freq_right = freq[coordinates.csize:]
    freq = numpy.concatenate((freq_left[:, None, :], freq_right[:, None, :]), 1)
    shtf.freq = freq

    # force the (almost) 0 Hz bin and bin at the Nyquist frequency to be real
    shtf.freq[..., 0] = numpy.abs(shtf.freq[..., 0])
    if n_samples % 2:
        shtf.freq[..., -1] = numpy.abs(shtf.freq[..., -1])

    # make it a Signal
    shtf = pyfar.Signal(shtf.freq, sampling_rate, n_samples, domain='freq')

    # shift peak to positive times
    shift_samples = numpy.round(
        float(head.radius[0]) / speed_of_sound * sampling_rate) + 20
    shtf = pyfar.dsp.time_shift(shtf, shift_samples, mode='cyclic')

    return shtf


def sound_pressure_on_sphere(
        sphere_radius, theta, distance, reference_distance, frequencies,
        speed_of_sound, n_max):
    r"""
    Compute complex valued sound pressure on the surface of a sphere.

    The pressure emitted by a free field point source at a point on a rigid
    sphere :math:`p_\text{s}` is computed according to the iterative
    formulation given by Duda & Martens [#]_. It is referenced to the pressure
    of a point source :math:`p_\text{s}` observed at the center of the sphere
    with the sphere being absent

    :math:`H = \frac{p_\text{s}}{p_\text{ff}}\,.`

    This implementation is an extended version, that allows different distances
    for :math:`p_\text{s}` and :math:`p_\text{ff}`.

    .. note ::

        The parameters `theta`, `distance`, or `reference_distance` can be
        array likes but must be broadcastable to a common shape.

    Parameters
    ----------
    sphere_radius : float
        The radius of the sphere in meter.
    theta : float, array like
        Angle (great circle distance) in radians between the point source
        :math:`p_\text{s}` and the point on the sphere at which the pressure is
        computed.
    distance : float, array like
        Distance of the point source :math:`p_\text{s}` to the center of the
        sphere in meter.
    reference_distance : float, array like
        Distance of the referencing point source :math:`p_\text{ff}` to the
        center of the sphere in meter. Often the same as `distance`.
    frequencies : float, array like
        The frequencies in Hz, for which the sound pressure is computed. Must
        be greater than 0.
    speed_of_sound : float
        The speed of sound in meters per second.
    n_max : int
        The spherical harmonic oder up to which the sound pressure is computed.

    Returns
    -------
    pressure : pyfar.FrequencyData
        The sound pressure on the sphere.

    References
    ----------
    [#] Duda, R. O., and Martens, W. L. (1998). “Range dependence of the
        response of a spherical head model,” J. Acoust. Soc. Am., 104,
        3048-3058.
    """
    # Note: Duda and Martens provide a Matlab implementation in their appendix.
    # This function uses the same variable names to make it easier to compare
    # the code to the publication.

    # input checks
    frequencies = numpy.array(frequencies)
    if numpy.any(frequencies <= 0):
        raise ValueError('all frequencies must be greater than zero')

    # allocate array
    pressure = numpy.ones((theta.size, frequencies.size), dtype=complex)

    # Normalized frequencies according to Eq. (4)
    mu = frequencies * 2 * numpy.pi * sphere_radius / speed_of_sound

    # Normalized distances according to Eq. (5)
    rho = distance / sphere_radius
    rho_0 = reference_distance / sphere_radius

    for nn in range(len(theta)):

        # argument and initialization for recursive computation of Legendre
        # polynomial according to Eq. (3) and (A9)
        # initialize legendre Polynom for order m=0 (P2) and m=1 (P1)
        x = numpy.cos(theta[nn])
        P2 = 1
        P1 = x

        # initialize the calculation of the Hankel fraction. Appendix 1 and 2
        # in Duda % Martens
        zr = 1 / (1j * mu * rho[nn])
        za = 1 / (1j * mu)
        Qr2 = zr
        Qr1 = zr * (1 - zr)
        Qa2 = za
        Qa1 = za * (1 - za)

        # initialize the sum in Eq. (A10)
        sum = 0

        # calculate the sum for m=0
        term = zr / (za * (za - 1))
        sum = sum + term

        # calculate sum for m=1
        if n_max > 0:
            term = (3 * x * zr * (zr - 1)) / (za * (2 * za**2 - 2 * za + 1))
            sum = sum + term

        # Computations below can become NaN for high SH orders and low
        # frequencies. However, we don't need the high orders for low
        # frequencies and simply discard invalid values. These warnings only
        # occur for very low frequencies, and are filtered out to avoid
        # command line clutter in these cases.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', 'overflow encountered', RuntimeWarning)
            warnings.filterwarnings(
                'ignore', 'invalid value encountered', RuntimeWarning)

            # calculate the sum for 2 <= m <= Nsh
            for m in range(2, n_max + 1):

                # recursive calculation of the Legendre polynomial of order m
                # (see doc legendreP)
                P = ((2 * m - 1) * x * P1 - (m - 1) * P2) / m

                # recursive calculation of the Hankel fraction
                Qr = - (2 * m - 1) * zr * Qr1 + Qr2
                Qa = - (2 * m - 1) * za * Qa1 + Qa2

                # update the sum and recursive terms
                term = ((2 * m + 1) * P * Qr) / ((m + 1) * za * Qa - Qa1)

                # only consider valid values
                idx = ~numpy.isnan(term)
                sum[idx] += term[idx]

                # update variables
                Qr2 = Qr1
                Qr1 = Qr
                Qa2 = Qa1
                Qa1 = Qa
                P2  = P1
                P1  = P

        # calculate the pressure - Eq. (A10) in Duda & Martens
        pressure[nn] = (
            rho_0[nn] * numpy.exp(1j * (mu * rho[nn] - mu * rho_0[nn] - mu) ) *
            sum) / (1j * mu)

    # Duda & Marten use Fourier convention with the negative exponent for the
    # inverse transform - cf. Eq. (13). Since pyfar uses the opposite
    # convention the pressure is conjugated
    pressure = pyfar.FrequencyData(numpy.conj(pressure), frequencies)

    return pressure
