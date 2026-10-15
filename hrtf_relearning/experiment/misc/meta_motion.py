from __future__ import print_function
import time
import numpy
from scipy.stats import zscore
import logging
try:
    from mbientlab.warble import *
    from mbientlab.metawear import *
except  ModuleNotFoundError:
    mbientlab = None
    logging.warning('Could not import mbientlab - working with motion sensor is disabled')

class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
        self.pose = None
    # callback
    def data_handler(self, ctx, data):
        self.pose = parse_value(data)
        self.samples += 1

def get_device():
    """
    Scan Bluetooth environment for the motion sensor and connect to the device.
    Returns:
        (instance of Sensor): Object for handling the initialized sensor.
    """
    def handler(result):
        devices[result.mac] = result.name
    devices = {}
    BleScanner.set_handler(handler)
    BleScanner.start()
    # make a choice if multiple sensors are found
    while True:
        logging.info("Scanning for motion sensor")
        t_start = time.time()
        while not time.time() > t_start + 5:
            time.sleep(0.1)  # scanning for devices time
        if 'MetaWear' in devices.values():
            mac_list = []
            for idx, device in enumerate(devices.values()):
                if device == 'MetaWear':
                    mac_list.append(list(devices.keys())[idx])
            if len(mac_list) > 1:
                logging.warning('More than one motion sensor detected.\nChoose a sensor:')
                for idx, mac_id in enumerate(mac_list):
                    print(f'{idx} {mac_id}\n')
                address = mac_list[int(input())]
            else:
                address = mac_list[0]
            break
    BleScanner.stop()
    logging.info("Connecting to motion sensor (MAC: %s)" % (address))
    device = MetaWear(address)
    while not device.is_connected:
        try:
            device.connect()
        except:
            logging.debug('Connecting to motion sensor')
    return device

class Sensor:
    def __init__(self, state):
        self.state = self._connect(state)
        self.convention = 'psychoacoustics'
        self.is_calibrated = False
        #: Raw (gravity-referenced) pose at every `calibrate()` call this
        #: session, in order: one row per trial, [azimuth, elevation]. The
        #: sensor runs in NDOF, so the elevation column is a real pitch rather
        #: than a relative reading -- `calibrate` then zeroes it, which is why
        #: nothing downstream can see what it was.
        #:
        #: BOOKKEEPING ONLY. Nothing checks it live, deliberately: the
        #: headband comes off and is refit within a session, which moves the
        #: sensor's mounting offset, so there is no value a warning could be
        #: written against that would not fire on normal procedure. What it IS
        #: good for is afterwards -- a step change between consecutive trials
        #: marks a refit, so a session's blocks can be segmented into
        #: alignment epochs and asked whether anything moved WITHIN one. The
        #: guarantee that the laser matches the head is manual (Paul,
        #: 2026-09-11); this only makes it auditable.
        self.calibration_log = []
        #: Details of the most recent `calibrate()` (see its docstring).
        self.last_calibration = None

    @staticmethod
    def _connect(state):
        logging.debug("Configuring motion sensor")
        # setup ble
        libmetawear.mbl_mw_settings_set_connection_parameters(state.device.board, 7.5, 7.5, 0, 6000)
        # Calculates absolute orientation from accelerometer, gyro, and magnetometer:
        libmetawear.mbl_mw_sensor_fusion_set_mode(state.device.board, SensorFusionMode.NDOF)
        # Calculates relative orientation in space from accelerometer and gyro data:
        # libmetawear.mbl_mw_sensor_fusion_set_mode(state.device.board, SensorFusionMode.IMU_PLUS)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(state.device.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(state.device.board, SensorFusionGyroRange._2000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(state.device.board)
        # get quat signal and subscribe
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(state.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_datasignal_subscribe(signal, None, state.callback)
        # start acc, gyro, mag
        libmetawear.mbl_mw_sensor_fusion_enable_data(state.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_sensor_fusion_start(state.device.board)
        logging.info('Motion sensor connected and running')
        state.pose_offset = None
        return state

    @staticmethod
    def _validate(pose):
        if not any(numpy.isnan(pose)) and all(1e-6 <= abs(_pose) <= 360 for _pose in pose):
            return True
        else:
            return False

    def get_pose(self, n_datapoints=1, calibrate=True, print_pose=False):
        if not self.state.device.is_connected:
            logging.warning('Sensor connection lost! Reconnect? (Y/n)')
            if input().upper() == 'Y':
                self.halt()
                self.state.device.connect()
        pose_log = []
        while len(pose_log) < n_datapoints:  # filter invalid values
            pose = numpy.array((self.state.pose.yaw, self.state.pose.roll))
            if self._validate(pose):
                pose_log.append(pose)
        # self._remove_outliers(pose_log, threshold=1)
        pose = numpy.mean(pose_log, axis=0).astype('float16')
        if self.convention == 'psychoacoustics':
            pose[0] = (pose[0] + 180) % 360 - 180
        if calibrate:
            if not self.is_calibrated:
                logging.warning("Device not calibrated")
            else:
                if self.convention == 'psychoacoustics':  # normalize AZ to [0, 360] range
                    pose[0] = (pose[0] - self.pose_offset[0] + 180) % 360 - 180
                if self.convention == 'physics':
                    pose[0] = (pose[0] - self.pose_offset[0]) % 360
                    if pose[0] < 0: pose[0] = pose[0] + 360
                pose[1] = (pose[1] - self.pose_offset[1])
        if print_pose:
                logging.info('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        return pose

    def calibrate(self, window_s=0.25, stable_mad=0.5, timeout_s=1.5,
                  check_n=5, check_tol=3.0, max_attempts=3):
        """
        Set the 0 deg azimuth / 0 deg elevation offset to the current head pose.

        The offset is the MEDIAN over `window_s` of DISTINCT sensor updates
        (counted by `State.samples`), taken once that window is still: median
        absolute deviation <= `stable_mad` deg on both axes. The previous version
        re-read the cached `state.pose` ~100 times within a millisecond, so its
        stability test always passed on one or two sensor packets and a single
        glitched packet became the zero for a whole trial (AM 15.10: still head
        reading +56 deg / +152 deg from trial start). It also averaged in a
        leading row of zeros, biasing the zero by raw_heading/102 (~1.5 deg).

        After setting the offset, `check_n` fresh calibrated samples must have a
        median within `check_tol` deg of (0, 0); otherwise it re-calibrates, up
        to `max_attempts` times. Never blocks: if the head never settles within
        `timeout_s`, or every attempt fails the check, it keeps the last result
        and logs a warning.

        Returns `self.last_calibration` (dict): offset_az, offset_el (raw
        sensor frame), spread_az, spread_el (MAD, deg), stable, n_samples,
        residual (deg, from the check), attempts.
        """
        info = None
        for attempt in range(1, max_attempts + 1):
            az, el, mad_az, mad_el, stable, n = self._measure_offset(
                window_s, stable_mad, timeout_s)
            self.pose_offset = numpy.array((az, el), dtype=float)
            self.is_calibrated = True
            residual = self._check_zero(check_n)
            info = dict(offset_az=az, offset_el=el, spread_az=mad_az,
                        spread_el=mad_el, stable=bool(stable), n_samples=int(n),
                        residual=residual, attempts=attempt)
            if residual is not None and residual <= check_tol:
                break
            logging.warning('Calibration check failed (attempt %d/%d): pose %s deg '
                            'from zero right after calibrating -- recalibrating',
                            attempt, max_attempts,
                            'unknown' if residual is None else '%.1f' % residual)
        else:
            logging.warning('Calibration check still failing after %d attempts; '
                            'keeping the last offset', max_attempts)
        if not info['stable']:
            logging.warning('Head did not settle within %.1f s; calibrated on the '
                            'last %.2f s anyway (spread az %.1f, el %.1f deg)',
                            timeout_s, window_s, info['spread_az'], info['spread_el'])
        self.last_calibration = info
        self.calibration_log.append([float(info['offset_az']),
                                     float(info['offset_el'])])
        logging.debug('Sensor calibration complete: %s', info)
        return info

    def _wait_new_sample(self, last, timeout=0.5):
        """Block until the sensor delivers a packet after count `last`. Returns
        the new count, or None if nothing arrived within `timeout` s."""
        t_end = time.time() + timeout
        while self.state.samples == last:
            if time.time() > t_end:
                return None
            time.sleep(0.001)
        return self.state.samples

    @staticmethod
    def _robust_center(az, el):
        """Median and MAD of yaw (wrap-safe around +-180) and pitch."""
        az, el = numpy.asarray(az, float), numpy.asarray(el, float)
        ref = az[-1]
        d = (az - ref + 180) % 360 - 180
        med_d = numpy.median(d)
        az_c = (ref + med_d + 180) % 360 - 180
        el_c = numpy.median(el)
        return (float(az_c), float(el_c),
                float(numpy.median(numpy.abs(d - med_d))),
                float(numpy.median(numpy.abs(el - el_c))))

    def _measure_offset(self, window_s, stable_mad, timeout_s):
        """Sliding window of distinct raw samples until it is still (or timeout)."""
        buf = []                       # (t, az, el), raw: calibrate=False
        t_stop = time.time() + timeout_s
        last = self.state.samples
        stable = False
        while True:
            nxt = self._wait_new_sample(last)
            if nxt is None:
                logging.warning('Motion sensor delivered no new samples for 0.5 s '
                                'during calibration')
                if time.time() > t_stop:
                    break
                continue
            last = nxt
            p = self.get_pose(calibrate=False)
            now = time.time()
            buf.append((now, float(p[0]), float(p[1])))
            while buf[0][0] < now - window_s:
                buf.pop(0)
            if len(buf) >= 5 and now - buf[0][0] >= 0.8 * window_s:
                _, _, mad_az, mad_el = self._robust_center(
                    [b[1] for b in buf], [b[2] for b in buf])
                if mad_az <= stable_mad and mad_el <= stable_mad:
                    stable = True
                    break
            if now > t_stop:
                break
        if not buf:  # sensor silent the whole time: fall back to one read
            p = self.get_pose(calibrate=False)
            buf = [(time.time(), float(p[0]), float(p[1]))]
        az, el, mad_az, mad_el = self._robust_center(
            [b[1] for b in buf], [b[2] for b in buf])
        return az, el, mad_az, mad_el, stable, len(buf)

    def _check_zero(self, n):
        """Median distance (deg) of `n` fresh calibrated samples from (0, 0);
        None if the sensor delivered nothing."""
        az, el = [], []
        last = self.state.samples
        for _ in range(n):
            nxt = self._wait_new_sample(last)
            if nxt is None:
                break
            last = nxt
            p = self.get_pose(calibrate=True)
            az.append(float(p[0])); el.append(float(p[1]))
        if not az:
            return None
        return float(numpy.hypot(numpy.median(az), numpy.median(el)))

    def halt(self):
        """
        Disconnect the motion sensor.
        """
        if self.state:
            libmetawear.mbl_mw_sensor_fusion_stop(self.state.device.board)
            # unsubscribe to signal
            signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.state.device.board, SensorFusionData.EULER_ANGLE)
            libmetawear.mbl_mw_datasignal_unsubscribe(signal)
            # disconnect
            libmetawear.mbl_mw_debug_disconnect(self.state.device.board)
            # while not sensor.device.is_connected:
            time.sleep(0.5)
            self.state.device.disconnect()
            self.state = None
            logging.info('Motion sensor disconnected')

    def set_fusion_mode(self, fusion_mode):
        """
        Change the fusion mode of the sensor.
        Arguments:
            fusion_mode (str): Fusion mode of the sensor.
            NDoF: Calculates absolute orientation from accelerometer, gyro, and magnetometer
            IMUPlus: Calculates relative orientation in space from accelerometer and gyro data
            Compass: Determines geographic direction from th Earth’s magnetic field
            M4G: Similar to IMUPlus except rotation is detected with the magnetometer
        """
        mode = getattr(SensorFusionMode, fusion_mode.upper())
        libmetawear.mbl_mw_sensor_fusion_set_mode(self.state.device.board, mode)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.state.device.board)
        logging.info(f'Sensor fusion mode set to {fusion_mode}')

    @staticmethod
    def _remove_outliers(pose_log, threshold=3.0):
        """
        Remove outliers from a 2D coordinate array using z-score.

        Parameters:
        - pose_log: list of length n_datapoints
        - threshold: float, z-score threshold

        Returns:
        - filtered_coords: numpy array with outliers removed
        """
        # pose_log = numpy.asarray(pose_log)
        # z_scores = numpy.abs(zscore(pose_log, axis=0))
        # mask = (z_scores < threshold).all(axis=1)
        # return pose_log[mask]
        pose_log = numpy.asarray(pose_log)
        d = numpy.abs(pose_log - numpy.median(pose_log))  # deviation from median
        mdev = numpy.median(d)  # median deviation
        s = d / mdev if mdev else numpy.zeros_like(d)  # factorized mean deviation to detect outliers
        pose = numpy.array((numpy.mean(pose_log[:, 0][(s < 2)[:, 0]]), numpy.mean(pose_log[:, 1][(s < 2)[:, 1]])))
        return pose

