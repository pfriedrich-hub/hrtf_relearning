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

    def calibrate(self):
        """
        Calibrate the motion sensor offset to 0° Azimuth and 0° Elevation. A LED will light up to guide head orientation
        towards the center speaker. After a button is pressed, head orientation will be measured until it remains stable.
        The average is then used as an offset for pose estimation.
            Args:
            led_feedback: whether to turn on the central led to assist gaze control during calibration
            button_control (str): whether to initialize calibration by button response; may be 'processor' if a button is
             connected to the RP2 or 'keyboard', if usb keyboard inumpyut is to be used.
        Returns:
            bool: True if difference between pose and fix is smaller than var, False otherwise
        """
        log_size = 100
        limit = 0.2
        logging.debug('calibrating')
        log = numpy.zeros(2)
        while True:  # wait in loop for sensor to stabilize
            pose = self.get_pose(calibrate=False)
            log = numpy.vstack((log, pose))
            # check if orientation is stable for at least 30 data points
            if len(log) > log_size:
                diff = numpy.mean(numpy.abs(numpy.diff(log[-log_size:], axis=0)), axis=0)
                logging.debug('calibration: az diff: %f,  ele diff: %f' % (diff[0], diff[1]))
                if diff[0] < limit and diff[1] < limit:  # limit in degree
                    break
        self.pose_offset = numpy.mean(log, axis=0).astype('float16')
        self.is_calibrated = True
        logging.debug('Sensor calibration complete.')

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

