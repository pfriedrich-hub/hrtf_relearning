from mbientlab.metawear import *
import time
import freefield
import numpy
import numpy as np
from matplotlib import pyplot as plt

class State:
    # init
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
        self.pose = None
    # callback
    def data_handler(self, ctx, data):
        # print("QUAT: %s -> %s" % (self.device.address, parse_value(data)))
        self.pose = parse_value(data)
        self.samples+= 1

def start_sensor(device=MetaWear('E1:CD:49:19:08:19')):
    while not device.is_connected:
        try:
            device.connect()
        except:
            print('connecting to sensor', end="\r")
    # states = []
    s = (State(device))
    # configure
    print("configuring sensor")
    # setup ble
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    time.sleep(1.5)
    # setup quaternion
    libmetawear.mbl_mw_sensor_fusion_set_mode(s.device.board, SensorFusionMode.NDOF)
    libmetawear.mbl_mw_sensor_fusion_set_mode(s.device.board, SensorFusionMode.IMU_PLUS)
    libmetawear.mbl_mw_sensor_fusion_set_acc_range(s.device.board, SensorFusionAccRange._8G)
    libmetawear.mbl_mw_sensor_fusion_set_gyro_range(s.device.board, SensorFusionGyroRange._2000DPS)
    libmetawear.mbl_mw_sensor_fusion_write_config(s.device.board)
    # get quat signal and subscribe
    signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(s.device.board, SensorFusionData.EULER_ANGLE)
    libmetawear.mbl_mw_datasignal_subscribe(signal, None, s.callback)
    # start acc, gyro, mag
    libmetawear.mbl_mw_sensor_fusion_enable_data(s.device.board, SensorFusionData.EULER_ANGLE)
    libmetawear.mbl_mw_sensor_fusion_start(s.device.board)
    print('sensor started!')
    time.sleep(1.5)
    return s

# tear down
def disconnect(sensor):
        # stop
        libmetawear.mbl_mw_sensor_fusion_stop(sensor.device.board);
        # unsubscribe to signal
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sensor.device.board, SensorFusionData.EULER_ANGLE);
        libmetawear.mbl_mw_datasignal_unsubscribe(signal)
        # disconnect
        libmetawear.mbl_mw_debug_disconnect(sensor.device.board)
        while not sensor.device.is_connected:
            time.sleep(0.1)
        del sensor
        print('sensor disconnected')

def get_pose(sensor, n_datapoints=100):
    pose_log = numpy.zeros((n_datapoints, 2))
    n = 0
    while n < n_datapoints:  # filter invalid values
        pose = numpy.array((sensor.pose.yaw, sensor.pose.roll))
        if not any(numpy.isnan(pose)) and all(-180 <= _pose <= 360 for _pose in pose)\
                and not any(-1e-3 <= _pose <= 1e-3 for _pose in pose):
            if pose[0] > 180:
                pose[0] -= 360
            pose_log[n] = pose
            n += 1
    d = numpy.abs(pose_log - numpy.median(pose_log))  # deviation from median
    mdev = numpy.median(d)  # median deviation
    s = d / mdev if mdev else numpy.zeros_like(d)  # factorized mean deviation to detect outliers
    pose = numpy.array((numpy.mean(pose_log[:, 0][(s < 2)[:, 0]]), numpy.mean(pose_log[:, 1][(s < 2)[:, 1]])))
    return pose

def print_pose(pose):
    if all(pose):
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:
        print('no head pose detected', end="\r", flush=True)

def test_sensor(sensor, n_datapoints=100, timer=False):
    # sensor = start_sensor()
    log = get_pose(sensor, n_datapoints)
    t_start = time.time()
    t = True
    try:
        while t:
            pose = get_pose(sensor, n_datapoints)
            print_pose(pose)
            log = numpy.vstack((log, pose))
            if timer and (time.time() < t_start + 30):
                t = False
    except KeyboardInterrupt:
        pass
    # disconnect(sensor)
    # print('test completed')
    return log

def calibrate_pose(sensor, limit=0.2, report=False):
    [led_speaker] = freefield.pick_speakers(23)  #s get object for center speaker LED
    freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                    processors=led_speaker.digital_proc)  # illuminate LED
    # print('rest at center speaker and press button to start calibration...', end="\r", flush=True)
    freefield.wait_for_button()  # start calibration after button press
    # print('calibrating', end="\r", flush=True)
    log = numpy.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        pose = get_pose(sensor)
        # print(pose)
        log = numpy.vstack((log, pose))
        # check if orientation is stable for at least 30 data points
        max_logsize = 100
        if len(log) > max_logsize:
            diff = numpy.mean(numpy.abs(numpy.diff(log[-max_logsize:], axis=0)), axis=0).astype('float16')
            if report:
                print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
            if diff[0] < limit and diff[1] < limit:  # limit in degree
                break
    freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    pose_offset = numpy.around(numpy.mean(log[-int(max_logsize/2):].astype('float16'), axis=0), decimals=2)
    # print('calibration complete.', end="\r", flush=True)
    return pose_offset



# def config_handler():
#     print('test')
# wrapper = FnVoid_VoidP_VoidP_FnVoidVoidPtrInt(config_handler)
# config = libmetawear.mbl_mw_sensor_fusion_read_config(sensor.device.board, None, FnVoid_VoidP_Int())


# def calibrate_pose(sensor):
#     [led_speaker] = freefield.pick_speakers(23)  #s get object for center speaker LED
#     freefield.write(tag='bitmask', value=led_speaker.digital_channel,
#                     processors=led_speaker.digital_proc)  # illuminate LED
#     print('rest at center speaker and press button to start calibration...')
#     freefield.wait_for_button()  # start calibration after button press
#     libmetawear.mbl_mw_sensor_fusion_reset_orientation(sensor.device.board)


"""
# remove outliers
    # for i in range(2):
    #     d = numpy.abs(_pose[:, i] - numpy.median(_pose[:, i]))  # deviation from median
    #     mdev = numpy.median(d)  # mean deviation
    #     s = d / mdev if mdev else numpy.zeros_like(d)  # factorized mean deviation of each element in pose
    #     pose[i] = numpy.mean(_pose[s < 2, i])
    pose = numpy.mean(_pose, axis=0)
    return pose
"""