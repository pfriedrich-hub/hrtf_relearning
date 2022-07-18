from mbientlab.metawear import *
from time import sleep
# import freefield
import numpy

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
            print('Connecting to sensor...', end="\r", flush=True)
    # states = []
    s = (State(device))
    # configure
    print("Configuring..")
    # setup ble
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)
    # setup quaternion
    libmetawear.mbl_mw_sensor_fusion_set_mode(s.device.board, SensorFusionMode.NDOF)
    libmetawear.mbl_mw_sensor_fusion_set_acc_range(s.device.board, SensorFusionAccRange._8G)
    libmetawear.mbl_mw_sensor_fusion_set_gyro_range(s.device.board, SensorFusionGyroRange._2000DPS)
    libmetawear.mbl_mw_sensor_fusion_write_config(s.device.board)
    # get quat signal and subscribe
    signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(s.device.board, SensorFusionData.EULER_ANGLE)
    libmetawear.mbl_mw_datasignal_subscribe(signal, None, s.callback)
    # start acc, gyro, mag
    libmetawear.mbl_mw_sensor_fusion_enable_data(s.device.board, SensorFusionData.EULER_ANGLE)
    libmetawear.mbl_mw_sensor_fusion_start(s.device.board)
    print('Sensor started!')
    sleep(1.5)
    return s

# tear down
def disconnect(sensor):
        # stop
        libmetawear.mbl_mw_sensor_fusion_stop(s.device.board);
        # unsubscribe to signal
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(s.device.board, SensorFusionData.QUATERNION);
        libmetawear.mbl_mw_datasignal_unsubscribe(signal)
        # disconnect
        libmetawear.mbl_mw_debug_disconnect(sensor.device.board)
        while not sensor.device.is_connected:
            sleep(0.1)
        print('sensor disconnected')

def get_pose(sensor, n_datapoints):
    _pose = numpy.zeros((n_datapoints, 2))
    for n in range(n_datapoints):
        _pose[n] = numpy.array((sensor.pose.yaw, sensor.pose.roll))
    # remove outliers
    d = numpy.abs(_pose - numpy.median(_pose, axis=0))  # deviation from median
    mdev = numpy.median(d, axis=0)  # mean deviation
    s = d / mdev if all(mdev) else numpy.zeros_like(d)  # factorized mean deviation of each element in pose
    # _pose[:, 0] = _pose[s[:, 0] < 2][:, 0]
    # _pose[:, 1] = _pose[s[:, 1] < 2][:, 1]
    # remove outliers
    pose = numpy.mean(_pose, axis=0)
    # print(pose)
    return pose

def print_pose(pose):
    if all(pose):
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:
        print('no head pose detected', end="\r", flush=True)

def test_pose(n_datapoints=30):
    sensor = start_sensor()
    while True:
        pose = get_pose(sensor, n_datapoints)
        print_pose(pose)

def calibrate_pose(s, limit=0.11, report=True):
    # [led_speaker] = freefield.pick_speakers(23)  #s get object for center speaker LED
    # freefield.write(tag='bitmask', value=led_speaker.digital_channel,
    #                 processors=led_speaker.digital_proc)  # illuminate LED
    print('rest at center speaker and press button to start calibration...')
    # freefield.wait_for_button()  # start calibration after button press
    log = numpy.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        pose = get_pose(s)
        # print(pose)
        log = numpy.vstack((log, pose))
        # check if orientation is stable for at least 30 data points
        if len(log) > 500:
            diff = numpy.mean(numpy.abs(numpy.diff(log[-500:], axis=0)), axis=0).astype('float16')
            if report:
                print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
            if diff[0] < limit and diff[1] < limit:  # limit in degree
                break
    # freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    pose_offset = numpy.around(numpy.mean(log[-20:].astype('float16'), axis=0), decimals=2)
    print('calibration complete, thank you!')
    return pose_offset
