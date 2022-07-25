from mbientlab.metawear import *
from time import sleep
import freefield
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
        libmetawear.mbl_mw_sensor_fusion_stop(sensor.device.board);
        # unsubscribe to signal
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sensor.device.board, SensorFusionData.EULER_ANGLE);
        libmetawear.mbl_mw_datasignal_unsubscribe(signal)
        # disconnect
        libmetawear.mbl_mw_debug_disconnect(sensor.device.board)
        while not sensor.device.is_connected:
            sleep(0.1)
        print('sensor disconnected')

def get_pose(sensor, n_datapoints=20):
    pose_log = numpy.zeros((n_datapoints, 2))
    pose = numpy.array((sensor.pose.yaw, sensor.pose.roll))
    n = 0
    while n < n_datapoints:  # filter invalid values
        if not any(numpy.isnan(pose)) and all(-180 <= _pose <= 360 for _pose in pose)\
                and not all(-1e-3 <= _pose <= 1e-3 for _pose in pose):
            if pose[0] > 180:
                pose[0] -= 360
            pose_log[n] = pose
        n += 1
    d = numpy.abs(pose_log - numpy.median(pose_log))  # deviation from median
    mdev = numpy.median(d)  # median deviation
    s = d / mdev if mdev else numpy.zeros_like(d)  # factorized mean deviation to detect outliers
    pose = numpy.array((numpy.mean(pose_log[:,0][(s < 2)[:,0]]), numpy.mean(pose_log[:,1][(s < 2)[:,1]])))
    return pose

def print_pose(pose):
    if all(pose):
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:
        print('no head pose detected', end="\r", flush=True)

def test_pose(n_datapoints=50):
    sensor = start_sensor()
    while True:
        pose = get_pose(sensor, n_datapoints)
        print_pose(pose)

def calibrate_pose(sensor, limit=0.5, report=True):
    [led_speaker] = freefield.pick_speakers(23)  #s get object for center speaker LED
    freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                    processors=led_speaker.digital_proc)  # illuminate LED
    print('rest at center speaker and press button to start calibration...')
    freefield.wait_for_button()  # start calibration after button press
    log = numpy.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        pose = get_pose(sensor)
        # print(pose)
        log = numpy.vstack((log, pose))
        # check if orientation is stable for at least 30 data points
        max_logsize = 2000
        if len(log) > max_logsize:
            diff = numpy.mean(numpy.abs(numpy.diff(log[-max_logsize:], axis=0)), axis=0).astype('float16')
            if report:
                print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
            if diff[0] < limit and diff[1] < limit:  # limit in degree
                break
    freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    pose_offset = numpy.around(numpy.mean(log[-int(max_logsize/2):].astype('float16'), axis=0), decimals=2)
    print('calibration complete, thank you!')
    return pose_offset


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