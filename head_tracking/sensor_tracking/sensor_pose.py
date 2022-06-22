import freefield
from pathlib import Path
import numpy
data_dir = Path.cwd() / 'data'
proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
             ['RX82', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
             ['RP2', 'RP2', data_dir / 'rcx' / 'arduino_analog.rcx']]
# freefield.initialize('dome', zbus=True, device=proc_list)
# freefield.set_logger('WARNING')

# read arduino data
def get_pose(report=False):
    az = freefield.read(tag='azimuth', processor='RP2', n_samples=1)
    ele = freefield.read(tag='elevation', processor='RP2', n_samples=1)
    az = numpy.interp(az, [0.55, 2.75], [0, 360])
    ele = numpy.interp(ele, [0.55, 2.75], [-90, 90])
    if report:
        print('az: %f,  ele: %f' % (az, ele), end="\r", flush=True)
    return numpy.array((az, ele))

def calibrate_pose(limit=0.11, report=True):
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
        freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                        processors=led_speaker.digital_proc)  # illuminate LED
        print('rest at center speaker and press button to start calibration...')
        freefield.wait_for_button()  # start calibration after button press
        log = numpy.zeros(2)
        while True:  # wait in loop for sensor to stabilize
            pose = get_pose()
            # print(pose)
            log = numpy.vstack((log, pose))
            # check if orientation is stable for at least 30 data points
            if len(log) > 500:
                diff = numpy.mean(numpy.abs(numpy.diff(log[-500:], axis=0)), axis=0).astype('float16')
                if report:
                    print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
                if diff[0] < limit and diff[1] < limit:  # limit in degree
                    break
        freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
        pose_offset = numpy.around(numpy.mean(log[-20:].astype('float16'), axis=0), decimals=2)
        print('calibration complete, thank you!')
        return pose_offset

def test_sensor():
    if not freefield.PROCESSORS.mode:  # avoid reinitializing every time
        freefield.initialize('dome', zbus=True, device=proc_list)
    freefield.set_logger('warning')
    offset = calibrate_pose(report=True)
    response = 0
    while not response:
        pose = get_pose()
        if pose[0] != None and pose[1] != None:
            pose = pose - offset
            print(pose, end="\r", flush=True)
            response = freefield.read('response', processor='RP2')
        else:
            print('no marker detected', end="\r", flush=True)
