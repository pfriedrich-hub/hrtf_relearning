import freefield
from pathlib import Path
import numpy
DIR = Path.cwd()
# proc_list = [['RP2', 'RP2',  DIR / 'data' / 'rcx' / 'arduino_analog.rcx']]
# freefield.initialize('dome', zbus=True, device=proc_list)
# freefield.set_logger('WARNING')

# read arduino data
def get_pose():
    az = freefield.read(tag='azimuth', processor='RP2', n_samples=1)
    ele = freefield.read(tag='elevation', processor='RP2', n_samples=1)
    az = numpy.interp(az, [0.55, 2.75], [90, 270])
    ele = numpy.interp(ele, [0.55, 2.75], [-90, 90])
    print('azimuth: %i, elevation: %i '%(int(az), int(ele)))
    return(numpy.array[az, ele])

def calibrate_sensor(limit=0.5, report=True):
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
            if len(log) > 30:
                diff = numpy.mean(numpy.abs(numpy.diff(log[-20:], axis=0)), axis=0).astype('float16')
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
        freefield.initialize('dome', default="loctest_freefield")
    freefield.set_logger('warning')
    offset = calibrate_sensor(limit=0.5, report=True)
    response = 0
    while not response:
        pose = get_pose()
        if pose[0] != None and pose[1] != None:
            pose = pose - offset
            print(pose, end="\r", flush=True)
            response = freefield.read('response', processor='RP2')
        else:
            print('no marker detected', end="\r", flush=True)
