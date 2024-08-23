import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import freefield
from pathlib import Path
import time
import numpy
import slab
data_path = Path.cwd() / 'data'
fs = 24414  # RP 2 / RM1 limitation

class HRTF_test():
    def __init__(self, processor='RM1', target=(0, 0)):
        self.processor = processor
        if self.processor == 'RP2':
            self.zbus = True
            self.connection = 'zBus'
            self.led_feedback = False
            self.button_control = True
        elif self.processor == 'RM1':
            self.zbus = False
            self.connection = 'USB'
            self.led_feedback = False
            self.button_control = False
        self.target = target

    def run(self):
        # if not freefield.PROCESSORS.mode:
        freefield.initialize(setup='dome', zbus=self.zbus, connection=self.connection, sensor_tracking=True,
        device=[self.processor, self.processor, data_path / 'rcx' / f'HRTF_test_{self.processor}.rcx'])
        self.set_target()  # get next target
        self.wait_for_button()
        freefield.calibrate_sensor(led_feedback=self.led_feedback, button_control=self.button_control)
        freefield.play(1, self.processor)  # start pulse train
        try:
            while True:
                self.update_headpose()  # read headpose from sensor and send to dsp
                dst = self.pose - self.target
                distance = numpy.sqrt(numpy.sum(numpy.square(dst)))  # faster than reading from DSP
                print('Distance from target: azimuth %.1f, elevation %.1f, total %.2f'
                      % (dst[0], dst[1], distance), end="\r", flush=True)
        except KeyboardInterrupt:
            freefield.play(2, self.processor)

    def stop(self):
        freefield.halt()

    def set_target(self):
        freefield.write('target_az', self.target[0], self.processor)
        freefield.write('target_ele', self.target[1], self.processor)
        print('\n TARGET| azimuth: %.1f, elevation %.1f' % (self.target[0], self.target[1]))

    def update_headpose(self):
        # get headpose from sensor and write to processor
        self.pose = freefield.get_head_pose()
        freefield.write('head_az', self.pose[0], self.processor)
        freefield.write('head_ele', self.pose[1], self.processor)

    def wait_for_button(self):
        if self.processor == 'RP2':  # calibrate (wait for button)
            print('Press button to start sensor calibration')
        elif self.processor == 'RM1':
            input('Press Enter to start sensor calibration')

if __name__ == "__main__":
    test = HRTF_test('RM1', target=(0,0))
    test.run()