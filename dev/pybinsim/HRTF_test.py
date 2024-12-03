import matplotlib
matplotlib.use('TkAgg')
from dev.pybinsim.Pulse_Stream import Pulse_Stream
import pybinsim
import logging
import argparse
import freefield
import numpy
import slab
from pathlib import Path
from pythonosc import udp_client
import threading

data_dir = Path.cwd() / 'data'
filename='kemar'
freefield.initialize(setup='dome', default=None, device=None, sensor_tracking=True)

class Test:
    def __init__(self, filename='kemar'):
        # general parameters
        self.filename = filename
        self.hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / str(filename + '.sofa'))
        self.target = numpy.array((0, 0))
        self.osc_client = self._make_osc_client()

        # todo fix audio glitches when threading
        # self.pulse_stream = Pulse_Stream(self.filename)
        self.binsim = self._init_pybinsim(self.filename)
        self.audio_stream = threading.Thread(target=self._binsim_start, args=(self.binsim,))

        self.filter_idx = 0  # filter index in the initial osc message

    def run(self):
        self._wait_for_button('Press Enter to start.')
        freefield.calibrate_sensor(led_feedback=False, button_control=False)
        self.audio_stream.start()
        # self.pulse_stream.start()
        # self.pulse_stream.interval_duration = 0
        while True:
            self.headpose = freefield.get_head_pose()  # read headpose from sensor
            self.set_filter()  # set the HRIR to pybinsim

    def set_filter(self):
        rel_coords = self.target - self.headpose
        # convert coordinates to HRTF convention (=physics convention)
        if rel_coords[0] > 0:
            rel_coords[0] = 360 - rel_coords[0]
        elif rel_coords[0] < 0:
            rel_coords[0] *= -1
        # find idx of the nearest filter in the hrtf
        filter_coords = self.hrtf._get_coordinates((rel_coords[0], rel_coords[1],
                        self.hrtf.sources.vertical_polar[0,2]), 'spherical').cartesian
        distances = numpy.sqrt(((filter_coords - self.hrtf.sources.cartesian) ** 2).sum(axis=1))
        next_idx = int(numpy.argmin(distances))
        if next_idx != self.filter_idx:
            # change filter
            filter_msg = [0, next_idx, 0, 0, 0, 0, 0]
            self.osc_client.send_message('/pyBinSim', filter_msg)
            self.filter_idx = next_idx

    @staticmethod
    def _make_osc_client():
        # Create OSC client
        host = '127.0.0.1'
        mode = 'client'
        ip = '127.0.0.1'
        port = 10000
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default=host)
        parser.add_argument("--mode", default=mode)
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
        return osc_client

    @staticmethod
    def _wait_for_button(*msg, button=''):
        response = None
        while response != button:
            if msg: response = input(msg)
            else: response = input('Waiting for button.')
        return response

    @staticmethod
    def _init_pybinsim(filtername):
        # init binsim object
        binsim = pybinsim.BinSim(Path.cwd() / 'data' / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
        pybinsim.logger.setLevel(logging.DEBUG)  # defaults to INFO
        return binsim

    @staticmethod
    def _binsim_start(binsim):
        binsim.stream_start()

if __name__ == "__main__":
    self = Test('kemar')
    self.run()