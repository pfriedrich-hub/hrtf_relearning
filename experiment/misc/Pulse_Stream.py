import pybinsim
import logging
from pathlib import Path
import threading
import queue
import time
wav_dir = Path.cwd() / 'data' / 'sounds'

class Pulse_Stream:
    def __init__(self, filtername):
        """
        filtername (str): Name of the folder containing the filter_list.txt, settings.txt and direction filters
        """
        # init binsim
        self.binsim = self._init_pybinsim(filtername)  # init binsim object
        self.binsim.soundHandler.loopSound = True
        self.binsim.stream_start()

        self.pulse = threading.Thread(target=self.pulse, args=())  # thread to control pulse interval
        self.interval_duration = -1  # initial interval duration, -1 pauses the stream
        self.binsim.stream.start_stream()
        self.pulse.start()

    @staticmethod
    def _init_pybinsim(filtername):
        binsim = pybinsim.BinSim(Path.cwd() / 'data' / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
        pybinsim.logger.setLevel(logging.DEBUG)  # defaults to INFO
        return binsim

    def pulse(self):
        while True:
            interval_duration = getattr(self, "interval_duration")
            if interval_duration == -1 and self.binsim.stream.is_active():
                self.binsim.stream.stop_stream()
            elif interval_duration == 0 and self.binsim.stream.is_stopped():
                self.binsim.stream.start_stream()
            elif interval_duration > 0:
                if self.binsim.stream.is_stopped():
                    self.binsim.stream.start_stream()
                time.sleep(interval_duration / 1000)
                self.binsim.config.configurationDict['loudnessFactor'] = 0
                time.sleep(interval_duration / 1000)
                self.binsim.config.configurationDict['loudnessFactor'] = 0.5
            time.sleep(.001)

