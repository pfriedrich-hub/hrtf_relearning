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

        # init binsim object and assign thread to handle stream
        self.binsim = self._init_pybinsim(filtername)
        self.audio_stream = threading.Thread(target=self._binsim_start, args=(self.binsim,))

        # init thread and input queue to control pulse interval
        self.interval_queue = queue.Queue()
        self.set_interval = threading.Thread(target=self.make_pulse, args=())

        # set initial interval duration
        self.interval_duration = -1

    @staticmethod
    def _init_pybinsim(filtername):
        # init binsim object
        binsim = pybinsim.BinSim(Path.cwd() / 'data' / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
        pybinsim.logger.setLevel(logging.INFO)  # defaults to INFO
        return binsim

    @staticmethod
    def _binsim_start(binsim):
        binsim.stream_start()

    def make_pulse(self):
        while True:
            # get input queue value and set pulse interval duration
            try:
                self.interval_duration = self.interval_queue.get(timeout=1e-6)  # update interval duration
            except queue.Empty:
                pass
            if self.interval_duration == -1:
                self.binsim.config.configurationDict['loudnessFactor'] = 0
            elif self.interval_duration == 0:
                self.binsim.config.configurationDict['loudnessFactor'] = 0.5
            elif self.interval_duration > 0:
                time.sleep(self.interval_duration / 1000)
                self.binsim.config.configurationDict['loudnessFactor'] = 0
                time.sleep(self.interval_duration / 1000)
                self.binsim.config.configurationDict['loudnessFactor'] = 0.5

    # start audio streaming and pulse interval threads
    def start(self):
        self.audio_stream.start()
        self.set_interval.start()

    # update pulse interval
    def update_interval(self, interval_duration):
        """
        interval duration: (float, int): pulse interval duration in ms
        """
        with self.interval_queue.mutex:  # clear queue
            self.interval_queue.queue.clear()
        self.interval_queue.put(interval_duration)  # add new value

    def halt(self):
        self.update_interval(-1)


# self = Pulse_Stream('kemar')
# self.start()
# time.sleep(3)
# self.halt()



# self.binsim.stream.stop_stream()
# self.start()


# if __name__ == "__main__":
#     pulse = Pulse_Stream('kemar')
#     pulse.start()

# import threading
# import queue
#
# def drive(speed_queue):
#     speed = 1
#     while True:
#         try:
#             speed = speed_queue.get(timeout=1)
#             if speed == 0:
#                 break
#         except queue.Empty:
#             pass
#         print("speed:", speed)
#
# def main():
#     speed_queue = queue.Queue()
#     threading.Thread(target=drive, args=(speed_queue,)).start()
#     while True:
#         speed = int(input("Enter 0 to Exit or 1/2/3 to continue: "))
#         speed_queue.put(speed)
#         if speed == 0:
#             break
#
# main()