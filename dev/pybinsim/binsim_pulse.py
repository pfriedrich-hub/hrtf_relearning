import pybinsim
import logging
from pathlib import Path
import threading
import argparse
from pythonosc import udp_client
import queue
import time
wav_dir = Path.cwd() / 'data' / 'sounds'

class Binsim_Pulse:
    def __init__(self, filepath):
        self.interval_duration = 0
        self.filterpath = Path.cwd() / 'data' / 'hrtf' / 'wav' / 'kemar'
        self.osc_msgr = self.make_osc_msgr()
        self.set_pulse = threading.Thread(target=self.start, args=('kemar',))
        self.interval_queue = queue.Queue()
        
    @staticmethod
    def make_osc_msgr():
        ip = '127.0.0.1'
        port = 10000
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        return udp_client.SimpleUDPClient(args.ip, args.port)

    def start(self):
        pybinsim.logger.setLevel(logging.DEBUG)  # defaults to INFO
        # Use logging.WARNING for printing warnings only
        with pybinsim.BinSim(f'{self.filterpath}_settings.txt') as binsim:
            binsim.stream_start()

        while True:
            try:
                interval = self.interval_queue.get(timeout=1)  # update interval duration
            except queue.Empty:
                pass
            self.osc_msgr.send_message('/pyBinSimFile', wav_dir / 'pinknoise_44100.wav')
            time.sleep(interval)
            self.osc_msgr.send_message('/pyBinSimFile', wav_dir / 'silence_44100.wav')
            time.sleep(interval)

    def halt(self):
        self.binsim_thread.join()  # end binsim thread




# queueing example
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