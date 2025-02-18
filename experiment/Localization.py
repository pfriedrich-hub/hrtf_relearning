import freefield
import slab
import datetime
import pybinsim
date = datetime.datetime.now()
date = f'{date.strftime("%d")}_{date.strftime("%m")}'
from pathlib import Path
import argparse
from pythonosc import udp_client
from experiment.Subject import Subject
from localization_analysis import *
from hrtf.processing.hrtf2wav import hrtf2wav
import logging
import threading

data_dir = Path.cwd() / 'data'

subject_id = 'Paul'
hrtf_filename = 'KU100_HRIR_L2702.sofa'

class Localization:
    def __init__(self, subject_id, hrtf_filename, n_trials=150, az_range=(-52.5, 52.5), ele_range=(-37.5, 37.5), min_dist=35):
        self.sequence = self._make_sequence(n_trials, az_range, ele_range, min_dist)
        self.filename = subject_id + '_' + date
        self.n_trials = n_trials
        self.subject = Subject(subject_id)
        self.beep = slab.Sound.tone(frequency=1000, level=70)
        self.hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / hrtf_filename)
        slab.set_default_samplerate(self.hrtf.samplerate)

        self.osc_client = self._make_osc_client()
        self.filter_idx = 0  # filter index in the initial osc message
        self.binsim = self._init_pybinsim(Path(hrtf_filename).stem)
        self.audio_stream = threading.Thread(target=self._binsim_start, args=(self.binsim,))

    # def __repr__(self):
    #     return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        freefield.initialize(setup='dome', default=None, device=None, sensor_tracking=True)
        self._wait_for_button('Press Enter to start.')
        freefield.calibrate_sensor(led_feedback=False, button_control=False)
        for self.target in self.sequence:
            # freefield.calibrate_sensor(led_feedback=False, button_control=False)
            self.sequence.add_response(self.localization_trial())
            self.wait_for_button('Press Enter to continue')

        return

    def localization_trial(self):
        self.play_sound(sound=self._make_stim(), coordinates=self.target)  # play stimulus
        self.wait_for_button('Aim at the sound and press Enter to confirm')  # wait for response
        response = freefield.get_head_pose()  # get pose
        print('Response| azimuth %.1f |elevation %.1f' % (response[0], response[1]))
        self.play_sound(sound=self.beep, coordinates=(0, 0))  # play stimulus
        return response

    def play_sound(self, sound, coordinates):
        # write sound to wav and send to pybinsim
        self.set_sound(sound)
        self.set_filter(coordinates)  # set the HRIR to pybinsim
        self.audio_stream.start()
        # self.binsim.stream_close()  #todo see if stream closes automatically when looping is set to false

    def set_sound(self, sound):
        sound.write(data_dir / 'sounds' / 'localization.wav')
        self.osc_client.send_message('/pyBinSimFile', 'localization.wav')

    def set_filter(self, coordinates):
        # get sound source coordinates relative to head pose
        self.headpose = freefield.get_head_pose()
        rel_coords = coordinates - self.headpose
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
    def _init_pybinsim(filtername):
        # init binsim object
        binsim = pybinsim.BinSim(Path.cwd() / 'data' / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
        pybinsim.logger.setLevel(logging.DEBUG)  # defaults to INFO
        return binsim

    @staticmethod
    def _binsim_start(binsim):
        binsim.stream_start()

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
            if msg:
                response = input(msg)
            else:
                response = input('Waiting for button.')
        return response

    @staticmethod
    def _make_sequence(n_trials, az_range, ele_range, min_dist):
        """
        Create a sequence of n_trials target locations
        with more than min_dist angular distance between successive targets
        """
        sequence = []
        target = (numpy.random.randint(az_range[0], az_range[1] + 1),
                  numpy.random.randint(ele_range[0], ele_range[1] + 1))
        for i in range(n_trials):
            distance = 0
            while distance < min_dist:
                next_target = (numpy.random.randint(az_range[0], az_range[1] + 1),
                               numpy.random.randint(ele_range[0], ele_range[1] + 1))
                distance = numpy.sqrt((target[0] - next_target[0]) ** 2 + (target[1] - next_target[1]) ** 2)
            sequence.append(next_target)
            target = next_target
        sequence = slab.Trialsequence(sequence)
        return sequence

    @staticmethod
    def _make_stim():
        noise = slab.Sound.pinknoise(duration=0.025, level=90).ramp(when='both')
        silence = slab.Sound.silence(duration=0.025)
        return slab.Sound.sequence(noise, silence, noise, silence, noise, silence, noise, silence, noise)



#
# if __name__ == "__main__":
#     subject = Subject(id=subject_id)
#     localization = Localization(n_repetitions=3)
#     localization_data = localization.run()
#     subject.localization_data.append({'data': localization_data, 'condition': filename, 'date': date})
#     subject.write()
#     if not (data_dir / 'hrtf' / 'wav' / str(Path(filename).stem)).exists():
#         hrtf2wav(filename)
#     localization_accuracy(data=localization_data)
