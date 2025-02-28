import pybinsim
import logging
from pathlib import Path
import multiprocessing
import time
import numpy
import slab
from pythonosc import udp_client
import freefield
import argparse
from collections import deque

data_dir = Path.cwd() / 'data'
fs = 44100
slab.set_default_samplerate(fs)


def init_osc_client():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default='127.0.0.1', help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=10000, help="The port the OSC server is listening on")
    args = parser.parse_args()
    return udp_client.SimpleUDPClient(args.ip, args.port)


def load_hrtf_sources(filename):
    hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / filename)
    return hrtf.sources


def head_tracking_worker(target, osc_client, hrtf_sources):
    while True:
        headpose = freefield.get_head_pose()
        relative_coords = headpose - target

        if relative_coords[0] > 0:
            relative_coords[0] = 360 - relative_coords[0]
        elif relative_coords[0] < 0:
            relative_coords[0] *= -1

        polar = numpy.array((relative_coords[0], relative_coords[1], hrtf_sources.vertical_polar[0, 2]))
        filter_coords = hrtf_sources._vertical_polar_to_cartesian(polar[numpy.newaxis, :])
        distances = numpy.sqrt(((filter_coords - hrtf_sources.cartesian) ** 2).sum(axis=1))
        filter_idx = int(numpy.argmin(distances))

        osc_client.send_message('/pyBinSim', [0, filter_idx, 0, 0, 0, 0, 0])
        time.sleep(0.005)


def pulse_stream_worker(filtername, pulse_queue):
    binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
    pybinsim.logger.setLevel(logging.DEBUG)
    binsim.soundHandler.loopSound = True
    binsim.stream_start()

    max_interval = 500  # Maximum pulse interval in ms
    binsim.config.configurationDict['loudnessFactor'] = 0  # Start muted

    while True:
        binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / filtername / f'{filtername}_settings.txt')
    pybinsim.logger.setLevel(logging.DEBUG)
    binsim.soundHandler.loopSound = True
    binsim.stream_start()

    max_interval = 500  # Maximum pulse interval in ms
    while True:
        distance = pulse_queue.get()
        if distance < 0:
            binsim.config.configurationDict['loudnessFactor'] = 0  # Mute between trials
        else:
            # Logarithmic scaling for organic feel
            interval_scale = (distance - 2 + 1e-9) / 30  # Normalize distance
            interval = max_interval * (numpy.log(interval_scale + 0.05) + 3) / 3  # Log scaling
            if distance < 5:
                interval = -1  # Continuous sound when within target zone

            binsim.config.configurationDict['loudnessFactor'] = 0.5
            time.sleep(interval / 1000)
            binsim.config.configurationDict['loudnessFactor'] = 0
            time.sleep(interval / 1000)


def set_target(az_range, ele_range, target_history, min_dist=30):
    while True:
        target = (numpy.random.randint(az_range[0], az_range[1]),
                  numpy.random.randint(ele_range[0], ele_range[1]))

        if all(numpy.linalg.norm(numpy.array(target) - numpy.array(prev_target)) >= min_dist for prev_target in
               target_history):
            target_history.append(target)
            return target


def play_trial_worker(target, pulse_queue, osc_client, hrtf_sources, trial_time, target_size, target_time):
    trial_start = time.time()
    time_on_target = 0
    count_down = False
    score = 0
    trial_start = time.time()
    time_on_target = 0
    count_down = False
    score = 0

    while time.time() - trial_start < trial_time:
        if pulse_queue.empty():
            continue
        distance = pulse_queue.get()

        if distance < target_size:
            if not count_down:
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False

        if time.time() > time_on_target + target_time:
            if time.time() - trial_start <= 3:
                score = 2
                pulse_queue.put(0)  # Temporarily set pulse interval to 0 to play game sound
                osc_client.send_message('/pyBinSim', [1, 0])
                pulse_queue.put(-1)  # Restore mute after sound  # Play 'coins' sound
            else:
                score = 1
                pulse_queue.put(0)  # Temporarily set pulse interval to 0 to play game sound
                osc_client.send_message('/pyBinSim', [1, 1])
                pulse_queue.put(-1)  # Restore mute after sound  # Play 'coin' sound
            break

        time.sleep(0.005)

        return score




def play_session(game_time, trial_time, target_size, target_time, az_range, ele_range, osc_client, hrtf_sources,
                 pulse_queue, pulse_worker):
    tracking_worker = multiprocessing.Process(target=head_tracking_worker, args=((0, 0), osc_client, hrtf_sources))
    tracking_worker.start()

    def calibrate_sensor():
        input("Look at the center of the screen and press Enter to calibrate the sensor...")
        freefield.calibrate_sensor(led_feedback=False, button_control=False)

    input("Press Enter to start the game...")
    start_time = time.time()
    scores = []
    target_history = deque(maxlen=5)

    while time.time() - start_time < game_time:
        target = set_target(az_range, ele_range, target_history)
        calibrate_sensor()

        pulse_queue.put(-1)  # Mute sound between trials
        trial_queue = multiprocessing.Queue()
        trial_process = multiprocessing.Process(target=play_trial_worker, args=(
        target, pulse_queue, osc_client, hrtf_sources, trial_time, target_size, target_time, trial_queue))
        trial_process.start()
        trial_process.join()
        score = trial_queue.get()
        if score is not None:
            scores.append(score)
        tracking_worker.terminate()
        tracking_worker.join()
        print(f'Trial complete: {score} points')

    print(f'Game Over! Total Score: {sum(scores)}')
    pulse_queue.put(0)  # Temporarily set pulse interval to 0 to play game sound
    osc_client.send_message('/pyBinSim', [1, 2])
    pulse_queue.put(-1)  # Restore mute after sound  # Play 'buzzer' sound
    tracking_worker.terminate()
    tracking_worker.join()
    pulse_worker.terminate()


def main(game_time, trial_time, target_size, target_time, az_range, ele_range, hrtf_filename):
    osc_client = init_osc_client()
    hrtf_sources = load_hrtf_sources(f'{hrtf_filename}.sofa')
    pulse_queue = multiprocessing.Queue()
    pulse_worker = multiprocessing.Process(target=pulse_stream_worker, args=(hrtf_filename, pulse_queue))
    pulse_worker.start()

    try:
        play_session(game_time, trial_time, target_size, target_time, az_range, ele_range, osc_client, hrtf_sources,
                     pulse_queue, pulse_worker)
    finally:
        pulse_worker.terminate()
        pulse_worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the auditory training experiment.")
    parser.add_argument("--game_time", type=int, default=90, help="Total session duration in seconds")
    parser.add_argument("--trial_time", type=int, default=10, help="Time limit per trial in seconds")
    parser.add_argument("--target_size", type=int, default=5, help="Size of the target zone")
    parser.add_argument("--target_time", type=float, default=3, help="Time required to hold gaze on target")
    parser.add_argument("--az_range", type=int, nargs=2, default=[-30, 30], help="Azimuth range for target selection")
    parser.add_argument("--ele_range", type=int, nargs=2, default=[-30, 30],
                        help="Elevation range for target selection")
    parser.add_argument("--hrtf_filename", type=str, default='KU100_HRIR_L2702', help="HRTF filename for PyBinSim")
    if not (data_dir / 'hrtf' / 'wav' / str(Path(filename).stem)).exists():
        hrtf2wav(filename)
    args = parser.parse_args()
    main(args.game_time, args.trial_time, args.target_size, args.target_time, tuple(args.az_range),
         tuple(args.ele_range), args.hrtf_filename)
