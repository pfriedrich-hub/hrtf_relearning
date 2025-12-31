import matplotlib
# matplotlib.rcParams['figure.raise_window'] = False
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy
import slab
import time
import logging
from pathlib import Path
import multiprocessing as mp
from pythonosc import udp_client
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.experiment.misc.training_helpers import meta_motion

logging.getLogger().setLevel('INFO')
import hrtf_relearning

# ------------------------ CONFIG ------------------------

# HRTF selection
sofa_name = 'universal'

# unilateral vs. binaural
ear = None
# ear = 'left'

settings = dict(
    trial_time=300,
    gain=.2,                      # loudness
)

soundfile = None
# soundfile = 'c_chord_guitar.wav'
# soundfile = 'uso_225ms_9_.wav'

show = 'IR'

# ------------------------ HRTF LOAD ------------------------

hrir = hrtf2binsim(sofa_name, ear, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
hrir_dir = hrtf_relearning.PATH / 'data' / 'hrtf' / 'binsim' / hrir.name


# ------------------------ MAIN LOOP ------------------------
def play_session():
    """
    Test session: similar to training but allows staying at the center longer.
    """
    global osc_client
    osc_client = make_osc_client(port=10003)  # pyBinSim control

    sensor_state = mp.Value("i", 0)       # 1=ready, 2=calibrate, 3=track
    target = mp.Array("f", [0, 0])        # [az, el] in (-180,180], linear el
    filter_idx_shared = mp.Value("i", -1)

    # workers
    # 1) REMOVE daemon=True here
    plot_worker = mp.Process(target=plot_current_tf, args=(filter_idx_shared, show,))
    plot_worker.start()

    tracking_worker = mp.Process(
        target=head_tracker,
        args=(target, sensor_state, filter_idx_shared)
    )
    tracking_worker.start()

    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()

    # wait for tracker init
    while sensor_state.value != 1:
        time.sleep(0.1)

    sensor_state.value = 2  # calibrate
    time.sleep(0.1)
    while sensor_state.value != 3:
        sensor_state.value = 3
        time.sleep(0.1)

    osc_client.send_message('/pyBinSimLoudness', settings['gain'])
    play_sound(
        osc_client,
        soundfile=soundfile,
        duration=float(settings['trial_time']),
        sleep=False
    )

    # 2) KEEP MAIN PROCESS ALIVE for the trial, then clean up
    try:
        # Let the session run for trial_time seconds, or interrupt with Ctrl-C
        time.sleep(settings['trial_time'])
    except KeyboardInterrupt:
        logging.info("Session interrupted by user.")
    finally:
        logging.info("Stopping workers...")
        for p in (tracking_worker, binsim_worker, plot_worker):
            if p.is_alive():
                p.terminate()
        for p in (tracking_worker, binsim_worker, plot_worker):
            p.join()
        logging.info("All workers stopped.")


# ------------------------ SUB-PROCESSES ------------------------

def binsim_stream():
    import pybinsim
    pybinsim.logger.setLevel(logging.ERROR)
    logging.info(f'Loading {hrir.name}')
    binsim = pybinsim.BinSim(hrir_dir / f'{hrir.name}_training_settings.txt')
    binsim.stream_start()

def head_tracker(target, sensor_state, filter_idx_shared):
    osc_client = make_osc_client(port=10000)
    sources = hrir.sources.vertical_polar

    # init motion sensor
    device = meta_motion.get_device()
    state = meta_motion.State(device)
    motion_sensor = meta_motion.Sensor(state)
    logging.debug('motion sensor running')
    sensor_state.value = 1

    last_idx = -1
    while True:
        if sensor_state.value == 2:
            logging.debug('Calibrating sensor..')
            motion_sensor.calibrate()
            while not motion_sensor.is_calibrated:
                time.sleep(0.1)
            sensor_state.value = 1

        elif sensor_state.value == 3:
            pose = motion_sensor.get_pose()

            # distance to target in az/el plane
            relative_coords = target[:] - pose

            # convert to HRIR convention (0..360 az) & pick nearest filter
            relative_coords[0] = (-relative_coords[0] + 360) % 360
            rel_target = numpy.array((relative_coords[0], relative_coords[1], sources[0, 2]))
            filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - sources, axis=1))
            rel_hrtf_coords = sources[filter_idx]

            # send to pyBinSim
            osc_client.send_message('/pyBinSim_ds_Filter', [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                0, 0, 0
            ])
            logging.debug(f'head tracking: filter coords: {rel_hrtf_coords}')

            # publish to plotter if changed
            if filter_idx != last_idx:
                last_idx = filter_idx
                filter_idx_shared.value = int(filter_idx)

        time.sleep(0.01)

# ------------------------ HELPERS ------------------------

def play_sound(osc_client, soundfile=None, duration=None, sleep=False):
    """Wrapper that passes the soundfile to pyBinSim. Generates noise if None."""
    if duration:
        if soundfile:  # crop file to duration
            sound = slab.Sound.read(hrir_dir / 'sounds' / soundfile)
            soundfile = 'cropped_' + soundfile
            (slab.Sound(sound.data[:int(hrir.samplerate * duration)]).ramp(duration=.03)
             .write(hrir_dir / 'sounds' / soundfile))
        else:          # generate noise with pulse duration
            soundfile = 'noise_pulse.wav'
            slab.Sound.pinknoise(duration).ramp(duration=.03).write(hrir_dir / 'sounds' / soundfile)
    else:
        # use full length
        if soundfile is None:
            soundfile = 'noise_pulse.wav'
            # ensure it exists (1s default)
            slab.Sound.pinknoise(1.0).ramp(duration=.03).write(hrir_dir / 'sounds' / soundfile)
        duration = slab.Sound(hrir_dir / 'sounds' / soundfile).duration

    logging.debug(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(hrir_dir / 'sounds' / soundfile))
    if sleep:
        time.sleep(duration)


def make_osc_client(port, ip='127.0.0.1'):
    return udp_client.SimpleUDPClient(ip, port)


# ------------------------ LIVE TF PLOTTER ------------------------

def plot_current_tf(filter_idx_shared, show, redraw_interval_s=0.05, ):
    """
    Lives in its own process. Opens a Qt figure and plots the TF of the
    current HRTF (hrir[filter_idx]) whenever the filter index changes.
    """
    global hrir
    sources = hrir.sources.vertical_polar

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Current HRTF Transfer Function")
    try:
        fig.canvas.manager.set_window_title("Live HRTF TF")
    except Exception:
        pass

    last_idx = -1
    while True:
        idx = filter_idx_shared.value
        if idx >= 0 and idx != last_idx:
            last_idx = idx
            try:
                ax.cla()
                # slab's helper draws into the provided axis
                if show == 'TF':
                    hrir[idx].tf(show=True, axis=ax)
                elif show == 'IR':
                    times = numpy.linspace(0, hrir[idx].n_samples / hrir.samplerate, hrir[idx].n_samples)
                    ax.plot(times, hrir[idx].data)
                az0, el0 = sources[idx, 0], sources[idx, 1]
                az180 = (az0 + 180) % 360 - 180
                ax.set_title(f"TF idx {idx}  |  az={az180:.1f}°, el={el0:.1f}°")
                ax.grid(True, which='both', linestyle=':', linewidth=0.6)
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception as e:
                ax.cla()
                ax.text(0.5, 0.5, f"Plot error for idx {idx}:\n{e}", ha='center', va='center')
                fig.canvas.draw_idle()
                plt.pause(0.001)

        # throttle loop
        plt.pause(redraw_interval_s)


# ------------------------ ENTRY ------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer on Windows/Qt
    play_session()
