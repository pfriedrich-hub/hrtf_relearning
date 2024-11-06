import argparse
from pythonosc import udp_client

import time
import numpy
from pathlib import Path
# import freefield
import slab

# choose hrtf
hrtf = slab.HRTF.kemar()
# hrtf = slab.HRTF('/Users/paulfriedrich/projects/hrtf_relearning/data/hrtf/sofa/aachen_database/MRT02.sofa')

def filters(hrtf):

    # Default values
    filt_identifier = '/pyBinSim'
    file_identifier = '/pyBinSimFile'
    ip = '127.0.0.1'
    port = 10000
    comPort = 'COM4'    # please choose the correct COM-Port
    baudRate = 57600
    print(['ComPort :', comPort])
    print(['Baudrate: ', baudRate])
    print(['IP: ', ip])
    print(['Using Port ', port])

    # Create OSC client
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
    args = parser.parse_args()
    client = udp_client.SimpleUDPClient(args.ip, args.port)

    # get list of sound files
    soundpath = Path.cwd() / 'data' / 'sounds' / 'pinknoise_pulses'
    pulses = sorted(list(soundpath.glob('*wav')))

    # cartesian sources for distance
    sources = hrtf.sources.cartesian
    r = hrtf.sources.vertical_polar[:, 2].mean()

    for i, (az, ele) in enumerate(zip(numpy.linspace(0, 360, 37),
                                      numpy.linspace(-60, 60, 37))):

        # find idx of the nearest source coordinates
        target = hrtf._get_coordinates((az, ele, r), 'spherical').cartesian
        distances = numpy.sqrt(((target - sources) ** 2).sum(axis=1))
        idx = int(numpy.argmin(distances))

        # change filter
        filter_msg = [0, idx, 0, 0, 0, 0, 0]
        client.send_message(filt_identifier, filter_msg)
        print(f'sending parameters: az: {hrtf.sources.vertical_polar[idx, 0]}'
              f' ele: {hrtf.sources.vertical_polar[idx, 1]}')

        # change sound file
        sound_msg = str(pulses[i])
        client.send_message(file_identifier, sound_msg)

        time.sleep(0.1)

if __name__ == "__main__":
    filters(hrtf)
