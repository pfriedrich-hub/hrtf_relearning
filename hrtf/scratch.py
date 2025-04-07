import argparse
from pythonosc import udp_client
import numpy
import slab
from pathlib import Path
import time

filename ='KU100_HRIR_L2702'
data_dir = Path.cwd() / 'data' / 'hrtf'
hrtf_sources = slab.HRTF(data_dir / 'sofa' / f'{filename}.sofa').sources.vertical_polar

def make_osc_client(port=10000):
    host = '127.0.0.1'
    mode = 'client'
    ip = '127.0.0.1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--mode", default=mode)
    parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
    args = parser.parse_args()
    return udp_client.SimpleUDPClient(args.ip, args.port)


osc_client = make_osc_client()

filter_idx = 130
# osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                                                     float(hrtf_sources[filter_idx, 0]), float(hrtf_sources[filter_idx, 1]), 0,
#                                                     0, 0, 0])

# osc_client.send_message('/pyBinSimPauseAudioPlayback','True')
# osc_client.send_message('/pyBinSimLoudness',0.1)
# osc_client.send_message('/pybinsimPauseConvolution','True')
# osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / 'coin.wav'))
osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / 'pinknoise.wav'))


# /pyBinSimLoudness {loudness: float32}
# for source in hrtf_sources:
#     print(source)
#     osc_client.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                                                     float(source[0]), float(source[1]), 0,
#                                                     0, 0, 0])
#     time.sleep(.1)

# for interval in numpy.arange(0,2000,100):
#     print(interval)
#     osc_client.send_message('/pyBinSimPauseAudioPlayback', True)
#     time.sleep(interval / 1000)
#     osc_client.send_message('/pyBinSimPauseAudioPlayback', False)
#     time.sleep(interval / 1000)
#     time.sleep(0.01)   # these intervals mainly determines CPU load
