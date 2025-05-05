import argparse
import logging
import multiprocessing as mp
from pythonosc import udp_client
import pybinsim
from hrtf.processing.hrtf2wav import *

logging.getLogger().setLevel('INFO')
pybinsim.logger.setLevel(logging.WARNING)

filename = 'KU100_HRIR_L2702'
data_dir = Path.cwd() / 'data' / 'hrtf'
soundfile = 'c_chord_guitar.wav'  # choose file from sounds folder

def play():
    hrtf = slab.HRTF(data_dir / 'sofa' / f'{filename}.sofa')
    # hrtf = slab.HRTF.kemar()
    sources = hrtf.cone_sources(2, full_cone=False)

    global osc_client
    osc_client = make_osc_client(port=10003)  # loudness / files
    osc_client_1 = make_osc_client(port=10000)  # filter

    # init pybinsim
    binsim_worker = mp.Process(target=binsim_stream, args=())
    binsim_worker.start()

    input('Press button to play')
    osc_client.send_message('/pyBinSimLoudness', .2)


    for source in sources:
        source = hrtf.sources.vertical_polar[source]

        logging.info(f'Playing from {source}')
        osc_client_1.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        float(source[0]), float(source[1]), 0,
                                                        0, 0, 0])
        # record
        # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
            frames.append(data)

        time.sleep(.25)
    binsim_worker.join()

def binsim_stream():
    binsim = pybinsim.BinSim(data_dir / 'wav' / filename / f'{filename}_settings.txt')
    binsim.soundHandler.loopSound = True
    binsim.stream_start()  # run binsim loop

def make_osc_client(port):
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

def set_soundfile(soundfile, osc_client):
    logging.info(f'Setting soundfile: {soundfile}')
    osc_client.send_message('/pyBinSimFile', str(data_dir / 'wav' / filename / 'sounds' / soundfile))

def record_stream():
    import wave
    wf = wave.open('test', 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    make_wav(filename)
    play()
