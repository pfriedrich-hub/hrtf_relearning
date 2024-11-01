import pybinsim
import logging
from pathlib import Path
import pythonosc

pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO
#Use logging.WARNING for printing warnings only
filename = 'kemar'
data_dir = Path.cwd() / 'data' / 'hrtf' / 'wav' / filename

with pybinsim.BinSim(data_dir / f'{filename}_settings.txt') as binsim:
    binsim.stream_start()