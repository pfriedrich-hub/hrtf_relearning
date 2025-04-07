import pybinsim
import logging
from pathlib import Path
filename ='KU100_HRIR_L2702'
data_dir = Path.cwd() / 'data' / 'hrtf'

pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO

with pybinsim.BinSim(data_dir / 'wav' / filename / f'{filename}_settings.txt') as binsim:
    binsim.stream_start()