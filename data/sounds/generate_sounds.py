import slab
slab.set_default_samplerate(48828)
from pathlib import Path
data_path = Path.cwd() / 'data' / 'sounds'

noise = slab.Sound.pinknoise(duration=5.0)
noise.write(data_path / 'pinknoise.wav')
