import slab
slab.set_default_samplerate(48828)
from pathlib import Path
data_path = Path.cwd() / 'data' / 'sounds'

# training stimulus
noise = slab.Sound.pinknoise(duration=60.0)
noise.write(data_path / 'training_noise.wav')

# test stimulus
