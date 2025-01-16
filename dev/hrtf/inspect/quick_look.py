from dev.hrtf.inspect.movie import movie
from pathlib import Path
import slab
sofa_path = Path.cwd() / 'data' / 'hrtf' / 'sofa'

filename = 'FABIAN.sofa'

hrtf = slab.HRTF(sofa_path / str(filename + '.sofa'))


movie(

)