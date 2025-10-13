from hrtf.analysis.plot import plot
from hrtf.processing.binsim.tf2ir import hrtf2hrir
from hrtf.processing.binsim.flatten import flatten_dtf
from hrtf.processing.binsim.hrir2wav import hrir2wav
from pathlib import Path
import slab
data_dir = Path.cwd() / 'data' / 'hrtf'

def hrtf2binsim(sofa_name, ear=None, overwrite=False):
    hrir = slab.HRTF(data_dir / 'sofa' / f'{sofa_name}.sofa')  # read original sofa file
    hrir.name = sofa_name
    # convert to IR if necessary
    if hrir.datatype != 'FIR':
        hrir = hrtf2hrir(hrir)
    if ear:        # flatten DTF at specified ear
        hrir = flatten_dtf(hrir, ear)
        hrir.name += f'_{ear}'
    if (data_dir / 'wav' / hrir.name).exists() and not overwrite:
        return hrir  # dont create wav files
    else:
        # create folder structure for HRTF wav files
        (data_dir / 'wav' / hrir.name / 'IR_data').mkdir(parents=True, exist_ok=True)
        (data_dir / 'wav' / hrir.name / 'sounds').mkdir(exist_ok=True)
        (data_dir / 'wav' / hrir.name / 'plot').mkdir(exist_ok=True)
        # write to wav files for pybinsim
        hrir = hrir2wav(hrir)
        plot(hrir, title=f'{hrir.name} raw')  # plot raw example IR at 90°
        return hrir

