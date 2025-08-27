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
    if hrir.datatype != 'FIR':  # convert to IR if necessary
        hrir = hrtf2hrir(hrir)
    if ear:  # flatten DTF at specified ear (modifies .name)
        hrir = flatten_dtf(hrir, ear)
        hrir.name += f'_{(ear[0]).upper()}_flat'
    if (data_dir / 'wav' / hrir.name).exists() and not overwrite:
        return hrir  # dont create wav files
    # create folder structure for HRTF wav files
    (data_dir / 'wav' / hrir.name / 'IR_data').mkdir(parents=True, exist_ok=True)
    (data_dir / 'wav' / hrir.name / 'sounds').mkdir(exist_ok=True)
    (data_dir / 'wav' / hrir.name / 'plot').mkdir(exist_ok=True)
    hrir = hrir2wav(hrir)  # write to wav files for use with pybinsim
    plot(hrir, title=f'{hrir.name} raw')  # plot raw
    return hrir

