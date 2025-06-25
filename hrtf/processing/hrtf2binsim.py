from hrtf.analysis.plot import plot
from hrtf.processing.binsim.tf2ir import hrtf2hrir
from hrtf.processing.binsim.flatten import flatten_dtf
from hrtf.processing.binsim.hrir2wav import hrir2wav
from pathlib import Path
import slab
data_dir = Path.cwd() / 'data' / 'hrtf'

def hrtf2binsim(sofa_name, ear=None, overwrite=False):
    hrtf = slab.HRTF(data_dir / 'sofa' / f'{sofa_name}.sofa')  # get original HRTF/HRIR from sofa file
    hrtf.name = sofa_name
    hrir = hrtf2hrir(hrtf)  # convert to IR if necessary
    if ear:
        hrir.name += f'_{(ear[0]).upper()}_flat'
    if (data_dir / 'wav' / hrir.name).exists() and not overwrite:
        return hrir  # dont create wav files
    # create folder structure for HRTF wav files
    (data_dir / 'wav' / hrir.name / 'IR_data').mkdir(parents=True, exist_ok=True)
    (data_dir / 'wav' / hrir.name / 'sounds').mkdir(exist_ok=True)
    (data_dir / 'wav' / hrir.name / 'plot').mkdir(exist_ok=True)
    plot(hrir, title=f'{hrtf.name} raw')  # plot raw
    hrir = flatten_dtf(hrir, ear)  # flatten DTF at specified ear (modifies hrir.name)
    hrir = hrir2wav(hrir)  # write to wav files for use with pybinsim
    return hrir

