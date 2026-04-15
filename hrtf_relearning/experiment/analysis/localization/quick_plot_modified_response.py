import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import slab
from hrtf_relearning import PATH
hrtf_dir = PATH / 'data' / 'hrtf' / 'sofa'
from hrtf_relearning.hrtf.analysis.vsi import vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy, plot_localization, plot_elevation_response,
)
import hrtf_relearning
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.hrtf.processing.modify import plot

idx = 14.05

# load subject localization
sub_id = 'Nka'
subject = hrtf_relearning.Subject(sub_id)
loc = subject.localization['NKa_14.04_10-52_NKa']

# load modified
hrtf_original = slab.HRTF('C:/projects/hrtf_relearning/hrtf_relearning/data/hrtf/sofa/VD.sofa')

# load original
hrtf_modified = slab.HRTF('C:/projects/hrtf_relearning/hrtf_relearning/data/hrtf/sofa/VD_notch.sofa')

VSI_BW = (5700, 11300)
vsi_orig = _vsi(hrtf_original, bandwidth=VSI_BW)
vsi_mod = _vsi(hrtf_modified, bandwidth=VSI_BW)
vsi_dis = _vsi_dissimilarity(hrtf_original, hrtf_modified, bandwidth=VSI_BW)

fig_h = plot(hrtf_original, hrtf_modified, 'image', ear='right',
           vsi_orig=vsi_orig, vsi_mod=vsi_mod, vsi_dis=vsi_dis, vsi_bw=VSI_BW)
fig_h.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + f'_modified_{idx}.png'),
            bbox_inches='tight')

fig = plot_elevation_response(loc)
fig.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + f'loc_modified_{idx}.png'),
            bbox_inches='tight')