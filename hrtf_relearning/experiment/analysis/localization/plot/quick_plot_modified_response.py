import matplotlib
matplotlib.use('TkAgg')
import slab
from hrtf_relearning import PATH
from hrtf_relearning.utils import paths
hrtf_dir = paths.SOFA_DIR
from hrtf_relearning.hrtf.analysis.vsi import vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_elevation_response,
)
import hrtf_relearning
from hrtf_relearning.hrtf.modify.plot_compare import plot

idx = 14.05

# load subject localization
sub_id = 'NKa'
subject = hrtf_relearning.Subject(sub_id)

# ---- elevation response
sequence = 'NKa_15.04_10-50_NKa'
loc = subject.localization[sequence]
fig = plot_elevation_response(loc)
fig.savefig(paths.subject_plot_dir(sub_id) / f'{sequence}.png', bbox_inches='tight')

# --- plot hrtf
# load original
hrtf_original = slab.HRTF(hrtf_dir / 'VD' / 'VD.sofa')

# load modified
hrtf_modified = slab.HRTF(hrtf_dir / 'VD' / 'VD_notch.sofa')

VSI_BW = (5700, 11300)
vsi_orig = _vsi(hrtf_original, bandwidth=VSI_BW)
vsi_mod = _vsi(hrtf_modified, bandwidth=VSI_BW)
vsi_dis = _vsi_dissimilarity(hrtf_original, hrtf_modified, bandwidth=VSI_BW)

fig_h = plot(hrtf_original, hrtf_modified, 'image', ear='right',
           vsi_orig=vsi_orig, vsi_mod=vsi_mod, vsi_dis=vsi_dis, vsi_bw=VSI_BW)
fig_h.savefig(paths.subject_plot_dir(sub_id) / str(sub_id + f'_modified_{idx}.png'),
            bbox_inches='tight')

