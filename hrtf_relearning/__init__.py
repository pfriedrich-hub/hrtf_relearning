# Pick the matplotlib backend before anything below imports pyplot — the choice
# only takes effect while no figure exists (see hrtf_relearning.utils.mpl_backend).
from hrtf_relearning.utils.mpl_backend import use_interactive as _use_interactive
_use_interactive()

from hrtf_relearning.utils import paths
# Absolute path to the installed package root (canonical definitions in hrtf_relearning.utils.paths)
from hrtf_relearning.utils.paths import PATH
from hrtf_relearning.experiment.misc.Subject import Subject
from hrtf_relearning.experiment.analysis.localization.localization_analysis import localization_accuracy

__all__ = ["PATH", "paths", "Subject", "localization_accuracy"]
