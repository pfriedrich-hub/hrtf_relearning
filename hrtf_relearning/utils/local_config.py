"""Machine-local settings that must not be committed.

A few settings depend on the machine rather than on the experiment: the rig has a
CUDA GPU, the laptops do not, so ``convolution`` / ``storage`` have to be
``'cuda'`` in one place and ``'cpu'`` in another. Hardcoding either value in the
protocol scripts means the scripts differ per checkout and every pull is a merge
conflict.

Instead, put the machine-specific values in ``local_config.json`` in the repo
root. That file is gitignored; ``local_config.example.json`` next to it is
committed and shows the available keys.

    {
      "torch_device": "cuda",
      "mpl_backend": "TkAgg",
      "audio_output": "Sound Blaster",
      "audio_volume": 50
    }

``mpl_backend`` is read by :mod:`hrtf_relearning.utils.mpl_backend` and is only
needed on a machine where the auto-detected plot backend is wrong.

``audio_output`` / ``audio_volume`` are read by
:mod:`hrtf_relearning.experiment.misc.audio_device`. Set them on a machine with a
fixed audio interface and the training and localization scripts refuse to start
unless that device is the default output at that volume. Omit them and the check
is a no-op, so the same commit runs on a laptop with no interface attached.
See that module for the optional ``audio_strict``, ``audio_autofix`` and
``soundvolumeview`` keys.

Resolution order for ``torch_device()`` (first hit wins):

1. environment variable ``HRTF_TORCH_DEVICE``  -- one-off override
2. ``torch_device`` in ``local_config.json``   -- this machine
3. whatever the calling script asked for       -- e.g. hrir_settings['convolution']
4. auto-detection via ``torch.cuda.is_available()``

A resolved ``'cuda'`` is always validated against the actual machine and falls
back to ``'cpu'`` with a warning, so a script that hardcodes ``'cuda'`` still runs
on a laptop even with no local_config.json present.
"""
import json
import logging
import os
from pathlib import Path

from hrtf_relearning.utils import paths

logger = logging.getLogger(__name__)

# Repo root (hrtf_relearning/hrtf_relearning/ -> hrtf_relearning/), with the
# package directory as fallback for a non-editable install.
CONFIG_PATHS = (
    paths.PATH.parent / "local_config.json",
    paths.PATH / "local_config.json",
)

ENV_PREFIX = "HRTF_"

# What people actually write, mapped to what torch and the pyBinSim settings key
# accept.
ALIASES = {"gpu": "cuda", "nvidia": "cuda", "cpu": "cpu", "cuda": "cuda"}

_cache = None


def config_path():
    """The local_config.json in use, or None if the machine has none."""
    for path in CONFIG_PATHS:
        if path.exists():
            return path
    return None


def load(reload: bool = False):
    """Contents of local_config.json as a dict ({} if absent or unreadable)."""
    global _cache
    if _cache is not None and not reload:
        return _cache

    path = config_path()
    if path is None:
        _cache = {}
        return _cache

    try:
        _cache = json.loads(path.read_text())
        logger.debug("Local config loaded from %s: %s", path, _cache)
    except (json.JSONDecodeError, OSError) as err:
        logger.warning("Could not read %s (%s) — using defaults.", path, err)
        _cache = {}
    return _cache


def get(key, default=None):
    """Machine-local value for ``key``: env var, then local_config.json, then default."""
    env = os.environ.get(ENV_PREFIX + key.upper())
    if env is not None:
        return env
    return load().get(key, default)


def cuda_available():
    """True when this machine can actually run torch on CUDA."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def torch_device(requested=None, key="torch_device"):
    """
    Resolve the torch device for pyBinSim ('cpu' or 'cuda').

    Parameters
    ----------
    requested : str or None
        What the calling script asked for (e.g. ``hrir_settings['convolution']``).
        Used only when neither the environment nor local_config.json says
        otherwise, so a machine-local setting always wins over a hardcoded one.
    key : str
        Key to look up in local_config.json / ``HRTF_TORCH_DEVICE``.

    Returns
    -------
    str
        'cpu' or 'cuda', guaranteed to work on this machine.
    """
    device = get(key) or requested
    source = "local config" if get(key) else "caller"

    if device is None:
        device = "cuda" if cuda_available() else "cpu"
        source = "auto-detect"

    # 'gpu' and 'cuda:0' are the obvious things to write in local_config.json,
    # but torch (and the pyBinSim settings key) only know 'cpu' and 'cuda', and
    # an unrecognised value used to fall through to cpu with nothing but a
    # warning -- i.e. the whole rig silently ran on the CPU.
    device = ALIASES.get(str(device).strip().lower(), str(device).strip().lower())
    if device.startswith("cuda:"):
        device = "cuda"
    if device not in ("cpu", "cuda"):
        logger.warning("Unknown torch device %r (%s) — falling back to cpu.", device, source)
        return "cpu"

    if device == "cuda" and not cuda_available():
        logger.warning(
            "torch device 'cuda' requested (%s) but CUDA is unavailable here — using cpu. "
            "Set \"torch_device\" in %s to silence this.",
            source, config_path() or CONFIG_PATHS[0],
        )
        return "cpu"

    logger.debug("torch device %s (%s)", device, source)
    return device
