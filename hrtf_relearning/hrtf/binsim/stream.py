"""Standalone pyBinSim audio stream.

Every experiment script used to define its own ``binsim_stream`` closure and run
it with multiprocessing. Under the 'spawn' start method (macOS/Windows) the
child re-imports the parent script, which breaks any script without an
``if __name__ == '__main__'`` guard — i.e. every cell-by-cell script.

``start_stream`` avoids multiprocessing altogether: it launches this module as
its own interpreter, so it is immune to spawn/import semantics and can be
started from a REPL or a single cell:

    from hrtf_relearning.hrtf.binsim.stream import start_stream
    proc = start_stream(hrir.name, 'test')
    ...
    proc.terminate()

It can also be run by hand:  python -m hrtf_relearning.hrtf.binsim.stream JS test
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

from hrtf_relearning.utils import paths

logger = logging.getLogger(__name__)


def settings_file(hrir_name, settings='test'):
    """Path of the pyBinSim settings file written by hrtf2binsim.write_settings."""
    return paths.BINSIM_DIR / hrir_name / f'{hrir_name}_{settings}_settings.txt'


def binsim_stream(hrir_name, settings='test'):
    """Run the pyBinSim audio loop (blocking) — intended as an mp.Process target.

    hrir_name : name of the binsim database directory, i.e. hrir.name as
        returned by hrtf2binsim (includes any _left / _env4 / _mirrored suffix).
    settings : 'test' or 'training' — which settings file to load.
    """
    import pybinsim
    pybinsim.logger.setLevel(logging.ERROR)
    fname = Path(settings_file(hrir_name, settings))
    if not fname.exists():
        raise FileNotFoundError(
            f'{fname} not found — run hrtf2binsim(..., overwrite=True) first.')
    binsim = pybinsim.BinSim(fname)
    binsim.stream_start()


def start_stream(hrir_name, settings='test', wait=3.):
    """Launch the pyBinSim stream in its own interpreter and return the Popen.

    Blocks for `wait` seconds so the audio device is up before the caller sends
    OSC, and raises if the process died during startup (bad settings file,
    torchConvolution 'cuda' without a GPU, no audio device, ...). Stop it with
    proc.terminate().
    """
    fname = Path(settings_file(hrir_name, settings))
    if not fname.exists():
        raise FileNotFoundError(
            f'{fname} not found — run hrtf2binsim(..., overwrite=True) first.')
    proc = subprocess.Popen(
        [sys.executable, '-m', __spec__.name if __spec__ else __name__,
         str(hrir_name), str(settings)])
    logger.info('pyBinSim stream started (pid %d) | %s', proc.pid, fname.name)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f'pyBinSim exited during startup (code {proc.returncode}) — '
                f'see the traceback above; check {fname.name}')
        time.sleep(.2)
    return proc


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    name = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else 'test'
    binsim_stream(name, kind)
