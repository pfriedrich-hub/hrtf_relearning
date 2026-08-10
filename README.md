# hrtf_relearning

Experiment code for HRTF relearning: HRIR recording, AR/VR localization tests,
and the audio-feedback training game.

## Setup on a new machine

Prerequisites: **git** on `PATH` (two dependencies install straight from GitHub,
so a machine without git fails with `slab` / `pybinsim` missing), and a C/C++
toolchain is *not* needed — every dependency has a Windows wheel.

```bash
conda create -n hrtf_relearning python=3.11.9 -y
```

```bash
conda activate hrtf_relearning
```

Then from the project root:

```bash
python -m pip install -e .
```

That pulls in everything, including the two git dependencies:

- `slab` — <https://github.com/pfriedrich-hub/slab>
- `pybinsim` — <https://github.com/pfriedrich-hub/pybinsim_tuil>

Verify the install:

```bash
python -c "import slab, pybinsim, hrtf_relearning; print('ok')"
```

If `slab` or `pybinsim` are missing after this, the pip run failed partway —
re-run it and read the error rather than the tail of the log; a missing `git`
executable is the usual cause.

### Optional extras

CUDA build of torch (RTX 30xx and newer):

```bash
pip uninstall -y torch && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Freefield recording with TDT hardware:

```bash
pip install git+https://github.com/pfriedrich-hub/freefield.git
```

```bash
conda install pywin32
```

## Running

The install puts an `hrtf` command on `PATH`. Run it with no arguments to list
the commands:

```bash
hrtf
```

| Command | Module | What it does |
| --- | --- | --- |
| `hrtf training` | `experiment.training.Training_AR` | Audio-feedback training game (AR) |
| `hrtf localize-ar` | `experiment.localization.Localization_AR` | Localization test, AR headset |
| `hrtf localize-vr` | `experiment.localization.Localization_VR` | Localization test, VR / pybinsim |
| `hrtf analyze` | `experiment.analysis.localization.localization_analysis` | Localization analysis + plots |

Each is equivalently runnable as a module, which is what you want when
attaching a debugger in PyCharm:

```bash
python -m hrtf_relearning.experiment.training.Training_AR
```

### Not CLI commands

`hrtf_relearning/experiment/protocols/HRIR_Recording.py` is the first-session
pipeline (record HRIR → calibrate headphones → dome vs. virtual localization).
It is written as `# %%` cells and is meant to be stepped through in the IDE —
do **not** run it top to bottom.

## Layout

```
hrtf_relearning/
  cli.py          entry point behind the `hrtf` command
  experiment/     training, localization, protocols, analysis
  hrtf/           HRIR / HRTF processing
  data/           subject data, HRIRs, settings
  utils/
```

## Notes

- Editable install (`-e`) is the intended mode — the code reads and writes
  `hrtf_relearning/data/` in place.
- Only `hrtf_relearning*` is packaged; the top-level `docs/` and
  `analysis_results/` directories are deliberately excluded from the install.
