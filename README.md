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

### Plot windows

Check that plots open:

```bash
hrtf backend
```

It prints the backend in use and opens one test window; close it to finish. If
it reports `Agg`, no window will ever open — the message tells you which of the
two fixes applies.

The backend is chosen once, in `hrtf_relearning/utils/mpl_backend.py`, when the
package is imported: **TkAgg**, falling back to QtAgg and then to file-only Agg.
Nothing else in the package sets a backend, so there is no longer a machine
where plots open in one script and hang in the next. Override per machine in
`local_config.json`:

```json
{ "mpl_backend": "QtAgg" }
```

or for one run with `HRTF_MPL_BACKEND=QtAgg` (`"none"` means never open a
window). Qt is *not* the default: PyQt5 is installed (the game UI is written in
it) and `QtAgg` imports fine, but its plot windows do not come up — see the
module docstring.

**PyCharm:** turn off *Settings → Tools → Python Plotting → Show plots in tool
window* (older versions: *Python Scientific → Show plots in tool window*). That
setting routes figures through PyCharm's own SciView backend, which hangs on
scripts that run subprocesses or an event loop — i.e. every protocol here. The
package unsets the corresponding `MPLBACKEND` for run configurations, but in the
**Python Console** PyCharm patches `plt.show` before any of our code executes,
so that one has to be switched off in the settings.

### Why plots used to freeze in the console

In the Python Console (the `# %%` cell workflow), a bare `plt.show()` calls the
toolkit's `mainloop()` and blocks the prompt until you close the window. Ctrl-C
gives the prompt back but tears the event loop out from under a window that is
still on screen, and it stops repainting from then on:

```
File "matplotlib/backends/_backend_tk.py", line 583, in start_main_loop
  first_manager.window.mainloop()
KeyboardInterrupt
```

That is not a backend problem — it happens on Tk and Qt alike. On import, the
package now hands the GUI event loop to the console's input hook (PyCharm's
`pydev_ipython`, or IPython's `%matplotlib`) and turns on interactive mode, so
`plt.show()` returns immediately and figures stay live and redrawable while you
keep typing. **You should not need Ctrl-C any more.** Interactive mode is only
enabled when a hook was actually installed — without one it would produce a
window that neither blocks nor responds.

Script runs are deliberately left alone: there, `plt.show()` blocking at the end
is the only thing keeping the window on screen until the process exits.

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
