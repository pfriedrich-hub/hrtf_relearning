## Install instructions ##
	pycharm 2025.2
	conda create --n hrtf_relearning python=3.11.9
	cd project root dir 
	python -m pip install -e.

## will automatically install following dependencies:
	git+https://github.com/pfriedrich-hub/slab.git
	git+https://github.com/pfriedrich-hub/pybinsim_tuil.git
	h5py h5netcdf metawear pyqt5 pynput pyfar

## for cuda support (>=RTX 30xx):
	pip uninstall torch
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
## for freefield recording with TDT:
	pip install git+https://github.com/pfriedrich-hub/freefield.git
	conda install pywin32

## Run instructions: ##
	call with python -m hrtf_relearning.experiment.Training

todo: move data folder for git install
