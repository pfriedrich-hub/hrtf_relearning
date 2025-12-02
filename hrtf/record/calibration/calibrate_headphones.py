import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import freefield
import slab
import numpy
import pyfar
import warnings
warnings.filterwarnings("ignore", category=pyfar._utils.PyfarDeprecationWarning)
from pathlib import Path
from copy import deepcopy
import pickle
fs = 48828
slab.set_default_samplerate(fs)

# initialize setup with standard samplerate (48824)
freefield.initialize('headphones', default='bi_play_rec')
freefield.set_logger('info')


# signal
low_cutoff = 20
high_cutoff = 20000
rec_repeat = 5  # how often to repeat measurement for averaging
# signal for loudspeaker calibration
signal_length = 1.0  # how long should the chirp be?
ramp_duration = signal_length / 50
signal = slab.Binaural.chirp(duration=signal_length, level=85, from_frequency=low_cutoff, to_frequency=high_cutoff,
                          kind='logarithmic')
# signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)


recs = []
for i in range(rec_repeat):
    recs.append(freefield.play_and_record_headphones(speaker='both', sound=signal, compensate_delay=True, distance=0,
                                     equalize=False, recording_samplerate=fs))
    freefield.wait_to_finish_playing()
hp_raw = slab.Sound(data=numpy.mean(recs, axis=0))

hp_raw = slab.Binaural.read('/Users/paulfriedrich/projects/hrtf_relearning/hrtf/record/calibration/hp_raw.wav')


# pyfar solution
# convert to pyfar
hp_raw = pyfar.Signal(hp_raw.data.T, hp_raw.samplerate)
signal = pyfar.Signal(signal.data.T, signal.samplerate)

# headphone transfer function
signal_inv = pyfar.dsp.regularized_spectrum_inversion(signal, frequency_range=(60, high_cutoff))
hp_tf = hp_raw * signal_inv

# invert to obtain equalizing filter
hp_filt = pyfar.dsp.regularized_spectrum_inversion(hp_tf, frequency_range=(low_cutoff, high_cutoff))
hp_filt = pyfar.dsp.time_shift(hp_filt, 2000)

# window
# onsets = pyfar.dsp.find_impulse_response_start(hp_filt, threshold=15)
# onsets_min = numpy.min(onsets) / fs  # earliest onset in seconds
# times = (onsets_min - .00025,  # start of fade-in
#          onsets_min,  # end if fade-in
#          onsets_min + .0048,  # start of fade_out
#          onsets_min + .0058)  # end of_fade_out
# hp_filt, window = pyfar.dsp.time_window(hp_filt, times, "hann", unit="s", crop="none", return_window=True)

# shift to 1ms
shift_s = .0001 - pyfar.dsp.find_impulse_response_start(hp_filt, threshold=20)
hp_filt = pyfar.dsp.time_shift(hp_filt, shift_s)

# crop
# times_samples = [0, 10, 246, 255]
# hp_filt = pyfar.dsp.time_window(hp_filt, times_samples, "hann", crop="end")

# test
equalized = pyfar.dsp.convolve(hp_raw, hp_filt, mode='full', method='overlap_add')
plt.figure()
pyfar.plot.time_freq(equalized)
plt.title('not windowed, not cropped, req inv')


# slab equalization
# parameters
freq_bins = 1000
bandwidth = 1 / 50
alpha = 1.0
filter_bank = slab.Filter.equalizing_filterbank(signal.channel(0), hp_raw, length=freq_bins, low_cutoff=low_cutoff,
                                                high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)

# test
attenuated = deepcopy(signal)
attenuated = filter_bank.apply(attenuated)

recs = []
for i in range(rec_repeat):
    recs.append(freefield.play_and_record_headphones(speaker='both', sound=attenuated, compensate_delay=True, distance=0,
                                     equalize=False, recording_samplerate=fs))
    freefield.wait_to_finish_playing()
equalized_recording = slab.Sound(data=numpy.mean(recs, axis=0))


# equalization = dict()  # dictionary to hold equalization parameters
# array_equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
#                       for i in range(len(speakers))}
# equalization.update(array_equalization)
# # write final equalization to pkl file
# # freefield_path = freefield.DIR / 'data'
# project_path = Path.cwd() / 'data' / 'calibration'
# equalization_path = project_path / f'calibration_dome_100k_31.10'
# with open(equalization_path, 'wb') as f:  # save the newly recorded calibration
#     pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)
