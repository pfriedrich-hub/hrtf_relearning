# record from in-ear microphones
import numpy
from pathlib import Path
import slab
import freefield
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
import pyfar as pf
import time
import os
import pickle

date = datetime.datetime.now()
from copy import deepcopy
fs = 48828  # 97656, 195312.5
slab.set_default_samplerate(fs)

conditions = "mems_long_full_15_10" # how high is the gain, how long is the sweep
path = f"{conditions}.pkl"
with open(path, 'rb') as dat:
    mics = pickle.load(dat)

# file settings
subject_id = 'pf'
condition = 'center'  # can be 'ears_free' or 'earmolds' - important for file naming! # in_ear_mic
# center or up -> up means fixpoint is between center and the speaker above, the recordings are added inbetween while processing
kemar = False  # requires no button press if true
safe = 'both'  # decide if additionally save in-ear-recordings
data_dir = Path.cwd() / 'data' / 'experiment' / subject_id / condition

# HRTF recording settings
speakers = numpy.array([i for i in range(5, 42) if i not in (19, 27)]).tolist()
#speaker = numpy.array([i for i in range(20,27)]).tolist()
n_directions = 1  # only from the front (1) or front-back recordings (2)
level = 80  # minimize to reduce reverb ripple effect, apparently kemar recordings are not affected?
duration = 0.2  # short chirps <0.05s introduce variability in low freq (4-5 kHz). improvement at 0.1s for kemar vsi # todo 0.1 0.2
low_freq = 20 # 1000
high_freq = 24000  # window of interest is 4-16 # 17000
repetitions = 20  # 10 work for kemar, 30-50 for in ear mics

ramp_duration = duration/20
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.

signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq, kind="log") #"'linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)

# plot options
dfe = False  # whether to use diffuse field equalization to plot hrtf and compute vsi
plot_bins = 2400  # number of bins also used to calculate vsi across bands (use 80 to minimize´frequency-resolution dependend vsi change)
plot_ear = 'left'  # ear for which to plot HRTFs


def record_hrtf(subject_id, data_dir, condition, signal, repetitions, n_directions, safe, speakers, kemar=False):
    global filt, sources
    # filt = slab.Filter.band('bp', (low_freq, high_freq))
    filt = slab.Filter.band('hp', (200))  # makes no diff
    if not freefield.PROCESSORS.mode:
        proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_rec_buf.rcx'],
                     ['RX81', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx'],
                     ['RX82', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx']]
        freefield.initialize('dome', device=proc_list)
        freefield.PROCESSORS.mode = 'play_birec'
        #freefield.load_equalization(file=Path.cwd() / 'data' / 'calibration' / 'calibration_central_22_07_left_in_ear')#'calibration_dome_100k_31.10') #TODO change to correct calibration
    freefield.set_logger('warning')
    table_file = 'speakertable_dome_mirror.txt'
    #table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')  # get speaker coordinates
    if isinstance(speakers, str) and speakers == 'all':
        source_locations = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                         delimiter=",", dtype=float)
        source_locations = numpy.delete(source_locations, [numpy.where(source_locations[:, 0] == 19),
                           numpy.where(source_locations[:, 0] == 27)], axis=0)  # remove 0, 50 and -50 speakers
    elif isinstance(speakers, list):
        source_locations = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                         delimiter=",", dtype=float)[speakers]
    else:
        raise ValueError('Speakers must be >>all<< or list of indices.')
    speaker_ids = source_locations[:, 0].astype('int')
    sources = deepcopy(source_locations)
    if not kemar:
        [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
        freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                    processors=led_speaker.digital_proc)  # illuminate LED
        print('Face fixpoint and press button to start recording.')
        freefield.wait_for_button()
    recordings = []
    for i in range(n_directions):  # record for n listener orientations, 2 = front + back
        recordings = recordings + (dome_rec(signal, speaker_ids, sources, repetitions))
        if i < n_directions-1:
            sources[:, 1] += 360/n_directions
            print('Rotate chair %i degrees clockwise and look at fixpoint. \nPress button to start recording.'
                  % 360/n_directions)
            freefield.wait_for_button()
    freefield.set_logger('INFO')
    if not kemar:
        freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    sources = create_src_txt(sources)  # create source coordinate array
    data_dir.mkdir(parents=True, exist_ok=True)  # create condition directory if it doesnt exist
    if safe == 'sofa' or safe == 'both':  # compute HRTFs and write to sofa file
        print('Creating sofa file...')
        recorded_hrtf = estimate_hrtf([rec[2] for rec in recordings], sources) # recordings, signal, sources
        recorded_hrtf.write_sofa(str(data_dir / (subject_id + '_' + condition + date.strftime('_%d.%m'))) + '.sofa')
    if safe == 'wav' or safe == 'both':  # write impulses to wav
        wav_dir = data_dir / str('in_ear_recordings' + date.strftime('_%d.%m'))
        wav_dir.mkdir(parents=True, exist_ok=True)
        print('Creating wav files...')
        for idx, bi_rec in enumerate(recordings):    # save recordings as .wav
            filename = '%s_%s_%02d_az%i_el%i.wav' % (subject_id, condition, idx, bi_rec[0], bi_rec[1])
            pf.io.write_audio(bi_rec[2],f"{wav_dir}/{filename}.wav")
        numpy.savetxt(wav_dir / ('sources_%s_%s.txt' % (subject_id, condition)), sources, fmt='%1.1f')
    recordings2, sources2 = add_elevation_sources(recordings)
    sources2 = create_src_txt(sources2)  # create source coordinate array

    if safe == 'sofa' or safe == 'both':  # compute HRTFs and write to sofa file
        print('Creating sofa file...')
        recorded_hrtf2 = estimate_hrtf([rec[2] for rec in recordings2], sources2)
        recorded_hrtf2.write_sofa(str(data_dir / (subject_id + '_' + condition + date.strftime('_%d.%m'))) + '_new_shape.sofa')

    if safe == 'wav' or safe == 'both':  # write recordings to wav files
        numpy.savetxt(wav_dir / ('new_sources_%s_%s.txt' % (subject_id, condition)), sources2, fmt='%1.1f')

    return recordings, sources, recorded_hrtf

# def dome_rec(signal, speaker_ids, sources, repetitions):
#     print('Recording...')
#     recordings = []  # list to store binaural recordings and source coordinates
#     for idx, speaker_id in enumerate(speaker_ids):
#         [speaker] = freefield.pick_speakers(speaker_id)
#         # get avg of n recordings from each sound source location
#         recs = []
#         for r in range(repetitions):
#             recs.append(freefield.play_and_record(speaker, signal, equalize=False, recording_samplerate=97656)) # TODO set equalize to true, recording_samplerate=97656
#         rec = slab.Binaural(numpy.mean(numpy.asarray(recs), axis=0), samplerate=fs)  # average
#         rec.data -= numpy.mean(rec.data, axis=0)  # baseline
#         rec = slab.Binaural.ramp(rec, when='both', duration=ramp_duration)
#         azimuth = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][1]
#         elevation = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][2]
#         recordings.append([azimuth, elevation, rec])
#         print('progress: %i %%' % (int((idx+1)/len(speaker_ids)*100)), end="\r", flush=True)
#     return recordings

def dome_rec(signal, speaker_ids, sources, repetitions):
    print('Recording...')
    ref = []
    for i in mics["rec_mean"]:
        ref.append(i) #(pf.Signal(i.data.T, i.samplerate))
    n_start = 0
    n_samples = 255
    recordings = []  # list to store binaural recordings and source coordinates
    speaker_rec = []
    for idx, speaker_id in enumerate(speaker_ids):
        [speaker] = freefield.pick_speakers(speaker_id)
        recs = []
        for r in range(repetitions):
            recs.append(freefield.play_and_record(speaker, signal, equalize=False, recording_samplerate=97656))
            time.sleep(0.05)
        speaker_rec.append(recs)



        # todo DONE -
        reference = pf.Signal(ref[idx].T,fs)
        rec = numpy.mean([record.data for record in recs],axis=0)
        hrir_recorded = pf.Signal(rec.T,fs)
        hrir_recorded.time -= numpy.mean(hrir_recorded.time, axis=1, keepdims=True)
        reference.time -= numpy.mean(reference.time, axis=1, keepdims=True)
        reference_inverted = pf.dsp.regularized_spectrum_inversion(reference, freq_range=(20, 18750)) # 18000? over this frequency the magnitude in reference spectrum falls
        hrir_deconvolved = hrir_recorded * reference_inverted
        n0 = min(int(numpy.argmax(numpy.abs(hrir_deconvolved.time[0]))), int(numpy.argmax(numpy.abs(hrir_deconvolved.time[1])))) # use the earlier peak (left or right) as reference in time
        hrir_windowed = pf.dsp.time_window(hrir_deconvolved, (max(0, n0-50), min(n0+100, len(hrir_deconvolved.time[0])-1)), 'boxcar', unit="samples",
                                           crop="window")  # (170,335)
        hrir_windowed = pf.dsp.pad_zeros(hrir_windowed, hrir_deconvolved.n_samples - hrir_windowed.n_samples)
        hrir_low = pf.dsp.filter.crossover(
            pf.signals.impulse(hrir_windowed.n_samples), 4, 400)[0] # *(10**(-1))
        hrir_low.sampling_rate = 48828
        hrir_high = pf.dsp.filter.crossover(hrir_windowed, 4, 400)[1]
        hrir_low_delayed = pf.dsp.fractional_time_shift(
            hrir_low, pf.dsp.find_impulse_response_delay(hrir_windowed)) # error if too noisy, overwrite function and use exactly the peakpoint?
        hrir_extrapolated = hrir_low_delayed + hrir_high
        hrir_final = pf.dsp.time_window(
            hrir_extrapolated, (n_start, n_start + n_samples), 'boxcar',
            crop='window')




        azimuth = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][1]
        elevation = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][2]
        #recordings.append([azimuth, elevation, rec])
        recordings.append([azimuth, elevation, hrir_final])
        print('progress: %i %%' % (int((idx+1)/len(speaker_ids)*100)), end="\r", flush=True)
    speaker_recs = {"speaker": speaker_rec}
    with open(f"recordings_{subject_id}_{condition}_{conditions}.pkl", "wb") as file:
        pickle.dump(speaker_recs, file, pickle.HIGHEST_PROTOCOL)
    return recordings

def estimate_hrtf(recordings, sources, listener=None):
    if isinstance(sources, (list, tuple)):
        sources = numpy.array(sources)
    if len(sources.shape) == 1:  # a single location (vector) needs to be converted to a 2d matrix
        sources = sources[numpy.newaxis, ...]
    if len(sources) != len(recordings):
        raise ValueError('Number of sound sources must be equal to number of recordings.')
    rec_samplerate = recordings[0].sampling_rate
    rec_n_samples = recordings[0].n_samples
    rec_data = []
    for recording in recordings:
        if not (recording.cshape == (2,) and recording.n_samples == recordings[0].n_samples
                and recording.sampling_rate == rec_samplerate):
            raise ValueError('Number of channels, samples and samplerate must be equal for all recordings.')
        rec = deepcopy(recording)
        rec_data.append(rec.time)
    rec_data = numpy.asarray(rec_data)
    with numpy.errstate(divide='ignore'):
        hrtf_data = rec_data
    if not listener:
        listener = {'pos': numpy.array([0., 0., 0.]), 'view': numpy.array([1., 0., 0.]),
                    'up': numpy.array([0., 0., 1.]), 'viewvec': numpy.array([0., 0., 0., 1., 0., 0.]),
                    'upvec': numpy.array([0., 0., 0., 0., 0., 1.])}
    return slab.HRTF(data=hrtf_data, datatype='FIR', samplerate=rec_samplerate, sources=sources,
                listener=listener)


def create_src_txt(sources):
    # convert interaural_polar to vertical_polar coordinates for sofa file
    # interaural polar to cartesian
    # interaural_polar = numpy.asarray(recordings)[:, :2].astype('float')  # deprecated numpy
    interaural_polar = sources[:, 1:]
    cartesian = numpy.zeros((len(interaural_polar), 3))
    vertical_polar = numpy.zeros((len(interaural_polar), 3))
    azimuths = numpy.deg2rad(interaural_polar[:, 0])
    elevations = numpy.deg2rad(90 - interaural_polar[:, 1])
    r = 1.4  # get radii of sound sources
    cartesian[:, 0] = r * numpy.cos(azimuths) * numpy.sin(elevations)
    cartesian[:, 1] = r * numpy.sin(azimuths)
    cartesian[:, 2] = r * numpy.cos(elevations) * numpy.cos(azimuths)
    # cartesian to vertical polar
    xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
    vertical_polar[:, 0] = numpy.rad2deg(numpy.arctan2(cartesian[:, 1], cartesian[:, 0]))
    vertical_polar[vertical_polar[:, 0] < 0, 0] += 360
    vertical_polar[:, 1] = 90 - numpy.rad2deg(numpy.arctan2(numpy.sqrt(xy), cartesian[:, 2]))
    vertical_polar[:, 2] = numpy.sqrt(xy + cartesian[:, 2] ** 2)
    return vertical_polar.astype('float16')

# just used as placeholders
def add_elevation_sources(recordings):
    new_recordings = []
    new_sources = []
    for idx, r in enumerate(recordings):
        new_recordings.append(r)
        new_sources.append([idx, r[0], r[1]])
        if idx not in (6, 13, 20, 27, 34):
            azi = r[0]
            ele = numpy.mean((recordings[idx][1], recordings[idx+1][1]))
            new_recordings.append([azi, ele, r[2]]) # just copies the impulse from other source (get´s overwritten later in processing)
            new_sources.append([idx+0.5, azi, ele])
        #if idx != 6:
            #azi = r[0]
            #ele = numpy.mean((recordings[idx][1], recordings[idx+1][1]))
            #new_recordings.append([azi, ele, r[2]]) # just copies the impulse from other source (get´s overwritten later in processing)
            #new_sources.append([idx+0.5, azi, ele])
    new_sources = numpy.array(new_sources)
    return new_recordings, new_sources

'''
# changed from pyfar
import numpy as np
import warnings
from scipy import signal as sgn

def find_impulse_response_delay(impulse_response, N=1):
    n = int(np.ceil((N+2)/2))
    start_samples = np.zeros(impulse_response.cshape)
    modes = ['real', 'complex'] if impulse_response.complex else ['real']
    start_sample = np.zeros((len(modes), 1), dtype=float)
    for ch in np.ndindex(impulse_response.cshape):
        n_samples = impulse_response.n_samples
        for idx, mode in enumerate(modes):
            ir = impulse_response.time[ch]
            ir = np.real(ir) if mode == 'real' else np.imag(ir)
            if np.max(np.abs(ir)) > 1e-16:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="h does not appear to by symmetric",
                        category=RuntimeWarning)
                    ir_minphase = sgn.minimum_phase(
                        ir, n_fft=4*n_samples)
                correlation = sgn.correlate(
                    ir,
                    np.pad(ir_minphase, (0, n_samples - (n_samples + 1)//2)),
                    mode='full')
                lags = np.arange(-n_samples + 1, n_samples)
                correlation_analytic = sgn.hilbert(correlation)
                argmax = np.argmax(np.abs(correlation_analytic))
                search_region_range = np.arange(argmax-n, argmax+n)
                search_region = np.imag(
                    correlation_analytic[search_region_range])
                search_region *= np.sign(correlation_analytic[argmax].real)
                mask = np.gradient(search_region, search_region_range) > 0
                try:
                    search_region_poly = np.polyfit(
                        search_region_range[mask]-argmax,
                        search_region[mask], N)
                    roots = np.roots(search_region_poly)
                    if np.all(np.isreal(roots)) and roots.size > 0:
                        root = roots[np.abs(roots) == np.min(np.abs(roots))]
                        start_sample[idx] = np.squeeze(lags[argmax] + root)
                    else:
                        raise ValueError("No valid real roots")
                except (np.linalg.LinAlgError, ValueError):
                    start_sample[idx] = lags[argmax]
            else:
                start_sample[idx] = np.nan
        start_samples[ch] = np.nanmin(start_sample)
    return start_samples
'''
if __name__ == "__main__":
    recordings, sources, hrtf = record_hrtf(subject_id, data_dir, condition, signal, repetitions, n_directions, safe, speakers, kemar)
    sources = list(range(hrtf.n_sources-1, -1, -1))  # works for 0°/+/-17.5° cone
    hrtf.plot_tf(sources, xlim=(20, 24000), ear=plot_ear)
    # fig, axis = plt.subplots(2, 1)
    # hrtf_analysis.plot_hrtf_image(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000), dfe=dfe)
    # hrtf_analysis.vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
    # axis[0].set_title(subject_id)


