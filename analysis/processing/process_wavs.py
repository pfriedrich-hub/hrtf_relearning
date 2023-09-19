from pathlib import Path
import analysis.processing.helper_functions as helper
import pandas
import slab
import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
wav_path = Path.cwd() / 'data' / 'experiment' / 'master'

signal = helper.hrtf_signal()
wav_path = Path.cwd() / 'data' / 'experiment' / 'master' / 'jl' / 'Ears Free' / 'in_ear_recordings'
recordings, file_list = helper.read_wav(wav_path)

# recordings = helper.apply_filterbank(recordings, type='triangular', bandwidth=0.0286)
recordings = helper.scepstral_filter_recordings(recordings, high_cutoff=1500)

sources = helper.read_source_txt(wav_path)

# todo compare this hrtf (from scepstral domain filtered recordings)
#  to hrtf smoothed at 15000 (also check for baselining effects and try dfe)
#  no differenece
hrtf = slab.HRTF.estimate_hrtf(recordings, signal, sources)

bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
hrtf.plot_tf(sourceidx=hrtf.cone_sources())
hrtf_plot.plot_vsi_across_bands(hrtf, bands)

# def get_localization_dataframe(path=Path.cwd() / 'final_data'  / 'experiment' / 'master', w2_exclude=['cs', 'lm', 'lk']):
wav_df = pandas.DataFrame({'subject': [], 'condition': [], 'wav_filenames': [], 'processed_wav': []})
subject_paths = list(wav_path.iterdir())
conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
for subject_path in subject_paths:
    subject = subject_path.name
    for condition in conditions:
        condition_path = subject_path / condition
        for file_name in sorted(list(condition_path.iterdir())):
            if file_name.is_dir() and file_name.name.contains('.wav'):
        condition_path = subject_path / condition / 'in_ear_recordings'
        for file_name in sorted(list(condition_path.iterdir())):
            if file_name.is_file() and file_name.suffix == '.wav':
                wav = slab.Sound()

        if processed:
            condition_path = subject_path / condition / 'processed_hrtf'
        else:
            condition_path = subject_path / condition
        for file_name in sorted(list(condition_path.iterdir())):
            if file_name.is_file() and file_name.suffix == '.sofa':
                hrtf = slab.HRTF(file_name)
                new_row = [subject, file_name.name, condition, hrtf]
                hrtf_df.loc[len(hrtf_df)] = new_row
