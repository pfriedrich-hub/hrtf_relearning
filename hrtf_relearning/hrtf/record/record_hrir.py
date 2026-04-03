"""
- directory handling
- recording + processing
"""
import matplotlib
matplotlib.use('TkAgg')
from hrtf_relearning.hrtf.record.recordings import *
from hrtf_relearning.hrtf.record.processing import *
import hrtf_relearning
base_dir = hrtf_relearning.PATH / "data" / "hrtf"
import logging

subject_id = 'kemar_pir'
head_radius = 0.0875
reference_id = 'ref_03.04'
overwrite = False
n_directions = 1
n_recordings = 10
n_samples_out = 512
fs = 48828  # 97656
hp_freq = 120
show = True
equalize_dome = True
align_interaural = True

slab.set_default_samplerate(fs)
freefield.set_logger("info")


# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------

def record_hrir(
    subject_id: str,
    reference_id: str,
    *,
    n_directions: int = 3,
    n_recordings: int = 10,
    fs: int = 48828,
    hp_freq: float = 120,
    equalize_dome: bool = False,
    overwrite: bool = False,
    align_interaural: bool = True,
    n_samples_out: int = 512,
    expand_az: bool = True,
    show: bool = True,
    base_dir: Path | str | None = None,
) -> slab.HRTF:
    """
    Full HRIR acquisition + processing pipeline for one subject.

    Steps:
    1) Record (or load) subject ear-pressure sweeps
    2) Record (or load) reference sweeps
    3) Deconvolve sweeps -> IRs
    4) Equalize subject IRs using reference IRs
    5) Low-frequency extrapolation (spherical head)
    6) Azimuth expansion + binaural cue imposition
    7) Export to slab.HRTF

    No DSP logic is implemented here – only orchestration.
    """

    logging.info(f"Starting HRIR pipeline for subject '{subject_id}'")

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------
    if base_dir is None:
        base_dir = hrtf_relearning.PATH / "data" / "hrtf"
    else:
        base_dir = Path(base_dir)

    subj_dir = base_dir / "rec" / subject_id
    ref_dir = base_dir / "rec" / "reference" / reference_id

    # -----------------------------------------------------------------
    # 1) Subject recordings
    # -----------------------------------------------------------------
    if overwrite or not subj_dir.exists():
        logging.info("Recording subject ear pressure")
        subj_dir.mkdir(parents=True, exist_ok=True)

        subject_rec = Recordings.record_dome(
            id=subject_id,
            n_directions=n_directions,
            n_recordings=n_recordings,
            hp_freq=hp_freq,
            fs=fs,
            equalize=equalize_dome,
            key=True)
        subject_rec.to_wav(subj_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject recordings from disk")
        subject_rec = Recordings.from_wav(subj_dir)

    # -----------------------------------------------------------------
    # 2) Reference recordings
    # -----------------------------------------------------------------
    """    
    if overwrite or not ref_dir.exists():
        logging.info("Recording reference")
        ref_dir.mkdir(parents=True, exist_ok=True)
        reference_rec = Recordings.record_dome(
            id=reference_id,
            n_directions=1,
            n_recordings=n_recordings,
            hp_freq=hp_freq,
            fs=fs,
            equalize=equalize_dome,
            key=False)
        reference_rec.to_wav(ref_dir, overwrite=overwrite)
    else:
    """
    logging.info("Loading reference recordings from disk")
    reference_rec = Recordings.from_wav(ref_dir)

    # -----------------------------------------------------------------
    # 3) Deconvolution: sweeps -> IRs
    # -----------------------------------------------------------------
    logging.info("Computing impulse responses")
    subject_ir = compute_ir(subject_rec, inversion_range_hz=(hp_freq, 20e3),
                            onset_threshold_db=10, align_interaural=align_interaural)
    reference_ir = compute_ir(reference_rec, inversion_range_hz=(hp_freq, 20e3),
                              onset_threshold_db=10, align_interaural=align_interaural)

    # -----------------------------------------------------------------
    # 4) Equalization + windowing
    # -----------------------------------------------------------------
    logging.info("Applying equalization")
    hrir_equalized = equalize(
        measured=subject_ir,
        reference=reference_ir,
        n_samples_out=n_samples_out,
        inversion_range_hz=(hp_freq, 18e3),
        onset_threshold_db=10,
    )

    # -----------------------------------------------------------------
    # 5) Low-frequency extrapolation
    # -----------------------------------------------------------------
    logging.info("Low-frequency extrapolation")
    hrir_extrapol = lowfreq_extrapolate(
        hrir_equalized,
        f_extrap=400.0,
        f_target=150.0,
        head_radius=head_radius,
    )

    # -----------------------------------------------------------------
    # 6) Azimuth expansion + binaural cues
    # -----------------------------------------------------------------
    if expand_az:
        logging.info("Expanding azimuths and imposing binaural cues")
        hrir_az_exp = expand_azimuths_with_binaural_cues(
            hrir_extrapol,
            az_range=(-50, 50),
            head_radius=head_radius,
            show=False,
        )
    else:
        hrir_az_exp = hrir_extrapol

    # -----------------------------------------------------------------
    # 7) Export to slab.HRTF
    # -----------------------------------------------------------------
    logging.info("Converting to slab.HRTF")
    hrtf = hrir_az_exp.to_slab_hrtf(datatype="FIR")
    hrtf.write_sofa(base_dir / 'sofa' / f'{subject_id}.sofa')

    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2)
        hrtf.plot_tf(hrtf.cone_sources(0), axis=axes, ear='both')
        plt.show()

    logging.info("HRIR pipeline finished successfully")
    return hrtf


from pynput import keyboard
def wait_for_button(msg=None):
    if msg:
        logging.info(msg)

    def on_press(key):
        if key == keyboard.Key.enter:
            listener.stop()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# if __name__ == "__main__":
#     hrtf = record_hrir(
#         subject_id=subject_id,
#         reference_id=reference_id,
#         n_directions=n_directions,
#         n_recordings=n_recordings,
#         fs=fs,
#         hp_freq=hp_freq,
#         n_samples_out=n_samples_out,
#         equalize_dome=equalize_dome,
#         align_interaural=align_interaural,
#         overwrite=overwrite,
#         show=show,
#         base_dir=base_dir,
#     )