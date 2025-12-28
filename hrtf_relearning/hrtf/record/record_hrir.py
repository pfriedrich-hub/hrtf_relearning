"""
High-level HRIR recording + processing wrapper.

Responsibilities:
- directory handling
- overwrite / reload logic
- calling recording + processing steps in the correct order
- no signal processing code lives here
"""
from hrtf_relearning.hrtf.record.recordings import *
from hrtf_relearning.hrtf.record.processing import *
import hrtf_relearning
base_dir = hrtf_relearning.PATH / "data" / "hrtf"
import logging

subject_id = 'kemar_test'
reference_id = 'kemar_reference'
overwrite = False
n_directions = 1
n_recordings = 20
n_samples_out = 256
fs = 48828  # 97656
hp_freq = 120

show = True

slab.set_default_samplerate(fs)
freefield.set_logger("info")
# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------

def record_hrir(
    subject_id: str,
    reference_id: str,
    *,
    n_directions: int = 5,
    n_recordings: int = 5,
    fs: int = 48828,
    hp_freq: float = 120,
    n_samples_out: int = 256,
    overwrite: bool = False,
    show: bool = False,
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
        base_dir = Path.cwd() / "data" / "hrtf"
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
            n_directions=n_directions,
            n_recordings=n_recordings,
            hp_freq=hp_freq,
            fs=fs,
        )
        subject_rec.params["subject_id"] = subject_id
        subject_rec.to_wav(subj_dir, overwrite=overwrite)
    else:
        logging.info("Loading subject recordings from disk")
        subject_rec = Recordings.from_wav(subj_dir)

    # -----------------------------------------------------------------
    # 2) Reference recordings
    # -----------------------------------------------------------------
    if overwrite or not ref_dir.exists():
        logging.info("Recording reference")
        ref_dir.mkdir(parents=True, exist_ok=True)

        reference_rec = Recordings.record_dome(
            n_directions=1,
            n_recordings=n_recordings,
            hp_freq=hp_freq,
            fs=fs,
        )
        reference_rec.params["subject_id"] = reference_id
        reference_rec.to_wav(ref_dir, overwrite=overwrite)
    else:
        logging.info("Loading reference recordings from disk")
        reference_rec = Recordings.from_wav(ref_dir)

    # -----------------------------------------------------------------
    # 3) Deconvolution: sweeps -> IRs
    # -----------------------------------------------------------------
    logging.info("Computing impulse responses")
    subject_ir = compute_ir(subject_rec)
    reference_ir = compute_ir(reference_rec)

    # -----------------------------------------------------------------
    # 4) Equalization
    # -----------------------------------------------------------------
    logging.info("Applying equalization")
    hrir = equalize(
        measured=subject_ir,
        reference=reference_ir,
        n_samples_out=n_samples_out,
    )

    # -----------------------------------------------------------------
    # 5) Low-frequency extrapolation
    # -----------------------------------------------------------------
    logging.info("Low-frequency extrapolation")
    hrir = lowfreq_extrapolate(
        hrir,
        f_extrap=400.0,
        f_target=150.0,
        head_radius=0.0875,
    )

    # -----------------------------------------------------------------
    # 6) Azimuth expansion + binaural cues
    # -----------------------------------------------------------------
    logging.info("Expanding azimuths and imposing binaural cues")
    hrir = expand_azimuths_with_binaural_cues(
        hrir,
        az_range=(-50, 50),
        head_radius=0.0875,
        show=show,
    )

    # -----------------------------------------------------------------
    # 7) Export to slab.HRTF
    # -----------------------------------------------------------------
    logging.info("Converting to slab.HRTF")
    hrtf = hrir.to_slab_hrtf(datatype="FIR")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        hrtf.plot_tf(hrtf.cone_sources(0), axis=ax)
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
