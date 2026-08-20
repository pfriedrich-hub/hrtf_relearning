"""
Orchestration for one subject's HRIR acquisition: directory handling plus the
record -> process -> SOFA chain. No DSP lives here; every step delegates to
`recordings` (rig) or `processing` (signal path).

    record_reference()  mics on the STAND, no listener -- the in-situ
                        calibration of speakers + mics + room. Records with the
                        SAME equalize_dome as the subject, refuses to silently
                        reuse an existing id.
    record_hrir()       1) record (or load) subject ear-pressure sweeps
                        2) record (or load) reference sweeps
                        3) deconvolve sweeps -> IRs
                        4) equalize against the reference, window,
                           zero frontal ITD/ILD
                        5) low-frequency extrapolation (spherical head)
                        6) azimuth expansion + binaural cue imposition
                        7) export to slab.HRTF / SOFA

Parameters come from the PROTOCOL (`experiment/protocols/HRIR_Recording.py`),
not from the module-level values below, which exist only for running this file
interactively. `head_radius` in particular differs between the two -- see the
note on that variable.
"""
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
from hrtf_relearning.hrtf.record.recordings import *
from hrtf_relearning.hrtf.record.processing import *
from hrtf_relearning.utils import paths
base_dir = paths.HRTF_DIR
import logging
from datetime import datetime

subject_id = 'kemar_pir'
# NOTE: 0.0875 m (Woodworth) is NOT what the experiment uses. The protocol sets
# HEAD_RADIUS = 0.0725 from the acoustic fit (step 0) and passes it explicitly.
# This value only leaks into `record_hrir`'s signature default, so a caller that
# forgets to pass it silently builds a different head -- it scales every
# off-midline ITD by 12-16%. Take the value from the protocol, always.
head_radius = 0.0875
reference_id = 'ref_03.04'
overwrite = False
n_directions = 1
n_recordings = 10
n_samples_out = 512
fs = 48828  # 97656
hp_freq = 120
show = True
# Dome EQ: OFF, for subject AND reference alike. The reference division IS the
# per-speaker in-situ calibration; the dome EQ adds a stale filter and costs
# headroom, and if it is on for only one of the two it does not cancel -- the
# per-speaker division in equalize() then leaves HRTF / E_k, elevation-dependent
# because each midline elevation is a different speaker.
# Everything recorded before 2026-08-20 has that mismatch (subjects False, all
# references True) and keeps it by decision -- do not rebuild. From ref_20.08
# on, record_reference() (step 0b in HRIR_Recording.py) records the reference
# with the same flag as the subject, so the mismatch is closed by construction.
# See docs/hrir_recording_audit.md finding A / project_dome_eq_mismatch.md.
equalize_dome = False
align_interaural = True

slab.set_default_samplerate(fs)

# freefield is a rig-only dependency: recordings.py imports it lazily so that
# this module stays importable on a cue-editing install (no TDT, no drivers).
# Keep the same contract here -- only configure its logger if it is present.
try:
    import freefield
    freefield.set_logger("info")
except ImportError:
    logging.info("freefield not available -- recording disabled, processing only")


# ---------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------

def record_reference(
    reference_id: str,
    *,
    n_recordings: int = 10,
    fs: int = 48828,
    hp_freq: float = 120,
    equalize_dome: bool = False,
    base_dir: Path | str | None = None,
    overwrite: bool = False,
    prompt: bool = True,
) -> Recordings:
    """Record one reference sweep set: mics on the STAND, no listener.

    The reference is the in-situ calibration of the whole chain -- same
    speakers, same mics, same room -- so it must be recorded the same way the
    subjects are. In practice that means ONE thing: `equalize_dome` here must
    match `equalize_dome` in the subject recording. It divides out per speaker,
    so if only one side has the dome EQ applied the speaker response is left
    imprinted on every DTF, elevation-dependently (each midline elevation is a
    different speaker). Default here is False, matching every subject on disk.

    Refuses to overwrite an existing id. `record_hrir` records a reference only
    `if not ref_dir.exists()`, so reusing an id silently loads the old sweeps --
    which is how a stale reference stays in service for months without anyone
    noticing. Pick a fresh id, or pass overwrite=True deliberately.
    """
    base_dir = Path(base_dir) if base_dir is not None else paths.HRTF_DIR
    ref_dir = base_dir / "rec" / "reference" / reference_id

    if ref_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{ref_dir} already exists. Use a fresh reference_id (dated ids "
            f"like 'ref_20.08' work well), or pass overwrite=True if you really "
            f"mean to replace it. Silently reusing an id is how the stale "
            f"equalize_dome=True references stayed in service.")

    if prompt:
        input(f"Reference '{reference_id}' (equalize_dome={equalize_dome}). "
              f"Put the mics on the stand at the listening position, clear the "
              f"chair, and press Enter ...")

    logging.info(f"Recording reference '{reference_id}' with equalize_dome={equalize_dome}")
    reference_rec = Recordings.record_dome(
        id=reference_id,
        n_directions=1,
        n_recordings=n_recordings,
        hp_freq=hp_freq,
        fs=fs,
        equalize_dome=equalize_dome,
        key=False)
    ref_dir.mkdir(parents=True, exist_ok=True)
    # Fresh sweeps must always be stored. This used to read `overwrite=overwrite`,
    # which is not a parameter of record_hrir and silently resolved to the
    # module-level global False.
    reference_rec.to_npz(ref_dir, overwrite=True)
    logging.info(f"Wrote {ref_dir / 'recordings.npz'}")
    return reference_rec


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
    overwrite_rec: bool = False,
    overwrite_hrir: bool = True,
    align_interaural: bool = True,
    head_radius: float = head_radius,
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
        base_dir = paths.HRTF_DIR
    else:
        base_dir = Path(base_dir)

    out_file = base_dir / 'sofa' / subject_id / f'{subject_id}.sofa'

    # Early exit: load and return existing SOFA without re-running the pipeline
    if not overwrite_hrir and out_file.exists():
        logging.info(f"Loading existing HRTF from {out_file}")
        hrtf = slab.HRTF(str(out_file))
        if show:
            plot_hrtf(hrtf, subject_id)
        return hrtf

    subj_dir = base_dir / "rec" / subject_id
    ref_dir = base_dir / "rec" / "reference" / reference_id

    # -----------------------------------------------------------------
    # 1) Subject recordings
    # -----------------------------------------------------------------
    npz_file = subj_dir / "recordings.npz"
    if overwrite_rec or not npz_file.exists():
        logging.info("Recording subject ear pressure")
        subj_dir.mkdir(parents=True, exist_ok=True)

        subject_rec = Recordings.record_dome(
            id=subject_id,
            n_directions=n_directions,
            n_recordings=n_recordings,
            hp_freq=hp_freq,
            fs=fs,
            equalize_dome=equalize_dome,
            key=True)
        # Fresh sweeps were just recorded, so they must be stored -- passing the
        # caller's `overwrite` here silently discarded re-recordings of subjects
        # who already had a recordings.npz. Any previous session is archived
        # rather than replaced.
        if npz_file.exists():
            stamp = datetime.fromtimestamp(npz_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            archived = npz_file.with_name(f"recordings_{stamp}.npz")
            npz_file.rename(archived)
            logging.warning(f"Archived previous recordings to {archived}")
        subject_rec.to_npz(subj_dir, overwrite=True)
    else:
        logging.info("Loading subject recordings from disk")
        subject_rec = Recordings.load(subj_dir)

    # -----------------------------------------------------------------
    # 2) Reference recordings
    # -----------------------------------------------------------------
    if not ref_dir.exists():
        # Recorded with the SAME equalize_dome as the subject above -- that is
        # the whole point of doing it here rather than separately, and it is
        # what went wrong with every reference on disk before 2026-08-19
        # (project_dome_eq_mismatch).
        reference_rec = record_reference(
            reference_id, n_recordings=n_recordings, fs=fs, hp_freq=hp_freq,
            equalize_dome=equalize_dome, base_dir=base_dir)
    else:
        logging.info("Loading reference recordings from disk")
        reference_rec = Recordings.load(ref_dir)
        stored = reference_rec.params.get("equalize_dome")
        if stored is not None and bool(stored) != bool(equalize_dome):
            logging.warning(
                "Reference '%s' was recorded with equalize_dome=%s but this "
                "subject is being recorded with %s. The dome EQ will NOT cancel "
                "in equalize() -- the per-speaker speaker response is imprinted "
                "on the DTF, elevation-dependently, because each midline "
                "elevation is a different speaker. Record a matching reference "
                "with record_reference(). See project_dome_eq_mismatch.",
                reference_id, stored, equalize_dome)

    # -----------------------------------------------------------------
    # 3) Deconvolution: sweeps -> IRs
    # -----------------------------------------------------------------
    logging.info("Computing impulse responses")
    # align_interaural is deliberately NOT passed here. Zeroing the frontal
    # ITD/ILD on the subject and the reference separately happens before the
    # division in equalize() and leaves a systematic interaural residual
    # instead of removing one -- it is done post-equalization below.
    # These two calls invert the EXCITATION SWEEP; `equalize` below inverts the
    # measured REFERENCE IR. Different signals, hence different upper bounds --
    # the rationale is on the constants in processing.py.
    subject_ir = compute_ir(
        subject_rec,
        inversion_range_hz=(hp_freq, EXCITATION_INVERSION_TOP_HZ),
        onset_threshold_db=10, align_interaural=False)
    reference_ir = compute_ir(
        reference_rec,
        inversion_range_hz=(hp_freq, EXCITATION_INVERSION_TOP_HZ),
        onset_threshold_db=10, align_interaural=False)

    # -----------------------------------------------------------------
    # 4) Equalization + windowing (+ frontal ITD/ILD zeroing, post-division)
    # -----------------------------------------------------------------
    logging.info("Applying equalization")
    hrir_equalized = equalize(
        measured=subject_ir,
        reference=reference_ir,
        n_samples_out=n_samples_out,
        inversion_range_hz=(hp_freq, REFERENCE_INVERSION_TOP_HZ),
        onset_threshold_db=10,
        align_interaural=align_interaural,
    )

    # -----------------------------------------------------------------
    # 5) Low-frequency extrapolation
    # -----------------------------------------------------------------
    logging.info("Low-frequency extrapolation")
    hrir_extrapol = lowfreq_extrapolate(
        hrir_equalized,
        f_extrap=800.0,
        f_target=150.0,
        head_radius=head_radius,
    )

    # -----------------------------------------------------------------
    # 6) Azimuth expansion + binaural cues
    # -----------------------------------------------------------------
    if expand_az:
        logging.info(f"Expanding azimuths and imposing binaural cues. Head radius: {head_radius}")
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
    hrtf = hrir_az_exp.to_slab_hrtf(datatype="FIR")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing HRTF to {out_file}")
    hrtf.write_sofa(out_file)

    if show:
        plot_hrtf(hrtf, subject_id)

    logging.info("HRIR pipeline finished successfully")
    return hrtf


def plot_hrtf(hrtf: slab.HRTF, subject_id: str) -> None:
    """Plot midline-cone TFs and save the figure to RESULTS_DIR/<id>/plots/acoustic/."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    hrtf.plot_tf(hrtf.cone_sources(0), axis=axes, ear='both')
    plot_dir = paths.subject_acoustic_dir(subject_id)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_file = plot_dir / f'{subject_id}_hrtf.png'
    fig.savefig(fig_file, dpi=200, bbox_inches='tight')
    logging.info(f"Saved HRTF plot to {fig_file}")
    plt.show()




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