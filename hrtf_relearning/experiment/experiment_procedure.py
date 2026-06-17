import logging
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import slab
import freefield
import hrtf_relearning as hr
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.experiment.Localization.Localization_dome import LocalizationDome
from hrtf_relearning.experiment.Localization.Localization_AR import Localization
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_localization, plot_elevation_response)
import hrtf_relearning.experiment.Training as Training
logging.getLogger().setLevel("INFO")
freefield.set_logger("info")
ROOT = hr.PATH


SUBJECT_ID   = "NKa"
HP_ID        = "MYSPHERE"       # headphone model
subject  = Subject(SUBJECT_ID)

def main():
    # --- Localization settings ---------------------------------------------------
    HRIR_NAME = f"{SUBJECT_ID}"
    
    hrir_settings = dict(
        name=HRIR_NAME,
        subject_id=SUBJECT_ID,
        ear=None,
        mirror=False,
        reverb=True,
        drr=20,
        hp_filter=True,
        hp=HP_ID,
        convolution='cpu',
        storage='cpu',)

    loc_settings = dict(
        kind              = "sectors",
        azimuth_range     = (-35, 35),
        elevation_range   = (-35, 35),
        sector_size       = (14, 14),
        targets_per_sector = 3,
        replace           = False,
        min_distance      = 20,
        gain              = 0.2,
        stim              = "noise",)

    vr_loc = Localization(subject, hrir_settings, loc_settings=loc_settings)
    vr_loc.run()

    # -----------------------------------------------------------------------------
    # 4b.  TRAINING SESSION  (standalone — run daily after localization)

    training_settings = dict(
        target_size     = 4,
        target_time     = 0.5,
        min_dist        = 30,
        game_time       = 90,      # seconds per game
        trial_time      = 10,
        score_time      = 3,
        gain            = 0.10,
        azimuth_range   = (0, 35),
        elevation_range = (-35, 35),
    )

    Training.SUBJECT_ID = SUBJECT_ID
    Training.HRIR_NAME  = hrir_settings["name"]
    Training.EAR        = hrir_settings.get("ear")
    Training.HP         = HP_ID
    Training.AZ_RANGE   = training_settings["azimuth_range"]
    Training.settings.update(training_settings)
    # Reload HRIR inside Training with current settings
    Training.hrir = hrtf2binsim(hrir_settings)
    Training.subject = Subject(SUBJECT_ID)
    Training.play_session()

    # ------------------------------------------------------------------
    # 4. Dome localization (real speakers, vertical midline)
    # ------------------------------------------------------------------
    logging.info('--- Step 4: Dome localization ---')
    dome_loc_settings = {
        'targets_per_speaker': 3,
        'min_distance': 15,
        'gain': 1,
    }
    dome_loc = LocalizationDome(subject, hrir_settings, loc_settings=dome_loc_settings)
    dome_loc.run()

    # ------------------------------------------------------------------
    # 5. Virtual localization (pybinsim)
    #    hrtf2binsim + hp filter loading are handled inside Localization()
    logging.warning('--------- Switch HP jack to PC ---------')
    # ------------------------------------------------------------------

    logging.info('--- Step 5: Virtual localization ---')


    ar_loc_settings = {
        'kind': 'standard',
        'azimuth_range': (-1, 1), 'elevation_range': (-35, 35),
        'targets_per_speaker': 2, 'min_distance': 15,
        'gain': .2,
        'stim': 'noise',
    }
    ar_loc = Localization(subject, hrir_settings, loc_settings=ar_loc_settings)
    ar_loc.run()


    # -----------------------------------------------------------------------------
    # DOME LOCALIZATION  (real speakers — standalone call)
    # -----------------------------------------------------------------------------
    dome_loc = LocalizationDome(subject)
    dome_loc.run()
