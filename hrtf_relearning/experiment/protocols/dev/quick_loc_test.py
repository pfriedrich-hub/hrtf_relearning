"""
quick_loc_test.py

Minimal cell-by-cell protocol for a single AR localization test (headphones +
head tracker, Localization_AR). No phases, no conditions list -- just one
config cell you edit and one run cell you execute. Use it to quickly check a
subject on any HRTF (baseline or modified) with any target grid.

Typical loop:
    1. edit the CONFIG cell (subject, HRTF label, sectors)
    2. run `list_sofas()` / `preview_targets()` to check what you are about to run
    3. run the RUN cell
    4. look at the plots cell

Run cell by cell (# %%) -- do NOT execute top-to-bottom as a script.
"""

# %% imports -----------------------------------------------------------------
import slab
import matplotlib.pyplot as plt

import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.localization_helpers.make_sequence import (
    make_sequence, plot_sequence_targets,
)
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_localization, plot_elevation_response,
)
from hrtf_relearning.utils import paths

# %% CONFIG — edit this cell -------------------------------------------------
SUBJECT_ID = "SZ"          # participant id (data/results/<id>/)
HRTF_LABEL = None          # None -> baseline <id>.sofa
                           # else suffix, e.g. "whole_2" -> <id>_whole_2.sofa
EAR = None                 # None (binaural) | 'left' | 'right'
OTHER_EAR = "envelope"     # monaural only: 'flat' | 'envelope' | 'native'
MIRROR = False
HP = "DT990"
CONVOLUTION = "cpu"        # 'cpu' | 'cuda'

# --- target grid ---
KIND = "sectors"           # 'sectors' (sample per sector) | 'standard' (all sources x reps)
AZ_RANGE = (-35, 35)
EL_RANGE = (-35, 35)
SECTOR_SIZE = (7, 14)      # (az, el) — 'sectors' only
TARGETS_PER_SECTOR = 1     # 'sectors' only — 1 keeps the block short
TARGETS_PER_SPEAKER = 3    # 'standard' only
MIN_DISTANCE = 20          # deg between successive targets
EXCLUDE_MIDLINE = False    # drop az ~= 0 sources (hemifield/ear contrasts)
GAIN = 0.2
STIM = "noise"             # 'noise' | 'uso'

SOFA_DIR = paths.SOFA_DIR / SUBJECT_ID


def hrir_name():
    return SUBJECT_ID if HRTF_LABEL in (None, "baseline") else f"{SUBJECT_ID}_{HRTF_LABEL}"


def hrir_settings():
    return {
        "name": hrir_name(), "subject_id": SUBJECT_ID,
        "ear": EAR, "other_ear": OTHER_EAR, "mirror": MIRROR,
        "reverb": True, "drr": 20,
        "hp_filter": True, "hp": HP,
        "convolution": CONVOLUTION, "storage": CONVOLUTION,
    }


def loc_settings():
    settings = {
        "kind": KIND,
        "azimuth_range": AZ_RANGE, "elevation_range": EL_RANGE,
        "min_distance": MIN_DISTANCE, "exclude_midline": EXCLUDE_MIDLINE,
        "gain": GAIN, "stim": STIM,
    }
    if KIND == "sectors":
        settings.update({"sector_size": SECTOR_SIZE, "replace": False,
                         "targets_per_sector": TARGETS_PER_SECTOR})
    else:
        settings.update({"targets_per_speaker": TARGETS_PER_SPEAKER})
    return settings


def list_sofas():
    """Which HRTFs exist for this subject (i.e. valid HRTF_LABEL values)."""
    for path in sorted(SOFA_DIR.glob(f"{SUBJECT_ID}*.sofa")):
        label = path.stem[len(SUBJECT_ID) + 1:] or "(baseline)"
        print(f"{label:<24} {path.name}")


def preview_targets(show=True):
    """Build the sequence from the current config and plot the targets over the
    sector grid — checks the grid before a participant is sitting in the rig."""
    hrtf = slab.HRTF(str(SOFA_DIR / f"{hrir_name()}.sofa"))
    sequence = make_sequence(loc_settings(), hrtf.sources.vertical_polar)
    print(f"{hrir_name()}: {len(sequence.conditions)} trials")
    if show:
        plot_sequence_targets(sequence, title=f"{hrir_name()} targets")
    return sequence


# %% inspect: which HRTFs exist / what will be tested ------------------------
list_sofas()
preview_targets()

# %% RUN — one localization block --------------------------------------------
subject = hr.Subject(SUBJECT_ID)
loc = Localization(subject, hrir_settings(), loc_settings())
loc.run()
sequence = subject.localization[loc.filename]

# %% results ------------------------------------------------------------------
# (Localization.run already saves these to results/<id>/plots/; this reopens them)
plot_dir = paths.subject_plot_dir(SUBJECT_ID)
plot_localization(sequence, report_stats=["elevation", "azimuth"], filepath=plot_dir)
plot_elevation_response(sequence, filepath=plot_dir)
plt.show()

# %% optional: build a modified SOFA before running it -----------------------
# Writes <id>_<condition>_<delta>.sofa, then set HRTF_LABEL to that suffix above.
# from hrtf_relearning.hrtf.modify.edge_shift import save_condition_sofa, print_notch_summary
# CONDITION, DELTA = "whole", 2.0          # 'whole' | 'edge' | 'rising' | 'falling' | 'edge_only'
# label = f"{CONDITION}_{DELTA:g}".replace(".", "p")
# base = slab.HRTF(str(SOFA_DIR / f"{SUBJECT_ID}.sofa")); base.name = SUBJECT_ID
# _, reports = save_condition_sofa(base, CONDITION, DELTA,
#                                  SOFA_DIR / f"{SUBJECT_ID}_{label}.sofa",
#                                  feature_kw=dict(band=(4000., 15000.)), use_tracking=True)
# print_notch_summary(reports, label=label)
