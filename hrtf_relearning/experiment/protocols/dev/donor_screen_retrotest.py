"""Retro-test the day-1 donor screen against every screen already run.

The gate in `donor_screening` was written from two datasets and validated
against four subjects who were run BEFORE it existed. This script is that
validation, kept runnable so it can be re-run whenever a threshold moves --
change a constant in donor_screening and see immediately which historical
verdict flips.

Expected output (2026-08-31):
    NR   pilot/AH REJECT (azimuth gain 1.60, azimuth RMSE, EG 0.00)
         pilot/SW PASS      -- matches the call Paul made on the day
    FP   FS       REJECT (EG 0.00)
         pilot/AH PASS      -- matches the call, on a defensible reason:
                               Paul's own comparison rested on a 5.1 deg
                               difference against a 5.9 deg detection limit
    LS   pilot/AH REJECT (azimuth gain 1.38, EG 0.09)
                            -- never screened; she ran the whole study on it
    AS   GS       REJECT (impairment only +2.5 deg)
                            -- never screened; the manipulation never bit, and
                               her 0.41 -> 0.51 "recovery" is inside the noise

CAVEAT. Historically the own-HRTF reference is binaural FULL FIELD (n=150) and
the donor blocks are monaural HEMIFIELD (n=75). The restricted azimuth range
lowers RMSE mechanically, so the azimuth-RMSE gate is weaker here than it is
prospectively -- the screen proper matches the geometry of the native block on
purpose. Azimuth GAIN is geometry-robust and does most of the work above.
"""

import json

import numpy

from hrtf_relearning.experiment.analysis.localization.localization_analysis \
    import block_summary
from hrtf_relearning.experiment.protocols.learning_transfer import donor_screening
from hrtf_relearning.utils import paths

# (label, subject, own-HRTF reference block, [(donor, shortlist rank, block)])
CASES = [
    ("NR  — screened pilot/AH then pilot/SW, chose SW",
     "NR", "NR_25.08_12-05_NR",
     [("pilot/AH", 0, "NR_25.08_12-20_NR_donor_AH_env4_right"),
      ("pilot/SW", 1, "NR_25.08_12-43_NR_donor_SW_env4_right")]),
    ("FP  — screened FS then pilot/AH, chose AH",
     "FP", "FP_24.08_15-05_FP",
     [("FS", 0, "FP_24.08_15-21_FP_donor_FS_env4_left"),
      ("pilot/AH", 1, "FP_24.08_15-31_FP_donor_AH_env4_left")]),
    ("LS  — never screened; what a screen would have said",
     "LS", "LS_25.08_09-50_LS",
     [("pilot/AH", 0, "LS_25.08_09-57_LS_donor_AH_env4_right")]),
    ("AS  — never screened; the manipulation that did not bite",
     "AS", "AS_18.08_11-50_AS",
     [("GS", 0, "AS_18.08_12-35_AS_donor_GS_env4_left")]),
]


class _Seq:
    """Minimal stand-in so block_summary can score a run read back from JSON."""

    def __init__(self, data):
        self.data = data
        self.this_n = len(data) - 1
        self.n_remaining = 0
        self.stim = None


def _block(subject_id, key):
    path = paths.subject_dir(subject_id) / f"{subject_id}.json"
    if not path.exists():
        path = paths.RESULTS_DIR / "pilot" / subject_id / f"{subject_id}.json"
    return block_summary(_Seq(json.load(open(path))["localization"][key]["data"]))


def main():
    for label, subject_id, reference_key, candidates in CASES:
        print("\n" + "#" * 78)
        print("#", label)
        print("#" * 78)
        reference_block = _block(subject_id, reference_key)
        reference = dict(pe=reference_block["polar_error"],
                         eg=reference_block["elevation_gain"],
                         az_rmse=reference_block["azimuth_rmse"],
                         n=reference_block["n"])
        rows = []
        for donor_id, rank, key in candidates:
            measured = _block(subject_id, key)
            rows.append(dict(donor=donor_id, rank=rank, n=measured["n"],
                             pe=measured["polar_error"],
                             eg=measured["elevation_gain"],
                             az_gain=measured["azimuth_gain"],
                             az_rmse=measured["azimuth_rmse"]))
        evaluated, chosen = donor_screening.evaluate(reference, rows)
        donor_screening.report(evaluated, chosen, reference)


if __name__ == "__main__":
    main()
