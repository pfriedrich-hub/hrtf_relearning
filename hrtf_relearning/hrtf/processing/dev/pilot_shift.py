"""Apply the modify.py frequency-shift (shift_band) to pilot subjects and save a
2-panel original-vs-modified median-plane DTF image (slab plot_tf, kind='image').

Uses the same shift config as modify.py __main__ (SHIFT_CENTER 10 kHz, 1.5 oct,
factor 0.9, envelope n_keep 3). Deduplicates the 0/360 midline copy so plots
don't scramble. Idempotent: skips subjects whose PNG already exists.

Run:  python pilot_shift.py SUB1 SUB2 ...     (or no args -> default human set)
"""
import sys
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
matplotlib.use = lambda *a, **k: None
import matplotlib.pyplot as plt
import slab
import hrtf_relearning as hr
from hrtf_relearning.hrtf.processing.modify import shift_band, octave_band

SOFA = hr.PATH / "data" / "hrtf" / "sofa" / "pilot"
OUT = hr.PATH / "analysis_results" / "tac_figures" / "pilot_shift"
OUT.mkdir(parents=True, exist_ok=True)

# modify.py shift config
SHIFT_CENTER, SHIFT_OCTAVES, SHIFT_FACTOR = 10000, 1.5, 0.9
SHIFT_ENV_NKEEP, SHIFT_SKIRT = 3, 0.25
XLIM = (2000, 16000)

DEFAULT = ["AGV", "AS", "CZ", "IM", "JP", "JR", "JZ", "LS", "MB", "MD", "MS",
           "MSc", "NK", "NKa", "OS", "PC", "PF", "PFo", "RK", "SK", "SW", "UG",
           "VD", "VG"]


def midline_dedup(hrtf, az_tol=2.0):
    vp = hrtf.sources.vertical_polar
    az = vp[:, 0]
    azn = (az + 180) % 360 - 180
    idx = [i for i in range(len(vp)) if abs(azn[i]) <= az_tol and az[i] != 360.0]
    return sorted(idx, key=lambda i: vp[i, 1])


def one(sub):
    png = OUT / f"{sub}_shift.png"
    if png.exists():
        print("skip", sub, "(exists)"); return
    hrtf = slab.HRTF(str(SOFA / sub / f"{sub}.sofa"))
    low, high = octave_band(SHIFT_CENTER, fraction=SHIFT_OCTAVES)
    mod = shift_band(hrtf, low, high, factor=SHIFT_FACTOR,
                     envelope_n_keep=SHIFT_ENV_NKEEP, skirt_octaves=SHIFT_SKIRT,
                     onset_threshold_db=15.0)
    src = midline_dedup(hrtf)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    hrtf.plot_tf(src, ear="left", kind="image", xlim=XLIM, show=False, axis=ax[0])
    mod.plot_tf(src, ear="left", kind="image", xlim=XLIM, show=False, axis=ax[1])
    ax[0].set_title("original"); ax[1].set_title("modified (shift x0.9)")
    fig.suptitle(f"{sub}: median-plane DTF (left ear), modify.py frequency-shift")
    fig.tight_layout()
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", png.name)


if __name__ == "__main__":
    subs = sys.argv[1:] or DEFAULT
    for s in subs:
        try:
            one(s)
        except Exception as e:
            print(f"{s}: ERROR {type(e).__name__}: {e}")
