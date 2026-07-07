"""
tac_figures.py  (dev/analysis)

TAC-meeting figures: per subject and per manipulation, a two-panel figure --
LEFT  : compare_waterfall  (original grey vs manipulated red median-plane DTF,
        stacked by elevation), overlaid on the original DTF.
RIGHT : plot_elevation_response for the matching baseline localization run.

Two manipulations are shown, both at the same finalized delta (ERB):
    whole  -- whole-notch shift by <delta> ERB   (SOFA: <id>_whole_<D>.sofa)
    edge   -- rising-edge shift by <delta> ERB   (SOFA: <id>_edge_<D>.sofa)

The matching localization run is found in Subject(id).localization by suffix
(<id>_whole_<D> / <id>_edge_<D>), so the same call works for tomorrow's
participants once they are measured -- just change --subject.

Outputs (analysis_results/tac_figures/):
    <id>_whole_<D>.png / .pdf     one panel-pair per manipulation
    <id>_edge_<D>.png  / .pdf
    <id>_tac.pdf                  both manipulations, one multi-page PDF

Run (in the plotting venv, with LD_LIBRARY_PATH set for portaudio):
    python tac_figures.py --subject AS --delta 1
    python tac_figures.py --subject AS --delta 1 --conditions whole edge
"""
import argparse
import importlib.util as iu
from pathlib import Path

# --- headless matplotlib BEFORE any hrtf_relearning import (which hardcodes tkagg)
import matplotlib
matplotlib.use("Agg", force=True)
matplotlib.use = lambda *a, **k: None          # neutralize hardcoded backend switch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import slab
import hrtf_relearning as hr
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_elevation_response,
)

# load edge_shift by path (module is the source of compare_waterfall)
_ES_PATH = hr.PATH / "hrtf" / "processing" / "edge_shift.py"
_spec = iu.spec_from_file_location("edge_shift", _ES_PATH)
es = iu.module_from_spec(_spec)
_spec.loader.exec_module(es)

SOFA_DIR = hr.PATH / "data" / "hrtf" / "sofa"
OUTDIR = hr.PATH / "analysis_results" / "tac_figures"

# human-readable red label per condition
COND_LABEL = {"whole": "whole-notch shift", "edge": "rising-edge shift"}


def delta_tag(delta):
    """1 -> '1', 0.5 -> '0p5', 0.25 -> '0p25', 1.5 -> '1p5' (matches SOFA names)."""
    s = ("%g" % float(delta))
    return s.replace(".", "p")


def find_loc_run(subject, sid, cond, dtag):
    """Localization key ending in '<id>_<cond>_<dtag>' (keys carry a timestamp prefix)."""
    suffix = f"{sid}_{cond}_{dtag}"
    hits = [k for k in subject.localization if k.endswith(suffix)]
    if not hits:
        raise KeyError(
            f"no localization run ending '{suffix}' for {sid}. "
            f"available: {list(subject.localization)}")
    # newest-looking (last) if several
    return sorted(hits)[-1]


def make_panel(base_hrtf, manip_sofa, seq, delta, cond, ear="left"):
    """One 2-panel figure: waterfall overlay (left) + elevation response (right)."""
    manip_hrtf = slab.HRTF(str(manip_sofa))
    fig, ax = plt.subplots(1, 2, figsize=(13, 7),
                           gridspec_kw={"width_ratios": [1.15, 1.0]})
    red_label = f"{COND_LABEL.get(cond, cond)} (+{delta:g} ERB)"
    es.compare_waterfall(base_hrtf, manip_hrtf, ear=ear, axis=ax[0], show=False,
                         labels=("original", red_label))
    ax[0].set_title(f"median-plane DTF ({ear} ear)\noriginal vs {red_label}",
                    fontsize=10)
    plot_elevation_response(seq, axis=ax[1], add_fit=True)
    fig.suptitle(f"{seq.name}", fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="TAC figures: DTF waterfall + elevation response.")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--delta", type=float, default=1.0, help="ERB shift (default 1)")
    ap.add_argument("--conditions", nargs="+", default=["whole", "edge"])
    ap.add_argument("--ear", default="left")
    a = ap.parse_args()

    sid, delta = a.subject, a.delta
    dtag = delta_tag(delta)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    base_hrtf = slab.HRTF(str(SOFA_DIR / sid / f"{sid}.sofa"))
    subject = hr.Subject(sid)

    figs = []
    for cond in a.conditions:
        manip_sofa = SOFA_DIR / sid / f"{sid}_{cond}_{dtag}.sofa"
        if not manip_sofa.exists():
            print(f"[skip] {cond}: {manip_sofa} not found")
            continue
        key = find_loc_run(subject, sid, cond, dtag)
        seq = subject.localization[key]
        fig = make_panel(base_hrtf, manip_sofa, seq, delta, cond, ear=a.ear)
        stem = OUTDIR / f"{sid}_{cond}_{dtag}"
        fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"saved {stem}.png / .pdf   (loc run: {key})")
        figs.append(fig)

    if figs:
        pdf_path = OUTDIR / f"{sid}_tac.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"saved combined {pdf_path}")


if __name__ == "__main__":
    main()
