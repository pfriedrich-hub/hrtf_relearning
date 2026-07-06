"""
edge_label_plot.py  (dev/analysis)

Label rising EDGES on top of the Iida-Gaussian minima detection (same detector
config as verify_notch_method.py: light Gaussian sigma=122 Hz, which sits on the
true minima where cepstral n_keep merges/shifts them). For each Gaussian-detected
notch the rising edge is the steepest-slope point of its upper flank (the low->high
attenuation contrast a DCN type-IV edge detector keys on). Dotted red lines mark
where each edge lands after a +1 / +2 ERB shift -- the manipulation to be applied
to the original DTF in the shift-edge pilot.

Same 4-subject config as verify_notch_method.py: JS, CA (our data) + two Aachen
MRT subjects (external DB, cartesian coords). Aachen dir via AACHEN_DIR env.

Run: python3 hrtf_relearning/hrtf/processing/dev/edge_label_plot.py
"""
import os
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
AACHEN = Path(os.environ.get("AACHEN_DIR", "/sessions/friendly-exciting-planck/mnt/aachen_database"))
OUT = ROOT / "analysis_results/edge_labels.png"
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

SIGMA_HZ = 122.0                 # Iida Eq 2-3
BAND = (5000., 13500.)           # elevation-notch search band for minima
F_LO, F_HI = 3000., 17500.
ERB_STEPS = [1.0, 2.0]           # edge-shift magnitudes to preview


def load_median(sofa, aachen=False):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    if aachen:
        x, y, z = sp[:, 0], sp[:, 1], sp[:, 2]; r = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x)); el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1, 1)))
        med = np.where((np.abs(az) <= 3) & (x > 0))[0]
    else:
        azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]; med = np.where(np.abs(azn) <= 2)[0]
    med = med[np.argsort(el[med])]
    return ir, fs, el, med


def gauss(x, fs):
    n = len(x); nfft = int(2**np.ceil(np.log2(n)) * 4)
    fr = np.fft.rfftfreq(nfft, 1 / fs); mag = np.abs(np.fft.rfft(x, nfft))
    L = 20 * np.log10(np.maximum(mag, 1e-9))
    Lg = gaussian_filter1d(L, SIGMA_HZ / (fr[1] - fr[0]), mode="nearest")
    return fr, L, Lg


def edges_from_minima(fr, Lg, band=BAND):
    """Minima (Iida Gaussian) in `band`; for each, the rising (upper) edge =
    steepest-slope point of the flank between the minimum and the next peak."""
    b = (fr >= band[0]) & (fr <= band[1]); frb = fr[b]; Lb = Lg[b]
    mins, _ = find_peaks(-Lb, prominence=3.0); pks, _ = find_peaks(Lb, prominence=3.0)
    x = np.log2(frb); dL = np.gradient(Lb, x)
    out = []
    for mi in mins:
        after = pks[pks > mi]; hi = after[0] if len(after) else len(frb) - 1
        if hi <= mi:
            continue
        ei = mi + int(np.argmax(dL[mi:hi + 1]))
        out.append(dict(f_min=frb[mi], l_min=Lb[mi], f_edge=frb[ei], l_edge=Lb[ei]))
    return out


def main():
    subjects = [("JS", SOFA / "JS/JS.sofa", False), ("CA", SOFA / "CA/CA.sofa", False),
                ("MRT01 (Aachen)", AACHEN / "MRT01.sofa", True), ("MRT08 (Aachen)", AACHEN / "MRT08.sofa", True)]
    els_want = [-16.0, 16.0]
    fig, axes = plt.subplots(len(subjects), len(els_want), figsize=(13, 14), sharex=True)
    for r, (name, path, aach) in enumerate(subjects):
        ir, fs, el, med = load_median(path, aachen=aach)
        for c, ew in enumerate(els_want):
            ax = axes[r, c]; d = med[np.argmin(np.abs(el[med] - ew))]
            fr, L, Lg = gauss(ir[d, 0, :], fs); m = (fr >= F_LO) & (fr <= F_HI)
            ax.plot(fr[m] / 1000, L[m], color="0.78", lw=0.8)
            ax.plot(fr[m] / 1000, Lg[m], color="#185FA5", lw=2.0)
            eds = edges_from_minima(fr, Lg); elabels = []; ymin = L[m].min()
            for e in eds:
                ax.plot(e["f_min"] / 1000, e["l_min"], "v", color="#185FA5", ms=10, mec="k", mew=0.5, zorder=6)
                ax.plot(e["f_edge"] / 1000, e["l_edge"], "o", color="#2CA02C", ms=8, mec="k", mew=0.5, zorder=7)
                ax.axvline(e["f_edge"] / 1000, color="#2CA02C", lw=1.0, alpha=0.6)
                for k, n in enumerate(ERB_STEPS):
                    fs_erb = float(es.erb_to_hz(es.hz_to_erb(e["f_edge"]) + n))
                    ax.axvline(fs_erb / 1000, color="#D62728", lw=1.0, ls=":", alpha=0.7)
                    ax.annotate(f"+{n:.0f}", (fs_erb / 1000, ymin + 1.5 + 2 * k), color="#D62728", fontsize=6, ha="center")
                elabels.append(round(e["f_edge"] / 1000, 1))
            ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 17])
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.set_xlim(F_LO / 1000, F_HI / 1000); ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{name}, el={el[d]:.0f}°  | rising edges {elabels} kHz", fontsize=9)
            if r == len(subjects) - 1:
                ax.set_xlabel("frequency (kHz)")
            if c == 0:
                ax.set_ylabel("mag (dB)")
    leg = [Line2D([], [], color="0.78", lw=0.8, label="raw"),
           Line2D([], [], color="#185FA5", lw=2, label="Iida Gaussian 122 Hz"),
           Line2D([], [], marker="v", color="#185FA5", lw=0, mec="k", label="minimum"),
           Line2D([], [], marker="o", color="#2CA02C", lw=0, mec="k", label="rising edge (steepest slope)"),
           Line2D([], [], color="#D62728", lw=1, ls=":", label="edge shifted +1/+2 ERB")]
    axes[0, 0].legend(handles=leg, fontsize=7, loc="lower left")
    fig.suptitle("Edge labelling: rising edge = steepest point of each Gaussian-detected notch's upper flank; "
                 "dotted red = target after +1/+2 ERB shift", fontsize=12, y=0.997)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=135, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
