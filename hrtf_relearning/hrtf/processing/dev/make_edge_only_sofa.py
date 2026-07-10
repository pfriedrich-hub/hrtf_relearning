"""
make_edge_only_sofa.py  (dev)

Build the edge-only condition SOFA for a subject: a low-n_keep externalisation
baseline with the rising edge(s) of the N most prominent relevant cue(s) added
back at their measured shape (edge_shift.edge_only_ir). Writes
<subject>_edge_only.sofa next to the source (Data.IR replaced, all other SOFA
metadata copied verbatim) and a median-plane QC waterfall.

Run: python3 hrtf_relearning/hrtf/processing/dev/make_edge_only_sofa.py AS \
        --n_keep 4 --n_cues 3 --tilt 4
"""
import sys, shutil, argparse
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
OUTDIR = ROOT / "analysis_results/edge_cascade"
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)


def sofa_path(name):
    for p in (SOFA / name / f"{name}.sofa", SOFA / "pilot" / name / f"{name}.sofa",
              SOFA / "pilot" / f"{name}.sofa"):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def raw_db(x, fs):
    nfft = int(2 ** np.ceil(np.log2(len(x))) * 4)
    return np.fft.rfftfreq(nfft, 1 / fs), 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x, nfft)), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--n_keep", type=int, default=4)
    ap.add_argument("--n_cues", type=int, default=3)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--band", type=float, nargs=2, default=(4000., 16000.))
    a = ap.parse_args()

    src = sofa_path(a.subject)
    with h5py.File(src, "r") as f:
        IR = f["Data.IR"][:]                       # (n_dir, n_ear, n_tap)
        sp = f["SourcePosition"][:]
        fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    kw = dict(n_keep_baseline=a.n_keep, n_cues=a.n_cues, tilt_db_per_oct=a.tilt,
              feature_kw=dict(band=tuple(a.band), sep_min_oct=0.0))

    out = np.empty_like(IR)
    cue_counts = []
    for d in range(IR.shape[0]):
        for ear in range(IR.shape[1]):
            y, rep = es.edge_only_ir(IR[d, ear, :], fs, return_report=True, **kw)
            out[d, ear, :] = y
            cue_counts.append(rep["n_edges"])
    cc = np.bincount(cue_counts)
    print(f"{a.subject}: n_keep={a.n_keep} n_cues<={a.n_cues} tilt={a.tilt} band={a.band}")
    print("  edges-per-direction/ear histogram:", {i: int(c) for i, c in enumerate(cc) if c})

    # write SOFA: copy source, replace Data.IR
    dst = src.parent / f"{a.subject}_edge_only.sofa"
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        f["Data.IR"][...] = out
    print("wrote", dst)

    # median-plane QC waterfall
    azn = (sp[:, 0] + 180) % 360 - 180
    el = sp[:, 1]
    med = np.where(np.abs(azn) <= 2)[0]
    med = med[(el[med] >= -35) & (el[med] <= 35)]
    med = med[np.argsort(el[med])]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 9))
    off = 0.0
    for d in med:
        fr, L0 = raw_db(IR[d, 0, :], fs); _, L1 = raw_db(out[d, 0, :], fs)
        b = (fr >= 3000) & (fr <= 16000)
        ax.plot(fr[b] / 1000, L0[b] + off, color="0.7", lw=1.1)
        ax.plot(fr[b] / 1000, L1[b] + off, color="#D62728", lw=1.3)
        ax.text(16 * 1.02, L1[b][-1] + off, f"{el[d]:+.0f}", fontsize=7, va="center")
        off += 30
    ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 16])
    ax.get_xaxis().set_major_formatter(ScalarFormatter()); ax.set_xlim(3, 16)
    ax.set_yticks([]); ax.set_xlabel("frequency (kHz)")
    ax.legend(handles=[Line2D([], [], color="0.7", lw=1.2, label="original"),
                       Line2D([], [], color="#D62728", lw=1.3,
                              label=f"edge-only ({a.n_cues} cues, n_keep {a.n_keep}, tilt {a.tilt:.0f})")],
              loc="upper left", fontsize=9)
    ax.set_title(f"{a.subject}: median-plane DTF, original vs edge-only (left ear)")
    fig.tight_layout()
    qc = OUTDIR / f"{a.subject}_edge_only_waterfall.png"
    fig.savefig(qc, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", qc)


if __name__ == "__main__":
    main()
