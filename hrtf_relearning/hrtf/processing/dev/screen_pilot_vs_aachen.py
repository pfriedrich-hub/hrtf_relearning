"""Summary of the corrected (Gaussian-detector) resolvability screen:
pilot set vs external Aachen database. Counts from screen_resolvable_gaussian.py."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs/screen_pilot_vs_aachen.png")
# pass / marginal / fail counts (>=50% / 20-50% / <20% of tested elevations resolvable)
data = {"pilot (our recordings)": dict(n=25, passv=0, marg=14, fail=11),
        "Aachen (external DB)":    dict(n=46, passv=20, marg=20, fail=6)}

fig, ax = plt.subplots(figsize=(8.5, 5))
labels = list(data)
y = np.arange(len(labels))
cols = {"pass": "#1D9E75", "marg": "#EF9F27", "fail": "#D85A30"}
for i, lab in enumerate(labels):
    d = data[lab]; n = d["n"]
    left = 0
    for key, disp in [("passv", "resolvable ≥50%"), ("marg", "marginal 20–50%"), ("fail", "fail <20%")]:
        frac = 100 * d[key] / n
        c = cols["pass" if key == "passv" else ("marg" if key == "marg" else "fail")]
        ax.barh(i, frac, left=left, color=c, edgecolor="white")
        if frac > 5:
            ax.text(left + frac/2, i, f"{d[key]}", va="center", ha="center",
                    color="white", fontsize=11, fontweight="medium")
        left += frac
ax.set_yticks(y); ax.set_yticklabels([f"{l}\n(n={data[l]['n']})" for l in labels])
ax.set_xlabel("% of subjects")
ax.set_xlim(0, 100)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols["pass"], label="resolvable (≥50% of elevations)"),
                   Patch(color=cols["marg"], label="marginal (20–50%)"),
                   Patch(color=cols["fail"], label="fail (<20%)")],
          fontsize=9, loc="lower right")
ax.set_title("Corrected (Gaussian) N1/N2 resolvability: pilot vs external database\n"
             "gate = adjacent sep >0.5 oct AND saddle >4 dB AND N1 width >0.17 oct",
             fontsize=11)
fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
