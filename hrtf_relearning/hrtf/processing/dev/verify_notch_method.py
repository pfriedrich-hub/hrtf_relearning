"""
verify_notch_method.py  (dev/analysis)

Verify notch center frequencies are evaluated correctly, and show why the
cepstral n_keep smoothing is the wrong tool for LOCATING minima: it reshapes the
spectrum and moves/merges the minima. Iida et al. (2007) instead use a light
Gaussian convolution (their Eq. 2-3: window ±4 bins, sigma=1.3 bins on a
512-pt/48-kHz grid ~ sigma ~ 122 Hz) that de-ripples without moving the extrema.

Overlays, per subject/elevation:
  raw log-mag (grey) | Iida Gaussian ~122 Hz (blue, ▼ minima) | cepstral n_keep=30 (orange, ▼ minima)
on JS, CA (our data) and two Aachen MRT subjects (external database, cartesian coords).

Run: python3 hrtf_relearning/hrtf/processing/dev/verify_notch_method.py
"""
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa"
AACHEN = Path("/sessions/friendly-exciting-planck/mnt/aachen_database")
OUT = Path("/sessions/friendly-exciting-planck/mnt/outputs/verify_notch_method.png")
spec = iu.spec_from_file_location("es", ES); es = iu.module_from_spec(spec); spec.loader.exec_module(es)

SIGMA_HZ = 122.0     # Iida Eq 2-3
NK = 30
BAND = (5000., 13500.)
F_LO, F_HI = 3000., 17500.


def load_median(sofa, aachen=False):
    with h5py.File(sofa, "r") as f:
        ir = f["Data.IR"][:]; sp = f["SourcePosition"][:]; fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
    if aachen:  # cartesian x,y,z (m) -> spherical
        x, y, z = sp[:, 0], sp[:, 1], sp[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x)); el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1, 1)))
        med = np.where((np.abs(az) <= 3) & (x > 0))[0]
    else:
        azn = (sp[:, 0] + 180) % 360 - 180; el = sp[:, 1]
        med = np.where(np.abs(azn) <= 2)[0]
    med = med[np.argsort(el[med])]
    return ir, fs, el, med


def curves(x, fs):
    n = len(x); nfft = int(2**np.ceil(np.log2(n))*4)
    fr = np.fft.rfftfreq(nfft, 1/fs); mag = np.abs(np.fft.rfft(x, nfft))
    L = 20*np.log10(np.maximum(mag, 1e-9))
    Lg = gaussian_filter1d(L, SIGMA_HZ/(fr[1]-fr[0]), mode="nearest")
    Lc = es._cepstral_smooth_db(mag, NK)
    return fr, L, Lg, Lc


def notch_cf(fr, L):
    b = (fr >= BAND[0]) & (fr <= BAND[1]); idx, _ = find_peaks(-L[b], prominence=3.0)
    return fr[b][idx], L[b][idx]


subjects = [("JS", SOFA/"JS/JS.sofa", False), ("CA", SOFA/"CA/CA.sofa", False),
            ("MRT01 (Aachen)", AACHEN/"MRT01.sofa", True), ("MRT08 (Aachen)", AACHEN/"MRT08.sofa", True)]
els_want = [-16.0, 16.0]

fig, axes = plt.subplots(len(subjects), len(els_want), figsize=(13, 14), sharex=True)
for r, (name, path, aach) in enumerate(subjects):
    ir, fs, el, med = load_median(path, aachen=aach)
    ear = 0
    for c, ew in enumerate(els_want):
        ax = axes[r, c]
        d = med[np.argmin(np.abs(el[med] - ew))]
        fr, L, Lg, Lc = curves(ir[d, ear, :], fs)
        m = (fr >= F_LO) & (fr <= F_HI)
        ax.plot(fr[m]/1000, L[m], color="0.75", lw=0.8, label="raw")
        ax.plot(fr[m]/1000, Lg[m], color="#185FA5", lw=2.0, label="Iida Gaussian 122 Hz")
        ax.plot(fr[m]/1000, Lc[m], color="#D85A30", lw=1.6, ls="--", label="cepstral n_keep=30")
        ngf, ngl = notch_cf(fr, Lg); ncf, ncl = notch_cf(fr, Lc)
        ax.plot(ngf/1000, ngl, "v", color="#185FA5", ms=11, mec="k", mew=0.5, zorder=6)
        ax.plot(ncf/1000, ncl, "v", color="#D85A30", ms=8, mec="k", mew=0.4, zorder=6)
        ax.set_xscale("log"); ax.set_xticks([3, 5, 8, 12, 17])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(F_LO/1000, F_HI/1000); ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"{name}, el={el[d]:.0f}°  | Gaussian {np.round(ngf/1000,1)} vs ceps {np.round(ncf/1000,1)} kHz", fontsize=9)
        if r == len(subjects)-1: ax.set_xlabel("frequency (kHz)")
        if c == 0: ax.set_ylabel("mag (dB)")
        if r == 0 and c == 0: ax.legend(fontsize=7.5, loc="lower left")

fig.suptitle("Notch evaluation: Iida's light Gaussian (▼ blue) sits on the true minima; "
             "cepstral n_keep (▼ orange) merges & shifts them", fontsize=13, y=0.997)
fig.tight_layout()
fig.savefig(OUT, dpi=135, bbox_inches="tight")
print("saved", OUT)
