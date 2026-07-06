"""
gammatone_notch_detect.py  (dev/analysis prototype)

Prototype of gammatone/ERB-domain notch + rising-edge detection for
edge_shift.py, per the resolution argument in
experiment/protocols/documentation/spectral_resolution_literature.md:
detect the notch as ONE coarse feature (Hebrank/Asano robust cue) and locate
its rising-edge frequency, rather than resolving sub-1/2-octave ripple (the
disputed Hebrank 1/4-octave peak, which n_keep=60 over-segments).

Detection lives in the same ~1-ERB gammatone RMS the Baumgartner front end /
verify_binned reads. Rising-edge frequency = gradient-weighted centroid of the
positive-slope flank above the notch minimum (same ec_edge logic as
verify_binned), giving sub-band precision from a coarse-identity representation.

Two test figures on JS median plane (left ear):
  1. spectra at several elevations, gammatone curve + raw, notch min + rising edge marked
  2. detected notch CF and rising-edge freq vs elevation, gammatone (stable)
     vs cepstral n_keep=60 (over-segmented)

Run: python3 hrtf_relearning/hrtf/processing/dev/gammatone_notch_detect.py
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[4]
ES = ROOT / "hrtf_relearning/hrtf/processing/edge_shift.py"
SOFA = ROOT / "hrtf_relearning/data/hrtf/sofa/JS/JS.sofa"
OUT_DIR = Path("/sessions/friendly-exciting-planck/mnt/outputs")

spec = iu.spec_from_file_location("es", ES)
es = iu.module_from_spec(spec); spec.loader.exec_module(es)

F_LO, F_HI = 3000.0, 17500.0
SEARCH = (5000.0, 12000.0)     # main elevation-notch region
PROM = 3.0
EAR = 0

with h5py.File(SOFA, "r") as f:
    ir_all = f["Data.IR"][:]
    sp = f["SourcePosition"][:]
    fs = float(np.array(f["Data.SamplingRate"]).ravel()[0])
azn = (sp[:, 0] + 180) % 360 - 180
el = sp[:, 1]
med = med0 = np.where(np.abs(azn) <= 2)[0]
med = med[np.argsort(el[med])]


def spectrum(ir):
    n = len(ir)
    nfft = int(2 ** np.ceil(np.log2(n)) * 4)
    mag = np.abs(np.fft.rfft(ir, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, mag


def gammatone(ir, flow=2500.0, fhigh=18000.0):
    freqs, mag = spectrum(ir)
    fc, ec, Lb = es.erb_filterbank_rms(freqs, mag, flow=flow, fhigh=fhigh, spacing=1.0)
    return fc, ec, Lb


def detect_gamma(ir):
    """Return dict(notch_hz, edge_hz, fc, ec, Lb, im, jsad) for the main notch."""
    fc, ec, Lb = gammatone(ir)
    win = (fc >= SEARCH[0]) & (fc <= SEARCH[1])
    mins, props = find_peaks(-Lb, prominence=PROM)
    mins_in = [m for m in mins if win[m]]
    if mins_in:
        im = min(mins_in, key=lambda m: Lb[m])          # deepest prominent notch in window
    else:
        idxs = np.where(win)[0]
        im = idxs[int(np.argmin(Lb[idxs]))]             # fallback: plain minimum
    j = im
    while j + 1 < len(Lb) and Lb[j + 1] >= Lb[j]:
        j += 1                                          # climb to saddle above the notch
    if j > im:
        g = np.maximum(np.gradient(Lb, ec), 0.0)[im:j + 1]
        edge_ec = np.sum(ec[im:j + 1] * g) / (g.sum() + 1e-9) if g.sum() > 0 else ec[im]
    else:
        edge_ec = ec[im]
    return dict(notch_hz=float(fc[im]), edge_hz=float(es.erb_to_hz(edge_ec)),
                fc=fc, ec=ec, Lb=Lb, im=im, jsad=j)


def cepstral_notches(ir, n_keep=60):
    freqs, mag = spectrum(ir)
    Lc = es._cepstral_smooth_db(mag, n_keep)
    band = (freqs >= SEARCH[0]) & (freqs <= SEARCH[1])
    fb, Lcb = freqs[band], Lc[band]
    idx, _ = find_peaks(-Lcb, prominence=PROM)
    return fb[idx]


def raw_logmag(ir):
    freqs, mag = spectrum(ir)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    return freqs[band], 20 * np.log10(np.maximum(mag[band], 1e-9))


# ============ FIGURE 1: spectra at several elevations ============
show_els = [-8.3, 0.0, 8.3, 16.7, 25.0]
dirs = [med[np.argmin(np.abs(el[med] - e))] for e in show_els]
fig1, axes = plt.subplots(1, len(dirs), figsize=(19, 4.2), sharey=True)
for ax, d in zip(axes, dirs):
    ir = ir_all[d, EAR, :]
    fr, Lr = raw_logmag(ir)
    ax.plot(fr / 1000, Lr, color="0.72", lw=0.9, label="raw")
    det = detect_gamma(ir)
    ax.plot(det["fc"] / 1000, det["Lb"], color="#185FA5", lw=2.2, label="gammatone ~1 ERB")
    ax.plot(det["notch_hz"] / 1000, det["Lb"][det["im"]], "v", color="#D85A30",
            ms=12, mec="k", mew=0.6, zorder=6, label="notch min")
    ax.axvline(det["edge_hz"] / 1000, color="#1D9E75", lw=2.0, ls="--", label="rising edge")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 8, 12, 17]); ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(F_LO / 1000, F_HI / 1000)
    ax.set_title(f"el = {el[d]:.0f}°   notch {det['notch_hz']/1000:.1f} kHz\nedge {det['edge_hz']/1000:.1f} kHz",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("kHz")
axes[0].set_ylabel("magnitude (dB)")
axes[0].legend(fontsize=8, loc="lower left")
fig1.suptitle("JS left DTF — gammatone-domain notch + rising-edge detection across elevation", fontsize=13)
fig1.tight_layout()
p1 = OUT_DIR / "JS_gammatone_detect_spectra.png"
fig1.savefig(p1, dpi=140, bbox_inches="tight")
print("saved", p1)

# ============ FIGURE 2: cue vs elevation, gammatone vs cepstral-60 ============
els = el[med]
notch_hz = np.array([detect_gamma(ir_all[d, EAR, :])["notch_hz"] for d in med])
edge_hz = np.array([detect_gamma(ir_all[d, EAR, :])["edge_hz"] for d in med])

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))

axL.plot(els, notch_hz / 1000, "-o", color="#185FA5", lw=2, label="notch minimum CF")
axL.plot(els, edge_hz / 1000, "--s", color="#1D9E75", lw=2, label="rising-edge frequency")
axL.set_xlabel("elevation (deg)"); axL.set_ylabel("frequency (kHz)")
axL.set_title("gammatone detector: single cue that climbs with elevation")
axL.grid(True, alpha=0.3); axL.legend(fontsize=9)

for d in med:
    cf = cepstral_notches(ir_all[d, EAR, :], 60)
    axR.plot([el[d]] * len(cf), cf / 1000, "o", color="#D85A30", ms=6, alpha=0.75)
axR.plot(els, notch_hz / 1000, "-o", color="#185FA5", lw=2, ms=4, label="gammatone notch (1 per el)")
axR.set_xlabel("elevation (deg)"); axR.set_ylabel("detected notch CF (kHz)")
axR.set_title("cepstral n_keep=60 (orange): multiple jittery notches per elevation")
axR.grid(True, alpha=0.3); axR.legend(fontsize=9)
axR.set_ylim(SEARCH[0] / 1000, SEARCH[1] / 1000)

fig2.suptitle("Detected elevation cue vs elevation — gammatone recovers one monotonic edge; "
              "n_keep=60 over-segments", fontsize=13)
fig2.tight_layout()
p2 = OUT_DIR / "JS_gammatone_cue_vs_elevation.png"
fig2.savefig(p2, dpi=140, bbox_inches="tight")
print("saved", p2)

# ---- console: monotonicity check ----
d_edge = np.diff(edge_hz)
print(f"\nrising-edge freq (kHz) by elevation:")
for e, nh, eh in zip(els, notch_hz, edge_hz):
    print(f"  el={e:6.1f}:  notch {nh/1000:5.2f}   edge {eh/1000:5.2f}")
print(f"\nrising-edge monotonic-up fraction: {(d_edge > 0).mean():.2f}  "
      f"(spearman-ish); notch CF range {notch_hz.min()/1000:.1f}-{notch_hz.max()/1000:.1f} kHz")
