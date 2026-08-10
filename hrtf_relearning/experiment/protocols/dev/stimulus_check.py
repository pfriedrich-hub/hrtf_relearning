"""
stimulus_check.py

Audition and QC the localization stimuli. Run cell by cell (# %%).

WHY. Plain gapped pinknoise has an across-trial SD of 0.40 dB in its own
1/6-octave log spectrum, while the elevation cue in a measured DTF is ~3.3 dB
SD across elevation. The sound at the eardrum therefore carries a fixed,
one-to-one map from ABSOLUTE spectrum to elevation, and a listener can solve
the task by template matching without ever separating source from filter. So a
noise-only design cannot tell a spectral-to-spatial recalibration apart from a
learned timbre->elevation lookup -- which is exactly what FS reported doing.

The USO composites were meant to break that, but `generate_uso` used to pick
its base texture in a default argument (evaluated once at import), so every
token in a session shared one base: 0.80 dB, only twice the noise. Randomising
the base properly gives 2.98 dB -- but that is as large as the cue itself, and
buries it.

`make_rippled_pinknoise` puts the variation where it does not compete with the
cue: large below 0.5 ripples/oct (broad colouration, normalisable) and small in
the 0.5-2 ripples/oct band where the pinna notch lives. Cell 4 checks that
against YOUR subject's own DTF -- do this per subject, the cue depth varies.

Listening in cell 2/3 is dry (no headphone EQ, no reverb, no head tracking), so
it is for judging timbre variation and whether elevation is still audible, not
for externalization. Cells 7-8 are the real check: run the full AR test on
YOURSELF with your OWN unmodified HRTF, noise vs ripple. If the envelope were
eating the cue, elevation gain would drop there -- with intact ears and an
intact HRTF there is nothing else that could cause it.
"""

SUBJECT_ID = "FS"
SOFA_NAME = "FS_donor_AS"      # the modified HRTF the subject is tested on
EAR = "left"                    # ear whose DTF carries the cue
AZIMUTH = -20.0                 # audition azimuth, experiment convention (neg = left)

RMS_TILT = 8.0                  # dB rms of the envelope below 0.5 ripples/oct
RMS_CUE = 0.0                   # dB rms inside the cue band -- 0 by design, see cell 6

# %% imports and helpers ------------------------------------------------------
import numpy
import slab
from scipy.fftpack import dct
from matplotlib import pyplot as plt

from hrtf_relearning.experiment.localization.localization_helpers import stimulus as stim_mod
from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise, shape_from_coefficients)
from hrtf_relearning.experiment.localization.localization_helpers.uso_generation import (
    generate_uso, BASES)
from hrtf_relearning.utils import paths

OUT_DIR = paths.SOUNDS_DIR / "stimulus_check"
FLO, FHI, NPTS = 3500.0, 16000.0, 192
LOGF = numpy.logspace(numpy.log10(FLO), numpy.log10(FHI), NPTS)
NOCT = numpy.log2(FHI / FLO)
BW = 2 ** (1 / 12)              # +/- 1/12 oct -> 1/6 octave window (~1 ERB at 8 kHz)
DENSITY = numpy.arange(NPTS) / (2 * NOCT)
CUE_BAND = (0.5, 2.0)


def log_spectrum(x, samplerate):
    """1/6-octave-smoothed log power spectrum on LOGF, mean removed (shape only)."""
    x = numpy.asarray(x, dtype=float).squeeze()
    power = numpy.abs(numpy.fft.rfft(x)) ** 2
    freqs = numpy.fft.rfftfreq(len(x), 1 / samplerate)
    cumulative = numpy.concatenate([[0], numpy.cumsum(power)])
    lo = numpy.searchsorted(freqs, LOGF / BW)
    hi = numpy.searchsorted(freqs, LOGF * BW)
    band = (cumulative[hi] - cumulative[lo]) / numpy.maximum(hi - lo, 1)
    spectrum = 10 * numpy.log10(band + 1e-20)
    return spectrum - spectrum.mean()


def ripple_rms(spectra, band):
    """rms per DCT coefficient of a set of log spectra, within a ripple band."""
    coeffs = dct(numpy.asarray(spectra), type=2, norm='ortho', axis=1).std(0)
    sel = (DENSITY >= band[0]) & (DENSITY < band[1])
    return float(numpy.sqrt((coeffs[sel] ** 2).mean()))


def cue_spectra(sofa_name=None, subject_id=None, ear=None, azimuth=None):
    """Log spectra of one DTF azimuth column across elevation -- the cue itself."""
    sofa_name = SOFA_NAME if sofa_name is None else sofa_name
    subject_id = SUBJECT_ID if subject_id is None else subject_id
    ear = EAR if ear is None else ear
    azimuth = AZIMUTH if azimuth is None else azimuth
    hrtf = slab.HRTF(str(paths.SOFA_DIR / subject_id / f"{sofa_name}.sofa"))
    sources = hrtf.sources.vertical_polar
    # the experiment negates azimuth relative to the SOFA convention
    sofa_az = numpy.mod(-azimuth, 360)
    az_grid = sources[:, 0]
    nearest = az_grid[numpy.argmin(numpy.abs(((az_grid - sofa_az + 180) % 360) - 180))]
    sel = (numpy.abs(((az_grid - nearest + 180) % 360) - 180) < 1) & (numpy.abs(sources[:, 1]) <= 35)
    channel = 0 if ear == 'left' else 1
    return numpy.array([log_spectrum(hrtf[i].data[:, channel], hrtf.samplerate)
                        for i in numpy.flatnonzero(sel)]), hrtf, numpy.flatnonzero(sel)


def make_set(kind, n=120, **kwargs):
    """n tokens of one stimulus kind -> (list of slab.Sound, array of log spectra)."""
    sounds = []
    for _ in range(n):
        if kind == 'noise':
            sounds.append(make_gapped_pinknoise())
        elif kind == 'ripple':
            sounds.append(make_rippled_pinknoise(
                rms_tilt=kwargs.get('rms_tilt', RMS_TILT),
                rms_cue=kwargs.get('rms_cue', RMS_CUE))[0])
        elif kind == 'uso':
            sounds.append(generate_uso(samplerate=slab.get_default_samplerate(),
                                       base=kwargs.get('base', None)))
        else:
            raise ValueError(kind)
    spectra = numpy.array([log_spectrum(s.data[:, 0], s.samplerate) for s in sounds])
    return sounds, spectra


# %% 1. quick listen: dry tokens, one after another ---------------------------
# Ripple tokens should sound plainly different from each other in colour while
# staying the same burst train. If they sound identical, rms_tilt is too low;
# if any sounds hollow or whistly, it is too high.
slab.set_default_samplerate(48828)
for i in range(6):
    token, params = make_rippled_pinknoise(rms_tilt=RMS_TILT, rms_cue=RMS_CUE)
    print(f"token {i}  shape rms {numpy.std(shape_from_coefficients(params['coeffs'])[1]):.1f} dB")
    token.play()

# %% 2. write dry tokens to disk for listening in your own player -------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for i in range(10):
    make_rippled_pinknoise(rms_tilt=RMS_TILT, rms_cue=RMS_CUE)[0].write(
        OUT_DIR / f"ripple_{i:02d}.wav")
    generate_uso(samplerate=slab.get_default_samplerate()).write(OUT_DIR / f"uso_{i:02d}.wav")
make_gapped_pinknoise().write(OUT_DIR / "noise.wav")
print(f"wrote to {OUT_DIR}")

# %% 3. THE listening test: is elevation still audible? -----------------------
# Same token family convolved with the subject's own modified DTF at a spread of
# elevations. Dry (no HP EQ, no reverb, no tracking) -- judge elevation, not
# externalization. Do it for 'noise' first to hear what the cue sounds like when
# the source spectrum is fixed, then for 'ripple': the elevation percept should
# survive, the timbre should not.
KIND = 'ripple'          # 'noise' | 'ripple' | 'uso'
ELEVATIONS = (-30, -15, 0, 15, 30)

_, hrtf, idx = cue_spectra()
sources = hrtf.sources.vertical_polar
slab.set_default_samplerate(hrtf.samplerate)
for elevation in ELEVATIONS:
    source = idx[numpy.argmin(numpy.abs(sources[idx, 1] - elevation))]
    token = (make_rippled_pinknoise(rms_tilt=RMS_TILT, rms_cue=RMS_CUE)[0] if KIND == 'ripple'
             else make_gapped_pinknoise() if KIND == 'noise'
             else generate_uso(samplerate=hrtf.samplerate))
    spatial = hrtf.apply(source, token)
    spatial.level = 70
    print(f"elevation {sources[source, 1]:+.1f}")
    spatial.play()
    spatial.write(OUT_DIR / f"{KIND}_el{int(sources[source, 1]):+03d}.wav")

# %% 4. does the cue stand above the source variation? ------------------------
# The number that matters is the ratio in the 0.5-2 ripples/oct band. Aim for
# ~3:1. Plain noise sits near 9:1 (no variation at all -> template matching is
# sufficient); fully randomised USOs sit near 1:1 (cue buried).
cue, _, _ = cue_spectra()
sets = {'noise': make_set('noise')[1],
        'ripple': make_set('ripple', rms_tilt=RMS_TILT, rms_cue=RMS_CUE)[1],
        'uso (base fixed)': make_set('uso', base=BASES[2])[1],
        'uso (base random)': make_set('uso')[1]}
cue_rms = ripple_rms(cue, CUE_BAND)
print(f"{SUBJECT_ID} {SOFA_NAME} {EAR} ear, az {AZIMUTH:+.0f}: "
      f"cue = {cue.std(0).mean():.2f} dB point-wise, {cue_rms:.2f} dB in the cue band\n")
print(f"{'stimulus':20s} {'point-wise SD':>13s} {'cue-band rms':>13s} {'cue:source':>11s}")
for name, spectra in sets.items():
    band = ripple_rms(spectra, CUE_BAND)
    print(f"{name:20s} {spectra.std(0).mean():13.2f} {band:13.2f} {cue_rms / band:10.1f}:1")

# %% 5. figure: where the variation sits, and how it compares to the cue ------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
colours = {'noise': '#1b6ca8', 'ripple': '#2e8b57',
           'uso (base fixed)': '#e0a33e', 'uso (base random)': '#c94f2e'}
axes[0].semilogx(LOGF, cue.std(0), 'k', lw=2, label='DTF elevation cue')
for name, spectra in sets.items():
    axes[0].semilogx(LOGF, spectra.std(0), color=colours[name], lw=1.6, label=name)
axes[0].set_xlabel('frequency (Hz)'); axes[0].set_ylabel('SD of 1/6-oct log spectrum (dB)')
axes[0].set_title('where the variation is'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.25, which='both')

coeff = lambda s: dct(numpy.asarray(s), type=2, norm='ortho', axis=1).std(0)
axes[1].plot(DENSITY[1:], coeff(cue)[1:], 'k', lw=2, label='DTF elevation cue')
for name, spectra in sets.items():
    axes[1].plot(DENSITY[1:], coeff(spectra)[1:], color=colours[name], lw=1.6, label=name)
axes[1].axvspan(*CUE_BAND, color='green', alpha=.08)
axes[1].set_xlim(0, 4.5); axes[1].set_xlabel('ripple density (ripples/octave)')
axes[1].set_ylabel('rms per DCT coefficient (dB)')
axes[1].set_title('ripple-density decomposition'); axes[1].legend(fontsize=7); axes[1].grid(alpha=.25)
fig.suptitle(f'{SUBJECT_ID} {SOFA_NAME} — stimulus spectral variation vs the elevation cue')
fig.tight_layout()
fig.savefig(paths.subject_acoustic_dir(SUBJECT_ID) / 'stimulus_spectral_variation.png',
            bbox_inches='tight')
plt.show()

# %% 6. tune rms_tilt for this subject ----------------------------------------
# rms_tilt buys audible timbre variation almost for free; rms_cue trades
# directly against the cue and is 0 in the experiment. Pick the rms_tilt whose
# cue-band ratio is nearest 3:1. The rms_cue > 0 rows are the diagnostic: they
# show how fast the cue goes once the envelope reaches into its band.
cue, _, _ = cue_spectra()
cue_rms = ripple_rms(cue, CUE_BAND)
print(f"{'rms_tilt':>9s} {'rms_cue':>8s} {'point-wise SD':>14s} {'tilt-band':>10s} {'cue-band':>9s}")
for rms_tilt in (6.0, 8.0, 10.0, 12.0):
    for rms_cue in (0.0, 0.5, 1.2):
        spectra = make_set('ripple', n=80, rms_tilt=rms_tilt, rms_cue=rms_cue)[1]
        tilt_ratio = ripple_rms(cue, (0.2, 0.5)) / ripple_rms(spectra, (0.2, 0.5))
        cue_ratio = cue_rms / ripple_rms(spectra, CUE_BAND)
        print(f"{rms_tilt:9.1f} {rms_cue:8.1f} {spectra.std(0).mean():14.2f} "
              f"{tilt_ratio:9.1f}:1 {cue_ratio:8.1f}:1")

# %% 7. SELF-TEST: full AR localization block, noise vs ripple ----------------
# The decisive check, and the one to run before any participant sees this.
# Own UNMODIFIED HRTF, binaural, full field. With intact ears and an intact
# HRTF, a drop in elevation gain from noise to ripple can only mean the
# envelope is eating the cue. Expect a small cost at most.
#
# Order matters: run the two blocks in one sitting and swap the order on a
# second run, otherwise practice/fatigue is confounded with stimulus.
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy)

SELF_ID = "PF"                        # own id — independent of the QC config above
BLOCK_ORDER = ('noise', 'ripple')     # swap on the second run
TARGETS_PER_SECTOR = 2                # 2 -> ~100 trials/block ~10 min; 1 -> ~50, ~5 min


def self_test_settings(stim):
    hrir = {"name": SELF_ID, "subject_id": SELF_ID, "ear": None,
            "other_ear": "envelope", "env_n_keep": 4, "native_sofa": SELF_ID,
            "mirror": False, "reverb": True, "drr": 20,
            "hp_filter": True, "hp": "DT990",
            "convolution": "cpu", "storage": "cpu"}
    loc = {"kind": "sectors", "azimuth_range": (-35, 35),
           "elevation_range": (-35, 35), "targets_per_speaker": 3,
           "targets_per_sector": TARGETS_PER_SECTOR, "min_distance": 20,
           "gain": 0.2, "stim": stim, "sector_size": (7, 14),
           "replace": False, "exclude_midline": False, "midline_tol": 1.0,
           "stim_settings": {"rms_tilt": RMS_TILT, "rms_cue": RMS_CUE}}
    return hrir, loc


self_runs = {}
for _stim in BLOCK_ORDER:
    print(f"\n{'=' * 60}\n  {_stim.upper()}  — own HRTF, binaural, full field\n{'=' * 60}")
    _subject = hr.Subject(SELF_ID)
    _test = Localization(_subject, *self_test_settings(_stim))
    _test.run()
    self_runs[_stim] = _subject.localization[_test.filename]

# %% 8. self-test result ------------------------------------------------------
# EG within ~0.05-0.1 of the noise block and no jump in SD -> the stimulus is
# safe to use. A large drop means rms_tilt is too high for your own cue depth
# (re-run cell 6) or the envelope is leaking into the cue band.
print(f"{'block':10s} {'n':>4s} {'EG':>6s} {'RMSE_el':>8s} {'SD_el':>7s} {'AG':>6s} {'RMSE_az':>8s}")
for _stim, _seq in self_runs.items():
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(_seq)
    print(f"{_stim:10s} {len(_seq.data):4d} {eg:6.2f} {ele_rmse:8.1f} {ele_sd:7.1f} "
          f"{(ag if ag is not None else float('nan')):6.2f} {az_rmse:8.1f}")
if len(self_runs) == 2:
    _n, _r = (localization_accuracy(self_runs[k])[0] for k in ('noise', 'ripple'))
    print(f"\nEG cost of the random envelope: {_n - _r:+.2f} "
          f"({_r / _n * 100:.0f}% of the noise gain)")

# %% 9. reload earlier self-test blocks (skip 7 if already run) ---------------
# Pulls the most recent noise and ripple blocks with the unmodified HRTF back
# out of the subject file, so cell 8 can be re-run without testing again.
_subject = hr.Subject(SELF_ID)
self_runs = {}
for _name, _seq in _subject.localization.items():
    if getattr(_seq, "hrir", None) != SELF_ID:      # unmodified HRTF only
        continue
    _stim = getattr(_seq, "stim", "noise")
    if _stim in ("noise", "ripple"):
        self_runs[_stim] = _seq                     # dict order -> last wins
print({k: v.name for k, v in self_runs.items()})
