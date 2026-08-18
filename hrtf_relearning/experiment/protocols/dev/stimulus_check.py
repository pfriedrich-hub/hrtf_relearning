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
for externalization. Cells 7-9 are the real check: run the full AR test on
YOURSELF with your OWN unmodified HRTF, noise vs ripple. If the envelope were
eating the cue, elevation gain would drop there -- with intact ears and an
intact HRTF there is nothing else that could cause it.

CHOOSING rms_tilt. It is the only knob (rms_cue is 0 by design). More of it
means more source variation, which is the point, but the band limit leaks, so
past some point it erodes the cue it was supposed to spare. Two steps:
  cell 6   acoustic screen -- sweep rms_tilt against BOTH of your ears and find
           the largest value that still leaves the cue >= 3:1 above the source
           in the 0.5-2 rip/oct band. Narrows the field; proves nothing about
           whether the result is still localizable.
  cells 11-13  DOME check -- real speakers, own ears, vertical midline, nothing
           else in the chain. Cheapest way to rule a depth out, and the one to
           run first. An upper bound: own ears are sharper than the modified
           HRTFs participants hear, so a pass here is necessary, not sufficient.
  cells 7-9  AR check -- the surviving candidates plus a noise reference, run on
           yourself through the actual presentation chain, scored by elevation
           gain. This is what decides it.
"""

SUBJECT_ID = "PF"
SOFA_NAME = "PF_donor_VD"      # the modified HRTF the subject is tested on
EAR = "left"                    # ear whose DTF carries the cue
AZIMUTH = -20.0                 # audition azimuth, experiment convention (neg = left)

RMS_TILT = 3.0               # dB rms of the envelope below 0.5 ripples/oct
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

for i in range(6):
    stim = make_gapped_pinknoise()
    stim.play()

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
KIND = 'noise'          # 'noise' | 'ripple' | 'uso'
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

# %% 6. sweep rms_tilt against BOTH ears --------------------------------------
# rms_tilt is the only knob (rms_cue stays 0 -- see 6b). Two ratios move in
# opposite directions as it rises, and the compromise is where they cross:
#
#   tilt band (0.2-0.5 rip/oct)  cue:source should be WELL BELOW 1 -- the source
#     spectrum must dominate here, that is what defeats a fixed template.
#   cue band  (0.5-2 rip/oct)    cue:source should stay ABOVE ~3 -- the notch
#     must still stand clear. This falls as rms_tilt rises even with rms_cue=0,
#     because a band-limited envelope cannot have a sharp edge at 0.5 rip/oct.
#     That leakage, not rms_cue, is the ceiling on the knob.
#
# Cue depth differs between ears, so the binding constraint is the WEAKER ear:
# pick the largest rms_tilt whose cue-band ratio is still >= TARGET_RATIO there.
# This is an acoustic screen only -- it narrows the field for the self-test in
# cells 7-9, it does not establish that the stimulus is still localizable.
TILT_GRID = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
TARGET_RATIO = 3.0              # minimum acceptable cue:source in the cue band
TILT_BAND = (0.2, 0.5)
N_TOKENS = 120

# tokens once per rms_tilt, scored against both ears -- the stimulus does not
# depend on the ear, only the cue it is measured against does
_sweep_spectra = {t: make_set('ripple', n=N_TOKENS, rms_tilt=t, rms_cue=0.0)[1]
                  for t in TILT_GRID}
sweep = {}
for _ear in ('left', 'right'):
    _cue, _, _ = cue_spectra(ear=_ear)
    _cue_band_rms, _cue_tilt_rms = ripple_rms(_cue, CUE_BAND), ripple_rms(_cue, TILT_BAND)
    sweep[_ear] = {
        'cue_band_rms': _cue_band_rms,
        'cue_pointwise': float(_cue.std(0).mean()),
        'cue_ratio': numpy.array([_cue_band_rms / ripple_rms(_sweep_spectra[t], CUE_BAND)
                                  for t in TILT_GRID]),
        'tilt_ratio': numpy.array([_cue_tilt_rms / ripple_rms(_sweep_spectra[t], TILT_BAND)
                                   for t in TILT_GRID])}
source_sd = numpy.array([_sweep_spectra[t].std(0).mean() for t in TILT_GRID])

print(f"{SUBJECT_ID} {SOFA_NAME}, az {AZIMUTH:+.0f}, {N_TOKENS} tokens per setting")
for _ear in sweep:
    print(f"  {_ear:5s} ear cue: {sweep[_ear]['cue_pointwise']:.2f} dB point-wise, "
          f"{sweep[_ear]['cue_band_rms']:.2f} dB in the cue band")
print(f"\n{'rms_tilt':>9s} {'source SD':>10s} "
      f"{'tilt L':>8s} {'tilt R':>8s} {'CUE L':>8s} {'CUE R':>8s}   verdict")
for i, t in enumerate(TILT_GRID):
    lo = min(sweep[e]['cue_ratio'][i] for e in sweep)
    print(f"{t:9.1f} {source_sd[i]:9.2f}dB "
          f"{sweep['left']['tilt_ratio'][i]:7.2f}:1 {sweep['right']['tilt_ratio'][i]:7.2f}:1 "
          f"{sweep['left']['cue_ratio'][i]:7.2f}:1 {sweep['right']['cue_ratio'][i]:7.2f}:1   "
          f"{'ok' if lo >= TARGET_RATIO else 'CUE ERODED'}")

_worst = numpy.minimum(sweep['left']['cue_ratio'], sweep['right']['cue_ratio'])
_ok = [t for t, r in zip(TILT_GRID, _worst) if r >= TARGET_RATIO]
CANDIDATES = ([_ok[-1]] if _ok else [TILT_GRID[0]])           # largest still safe
CANDIDATES += [t for t in TILT_GRID if t > CANDIDATES[0]][:1]  # one step past it
if _ok:
    print(f"\nlargest rms_tilt with cue:source >= {TARGET_RATIO:.0f}:1 in BOTH ears: "
          f"{CANDIDATES[0]:.0f} dB")
else:
    print(f"\n!! NO setting on the grid keeps cue:source >= {TARGET_RATIO:.0f}:1 in both "
          f"ears.\n   Either this subject's cue is unusually shallow (check the cue-band "
          f"rms above\n   against the ~3 dB typical), or the grid starts too high — extend "
          f"TILT_GRID\n   downwards before treating {CANDIDATES[0]:.0f} dB as a candidate.")
print(f"take to the self-test (cells 7-9): noise + ripple at "
      f"{' and '.join(f'{c:.0f}' for c in CANDIDATES)} dB\n"
      f"the second is deliberately past the acoustic criterion -- if it costs no\n"
      f"elevation gain either, the criterion is too conservative.")

# %% 6a. figure: the two ratios vs rms_tilt -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140, sharex=True)
for _ear, _style in (('left', '-o'), ('right', '--s')):
    axes[0].plot(TILT_GRID, sweep[_ear]['cue_ratio'], _style, color='#2e8b57',
                 label=f'{_ear} ear', ms=4)
    axes[1].plot(TILT_GRID, sweep[_ear]['tilt_ratio'], _style, color='#1b6ca8',
                 label=f'{_ear} ear', ms=4)
axes[0].axhline(TARGET_RATIO, color='k', ls=':', lw=1)
axes[0].axhspan(0, TARGET_RATIO, color='#c94f2e', alpha=.08)
axes[0].set_ylabel('cue : source'); axes[0].set_title(
    f'cue band {CUE_BAND[0]}-{CUE_BAND[1]} rip/oct — cue must stay above {TARGET_RATIO:.0f}:1')
axes[1].axhline(1.0, color='k', ls=':', lw=1)
axes[1].axhspan(1.0, 100, color='#c94f2e', alpha=.08)
axes[1].set_ylim(0, max(2.0, float(numpy.max([sweep[e]['tilt_ratio'] for e in sweep])) * 1.15))
axes[1].set_ylabel('cue : source'); axes[1].set_title(
    f'tilt band {TILT_BAND[0]}-{TILT_BAND[1]} rip/oct — source must dominate (< 1)')
for ax in axes:
    for c in CANDIDATES:
        ax.axvline(c, color='#e0a33e', lw=1.2, alpha=.7)
    ax.set_xlabel('rms_tilt (dB)'); ax.grid(alpha=.25); ax.legend(fontsize=7)
fig.suptitle(f'{SUBJECT_ID} — choosing rms_tilt (orange = self-test candidates)')
fig.tight_layout()
fig.savefig(paths.subject_acoustic_dir(SUBJECT_ID) / 'rms_tilt_sweep.png',
            bbox_inches='tight')
plt.show()

# %% 6b. diagnostic: what rms_cue > 0 costs -----------------------------------
# Not a setting to use -- this is the manipulation check. Deliberately push the
# envelope INTO the cue band and watch the ratio collapse. If a later behavioural
# effect tracks these rows, the cue band is doing the work.
cue, _, _ = cue_spectra()
cue_rms = ripple_rms(cue, CUE_BAND)
print(f"{'rms_tilt':>9s} {'rms_cue':>8s} {'point-wise SD':>14s} {'tilt-band':>10s} {'cue-band':>9s}")
for rms_tilt in (6.0, 8.0, 10.0, 12.0):
    for rms_cue in (0.0, 0.5, 1.2):
        spectra = make_set('ripple', n=80, rms_tilt=rms_tilt, rms_cue=rms_cue)[1]
        tilt_ratio = ripple_rms(cue, TILT_BAND) / ripple_rms(spectra, TILT_BAND)
        cue_ratio = cue_rms / ripple_rms(spectra, CUE_BAND)
        print(f"{rms_tilt:9.1f} {rms_cue:8.1f} {spectra.std(0).mean():14.2f} "
              f"{tilt_ratio:9.1f}:1 {cue_ratio:8.1f}:1")

# %% 7. SELF-TEST setup — config and helpers, runs nothing -------------------
# The decisive check, and the one to run before any participant sees this.
# Own UNMODIFIED HRTF, binaural, full field. With intact ears and an intact
# HRTF, a drop in elevation gain from noise to ripple can only mean the
# envelope is eating the cue. Expect a small cost at most.
#
# Order matters: run the two blocks in one sitting and swap BLOCK_ORDER on a
# second run, otherwise practice/fatigue is confounded with stimulus.
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    localization_accuracy)

SELF_ID = "PF"                        # own id — independent of the QC config above

# (stim, rms_tilt) per block. Fill the ripple rows from CANDIDATES in cell 6 --
# the noise block is the reference every ripple block is scored against, so it
# always runs. Keep it to 3 blocks: past that, fatigue costs more gain than the
# envelope does.
CONDITIONS = [('noise', None), ('ripple', 8.0), ('ripple', 12.0)]
REVERSE_ORDER = False                 # flip on the second sitting

# 'quick'  25 trials/block, ~3 min  — gross check only, EG is +/- ~0.15
# 'short'  50 trials/block, ~5 min  — EG +/- ~0.10
# 'full'  100 trials/block, ~10 min — EG +/- ~0.07, use before running anyone else
#
# The comparison is WITHIN a sitting, so block order is confounded with practice
# and fatigue. With 3 blocks a single order is not enough: run the whole set
# twice with REVERSE_ORDER flipped and average, or the middle condition is
# systematically flattered. 'short' x 3 x 2 sittings ~= 30 min of testing.
LENGTH = 'short'
_GRID = {'quick': ((14, 14), 1), 'short': ((7, 14), 1), 'full': ((7, 14), 2)}


def block_label(stim, rms_tilt=None):
    """Key for self_runs: 'noise', 'ripple@8', ..."""
    return stim if stim == 'noise' or rms_tilt is None else f"{stim}@{rms_tilt:g}"


def self_test_settings(stim, rms_tilt=None):
    sector_size, targets_per_sector = _GRID[LENGTH]
    hrir = {"name": SELF_ID, "subject_id": SELF_ID, "ear": None,
            "other_ear": "envelope", "env_n_keep": 4, "native_sofa": SELF_ID,
            "mirror": False, "reverb": True, "drr": 20,
            "hp_filter": True, "hp": "DT990",
            "convolution": "cpu", "storage": "cpu", "target_samplerate": 48000}
    loc = {"kind": "standard", "azimuth_range": (-1, 1),
           "elevation_range": (-35, 35), "targets_per_speaker": 3,
           "targets_per_sector": targets_per_sector, "min_distance": 10,
           "gain": 0.2, "stim": stim, "sector_size": sector_size,
           "replace": False, "exclude_midline": False, "midline_tol": 1.0,
           # goes to sequence.stim_settings, so the block records its own knob
           "stim_settings": {"rms_tilt": RMS_TILT if rms_tilt is None else rms_tilt,
                             "rms_cue": RMS_CUE}}
    return hrir, loc


def run_self_block(stim, rms_tilt=None):
    """One block. Returns the sequence and files it under its label."""
    label = block_label(stim, rms_tilt)
    print(f"\n{'=' * 60}\n  {label.upper()}  — own HRTF, binaural, full field"
          f"  [{LENGTH}]\n{'=' * 60}")
    subject = hr.Subject(SELF_ID)
    test = Localization(subject, *self_test_settings(stim, rms_tilt))
    test.run()
    self_runs[label] = subject.localization[test.filename]
    return self_runs[label]


self_runs = {}
_order = list(reversed(CONDITIONS)) if REVERSE_ORDER else list(CONDITIONS)
print(f"ready: {SELF_ID}, {LENGTH}, order "
      f"{[block_label(*c) for c in _order]} — run cell 8 to start")

# %% 8. RUN all blocks in order ------------------------------------------------
for _stim, _tilt in _order:
    run_self_block(_stim, _tilt)

# %% 8a. RUN one block at a time (alternative to cell 8) ---------------------
run_self_block('noise')

# %% 8b. ------------------------------------------------------------------------
run_self_block('ripple', 8.0)

# %% 8c. ------------------------------------------------------------------------
run_self_block('ripple', 12.0)

# %% 9. self-test result ------------------------------------------------------
# Read this against the noise block, not in absolute terms. EG within ~0.05-0.1
# of noise and no jump in elevation SD -> that rms_tilt is safe. A clear drop
# means the envelope is eating the cue at that setting: take the next one down.
# Note the resolution: at 'short' the EG error is ~0.10, so a 0.05 difference
# between two ripple settings is not a difference -- prefer the LARGER rms_tilt
# whenever the two are within noise of each other, since more source variation
# is the whole point.
print(f"{'block':12s} {'n':>4s} {'EG':>6s} {'RMSE_el':>8s} {'SD_el':>7s} "
      f"{'AG':>6s} {'RMSE_az':>8s}")
for _label, _seq in self_runs.items():
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(_seq)
    print(f"{_label:12s} {len(_seq.data):4d} {eg:6.2f} {ele_rmse:8.1f} {ele_sd:7.1f} "
          f"{(ag if ag is not None else float('nan')):6.2f} {az_rmse:8.1f}")

if 'noise' in self_runs and len(self_runs) > 1:
    _ref = localization_accuracy(self_runs['noise'])[0]
    print(f"\nEG cost of the random envelope (vs noise, EG {_ref:.2f}):")
    for _label, _seq in self_runs.items():
        if _label == 'noise':
            continue
        _eg = localization_accuracy(_seq)[0]
        print(f"  {_label:12s} {_ref - _eg:+.2f}  ({_eg / _ref * 100:3.0f}% of the noise gain)")
else:
    print("\nno noise reference in this set — run cell 8a before comparing")

# %% 10. reload earlier self-test blocks (skip the run cells) ----------------
# Pulls the most recent noise and ripple blocks with the unmodified HRTF back
# out of the subject file, so cell 9 can be re-run without testing again. Blocks
# are keyed by rms_tilt, so several ripple settings coexist; runs from before
# stim_settings was logged fall back to the module default.
_subject = hr.Subject(SELF_ID)
self_runs = {}
for _name, _seq in _subject.localization.items():
    if getattr(_seq, "hrir", None) != SELF_ID:      # unmodified HRTF only
        continue
    _stim = getattr(_seq, "stim", "noise")
    if _stim not in ("noise", "ripple"):
        continue
    _tilt = (getattr(_seq, "stim_settings", None) or {}).get('rms_tilt', RMS_TILT)
    self_runs[block_label(_stim, _tilt)] = _seq     # dict order -> last wins
print({k: v.name for k, v in self_runs.items()})

# %% 11. DOME check — real speakers, own ears, no processing -----------------
# The cheapest and cleanest localizability test, and the one to run BEFORE
# committing a value to the learning-transfer protocol. Free field, own ears,
# vertical midline: no headphone EQ, no binaural synthesis, no reverberation
# model, no tracked convolution. If elevation gain drops from noise to ripple
# here, it is the envelope, because nothing else in the chain differs.
#
# Read it as an UPPER BOUND. Own ears give deeper and sharper cues than the
# cepstrally smoothed, donor-modified transfer functions participants are
# tested on, so a setting that survives in the dome can still be too deep in
# AR. A failure here is decisive; a pass here still needs cells 7-9.
#
# 7 midline speakers, so a block is short: TARGETS_PER_SPEAKER=4 gives 28
# trials, ~3 min. Run the set twice with DOME_REVERSE flipped -- with three
# blocks in a sitting, order is otherwise confounded with practice.
from hrtf_relearning.experiment.localization.Localization_dome import (
    LocalizationDome, SAMPLERATE as DOME_FS)

# The stimulus helpers synthesise at slab's DEFAULT samplerate, and slab's own
# default is 8000 Hz. Cells 1 and 3 set it, but this cell must not depend on
# having run them: at 8 kHz the train plays back 6.1x too fast with no energy
# below ~500 Hz, which sounds like a broken stimulus rather than a wrong rate.
# LocalizationDome pins it too; this is belt and braces for the cells above.
slab.set_default_samplerate(DOME_FS)

DOME_ID = SELF_ID
DOME_CONDITIONS = [('noise', None), ('ripple', 3.0), ('ripple', 8.0)]
DOME_REVERSE = False
TARGETS_PER_SPEAKER = 4


def dome_settings(stim, rms_tilt=None):
    return {'targets_per_speaker': TARGETS_PER_SPEAKER, 'min_distance': 15,
            'stim': stim,
            'stim_settings': {'rms_tilt': RMS_TILT if rms_tilt is None else rms_tilt,
                              'rms_cue': RMS_CUE}}


def run_dome_block(stim, rms_tilt=None):
    label = block_label(stim, rms_tilt)
    print(f"\n{'=' * 60}\n  DOME {label.upper()}  — real speakers, own ears, "
          f"midline\n{'=' * 60}")
    subject = hr.Subject(DOME_ID)
    test = LocalizationDome(subject, dome_settings(stim, rms_tilt))
    test.run()
    dome_runs[label] = subject.localization[test.filename]
    return dome_runs[label]


dome_runs = {}
_dome_order = list(reversed(DOME_CONDITIONS)) if DOME_REVERSE else list(DOME_CONDITIONS)
print(f"ready: dome, {DOME_ID}, {TARGETS_PER_SPEAKER * 7} trials/block, order "
      f"{[block_label(*c) for c in _dome_order]} — run cell 12 to start")

# %% 12. RUN the dome blocks --------------------------------------------------
for _stim, _tilt in _dome_order:
    run_dome_block(_stim, _tilt)

# %% 12a. one dome block at a time --------------------------------------------
run_dome_block('noise')

# %% 12b. ----------------------------------------------------------------------
run_dome_block('ripple', 3.0)

# %% 12c. ----------------------------------------------------------------------
run_dome_block('ripple', 8.0)

# %% 13. dome result ----------------------------------------------------------
# Own ears in the free field should give EG near 1. A ripple block that holds
# within ~0.1 of the noise block is transparent to localization at that depth;
# a clear drop rules the depth out without spending an AR session on it.
print(f"{'block':12s} {'n':>4s} {'EG':>6s} {'RMSE_el':>8s} {'SD_el':>7s}")
for _label, _seq in dome_runs.items():
    eg, ele_rmse, ele_sd, *_ = localization_accuracy(_seq)
    print(f"{_label:12s} {len(_seq.data):4d} {eg:6.2f} {ele_rmse:8.1f} {ele_sd:7.1f}")
if 'noise' in dome_runs and len(dome_runs) > 1:
    _ref = localization_accuracy(dome_runs['noise'])[0]
    print(f"\nEG cost in the free field (vs noise, EG {_ref:.2f}):")
    for _label, _seq in dome_runs.items():
        if _label == 'noise':
            continue
        _eg = localization_accuracy(_seq)[0]
        print(f"  {_label:12s} {_ref - _eg:+.2f}  ({_eg / _ref * 100:3.0f}% of the noise gain)")

# %% 14. reload earlier dome blocks -------------------------------------------
# Dome runs from before the ripple option carry stim='pinknoise_burst'; they are
# noise blocks and are relabelled as such here.
_subject = hr.Subject(DOME_ID)
dome_runs = {}
for _name, _seq in _subject.localization.items():
    if getattr(_seq, "label", None) != 'dome':
        continue
    _stim = getattr(_seq, "stim", "noise")
    _stim = 'noise' if _stim == 'pinknoise_burst' else _stim
    _tilt = (getattr(_seq, "stim_settings", None) or {}).get('rms_tilt', RMS_TILT)
    dome_runs[block_label(_stim, _tilt)] = _seq
print({k: v.name for k, v in dome_runs.items()})
