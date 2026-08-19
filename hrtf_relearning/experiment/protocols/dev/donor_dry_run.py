"""
donor_dry_run.py — exercise the donor-detail pipeline on an existing participant.

Runs everything that does NOT need the rig, checking each step and printing a
PASS/FAIL report at the end. Nothing here plays sound, touches the head tracker
or needs a participant present; it writes one new SOFA
(<SUBJECT_ID>_donor_<DONOR>.sofa) and its QC figures, and optionally builds the
pyBinSim database.

Run cell by cell (# %%). Stop at the first FAIL and send the report back.

BEFORE RUNNING: add SUBJECT_ID to the 'subject' column of
learning_transfer/learning_transfer_block_order.csv (replace an '(assign)' cell),
the protocol's config cell raises.
"""

SUBJECT_ID = 'CO'          # any participant with a measured <id>.sofa
BUILD_BINSIM = True        # stage 4: write the pyBinSim database (no hardware,
                           # but needs the subject's DT990_equalization.npz)

# %% stage 0: imports ---------------------------------------------------------
import traceback

import numpy
import slab

from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.hrtf.modify.donor_detail import donor_detail_dtf, modification_params
from hrtf_relearning.hrtf.modify.edge_shift import (embed_modification_params,
                                                    read_modification_params)
from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
from hrtf_relearning.utils import paths

RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append((name, bool(condition), detail))
    print(f'  [{"PASS" if condition else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    return bool(condition)


def report():
    print('\n' + '=' * 70)
    failed = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'   ({detail})' if detail else ''))
    print('=' * 70)
    print(f'{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed'
          + ('' if not failed else f'  —  FAILED: {", ".join(n for n, _, _ in failed)}'))


sofa_dir = paths.SOFA_DIR / SUBJECT_ID
own_path = sofa_dir / f'{SUBJECT_ID}.sofa'
print(f'subject {SUBJECT_ID}   {own_path}')
check('native SOFA exists', own_path.exists(), str(own_path))
own = slab.HRTF(str(own_path))
own.name = SUBJECT_ID
print(f'  {own.n_sources} sources, {own[0].data.shape[0]} taps, {own.samplerate:.0f} Hz')

# %% stage 1: donor pool loads and conformance filtering works ----------------
candidates = selection.load_candidates(SUBJECT_ID)
check('donor pool non-empty', len(candidates) > 0, f'{len(candidates)} loaded: {", ".join(candidates)}')
check('subject excluded from own pool',
      SUBJECT_ID not in [k.split('/')[-1] for k in candidates])
check('all candidates conform',
      all(selection.conforms(h) for h in candidates.values()))
# a deliberately non-conforming file must be rejected
strict = selection.load_candidates(SUBJECT_ID, pool=None)
check('conformance filter rejects non-matching recordings',
      len(strict) >= len(candidates),
      f'{len(strict)} of all available recordings conform')

# %% stage 2: selection -------------------------------------------------------
chosen, rows = selection.select_donor(own, candidates)
reference, _ = selection.pairwise_r_match({SUBJECT_ID: own, **candidates})
selection.report(rows, reference)
check('a donor was selected', chosen is not None,
      f'{chosen["donor"]}  r_match {chosen["r_match"]:.2f}  '
      f'ridge {chosen["ridge_slope"]:+.2f}')
check('selection is not a fallback', not chosen['fallback'],
      'lowest-slope donor used — report this if it persists')
check('every candidate was scored', len(rows) == len(candidates),
      f'{len(rows)} rows')
check('chosen r_match inside the between-subject range',
      reference.min() <= chosen['r_match'] <= reference.max(),
      f'{chosen["r_match"]:.2f} in [{reference.min():.2f}, {reference.max():.2f}]')

# %% stage 3: build the composite and check the invariants --------------------
donor = candidates[chosen['donor']]
modified = donor_detail_dtf(own, donor, n_keep=selection.N_KEEP)
modified.name = f'{SUBJECT_ID}_donor_{chosen["donor"].split("/")[-1]}'

n = own.n_sources
check('source count preserved', modified.n_sources == n, f'{modified.n_sources}')


def itd_us(pair, fs):
    """GCC-PHAT interaural delay, positive = left leads."""
    size = 4 * pair.shape[0]
    left = numpy.fft.rfft(pair[:, 0], size)
    right = numpy.fft.rfft(pair[:, 1], size)
    cross = left * numpy.conj(right)
    cross /= numpy.maximum(numpy.abs(cross), 1e-20)
    correlation = numpy.fft.irfft(cross, size)
    correlation = numpy.concatenate((correlation[-64:], correlation[:65]))
    return -(int(numpy.argmax(correlation)) - 64) / fs * 1e6


def ild_db(hrtf, index):
    return 20 * numpy.log10(numpy.linalg.norm(hrtf[index].data[:, 0])
                            / numpy.linalg.norm(hrtf[index].data[:, 1]))


ild_change = max(abs(ild_db(modified, i) - ild_db(own, i)) for i in range(n))
itd_change = max(abs(itd_us(modified[i].data, own.samplerate)
                     - itd_us(own[i].data, own.samplerate))
                 for i in range(0, n, 7))
spectral_change = max(
    numpy.abs(numpy.abs(numpy.fft.rfft(modified[i].data[:, 0]))
              - numpy.abs(numpy.fft.rfft(own[i].data[:, 0]))).max()
    for i in range(0, n, 7))
check('broadband ILD preserved', ild_change < 1e-9, f'max {ild_change:.2e} dB')
check('ITD preserved', itd_change <= 1.1 * 1e6 / own.samplerate,
      f'max {itd_change:.1f} us (1 sample = {1e6 / own.samplerate:.1f} us)')
check('spectra actually changed', spectral_change > 1e-6, f'max |dH| {spectral_change:.3f}')

# %% stage 4: write, read back, and verify the round trip ---------------------
out_path = sofa_dir / f'{modified.name}.sofa'
modified.write_sofa(str(out_path))
check('SOFA written', out_path.exists(), str(out_path))

params = modification_params(
    SUBJECT_ID, chosen['donor'], n_keep=selection.N_KEEP,
    target_r_match=selection.TARGET_R_MATCH,
    band=selection.DEFAULT_BAND, resolution=selection.DEFAULT_RESOLUTION,
    max_ridge_slope=selection.MAX_RIDGE_SLOPE, pool=list(candidates),
    fallback=chosen['fallback'],
    scores={k: chosen[k] for k in ('r_match', 'ridge_slope', 'donor_strength')})
embed_modification_params(out_path, params)
recovered = read_modification_params(out_path)
check('modification params readable', recovered is not None)
check('donor id round-trips',
      (recovered or {}).get('donor_id') == chosen['donor'],
      f'{(recovered or {}).get("donor_id")}')

reloaded = slab.HRTF(str(out_path))
data_error = max(numpy.abs(reloaded[i].data - modified[i].data).max()
                 for i in range(0, n, 7))
check('reloaded data matches what was written', data_error < 1e-6,
      f'max {data_error:.2e}')
check('reloaded source count matches', reloaded.n_sources == n)

# %% stage 5: QC figures ------------------------------------------------------
plot_dir = paths.subject_acoustic_dir(SUBJECT_ID)
plot_dir.mkdir(parents=True, exist_ok=True)
try:
    fig = plot_ears(own, modified, vsi_dis=chosen['r_match'],
                    vsi_bw=selection.DEFAULT_BAND, band=selection.DEFAULT_BAND,
                    suptitle=f'{SUBJECT_ID}  + {chosen["donor"]} detail')
    fig.savefig(plot_dir / f'{modified.name}.png', bbox_inches='tight')
    check('plot_ears 2x2 figure', True, str(plot_dir / f'{modified.name}.png'))
except Exception as exc:
    traceback.print_exc()
    check('plot_ears 2x2 figure', False, repr(exc))

# %% stage 6: pyBinSim database builds under the new name ---------------------
# No hardware needed, but the subject's DT990_equalization.npz must exist.
if BUILD_BINSIM:
    from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
    for other_ear in ('flat', 'envelope', 'native'):
        try:
            hrir = hrtf2binsim({
                'name': modified.name, 'subject_id': SUBJECT_ID,
                'ear': 'left', 'other_ear': other_ear,
                'env_n_keep': selection.N_KEEP, 'native_sofa': SUBJECT_ID,
                'mirror': False, 'reverb': True, 'drr': 20,
                'hp_filter': True, 'hp': 'DT990',
                'convolution': 'cpu', 'storage': 'cpu'},
                overwrite=True, build=True)
            base = paths.BINSIM_DIR / hrir.name
            ok = ((base / f'{hrir.name}_filters.mat').exists()
                  and (base / f'filter_list_{hrir.name}.txt').exists()
                  and (base / f'{hrir.name}_test_settings.txt').exists())
            check(f'binsim database builds (other_ear={other_ear})', ok, hrir.name)
        except Exception as exc:
            traceback.print_exc()
            check(f'binsim database builds (other_ear={other_ear})', False, repr(exc))

# %% report -------------------------------------------------------------------
report()
