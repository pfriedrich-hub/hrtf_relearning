"""
backfill_comparison.py — write the before/after 2x2 for every modified SOFA.

WHY. :func:`...plot_compare.plot_ears` is saved by the current donor protocol,
but only there. Every earlier manipulation wrote its own figure somewhere else
or not at all: the deprecated erbshift protocol put a waterfall in
``plots/`` rather than ``plots/acoustic/``, the notch pilots wrote nothing, and
subjects recorded before the ``plots/acoustic/`` convention have no such folder.
The result is that most modified SOFAs on disk have no picture of what the
manipulation did to them, which is the same traceability hole that made FD
12:13 undiscardable-but-unusable.

This walks the SOFA tree, pairs each modified file with the native one beside
it, and writes ``results/<id>/plots/acoustic/<modified>.png``. Re-runnable and
idempotent; skips a pair whose figure already exists unless ``overwrite=True``.

    python -m hrtf_relearning.hrtf.modify.backfill_comparison            # dry run
    python -m hrtf_relearning.hrtf.modify.backfill_comparison --write
    python -m hrtf_relearning.hrtf.modify.backfill_comparison --write --only GS

NOTE ON IDS. A SOFA folder name is not always the results folder name: pilot
participants live under ``results/pilot/<id>`` while their SOFAs may sit at
either ``sofa/<id>`` or ``sofa/pilot/<id>``. Where an id exists in both places
those are DIFFERENT recordings sharing one results folder (see
experiment/analysis/subject/demerge_subject.py), so the figure name is prefixed
to keep them apart rather than silently overwriting.
"""

import argparse
import logging
import traceback

import matplotlib
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
import matplotlib.pyplot as plt
import slab

from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
from hrtf_relearning.utils import paths

logger = logging.getLogger(__name__)

#: Band drawn and measured on the figure. The donor-selection band, used for
#: every manipulation so the numbers on two figures mean the same thing.
DEFAULT_BAND = (5657.0, 11314.0)


def sofa_pairs(root=None, in_pilot_tree=False):
    """(results_id, native, [modified], from_pilot_tree) per subject folder.

    A SOFA under ``sofa/pilot/<id>`` ALWAYS belongs to ``results/pilot/<id>``,
    even when an active ``results/<id>`` exists. Resolving by "use results/<id>
    if it exists" routes pilot PF's files into the active PF's folder — the two
    are different recordings that happen to share initials, which is what
    experiment/analysis/subject/demerge_subject.py exists to undo.
    """
    root = paths.SOFA_DIR if root is None else root
    out = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == 'database':
            continue
        if directory.name == 'pilot' and not in_pilot_tree:
            out.extend(sofa_pairs(directory, in_pilot_tree=True))
            continue
        sofas = sorted(p for p in directory.glob('*.sofa'))
        native = next((p for p in sofas if p.stem == directory.name), None)
        if native is None:
            logger.debug('%s: no native <id>.sofa, skipping', directory.name)
            continue
        modified = [p for p in sofas if p != native]
        if not modified:
            continue
        if in_pilot_tree:
            results_id = f'pilot/{directory.name}'
        elif (paths.RESULTS_DIR / directory.name).exists():
            results_id = directory.name
        else:
            results_id = f'pilot/{directory.name}'
        out.append((results_id, native, modified, in_pilot_tree))
    return out


def backfill(overwrite=False, only=None, band=DEFAULT_BAND, write=False):
    """Write the 2x2 for every (native, modified) pair. Returns a report list."""
    report = []
    for results_id, native_path, modified_paths, from_pilot_tree in sofa_pairs():
        short = results_id.split('/')[-1]
        if only and short != only and results_id != only:
            continue
        # Both sofa/<id> and sofa/pilot/<id> can land in results/pilot/<id>.
        # Prefix the pilot-tree one so two different recordings cannot
        # overwrite each other's figure.
        collides = ((paths.SOFA_DIR / short).exists()
                    and (paths.SOFA_DIR / 'pilot' / short).exists())
        prefix = 'pilot_' if (from_pilot_tree and collides) else ''
        out_dir = paths.subject_acoustic_dir(results_id)
        native = None
        for modified_path in modified_paths:
            out_path = out_dir / f'{prefix}{modified_path.stem}.png'
            if out_path.exists() and not overwrite:
                report.append((results_id, modified_path.stem, 'exists', ''))
                continue
            if not write:
                report.append((results_id, modified_path.stem, 'would write',
                               str(out_path)))
                continue
            try:
                if native is None:
                    native = slab.HRTF(str(native_path)); native.name = short
                modified = slab.HRTF(str(modified_path))
                modified.name = modified_path.stem
                if modified.n_sources != native.n_sources:
                    raise ValueError(f'source grids differ: {native.n_sources} '
                                     f'vs {modified.n_sources}')
                figure = plot_ears(native, modified, band=band,
                                   suptitle=f'{short}   {native_path.stem} → '
                                            f'{modified_path.stem}')
                out_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(figure)
                report.append((results_id, modified_path.stem, 'written',
                               str(out_path)))
            except Exception as exc:                    # keep going
                plt.close('all')
                logger.debug(traceback.format_exc())
                report.append((results_id, modified_path.stem, 'FAILED', repr(exc)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true',
                        help='actually write; without it this is a dry run')
    parser.add_argument('--overwrite', action='store_true',
                        help='redo figures that already exist')
    parser.add_argument('--only', default=None, help='one subject id')
    parser.add_argument('--band', nargs=2, type=float, default=DEFAULT_BAND)
    args = parser.parse_args()

    report = backfill(overwrite=args.overwrite, only=args.only,
                      band=tuple(args.band), write=args.write)
    width = max((len(r[0]) for r in report), default=10)
    counts = {}
    for results_id, name, status, detail in report:
        counts[status] = counts.get(status, 0) + 1
        note = f'   {detail}' if status == 'FAILED' else ''
        print(f'{results_id:<{width}}  {name:<28} {status}{note}')
    print('\n' + '  '.join(f'{k}: {v}' for k, v in sorted(counts.items())))
    if not args.write:
        print('dry run — pass --write to save')


if __name__ == '__main__':
    main()
