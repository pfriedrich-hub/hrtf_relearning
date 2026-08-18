"""Detect pickles that have been silently mangled by a text editor.

THE FAILURE MODE. A subject pickle gets opened in an editor that treats it as
a text file, decodes it as UTF-8 with ``errors='replace'``, and writes it back.
Every byte that was not valid UTF-8 becomes U+FFFD (EF BF BD). The file keeps a
plausible size, git records it as a normal change, and nothing complains until
``pickle.load`` fails months later with ``invalid load key, '\\xef'``. The loss
is total and irreversible — U+FFFD does not record what byte it replaced.

HOW TO SPOT IT. A pickle written at protocol >= 2 starts with ``\\x80``, which
is a UTF-8 continuation byte with no lead byte, so a healthy pickle is never
valid UTF-8. If a .pkl decodes cleanly as UTF-8, it is not a pickle any more.
Checked against this repository's data folder: all healthy pickles fail to
decode, all mangled ones decode and contain U+FFFD. No false positives.

RECOVERY. Nothing can be reconstructed from the mangled file itself. Use, in
order of preference: the last good blob in git history
(``git log --follow -- <path>``, test each with :func:`is_mangled_pickle`);
the append-only ``<id>.json`` archive next to the pickle, via
``experiment/analysis/subject/restore_from_json.py``; the ``<id>.pkl.bak``
snapshot.

PREVENTION. ``.githooks/pre-commit`` refuses to commit a mangled pickle. Enable
it once per clone with::

    git config core.hooksPath .githooks
"""
from pathlib import Path

REPLACEMENT = b"\xef\xbf\xbd"  # U+FFFD encoded as UTF-8


def is_mangled_pickle(path):
    """True if `path` looks like a pickle destroyed by a text round-trip.

    A healthy pickle is not decodable as UTF-8. One that decodes *and* contains
    U+FFFD has been through an editor. Both conditions are required: a small
    protocol-0 pickle could in principle be pure ASCII and therefore decodable,
    but it would carry no replacement characters.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if not data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False  # still binary — healthy
    return REPLACEMENT in data


def scan(root=None, pattern="**/*.pkl"):
    """Every mangled pickle under `root` (defaults to the package data folder)."""
    if root is None:
        from hrtf_relearning.utils import paths
        root = paths.DATA_DIR
    return sorted(p for p in Path(root).glob(pattern) if is_mangled_pickle(p))


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    roots = [Path(a) for a in argv] or [None]
    bad = []
    for root in roots:
        bad.extend(scan(root) if root is None or root.is_dir()
                   else ([root] if is_mangled_pickle(root) else []))
    if not bad:
        print("no mangled pickles found")
        return 0
    print(f"{len(bad)} mangled pickle(s) — see hrtf_relearning/utils/integrity.py "
          f"for recovery:")
    for p in bad:
        print(f"  {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
