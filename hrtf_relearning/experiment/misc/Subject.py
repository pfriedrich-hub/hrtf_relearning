import json
import logging
import shutil
from pathlib import Path
import pickle
import hrtf_relearning
from hrtf_relearning.utils import paths
results_dir = paths.RESULTS_DIR
results_dir.mkdir(parents=True, exist_ok=True)


class Subject:
    """One participant's data (localization runs, training trials, highscore),
    persisted to RESULTS_DIR/<id>/<id>.pkl with a JSON backup alongside.

    Construct with the subject id; existing data is loaded from the pickle, or
    an empty record is created. Call write() to persist (it also refreshes the
    JSON archive). Instantiating never writes.

    Attributes
    ----------
    id : str
        Subject identifier; also the folder name under RESULTS_DIR.
    demographics : dict
        Age, gender and when they were recorded. Empty for records created
        before demographics were collected; fill with
        ``protocol_helpers.collect_demographics(subject)``.
    active_donor : dict
        Which donor this participant's composite HRTF is currently built from
        — ``{'donor', 'rank', 'tier', 'timestamp'}``, or ``{}`` if none has
        been set. Written by the learning-transfer protocol on build and on
        ``use_donor()``, and read back by ``load_existing_donor()`` so a later
        session reloads the donor the participant was actually trained on
        rather than whatever the selection rule ranks first today. The full
        candidate ranking is not here — it is embedded in the composite SOFA
        as ``GLOBAL_ModificationParams``.
    head_radius : float or None
        Effective head radius in metres, fitted acoustically from this
        participant's lateral ITDs by ``fit_head_radius.record_head_radius``
        (step 0 of the recording protocol) and passed to ``record_hrir`` as
        ``head_radius=``. ``None`` for records made before it was measured —
        those HRIRs used the pipeline default. The full fit (residual, both
        estimators, per-direction ITDs) is not kept here; it is logged when
        measured, and can be recovered from any SOFA with ``fit_from_sofa``.
    localization : dict[str, slab.Trialsequence]
        Localization runs keyed by filename ``<id>_<date>_<hrir>``, in
        insertion (chronological) order. Each value is the completed test
        sequence (targets, responses, settings, response_errors).
    trials : list[dict]
        Training trials appended sequentially; see the training data model.
    last_sequence : slab.Trialsequence or None
        Most recent localization run, used to weight training target
        probabilities. Repointed automatically if it is removed.
    highscore : int
        Best training session total, shown on the scoreboard.

    File layout
    -----------
        RESULTS_DIR/<id>/<id>.pkl        authoritative pickle
        RESULTS_DIR/<id>/<id>.pkl.bak    snapshot taken before each edit
        RESULTS_DIR/<id>/<id>.json       append-only text archive (all runs)

    Editing localization runs
    -------------------------
    Every removal backs up the pickle to <id>.pkl.bak, repoints last_sequence
    if it referenced a deleted run, and writes (pass write=False to preview).
    Removed runs are NOT lost: the <id>.json archive is append-only and keeps
    every run ever recorded, so the pickle can hold only the datapoints you
    want to analyse while the JSON retains the full record.

    From a ``# %%`` cell or REPL::

        s = Subject("AH")
        s.print_localization()              # numbered list of runs

        s.remove_localization_by_index(3)         # remove run #3 in the list
        s.remove_localization_by_index([1, 4])    # or several
        s.remove_localization("AH_02.06_...")     # or by exact key

        s.prune_localization()              # drop aborted + older duplicate runs

    Or interactively via experiment/analysis/subject/edit_subject.py::

        python edit_subject.py AH           # list runs, type numbers to remove
        python edit_subject.py AH --prune   # auto-drop aborted + duplicates

    Plotting localization runs / HRTFs
    ----------------------------------
    Same selection (1-based index from print_localization, or exact key)::

        s = Subject("AH")
        s.print_localization()
        s.plot_localization_run(3)                 # elevation gain / RMSE
        s.plot_localization_run(3, kind="both")    # + 2-D response grid
        s.plot_hrtf(3)                             # underlying HRTF waterfall
        s.plot_hrtf(3, compare_base=True)          # recorded-vs-modified HRTF
    """

    def __init__(self, id: str):
        # Everything for one subject lives under RESULTS_DIR/<id>/
        self.subject_dir = paths.subject_dir(id)
        self.file_path = paths.subject_pkl(id)
        self.backup_path = paths.subject_backup(id)
        self.id = id
        if self.file_path.exists():
            self._load()
        else:
            logging.info("Creating new subject.")
            self.localization = {}
            self.trials = []
            self.last_sequence = None
            self.highscore = 0
            self.demographics = {}
            self.active_donor = {}
            self.head_radius = None

    def _load(self):
        logging.info("Loading subject data.")
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            # 'invalid load key, \xef' is what a text-editor round-trip looks
            # like from here, and it says nothing about what to do next.
            from hrtf_relearning.utils.integrity import is_mangled_pickle
            if is_mangled_pickle(self.file_path):
                raise OSError(
                    f"{self.file_path.name} was destroyed by a text editor: it "
                    f"decodes as UTF-8 and is full of U+FFFD, so the original "
                    f"bytes are gone. Recover from git history (git log --follow "
                    f"-- {self.file_path}) or rebuild from {self.backup_path.name} "
                    f"with experiment/analysis/subject/restore_from_json.py. See "
                    f"hrtf_relearning/utils/integrity.py."
                ) from exc
            raise
        self.id = data.get("id", self.id)
        self.localization = data.get("localization", {})
        self.trials = data.get("trials", [])
        self.last_sequence = data.get("last_sequence", None)
        self.highscore = data.get("highscore", 0)
        # {} for records created before demographics were collected
        self.demographics = data.get("demographics", {})
        # {} for records created before the donor was tracked here (it used to
        # live in a <id>_donor_log.csv next to the plots)
        self.active_donor = data.get("active_donor", {})
        # None for records made before the head radius was fitted acoustically
        # (those HRIRs were built with the pipeline default, 0.0875 m).
        self.head_radius = data.get("head_radius", None)

    def write(self):
        logging.debug("Writing subject data.")
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "localization": self.localization,
            "trials": self.trials,
            "last_sequence": self.last_sequence,
            "highscore": self.highscore,
            "demographics": self.demographics,
            "active_donor": self.active_donor,
            "head_radius": self.head_radius,
        }
        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)
        self._write_backup()

    def _archivable_trials(self):
        """Training trials for the JSON archive: every field except pose_trace,
        with the trace replaced by its summary metrics.

        The raw ~48 Hz head trace is 97.6% of the volume of `trials` (IR: 7.9 MB
        with it, 190 KB without), which is why trials were left out of the
        archive entirely — and why a destroyed pickle used to take the whole
        training record with it. Summarising the trace (see
        analysis/training/pose_metrics.py) makes the archive complete at ~70x
        less volume, at the cost of the raw samples, which stay in the pickle.

        The metrics are also written back onto the in-memory trials, so they
        land in the pickle too and analysis need not recompute them.
        """
        try:
            from hrtf_relearning.experiment.analysis.training.pose_metrics import (
                add_pose_metrics)
            add_pose_metrics(self.trials)
        except Exception:
            logging.exception("Could not compute pose metrics for %s; archiving "
                              "trials without them.", self.id)
        out = []
        for trial in self.trials or []:
            if not trial:
                out.append({})
                continue
            out.append(_to_jsonable({k: v for k, v in trial.items()
                                     if k != "pose_trace"}))
        return out

    def _backup_pickle(self):
        """Copy the authoritative pickle to <id>.pkl.bak before a destructive
        edit, so a bad removal can always be undone. No-op if none exists yet."""
        if self.file_path.exists():
            shutil.copy2(self.file_path, self.file_path.with_suffix(".pkl.bak"))

    # ------------------------------------------------------------------ edit
    def localization_summary(self):
        """List stored localization runs as (key, label, n_trials, finished).

        Insertion order = chronological for a normally-grown file (filenames
        carry no year, so don't sort them lexically). Use to decide which runs
        to remove with remove_localization / prune_localization.
        """
        rows = []
        for key, seq in self.localization.items():
            rows.append((
                key,
                getattr(seq, "label", None),
                getattr(seq, "n_trials", None),
                bool(getattr(seq, "finished", False)),
            ))
        return rows

    def print_localization(self):
        """Pretty-print the localization runs with an index for interactive use."""
        rows = self.localization_summary()
        if not rows:
            print(f"{self.id}: no localization entries.")
            return
        print(f"\nSubject {self.id!r} — {len(rows)} localization run(s):\n")
        for i, (key, label, n, finished) in enumerate(rows):
            status = "✓" if finished else "…"
            n_str = f"{n:>3} trials" if isinstance(n, int) else "  ? trials"
            extra = f"  [{label}]" if label and label not in key else ""
            print(f"  [{i + 1:>2}]  {key:<45}  {status}  {n_str}{extra}")

    def _forget_removed_last_sequence(self, removed_keys):
        """If last_sequence pointed at a run we just removed, repoint it at the
        most recent surviving run (or None). Matched by the sequence's .name,
        which equals its localization key."""
        seq = self.last_sequence
        if seq is not None and getattr(seq, "name", None) in removed_keys:
            self.last_sequence = (next(reversed(self.localization.values()))
                                  if self.localization else None)

    def remove_localization(self, keys, write=True):
        """Remove one or more localization runs by key.

        `keys` may be a single key or an iterable of keys. Backs up the pickle
        first, fixes up last_sequence, and (by default) persists. Returns the
        list of keys actually removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        removed = []
        for key in keys:
            if key in self.localization:
                del self.localization[key]
                removed.append(key)
            else:
                logging.warning("No localization run %r for subject %s.", key, self.id)
        if not removed:
            return removed
        self._backup_pickle()
        self._forget_removed_last_sequence(set(removed))
        if write:
            self.write()
        logging.info("Removed %d localization run(s) from %s: %s",
                     len(removed), self.id, ", ".join(removed))
        return removed

    def _resolve_localization_key(self, sel):
        """Accept a 1-based index (as shown by print_localization) or an exact
        key and return the localization key. Raises IndexError / KeyError."""
        if isinstance(sel, int):
            keys = list(self.localization.keys())
            if not 1 <= sel <= len(keys):
                raise IndexError(f"index {sel} out of range (1..{len(keys)})")
            return keys[sel - 1]
        if sel in self.localization:
            return sel
        raise KeyError(f"no localization run {sel!r} for subject {self.id}")

    def remove_localization_by_index(self, indices, write=True):
        """Remove runs by 1-based index as shown by print_localization."""
        keys = list(self.localization.keys())
        if isinstance(indices, int):
            indices = [indices]
        to_remove = []
        for i in indices:
            if 1 <= i <= len(keys):
                to_remove.append(keys[i - 1])
            else:
                logging.warning("Index %d out of range (1..%d).", i, len(keys))
        return self.remove_localization(to_remove, write=write)

    def prune_localization(self, drop_unfinished=True, keep_last_duplicate=False,
                           write=True):
        """Remove redundant localization runs.

        drop_unfinished: remove aborted runs (no ``finished`` flag / no
            ``response_errors``) — these can't be analysed or used for target
            probabilities.
        keep_last_duplicate: when several runs share the same test condition
            (same hrir/ear/label/settings), keep only the most recent and drop
            the earlier repeats. OFF by default and to be used with care: the
            condition key ignores date, so genuine longitudinal repeats (the
            same test taken on different days — i.e. learning timepoints) look
            like duplicates and would be dropped. Preview with write=False and
            eyeball the result before trusting it on real data.
        Returns the list of removed keys.
        """
        def _freeze(x):
            if isinstance(x, (list, tuple)):
                return tuple(_freeze(v) for v in x)
            if isinstance(x, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
            return x

        def _condition_key(seq):
            s = getattr(seq, "settings", None)
            settings_sig = _freeze(s) if isinstance(s, dict) else None
            return (getattr(seq, "hrir", None), getattr(seq, "ear", None),
                    getattr(seq, "label", None), getattr(seq, "mirrored", None),
                    settings_sig)

        remove = []
        seen_condition = {}
        # iterate newest-first so the first time we see a condition is the keeper
        for key in reversed(list(self.localization.keys())):
            seq = self.localization[key]
            finished = bool(getattr(seq, "finished", False)) or \
                hasattr(seq, "response_errors")
            if drop_unfinished and not finished:
                remove.append(key)
                continue
            if keep_last_duplicate:
                cond = _condition_key(seq)
                if cond in seen_condition:
                    remove.append(key)  # an earlier (older) repeat
                else:
                    seen_condition[cond] = key
        return self.remove_localization(remove, write=write)

    # ------------------------------------------------------------------ plot
    def plot_localization_run(self, sel, kind="elevation", save=False, show=True):
        """Quickly plot one localization run.

        `sel` selects the run the same way as the editing methods: a 1-based
        index from print_localization, or the exact key. `kind` is 'elevation'
        (elevation gain / RMSE, the default quick QC), 'grid' (2-D response
        grid), or 'both'. Saves into RESULTS_DIR/<id>/plots/ when save=True.
        Returns the elevation Figure when drawn, else None.
        """
        from matplotlib import pyplot as plt
        from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
            plot_elevation_response, plot_localization)
        seq = self.localization[self._resolve_localization_key(sel)]
        plot_dir = paths.subject_plot_dir(self.id) if save else None
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
        fig = None
        if kind in ("elevation", "both"):
            res = plot_elevation_response(seq, filepath=plot_dir)
            fig = res if hasattr(res, "savefig") else None  # tuple => nothing to plot
        if kind in ("grid", "both"):
            plot_localization(seq, filepath=plot_dir)  # midline runs are skipped
        if show:
            plt.show()
        return fig

    def _sofa_path(self, hrir_name):
        """Resolve a SOFA file for an HRTF name. Individual HRTFs are foldered
        under the subject id (the part before the first underscore); reference
        HRTFs sit flat in SOFA_DIR. Falls back between the two layouts."""
        subject_id = hrir_name.split("_")[0]
        foldered = paths.SOFA_DIR / subject_id / f"{hrir_name}.sofa"
        if foldered.exists():
            return foldered
        flat = paths.SOFA_DIR / f"{hrir_name}.sofa"
        if flat.exists():
            return flat
        raise FileNotFoundError(
            f"no SOFA for {hrir_name!r} (looked in {foldered} and {flat})")

    def plot_hrtf(self, sel=None, hrir_name=None, compare_base=False):
        """Plot the underlying HRTF as a median-plane waterfall (both ears).

        Give a run selector `sel` (index/key — the HRTF name is read from the
        run's .hrir) or an explicit `hrir_name`. Saves the QC figure(s) to
        RESULTS_DIR/<id>/plots/hrtf/ and returns the saved Path(s).
        compare_base=True also plots recorded-vs-this HRTF (for modified HRTFs).
        """
        import slab
        from hrtf_relearning.hrtf.modify.edge_shift import (
            save_recorded_hrtf_waterfall, save_waterfall_qc)
        if hrir_name is None:
            if sel is None:
                raise ValueError("pass a run selector `sel` or an `hrir_name`")
            seq = self.localization[self._resolve_localization_key(sel)]
            hrir_name = getattr(seq, "hrir", None) or getattr(seq, "label", None)
        if not hrir_name:
            raise ValueError("run has no associated HRTF name")
        plot_dir = paths.subject_plot_dir(self.id) / "hrtf"
        sofa = self._sofa_path(hrir_name)
        hrtf = slab.HRTF(str(sofa))
        hrtf.name = hrir_name
        out = [save_recorded_hrtf_waterfall(hrtf, subject_id=hrir_name,
                                            plot_dir=plot_dir)]
        if compare_base:
            base_name = hrir_name.split("_")[0]
            base = slab.HRTF(str(self._sofa_path(base_name)))
            base.name = base_name
            out.append(save_waterfall_qc(base, hrtf, sofa, plot_dir=plot_dir))
        logging.info("Saved HRTF waterfall(s): %s", ", ".join(str(p) for p in out))
        return out[0] if len(out) == 1 else out

    def _write_backup(self):
        """Write a plain-text JSON archive of the localization data.

        Append-only: every localization run ever recorded stays here, keyed by
        its filename. Each write merges the current localization dict into
        whatever the JSON already holds, so runs removed from the pickle (via
        remove_localization / prune_localization) are NOT dropped from the
        JSON. The pickle can therefore hold only the runs you want as
        datapoints while the JSON keeps the full record; use a restore script
        to pull an archived run back if needed.

        Also read by game_ui.read_scoreboard() (id + highscore only) to render
        the cross-participant scoreboard during training. Failures here are
        logged but never raised, so a backup problem can't prevent the
        authoritative pickle write from succeeding. Lives at
        RESULTS_DIR/<id>/<id>.json.
        """
        try:
            # Start from whatever is already archived so removed runs survive.
            archived = {}
            if self.backup_path.exists():
                try:
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        archived = (json.load(f) or {}).get("localization", {}) or {}
                except Exception:
                    logging.exception(
                        "Could not read existing JSON archive for %s; rebuilding "
                        "from current data (older removed runs may be lost).",
                        self.id)
            # Current runs take precedence (updated in place); previously
            # archived keys not in the current pickle are preserved.
            merged = {**archived, **_to_jsonable(self.localization)}
            payload = {
                "id": self.id,
                "highscore": int(self.highscore) if self.highscore is not None else 0,
                "demographics": dict(self.demographics or {}),
                "active_donor": _to_jsonable(self.active_donor or {}),
                "head_radius": _to_jsonable(self.head_radius),
                # last_sequence is one of the localization runs; archive the key
                # rather than a second copy of the whole run.
                "last_sequence": getattr(self.last_sequence, "name", None),
                "localization": merged,
                "trials": self._archivable_trials(),
            }
            # Write via temp file + replace so a crash mid-write can't
            # leave a half-written backup on disk.
            tmp_path = self.backup_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.backup_path)
        except Exception:
            logging.exception("Failed to write JSON backup for subject %s", self.id)


def _to_jsonable(obj):
    """Recursively convert obj into JSON-serializable primitives.

    Handles numpy scalars/arrays and custom objects (like slab.Trialsequence)
    by dumping their `__dict__` under a `__class__` tag so the backup is
    self-describing enough for a restore script to reconstruct it.
    """
    # Fast path for primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy — imported lazily so this module works without numpy installed
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # any numpy scalar
            return obj.item()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)

    # Custom objects (slab.Trialsequence, etc.): dump instance state
    if hasattr(obj, "__dict__"):
        return {
            "__class__": f"{type(obj).__module__}.{type(obj).__name__}",
            **{k: _to_jsonable(v) for k, v in vars(obj).items()
               if not k.startswith("_")},
        }

    # Last-resort fallback: stringify
    return repr(obj)