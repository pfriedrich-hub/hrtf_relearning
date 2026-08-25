# donor_modification.py
"""
Donor-detail cue manipulation, as a workflow object.

    log|H_modified(dir)| = envelope_k( log|H_own| ) + detail( log|H_donor| )

with the participant's own phase and own per-direction broadband level. The
maths is in `donor_detail.py`; this is everything AROUND it -- choosing the
donor by the fixed rule, building and naming the composite, embedding the
selection record, staging the alternates and their pyBinSim databases, and
remembering which donor a participant is actually on.

WHY A CLASS. All of this lived as module-level functions inside
`learning_transfer.py` next door, reading a dozen module globals (SUBJECT_ID,
NATIVE_SOFA, DONOR_ID, MODIFIED_SOFA, TRAINED_EAR, OTHER_EAR, ENV_NKEEP,
PIPELINE) and writing three of them back with `global`. That works for one
participant in one console, and fails the moment you want two subjects in one
session, a unit test, or a second protocol -- and `global DONOR_ID` inside a
build function is exactly the kind of hidden state that lets a subject be
trained on one stimulus and tested on another. The state is per-participant, so
it belongs on an instance.

WHY IT LIVES HERE AND NOT IN hrtf/modify/. The split is along `build()`:
everything below it is a pure function on HRTFs and already lives in
`hrtf/modify/` (`donor_detail`) and `hrtf/processing/` (`envelope`, `midline`).
Everything here is the EXPERIMENT around that -- which donor this participant
is on, what the subject pickle says, which pyBinSim databases are staged, what
the QC figure is called. That is learning-transfer machinery, not a cue
manipulation, so it sits with the protocol it serves. `hrtf/modify/` stays
what it claims to be: manipulations, no I/O, no subject state.

Typical use, one participant:

    donor = DonorModification("AS", trained_ear="left")
    donor.prepare_shortlist(n=3)      # before the session, ~minutes
    donor.build()                     # day 1, picks rank 0
    donor.use_donor(rank=1, reason="EG 0.03 on baseline A, at chance")

Day 1 is where the donor is decided; the rest of the study just uses it. The
choice is recorded on the participant (subject.active_donor), and constructing
this object for that subject reads it back, so a session on day 3 is

    donor = DonorModification("AS", trained_ear="left")   # already on rank 1
    donor.load_existing()             # optional: confirm against the SOFA
"""
from __future__ import annotations

import datetime
import re
import shutil

import slab

import hrtf_relearning as hr
from hrtf_relearning.hrtf.analysis import donor_selection as selection
from hrtf_relearning.hrtf.analysis.vsi import vsi as vsi_of
from hrtf_relearning.hrtf.modify.donor_detail import donor_detail_dtf, modification_params
from hrtf_relearning.hrtf.modify.edge_shift import (embed_modification_params,
                                                    read_modification_params)
from hrtf_relearning.hrtf.modify.plot_compare import plot_ears
from hrtf_relearning.hrtf.processing.envelope import envelope_dtf, ENVELOPE_BAND
from hrtf_relearning.hrtf.processing.midline import (midline_arc, expand_from_midline,
                                                     qc_midline, format_qc)
from hrtf_relearning.utils import paths

OTHER_EAR_TREATMENTS = ("flat", "envelope", "native")


class DonorModification:
    """One participant's donor-detail manipulation: selection, build, staging.

    Parameters
    ----------
    subject_id : str
    trained_ear : {'left', 'right'}
        The ear that keeps the manipulated cue in monaural blocks.
    native_sofa : str, optional
        Stem of the measured HRTF. Defaults to `subject_id`.
    other_ear : {'flat', 'envelope', 'native'}
        What the non-listening ear gets. Only affects v1 naming and the
        pyBinSim database names; on v2 the reduction is baked into the SOFA.
    env_n_keep : int, optional
        Cepstral order of the envelope handed to the other ear. Defaults to
        `donor_selection.N_KEEP`.
    pipeline : {'v1', 'v2'}
        'v1' modifies the finished 475-direction SOFA and lets hrtf2binsim
        apply the monaural reduction at render time. 'v2' modifies the 19
        MEASURED az=0 DTFs and re-expands through the spherical head model.
        See the protocol's LEGACY_V1_SUBJECTS note for why subjects already
        run stay on v1.
    donor_id : str, optional
        Pre-set the active donor (skips selection when reloading a session).
    hp, reverb, drr, convolution, storage :
        Render settings passed through to `hrir_settings`, which is what
        hrtf2binsim consumes. Defaults match the learning-transfer protocol.
    """

    def __init__(self, subject_id, *, trained_ear, native_sofa=None,
                 other_ear="envelope", env_n_keep=None, pipeline="v2",
                 donor_id=None, hp="DT990", reverb=True, drr=20,
                 convolution="cpu", storage="cpu"):
        if trained_ear not in ("left", "right"):
            raise ValueError("trained_ear must be 'left' or 'right'")
        if other_ear not in OTHER_EAR_TREATMENTS:
            raise ValueError(f"other_ear must be one of {OTHER_EAR_TREATMENTS}")
        if pipeline not in ("v1", "v2"):
            raise ValueError("pipeline must be 'v1' or 'v2'")

        self.subject_id = subject_id
        self.native_sofa = native_sofa or subject_id
        self.trained_ear = trained_ear
        self.untrained_ear = "right" if trained_ear == "left" else "left"
        self.other_ear = other_ear
        self.env_n_keep = selection.N_KEEP if env_n_keep is None else env_n_keep
        self.pipeline = pipeline
        self.hp = hp
        self.reverb, self.drr = reverb, drr
        self.convolution, self.storage = convolution, storage

        # active state -- what the protocol's blocks will load.
        #
        # With no donor_id passed, it is read back from the participant's own
        # record (subject.active_donor in <id>.pkl, mirrored into <id>.json),
        # which build()/prepare_shortlist()/use_donor() wrote on day 1. That is
        # the whole point: the donor is chosen ONCE, per participant, and every
        # later session picks it up from their file without anyone having to
        # remember it or edit the protocol. Passing donor_id= overrides it and
        # is for rebuilding something already run, not for daily use.
        self.donor_from_record = False
        if donor_id is None:
            donor_id = self._recorded_donor()
            self.donor_from_record = donor_id is not None
        self.donor_id = donor_id
        self.modified_sofa = self.modified_name(donor_id) if donor_id else None

        self._shortlist = None

    def _recorded_donor(self):
        """The participant's donor from their pickle, or None -- never raises.

        Called from __init__, so a subject file that cannot be read must not
        stop the object being built: without it there is no way to run build()
        and create the record in the first place.
        """
        try:
            return self.last_active_donor()
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] could not read the donor from "
                  f"{self.subject_id}'s subject file: {exc}")
            return None

    def __repr__(self):
        return (f"<DonorModification {self.subject_id} "
                f"ear={self.trained_ear} {self.pipeline} "
                f"donor={self.donor_id} sofa={self.modified_sofa}>")

    # -- paths ------------------------------------------------------------

    @property
    def sofa_dir(self):
        return paths.SOFA_DIR / self.subject_id

    def own_hrtf(self):
        """The participant's measured HRTF, named so figures label it."""
        own = slab.HRTF(str(self.sofa_dir / f"{self.native_sofa}.sofa"))
        own.name = self.subject_id
        return own

    # -- naming -----------------------------------------------------------

    def modified_name(self, donor_id, n_keep=None):
        """The SOFA the protocol's blocks load.

        On v2 the monaural reduction is already inside the file, so it is part
        of the name -- there is no longer a render-time switch that could be
        set differently between two runs of the same block.
        """
        n_keep = selection.N_KEEP if n_keep is None else n_keep
        stem = f"{self.subject_id}_donor_{donor_id.split('/')[-1]}"
        if n_keep != selection.N_KEEP:
            stem = f"{stem}_n{n_keep}"
        if self.pipeline == "v2":
            stem = f"{stem}_env{self.env_n_keep}_{self.trained_ear}"
        return stem

    def is_training_sofa(self, stem):
        """True if ``stem`` is a composite the protocol would actually PLAY.

        Two kinds of file share the <subject>_donor_<D> prefix and must never
        be mistaken for it: the ladder's extra strengths (_n<k>, diagnostics
        only) and, on v2, the binaural pre-reduction QC file, which lacks the
        _env<k>_<ear> tail precisely because the monaural reduction is not in
        it yet. Getting this wrong means testing someone binaurally for a day.
        """
        prefix = f"{self.subject_id}_donor_"
        if not stem.startswith(prefix):
            return False
        rest = stem[len(prefix):]
        if re.search(r"_n\d+(_|$)", rest):     # ladder strength, not the protocol's
            return False
        if self.pipeline == "v2":
            return rest.endswith(f"_env{self.env_n_keep}_{self.trained_ear}")
        return not re.search(r"_env\d+_(left|right)$", rest)

    def binaural_name(self, donor_id, n_keep=None):
        """v2 only: the composite BEFORE the monaural reduction. QC reference."""
        n_keep = selection.N_KEEP if n_keep is None else n_keep
        stem = f"{self.subject_id}_donor_{donor_id.split('/')[-1]}"
        return stem if n_keep == selection.N_KEEP else f"{stem}_n{n_keep}"

    def binsim_names(self, sofa_name, mirror, other_ear=None):
        """The binsim database directory name hrtf2binsim would produce.

        `other_ear` overrides the instance setting -- `discard_unused` needs to
        enumerate the names every treatment could have produced, because a
        database may have been built when the setting was something else.
        """
        other_ear = self.other_ear if other_ear is None else other_ear
        name = sofa_name
        # v2 carries the reduction in the SOFA name already, so hrtf2binsim
        # adds nothing here -- see hrir_settings, which passes ear=None for
        # those files.
        baked = self.pipeline == "v2" and f"_env{self.env_n_keep}_" in sofa_name
        if self.trained_ear and not baked:
            if other_ear == "flat":
                name += f"_{self.trained_ear}"
            elif other_ear == "envelope":
                name += f"_{self.trained_ear}_env{self.env_n_keep}"
            elif other_ear == "native":
                name += f"_{self.trained_ear}_nat"
        if mirror:
            name += "_mirrored"
        return name

    def hrir_settings(self, sofa_name, ear=None, mirror=False, other_ear=None):
        """The dict hrtf2binsim / Localization consume for one block."""
        # v2 bakes the monaural reduction into the SOFA, so hrtf2binsim must
        # NOT apply it a second time -- ear=None means "take the file as it
        # is". `mirror` still happens at render time: it is a channel/source
        # swap and commutes with everything above it, so blocks C and D come
        # off this file.
        baked = self.pipeline == "v2" and f"_env{self.env_n_keep}_" in sofa_name
        return {
            "name": sofa_name,
            "subject_id": self.subject_id,
            "ear": None if baked else ear,
            "other_ear": self.other_ear if other_ear is None else other_ear,
            # pins the v1 render-time envelope to its historical behaviour;
            # unused when `baked`. See hrtf2binsim and processing.envelope.
            "env_elevation_average": False,
            "env_n_keep": self.env_n_keep,
            "native_sofa": self.native_sofa,
            "mirror": mirror,
            "reverb": self.reverb,
            "drr": self.drr,
            "hp_filter": True,
            "hp": self.hp,
            "convolution": self.convolution,
            "storage": self.storage,
        }

    # -- selection --------------------------------------------------------

    def shortlist(self, refresh=False, quiet=False):
        """This subject's candidate donors in rule order -- cached.

        ``[0]`` is the protocol pick, ``[1]`` and ``[2]`` are the alternates
        ``use_donor(rank=...)`` swaps to. Scoring the pool takes ~20 s, so it
        is computed once and reused; pass ``refresh=True`` after editing
        selection.DONOR_POOL.
        """
        if self._shortlist is not None and not refresh:
            return self._shortlist

        own = self.own_hrtf()
        candidates = selection.load_candidates(self.subject_id)
        if not candidates:
            raise RuntimeError("empty donor pool -- check selection.DONOR_POOL")
        if not quiet:
            print(f"donor pool ({len(candidates)}): {', '.join(candidates)}")

        self._shortlist = selection.shortlist(own, candidates)
        if not quiet:
            reference, _ = selection.pairwise_r_match(
                {self.subject_id: own, **candidates})
            selection.report(self._shortlist, reference)
        return self._shortlist

    def _pick(self, rows, rank=0, donor_id=None):
        """Resolve (rank | donor_id) to one shortlist row."""
        if donor_id is not None:
            matches = [row for row in rows if row["donor"] == donor_id]
            if not matches:
                raise ValueError(f"{donor_id} is not in this subject's pool: "
                                 f"{', '.join(r['donor'] for r in rows)}")
            return matches[0]
        if rank is None:
            raise ValueError("pass rank= or donor_id=")
        if rank >= len(rows):
            raise IndexError(f"requested rank {rank} but only {len(rows)} "
                             f"candidates were scored")
        return rows[rank]

    # -- the donor a participant is on ------------------------------------

    def _set_active_donor(self, donor_id, rank, reason=""):
        """Record the donor this participant is currently on, in their pickle.

        Stored as subject.active_donor so a later session can reload the donor
        the participant was actually trained on rather than whatever the
        selection rule ranks first today (the pool grows as subjects are
        added, so rank 0 is not stable over time). Overwritten on each change
        -- only the current donor is kept; the candidate ranking behind the
        choice lives in the composite SOFA as GLOBAL_ModificationParams.
        """
        subject = hr.Subject(self.subject_id)
        subject.active_donor = {
            "donor": donor_id,
            "rank": rank,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        subject.write()
        return subject.active_donor

    def last_active_donor(self):
        """The donor this participant is currently on, from their pickle."""
        return (hr.Subject(self.subject_id).active_donor or {}).get("donor")

    def show_log(self):
        """The donor this participant is on. Read before analysis."""
        record = hr.Subject(self.subject_id).active_donor or {}
        if not record:
            print(f"no donor recorded for {self.subject_id}")
            return
        print(f"{self.subject_id}: donor {record.get('donor')} "
              f"(rank {record.get('rank')}) set {record.get('timestamp')}")
        if record.get("reason"):
            print(f"  reason: {record['reason']}")
        return record

    # -- build ------------------------------------------------------------

    def _save_qc_figure(self, own, modified, name, chosen, n_keep, donor_id,
                        show=True, quiet=False, overwrite=False):
        """Write plots/acoustic/<name>.png -- the before/after 2x2.

        ALWAYS saved, `show` only decides whether it is displayed. It is
        provenance in the same sense as the ranking CSV: the picture of what
        this participant was actually given. Tying the save to a display flag
        is what left AS with six donor SOFAs and no figures -- staging builds
        pass show_qc=False, and by the time the day-1 cell ran the files
        existed, so the build returned early and never reached the plot.

        ``modified`` may be a path, so this also works on the already-built
        path where nothing is in memory.
        """
        plot_dir = paths.subject_acoustic_dir(self.subject_id)
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_png = plot_dir / f"{name}.png"
        if out_png.exists() and not overwrite and not show:
            return out_png
        if not isinstance(modified, slab.HRTF):
            modified = slab.HRTF(str(modified))
            modified.name = name
        fig = plot_ears(own, modified, vsi_dis=chosen["r_match"],
                        vsi_bw=selection.DEFAULT_BAND,
                        band=selection.DEFAULT_BAND,
                        suptitle=f"{self.subject_id}  own envelope "
                                 f"(n_keep={n_keep}) + {donor_id} detail",
                        show=show)
        fig.savefig(out_png, bbox_inches="tight")
        if not show:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        if not quiet:
            print(f"QC figure: {out_png}")
        return out_png

    def build(self, overwrite=False, show_qc=True, n_keep=None, rank=0,
              donor_id=None, set_active=True, quiet=False):
        """Build one composite, write it, return (path, donor_id).

        Writes <subject>_donor_<DONOR>.sofa next to the native one, embeds the
        full selection record as GLOBAL_ModificationParams, and saves the
        candidate ranking as a CSV for the supplement.

        ``rank`` picks which entry of :meth:`shortlist` to build -- 0 is the
        protocol choice, 1 and 2 are the alternates. ``donor_id`` overrides the
        rank and names a donor directly (use only to rebuild something already
        run). ``set_active=False`` builds without repointing this object, which
        is what :meth:`prepare_shortlist` uses to stage the alternates.
        """
        n_keep = selection.N_KEEP if n_keep is None else n_keep
        is_default = n_keep == selection.N_KEEP
        own = self.own_hrtf()

        candidates = selection.load_candidates(self.subject_id)
        if not candidates:
            raise RuntimeError("empty donor pool -- check selection.DONOR_POOL")

        rows = self.shortlist(quiet=quiet)
        chosen = self._pick(rows, rank=rank, donor_id=donor_id)

        if not quiet:
            print(f"\ndonor rank {chosen['rank']}: {chosen['donor']}   r_match "
                  f"{chosen['r_match']:.2f} (target "
                  f"{selection.TARGET_R_MATCH:.2f} ± {selection.TOLERANCE:.2f})"
                  f"   ridge slope {chosen['ridge_slope']:+.2f}"
                  f"   cue strength {chosen['donor_strength']:.1f} dB"
                  f"   [{chosen['tier']}]")
        if chosen["fallback"]:
            print("  !! FALLBACK: no candidate met the ridge criterion; lowest "
                  "slope used. This must be reported.")
        elif chosen["tier"] == "widened" and not quiet:
            print(f"  !  no candidate landed within {selection.TOLERANCE:.2f} "
                  f"of the target; nearest-to-target used. Report the tier.")

        # The donor is always selected at the protocol n_keep, so the two
        # ladder strengths differ ONLY in how much of the cue is handed over --
        # not in whose cue it is.
        donor_id = chosen["donor"]
        name = self.modified_name(donor_id, n_keep)
        if is_default and set_active:
            self.donor_id, self.modified_sofa = donor_id, name
            self._set_active_donor(donor_id, chosen["rank"], chosen["tier"])
        out_path = self.sofa_dir / f"{name}.sofa"
        if out_path.exists() and not overwrite:
            if not quiet:
                print(f"{out_path.name} already exists (overwrite=False) "
                      f"-- skipping build")
            # self-healing: the SOFA may have been written by a staging build
            # that never drew the figure. Re-read it rather than leave the
            # record short.
            self._save_qc_figure(own, out_path, name, chosen, n_keep, donor_id,
                                 show=show_qc, quiet=quiet)
            return out_path, donor_id

        # extra provenance for whatever this build did beyond the donor step.
        # The file name says _env4_<ear>; the embedded record has to say the
        # same, or it is another SOFA nobody can trace (see FD 12:13).
        pipeline_params = {}
        binaural_path = None

        if self.pipeline == "v1":
            modified = donor_detail_dtf(own, candidates[donor_id], n_keep=n_keep)
        else:
            # --- v2: modify the 19 MEASURED directions, then re-expand -------
            # The az=0 arc in the finished SOFA is magnitude-identical to what
            # went into the expansion (step 2 skips az=0), so it is read back
            # from there rather than replayed from the npz -- no reference
            # recording needed.
            own_arc = midline_arc(own)
            donor_arc = midline_arc(candidates[donor_id])
            arc = donor_detail_dtf(own_arc, donor_arc, n_keep=n_keep)
            # binaural composite, kept as the QC reference and for the ladder
            binaural = expand_from_midline(arc)
            binaural.name = self.binaural_name(donor_id, n_keep)
            binaural_path = self.sofa_dir / f"{binaural.name}.sofa"
            binaural.write_sofa(str(binaural_path))

            arc = envelope_dtf(arc, ear=self.trained_ear, n_keep=self.env_n_keep)
            report = qc_midline(own_arc, arc, processed_ear=self.untrained_ear,
                                raise_on_fail=True)
            print(format_qc(report))
            modified = expand_from_midline(arc)

            pipeline_params = {
                "pipeline": "v2",
                "chain": ("midline_arc -> donor_detail_dtf -> envelope_dtf -> "
                          "qc_midline -> expand_azimuths_with_binaural_cues"),
                "expansion": {"itd_method": "phase", "az_range": [-50, 50]},
                "monaural": {"other_ear": "envelope",
                             "ear_kept": self.trained_ear,
                             "ear_processed": self.untrained_ear,
                             "env_n_keep": self.env_n_keep,
                             "env_band_hz": list(ENVELOPE_BAND),
                             "elevation_average": True},
                "midline_qc": {k: v for k, v in report.items()},
            }

        modified.name = name
        own_vsi, modified_vsi = vsi_of(own), vsi_of(modified)
        print(f"VSI  own={own_vsi:.3f}  modified={modified_vsi:.3f}  "
              f"(diagnostic only -- not diffuse-field normalised, see vsi.py)")

        def _params(**extra):
            return modification_params(
                self.subject_id, donor_id, n_keep=n_keep,
                target_r_match=selection.TARGET_R_MATCH,
                tolerance=selection.TOLERANCE,
                band=selection.DEFAULT_BAND,
                resolution=selection.DEFAULT_RESOLUTION,
                max_ridge_slope=selection.MAX_RIDGE_SLOPE,
                pool=list(candidates),
                fallback=chosen["fallback"],
                scores={k: chosen[k] for k in ('r_match', 'ridge_slope',
                                               'donor_strength')},
                ranking=[{k: row[k] for k in ('donor', 'rank', 'tier',
                                              'donor_strength', 'r_match',
                                              'ridge_slope', 'eligible',
                                              'in_band')} for row in rows],
                **extra)

        modified.write_sofa(str(out_path))
        embed_modification_params(out_path, _params(**pipeline_params))
        print(f"wrote {out_path}")

        # the binaural composite is a MODIFIED file too -- label it, or it is
        # an unattributable SOFA sitting next to the native one
        if binaural_path is not None:
            embed_modification_params(binaural_path, _params(
                **{**pipeline_params, "monaural": None,
                   "note": "binaural composite, before the monaural reduction; "
                           "QC reference for the _env file next to it"}))
            print(f"wrote {binaural_path}")

        self._save_qc_figure(own, modified, name, chosen, n_keep, donor_id,
                             show=show_qc, quiet=quiet)
        return out_path, donor_id

    # -- staging and swapping ---------------------------------------------

    def prepare_shortlist(self, n=3, mirrored=True, overwrite=False):
        """Stage the top ``n`` donors so a mid-session swap costs seconds.

        Builds, for each of the first ``n`` entries of :meth:`shortlist`, the
        composite SOFA and the pyBinSim filter databases the protocol actually
        plays: trained ear + other_ear, un-mirrored (day-1 baseline A, the
        daily test, training) and mirrored (baseline D and the final C/D
        blocks).

        RUN THIS BEFORE THE PARTICIPANT ARRIVES. write_filters passes over all
        475 directions, so a database is minutes, not seconds; doing it with
        someone in the chair is what makes a donor swap expensive. Once
        staged, :meth:`use_donor` is instant.

        Only the rank-0 donor is left active -- the alternates are built but
        not selected, so nothing about the default path changes.
        """
        from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim

        rows = self.shortlist()
        n = min(n, len(rows))
        print(f"\nstaging {n} donors for {self.subject_id}: "
              f"{', '.join(r['donor'] for r in rows[:n])}")

        staged = []
        for row in rows[:n]:
            donor = row["donor"]
            print(f"\n--- rank {row['rank']}: {donor} "
                  f"(r_match {row['r_match']:.2f}, ridge "
                  f"{row['ridge_slope']:+.2f}, strength "
                  f"{row['donor_strength']:.1f} dB, {row['tier']}) ---")
            # set_active=False: staging must not silently repoint the protocol
            self.build(overwrite=overwrite, show_qc=False,
                       donor_id=donor, set_active=False, quiet=True)
            name = self.modified_name(donor)
            for mirror in ((False, True) if mirrored else (False,)):
                db = paths.BINSIM_DIR / self.binsim_names(name, mirror)
                if db.exists() and not overwrite:
                    print(f"    binsim {db.name} exists -- skipping")
                    continue
                print(f"    building binsim {db.name} ...")
                hrtf2binsim(self.hrir_settings(name, ear=self.trained_ear,
                                               mirror=mirror),
                            overwrite=overwrite, build=True)
            staged.append(donor)

        # rank 0 is the protocol donor; make it the active one
        self.build(overwrite=False, show_qc=False, rank=0, quiet=True)
        print(f"\nstaged: {', '.join(staged)}")
        print(f"active: {self.donor_id}  ({self.modified_sofa})")
        print("swap with  use_donor(rank=1, reason='...')")
        print("once the donor is settled: discard_unused()")
        return staged

    def use_donor(self, rank=None, donor_id=None, reason=""):
        """Switch the active donor mid-session. Instant if pre-staged.

        Use when a participant cannot localize at all with the current
        composite -- a floor-level block is uninformative, and the design needs
        an acute degradation, not an abolished cue. ``rank=1`` moves to the
        next donor on the rule's list.

        ``reason`` is stored on subject.active_donor and should say what was
        observed ('EG 0.02 on baseline A, at floor'), because a swap made after
        seeing the data has to be reportable as such. It is NOT free: the
        discarded block was still run, and both the discarded and the
        replacement donor belong in the participant's record.

        If the replacement's binsim database has not been staged it is built
        here, which takes minutes -- run :meth:`prepare_shortlist` before the
        session instead.
        """
        rows = self.shortlist(quiet=True)
        row = self._pick(rows, rank=rank, donor_id=donor_id)

        previous = self.donor_id
        name = self.modified_name(row["donor"])
        sofa_path = self.sofa_dir / f"{name}.sofa"
        if not sofa_path.exists():
            raise FileNotFoundError(
                f"{sofa_path.name} has not been built -- run "
                f"prepare_shortlist() before the session, or "
                f"build(rank={row['rank']}) now (slow).")

        missing = [self.binsim_names(name, m) for m in (False, True)
                   if not (paths.BINSIM_DIR / self.binsim_names(name, m)).exists()]
        if missing:
            print(f"  [!] binsim database(s) not staged: {', '.join(missing)}")
            print("      building now -- this takes minutes. Ctrl-C and run "
                  "prepare_shortlist() before the next session.")
            from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
            for mirror in (False, True):
                if self.binsim_names(name, mirror) in missing:
                    hrtf2binsim(self.hrir_settings(name, ear=self.trained_ear,
                                                   mirror=mirror),
                                overwrite=False, build=True)

        self.donor_id, self.modified_sofa = row["donor"], name
        self._set_active_donor(row["donor"], row["rank"],
                               reason or f"(no reason given; from {previous})")
        print(f"\nactive donor: {previous} -> {row['donor']}   "
              f"(rank {row['rank']}, {row['tier']}, "
              f"r_match {row['r_match']:.2f}, ridge "
              f"{row['ridge_slope']:+.2f}, strength "
              f"{row['donor_strength']:.1f} dB)")
        print(f"modified_sofa = {self.modified_sofa}")
        print(f"  recorded in {self.subject_id}'s subject file -- later "
              f"sessions load {row['donor']} automatically; nothing to edit.")
        if not reason:
            print("  !! no reason recorded -- re-run use_donor(..., reason=...) "
                  "to say what was observed.")
        return row

    def load_existing(self):
        """Point this object at the already-built donor SOFA on disk.

        Normally a no-op confirmation: __init__ already took the donor from the
        participant's record, and this re-reads the SOFA's embedded params to
        show that what is on disk is what was built. It still stands alone for
        a subject whose record predates active_donor, where the donor has to
        come off the filesystem.
        """
        if self.donor_id is not None:
            matches = [self.sofa_dir / f"{self.modified_name(self.donor_id)}.sofa"]
        else:
            matches = [path for path
                       in sorted(self.sofa_dir.glob(f"{self.subject_id}_donor_*.sofa"))
                       if self.is_training_sofa(path.stem)]

        if not matches:
            raise FileNotFoundError(
                f"no {self.subject_id}_donor_*.sofa in {self.sofa_dir} "
                f"-- run build() first")
        if len(matches) > 1:
            # Only reachable with no donor in the subject's record -- otherwise
            # __init__ resolved it and the single-file branch above was taken.
            # prepare_shortlist() stages alternates, so several composites on
            # disk is NORMAL; the filesystem cannot say which one was chosen.
            raise RuntimeError(
                f"several donor SOFAs for {self.subject_id}: "
                f"{', '.join(p.name for p in matches)}, and no donor recorded "
                f"in their subject file to say which is active. Pass "
                f"donor_id= to the constructor to name the one this subject "
                f"was trained on -- and re-run use_donor(donor_id=..., "
                f"reason='reconstructing the record') so the next session "
                f"does not have to ask again.")

        path = matches[0]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run build() first")

        params = read_modification_params(path) or {}
        embedded = params.get("donor_id")
        if embedded is None:
            print(f"  [warn] {path.name} carries no modification params -- donor "
                  f"id taken from the filename, which is not authoritative")
            embedded = path.stem.replace(f"{self.subject_id}_donor_", "")
        self.donor_id = embedded
        self.modified_sofa = path.stem

        scores = params.get("scores", {})
        print(f"using {path.name}   donor={self.donor_id}"
              + (f"   r_match {scores['r_match']:.2f}, ridge "
                 f"{scores['ridge_slope']:+.2f}" if 'r_match' in scores else "")
              + ("   [built with FALLBACK donor]" if params.get("fallback") else ""))
        return path

    # -- cleanup ----------------------------------------------------------

    def discard_unused(self, dry_run=True, keep=None):
        """Delete every staged donor for this subject except the active one.

        Staging costs ~4 MB of SOFA plus two pyBinSim databases per candidate,
        and after the day-1 baselines the choice is made -- the alternates are
        dead weight that also makes it ambiguous, months later, which composite
        the participant actually heard. Run this once the donor is settled.

        Keeps the active donor's SOFA and databases, and never touches the
        native <subject>.sofa. The discarded ones are reproducible at any time
        from build(donor_id=...), so nothing unrecoverable is lost -- the
        selection record stays in the surviving SOFA's GLOBAL_ModificationParams
        and in subject.active_donor.

        Defaults to ``dry_run=True``: it prints what it would remove and
        removes nothing. Call again with ``dry_run=False`` to actually delete.
        Pass ``keep=['XX']`` to spare extra donors (e.g. one you still mean to
        compare).
        """
        active = self.donor_id or self.last_active_donor()
        if active is None:
            raise RuntimeError(
                "no active donor for this subject -- run build() or "
                "load_existing() first, or this would delete everything.")

        # Match on the exact composite name, not on the donor id appearing in
        # the stem: '<SID>_donor_AS' is a PREFIX of '<SID>_donor_AS_n8', so a
        # substring or startswith test would silently spare the n_keep=8 ladder
        # build too. That build is a diagnostic, not what the participant was
        # trained on, so it goes with the rest unless named in `keep`.
        #
        # The binsim directory name depends on other_ear, which may have been a
        # different setting when a database was built than it is today. Spare
        # every variant the active composite could have produced, so a
        # since-changed setting can never make this delete the database the
        # participant is actually being tested on.
        spared_sofa, spared_binsim = set(), set()
        for donor in {active, *(keep or [])}:
            name = self.modified_name(donor)
            # On v2 a donor has TWO files: the reduced one the blocks load
            # (modified_name, '<SID>_donor_<D>_env4_<ear>') and the binaural
            # composite kept as its QC reference (binaural_name,
            # '<SID>_donor_<D>'). Sparing only the first would delete the QC
            # reference for the donor that was actually used. On v1 the two
            # names coincide.
            spared_sofa.update((name, self.binaural_name(donor)))
            for treatment in OTHER_EAR_TREATMENTS:
                spared_binsim.update(
                    self.binsim_names(name, m, other_ear=treatment)
                    for m in (False, True))

        victims = [p for p in sorted(self.sofa_dir.glob(f"{self.subject_id}_donor_*.sofa"))
                   if p.stem not in spared_sofa]
        victims += [p for p in sorted(paths.BINSIM_DIR.glob(f"{self.subject_id}_donor_*"))
                    if p.is_dir() and p.name not in spared_binsim]

        if not victims:
            print(f"{self.subject_id}: nothing to discard (active donor {active})")
            return []

        total = sum(f.stat().st_size for v in victims
                    for f in ([v] if v.is_file() else v.rglob("*")) if f.is_file())
        print(f"{self.subject_id}: keeping {', '.join(sorted(spared_sofa))}; "
              f"{'would remove' if dry_run else 'removing'} {len(victims)} "
              f"item(s), {total / 1e6:.1f} MB")
        for v in victims:
            print(f"    {'[dry-run] ' if dry_run else ''}"
                  f"{v.relative_to(paths.DATA_DIR)}")
            if not dry_run:
                shutil.rmtree(v) if v.is_dir() else v.unlink()
        if dry_run:
            print("  nothing deleted -- re-run with dry_run=False to apply")
        return victims
