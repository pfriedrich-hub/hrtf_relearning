"""
Per-participant cue-tuning log.

While fitting each subject's modified HRTF to the target starting performance
(elevation gain 0.2-0.4, RMSE 15-25 deg), you adjust the modification params,
regenerate the modified SOFA, run a binaural baseline loc test, check the
metrics, and repeat. This logs each iteration (params + resulting metrics +
which loc run produced them) so the choice is documented, and lets you finalize
one iteration and archive the rest at the end.

Params logged depend on cue type:
    shift : {low_hz, high_hz, factor, envelope_n_keep, skirt_octaves, ...}
    synth : {X1, X2, Y, ...}   (linear notch position/width/scaling)
You pass whatever dict you used; it's stored verbatim.

Log file (one per subject):  data/documentation/cue_tuning/<id>.json

USAGE (import from a REPL/notebook, or CLI):
    from cue_tuning_log import log_iteration, show_log, finalize
    log_iteration("JS", params={"X1":..,"X2":..,"Y":..},
                  loc_filename="JS_24.06_10-02_JS_synth", cue_type="synth",
                  note="raised low-band scaling")
    show_log("JS")
    finalize("JS", final_loc_filename="JS_24.06_10-31_JS_synth")            # dry run
    finalize("JS", final_loc_filename="JS_24.06_10-31_JS_synth", apply=True)

CLI:
    python cue_tuning_log.py show     --subject JS
    python cue_tuning_log.py log      --subject JS --loc <fname> --cue synth --params '{"X1":1,"X2":2,"Y":3}'
    python cue_tuning_log.py finalize --subject JS --final <fname> [--apply]
"""

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

# allow running directly even if the package isn't pip-installed in this env:
# walk up to the repo root (the dir that contains the hrtf_relearning package)
for _p in Path(__file__).resolve().parents:
    if (_p / "hrtf_relearning" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
import hrtf_relearning as hr

LOG_DIR     = hr.PATH / "data" / "documentation" / "cue_tuning"
ARCHIVE_DIR = hr.PATH / "data" / "results" / "archive"

# target starting-performance window the modification is tuned to hit
EG_TARGET   = (0.2, 0.4)
RMSE_TARGET = (15.0, 25.0)


def _log_path(subject_id):
    return LOG_DIR / f"{subject_id}.json"


def _read_log(subject_id):
    p = _log_path(subject_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"subject": subject_id, "iterations": []}


def _write_log(subject_id, log):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_path(subject_id).write_text(json.dumps(log, indent=2))


def _in(val, lo_hi):
    return lo_hi[0] <= val <= lo_hi[1]


def log_iteration(subject_id, params, loc_filename, cue_type=None, note=""):
    """Append one tuning iteration; metrics are computed from the named loc run."""
    subject = hr.Subject(subject_id)
    if loc_filename not in subject.localization:
        raise KeyError(
            f"'{loc_filename}' not in {subject_id}.localization. "
            f"Available: {list(subject.localization)[-5:]} ...")
    seq = subject.localization[loc_filename]
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = hr.localization_accuracy(seq)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cue_type": cue_type,
        "params": params,
        "loc_filename": loc_filename,
        "n_trials": len(getattr(seq, "data", []) or []),
        "elevation_gain": None if eg is None else round(float(eg), 3),
        "ele_rmse": None if ele_rmse is None else round(float(ele_rmse), 2),
        "ele_sd": None if ele_sd is None else round(float(ele_sd), 2),
        "note": note,
        "final": False,
    }
    log = _read_log(subject_id)
    log["iterations"].append(entry)
    _write_log(subject_id, log)

    eg_ok = entry["elevation_gain"] is not None and _in(entry["elevation_gain"], EG_TARGET)
    rmse_ok = entry["ele_rmse"] is not None and _in(entry["ele_rmse"], RMSE_TARGET)
    flag = "IN TARGET" if (eg_ok and rmse_ok) else "out of target"
    print(f"logged iter #{len(log['iterations'])} for {subject_id}: "
          f"EG={entry['elevation_gain']} RMSE={entry['ele_rmse']} SD={entry['ele_sd']}  [{flag}]")
    return entry


def show_log(subject_id):
    log = _read_log(subject_id)
    its = log["iterations"]
    print(f"\nCue-tuning log for {subject_id}  ({len(its)} iterations)")
    print(f"  target: EG {EG_TARGET}, RMSE {RMSE_TARGET}")
    print(f"  {'#':>2}  {'when':<16} {'cue':<6} {'EG':>5} {'RMSE':>6} {'SD':>5}  final  loc / note")
    for i, e in enumerate(its, 1):
        mark = "  *  " if e.get("final") else "     "
        print(f"  {i:>2}  {e['timestamp'][:16]:<16} {str(e['cue_type']):<6} "
              f"{str(e['elevation_gain']):>5} {str(e['ele_rmse']):>6} {str(e['ele_sd']):>5} {mark} "
              f"{e['loc_filename']}  {e.get('note','')}")
    print()


def finalize(subject_id, final_loc_filename, apply=False):
    """Mark the chosen iteration final; archive the intermediate baseline loc runs.

    Keeps only `final_loc_filename` among the logged tuning runs in
    subject.localization; the other logged runs are moved to
    data/results/archive/<id>_tuning_archive.pkl (NOT deleted).
    Non-logged runs (native, dome, daily, final tests) are untouched.
    """
    log = _read_log(subject_id)
    logged = [e["loc_filename"] for e in log["iterations"]]
    if final_loc_filename not in logged:
        raise KeyError(f"'{final_loc_filename}' is not a logged tuning iteration for {subject_id}.")

    intermediates = [f for f in logged if f != final_loc_filename]
    subject = hr.Subject(subject_id)
    present = [f for f in intermediates if f in subject.localization]

    print(f"{'APPLY' if apply else 'DRY RUN'} — finalize {subject_id}")
    print(f"  keep final     : {final_loc_filename}")
    print(f"  archive ({len(present)}) : {present}")
    missing = set(intermediates) - set(present)
    if missing:
        print(f"  (already absent from record: {missing})")

    # mark final flag in the log regardless
    for e in log["iterations"]:
        e["final"] = (e["loc_filename"] == final_loc_filename)

    if not apply:
        print("  DRY RUN — re-run with apply=True to back up, archive, and rewrite the subject.")
        _write_log(subject_id, log)   # flag update is safe/non-destructive
        return

    # back up subject pickle
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = hr.PATH / "data" / "results" / f"{subject_id}.pkl"
    shutil.copy2(src, src.with_name(f"{subject_id}_preFinalize_{ts}.pkl"))

    # load/append archive
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    arch_path = ARCHIVE_DIR / f"{subject_id}_tuning_archive.pkl"
    archive = pickle.loads(arch_path.read_bytes()) if arch_path.exists() else {}
    for f in present:
        archive[f] = subject.localization.pop(f)
    arch_path.write_bytes(pickle.dumps(archive))
    subject.write()

    _write_log(subject_id, log)
    print(f"  archived -> {arch_path}")
    print(f"  subject rewritten; {len(present)} intermediate runs removed from record.")


# ----------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description="Per-subject cue-tuning log.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("log"); s.add_argument("--subject", required=True)
    s.add_argument("--loc", required=True); s.add_argument("--cue", default=None)
    s.add_argument("--params", required=True, help="JSON dict of modification params")
    s.add_argument("--note", default="")

    s = sub.add_parser("show"); s.add_argument("--subject", required=True)

    s = sub.add_parser("finalize"); s.add_argument("--subject", required=True)
    s.add_argument("--final", required=True); s.add_argument("--apply", action="store_true")

    a = ap.parse_args()
    if a.cmd == "log":
        log_iteration(a.subject, json.loads(a.params), a.loc, cue_type=a.cue, note=a.note)
    elif a.cmd == "show":
        show_log(a.subject)
    elif a.cmd == "finalize":
        finalize(a.subject, a.final, apply=a.apply)


if __name__ == "__main__":
    _cli()
