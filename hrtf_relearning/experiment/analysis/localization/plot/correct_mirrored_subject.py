import numpy

def _as_tuple2(x):
    """Convert range-like to canonical (float,float) or None."""
    if x is None:
        return None
    try:
        return (float(x[0]), float(x[1]))
    except Exception:
        return None


def fix_mirrored_az_for_subject(
    subject_id: str,
    *,
    az_pair_a=(-35.0, 0.0),
    az_pair_b=(0.0, 35.0),
    dry_run: bool = True,
    verbose: bool = True,
):
    """
    Fix ONE subject: for mirrored Trialsequences, switch azimuth_range labeling
    (-35,0) <-> (0,35) AND flip az sign in targets (seq.data), conditions, sector_centers.
    Then write back via subject.write() unless dry_run=True.

    Assumptions based on your pipeline:
      - mirrored is detected via 'mirrored' in seq.hrir
      - targets are stored in seq.data as [response, target] pairs
      - seq.data can be reshaped to (n_trials, 2, 2) = (trial, resp/targ, az/el)
      - az flip is simply az *= -1
    """
    subj = hr.Subject(subject_id)

    # If your subject uses a different attribute name than `.localization`,
    # change this line accordingly.
    loc_dict = getattr(subj, "localization", None)
    if loc_dict is None:
        raise AttributeError("Subject has no `.localization` attribute (adjust for your project).")

    swap = {tuple(az_pair_a): tuple(az_pair_b), tuple(az_pair_b): tuple(az_pair_a)}

    n_total = 0
    n_changed = 0
    changed_keys = []

    for key, seq in loc_dict.items():
        n_total += 1

        hrir = (getattr(seq, "hrir", "") or "").lower()
        is_mirrored = "mirrored" in hrir
        if not is_mirrored:
            continue

        settings = getattr(seq, "settings", None)
        if not isinstance(settings, dict):
            continue

        az0 = _as_tuple2(settings.get("azimuth_range", None))
        if az0 is None or az0 not in swap:
            continue  # not one of the ranges we want to swap

        az_new = swap[az0]

        # ---- Apply fixes ----
        if verbose:
            print(f"\n[{subject_id}] fixing {key}")
            print(f"  hrir: {getattr(seq, 'hrir', None)}")
            print(f"  azimuth_range: {az0} -> {az_new}")

        # (1) azimuth_range relabel
        settings["azimuth_range"] = az_new

        # (2) sector_centers: flip az sign
        if "sector_centers" in settings and settings["sector_centers"] is not None:
            centers = numpy.asarray(settings["sector_centers"], dtype=float)  # (N,2)
            centers[:, 0] *= -1.0
            settings["sector_centers"] = [tuple(x) for x in centers.tolist()]
            if verbose:
                print("  sector_centers: az sign flipped")

        # (3) seq.conditions: flip az sign if present
        if hasattr(seq, "conditions") and getattr(seq, "conditions") is not None:
            try:
                cond = numpy.asarray(seq.conditions, dtype=float)
                if cond.ndim == 2 and cond.shape[1] >= 1:
                    cond[:, 0] *= -1.0
                    # write back preserving type
                    seq.conditions = cond
                    if verbose:
                        print("  conditions: az sign flipped")
            except Exception as e:
                if verbose:
                    print(f"  conditions: could not flip ({e})")

        # (4) seq.data targets: flip az sign in TARGETS only
        if getattr(seq, "data", None):
            try:
                loc_data = numpy.asarray(seq.data, dtype=float)
                loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)

                # targets are the SECOND row (index 1), azimuth is column 0
                loc_data[:, 1, 0] *= -1.0

                # write back with same structure as before
                # (list of trials; each trial: [response[2], target[2]])
                seq.data = loc_data.reshape(loc_data.shape[0], 2, 2).tolist()

                if verbose:
                    print("  data targets: az sign flipped (responses unchanged)")
            except Exception as e:
                if verbose:
                    print(f"  data targets: could not flip ({e})")

        n_changed += 1
        changed_keys.append(key)

        if hasattr(seq, "name") and isinstance(seq.name, str) and seq.name != key and verbose:
            print(f"  note: seq.name is {seq.name} (dict key is {key})")

    if verbose:
        print(f"\n[{subject_id}] scanned {n_total} sequences.")
        print(f"[{subject_id}] changed {n_changed} mirrored sequences with az in {list(swap.keys())}.")
        if changed_keys:
            print("Changed keys:")
            for k in changed_keys:
                print("  -", k)

    if not dry_run:
        subj.write()
        if verbose:
            print(f"[{subject_id}] subject.write() done.")
    else:
        if verbose:
            print(f"[{subject_id}] dry_run=True -> not writing. Set dry_run=False to save.")

    return subj, changed_keys

subj, changed = fix_mirrored_az_for_subject("RK", dry_run=True, verbose=True)
