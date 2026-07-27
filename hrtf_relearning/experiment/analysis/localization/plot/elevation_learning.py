import datetime
import re
from collections import OrderedDict

import numpy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import hrtf_relearning as hr
from hrtf_relearning.experiment.analysis.localization.localization_analysis import _azimuth_span
from hrtf_relearning.utils import paths


subject_id = "CA"


def learning_plot(
    subject_id,
    *,
    last_day_width=0.15,
    other_day_width=0.12,
    annotate_times=True,
    exclude_dome=True,
    exclude_midline=True,
    show_dome_baseline=True,
    split_final_transfer=True,
):
    """Plot learning metrics across days for one subject.

    By default only non-midline AR localization tests are shown: dome tests
    (free-field, hrir is None) and midline tests (azimuth span <= 2 deg, e.g.
    azimuth_range=(-1, 1)) are preparation/reference runs, not part of the
    learning curve. Set exclude_dome/exclude_midline to False to include them.

    With show_dome_baseline the *first* finished dome test is drawn as a
    faint, unconnected point left of day 1 — real-ear (free-field) baseline
    against which the AR learning curve can be compared.

    With split_final_transfer the final-day 2x2 (Ear x Side) transfer block is
    pulled off the daily axis: block A (trained ear, same side) stays as the
    curve's endpoint, while B/C/D (the transfer conditions) are shown side by
    side in a labelled cluster to the right, and the own-HRTF post test is
    drawn as a faint reference (like the dome baseline).
    """
    figsize_cm = (17.5, 6.5)
    fig_size = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    dpi = 264
    fs = 8
    lw = 0.7
    markersize = 2.5

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "axes.labelsize": fs,
            "lines.linewidth": lw,
            "ytick.direction": "in",
            "xtick.direction": "in",
            "ytick.major.size": 2,
            "xtick.major.size": 2,
            "axes.linewidth": lw,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )

    localization_dict = hr.Subject(subject_id).localization

    old_key = "AvS_1_KU100_loc_18.12_15.04"
    new_key = "AvS_KU100_loc_18.12_15.04"
    if old_key in localization_dict:
        localization_dict[new_key] = localization_dict.pop(old_key)

    items = sorted(localization_dict.items(), key=lambda kv: parse_loc_key(kv[0]))

    by_day = OrderedDict()
    for key, seq in items:
        if not _is_analysable(seq):
            continue  # aborted / incomplete run
        if exclude_dome and not getattr(seq, "hrir", None):
            continue  # dome (free-field) test — no HRIR
        if exclude_midline and _azimuth_span(seq) <= 2:
            continue  # midline test, e.g. azimuth_range=(-1, 1)
        day = extract_day(key)
        by_day.setdefault(day, []).append((key, seq))

    # First complete dome test = real-ear baseline (items are chronological)
    dome_baseline = None
    if show_dome_baseline:
        for key, seq in items:
            if _is_analysable(seq) and not getattr(seq, "hrir", None):
                dome_baseline = (key, numpy.asarray(hr.localization_accuracy(seq)))
                break

    # Which hemifield was trained (to place the final 2x2 relative to it).
    all_meta = [extract_seq_meta(s, fallback_name=k)
                for day in by_day.values() for k, s in day]
    trained_field = trained_hemifield(all_meta)

    # Pull the final-day transfer block (B/C/D) and the own-HRTF post test off
    # the daily axis. Block A (trained ear, same side) stays on the curve.
    transfer_by_block = {}   # 'B'/'C'/'D' -> (meta, metrics)
    post_baseline = None     # (key, metrics) own-HRTF final test
    if split_final_transfer and by_day:
        final_day = next(reversed(by_day))
        kept = []
        for key, seq in by_day[final_day]:
            meta = extract_seq_meta(seq, fallback_name=key)
            metrics = numpy.asarray(hr.localization_accuracy(seq))
            block = transfer_block(meta, trained_field)
            if block in ("B", "C", "D"):
                transfer_by_block[block] = (meta, metrics)      # last wins
            elif condition_of_meta(meta) == "baseline":
                post_baseline = (key, metrics)                  # faint reference
            else:
                kept.append((key, seq))                          # block A -> curve
        if kept:
            by_day[final_day] = kept
        else:
            del by_day[final_day]

    data_by_day = []
    times_by_day = []
    meta_by_day = []

    for _, loc_tests in by_day.items():
        loc_tests = sorted(loc_tests, key=lambda x: parse_loc_key(x[0]))

        day_data = []
        day_times = []
        day_meta = []

        for key, seq in loc_tests:
            day_data.append(hr.localization_accuracy(seq))
            day_times.append(key_time_str(key))
            day_meta.append(extract_seq_meta(seq, fallback_name=key))

        data_by_day.append(numpy.vstack(day_data))
        times_by_day.append(day_times)
        meta_by_day.append(day_meta)

    fig = plt.figure(figsize=fig_size, constrained_layout=True, dpi=dpi)
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    ax2 = plt.subplot2grid((2, 3), (1, 2))
    axes = [ax0, ax1, ax2]

    labels = ["Elevation Gain", "RMSE (deg)", "SD (deg)"]
    days = numpy.arange(1, len(data_by_day) + 1)

    def x_positions(day_idx, n_points):
        center = float(day_idx + 1)
        if n_points <= 1:
            return numpy.array([center], dtype=float)
        width = last_day_width if day_idx == len(data_by_day) - 1 else other_day_width
        return numpy.linspace(center - width, center + width, n_points)

    dome_x = 0.45  # left of day 1, clearly separate from the curve

    # --- geometry of the final-transfer cluster (right of the last day) ---
    # D (mirrored ear, mirrored side) is the MAIN transfer test — same ear-as-
    # field as the trained condition A, physically identical to A — so it sits
    # first, adjacent to the curve, at full prominence. B and C are the crossed
    # (ear != side) counterbalance conditions and are drawn faint.
    n_days = len(data_by_day)
    transfer_order = [b for b in ("D", "B", "C") if b in transfer_by_block]
    main_blocks = {"D"}
    divider_x = n_days + 0.55
    transfer_x = {b: n_days + 0.9 + 0.35 * i for i, b in enumerate(transfer_order)}
    postbase_x = (max(transfer_x.values()) + 0.45) if transfer_x else (n_days + 0.9)
    right_x = (postbase_x + 0.35 if post_baseline is not None
               else (max(transfer_x.values()) + 0.35 if transfer_x else n_days + 0.5))
    block_label = {"B": "B", "C": "C", "D": "D"}  # 2x2 Ear x Side block letters

    for metric_idx, axis in enumerate(axes):
        if dome_baseline is not None:
            dome_key, dome_metrics = dome_baseline
            axis.plot(
                [dome_x],
                [dome_metrics[metric_idx]],
                marker="o",
                markersize=markersize,
                color="0.65",
                linestyle="None",
                zorder=1,
            )

        # Connect only the trained condition (block A: trained ear, trained
        # side), chronologically — solid within a day, dotted across the day
        # break. Baseline / mirrored / opposite-side points are drawn but never
        # joined. Falls back to any learning point if the trained side is
        # unknown (e.g. no shifted tests to infer it from).
        def _on_curve(meta):
            if trained_field is None:
                return is_learning_point(meta)
            return transfer_block(meta, trained_field) == "A"

        le_pts = []  # (day_idx, x, y)
        for day_idx, day_data in enumerate(data_by_day):
            xs = x_positions(day_idx, day_data.shape[0])
            for point_idx, meta in enumerate(meta_by_day[day_idx]):
                if _on_curve(meta):
                    le_pts.append((day_idx, xs[point_idx], day_data[point_idx, metric_idx]))
        for (d0, xa, ya), (d1, xb, yb) in zip(le_pts, le_pts[1:]):
            if d0 == d1:
                axis.plot([xa, xb], [ya, yb], c="0", linewidth=lw, zorder=1)
            else:
                axis.plot([xa, xb], [ya, yb], linestyle=":", color="0.5",
                          linewidth=1, zorder=0)

        for day_idx, day_data in enumerate(data_by_day):
            x = x_positions(day_idx, day_data.shape[0])
            y = day_data[:, metric_idx]

            for point_idx, meta in enumerate(meta_by_day[day_idx]):
                open_marker = condition_of_meta(meta) == "baseline"
                axis.plot(
                    [x[point_idx]],
                    [y[point_idx]],
                    marker=marker_for_meta(meta),
                    markersize=markersize,
                    markerfacecolor="none" if open_marker else "0",
                    markeredgecolor="0",
                    color="0",
                    linestyle="None",
                    zorder=2,
                )

        # --- final-transfer cluster (blocks B/C/D) + faint own-HRTF post test ---
        if transfer_order or post_baseline is not None:
            axis.axvline(divider_x, color="0.8", linewidth=0.6, zorder=-1)
        for b in transfer_order:
            meta, metrics = transfer_by_block[b]
            is_main = b in main_blocks
            edge = "0" if is_main else "0.6"       # counterbalance drawn faint
            ms = markersize if is_main else markersize * 0.9
            axis.plot(
                [transfer_x[b]], [metrics[metric_idx]],
                marker=marker_for_meta(meta), markersize=ms,
                markerfacecolor=edge, markeredgecolor=edge, color=edge,
                linestyle="None", zorder=2 if is_main else 1,
            )
        if post_baseline is not None:
            axis.plot(
                [postbase_x], [post_baseline[1][metric_idx]],
                marker="D", markersize=markersize, markerfacecolor="none",
                markeredgecolor="0.65", color="0.65", linestyle="None", zorder=1,
            )

        xticks = list(days)
        xticklabels = [str(d) for d in days]
        for b in transfer_order:
            xticks.append(transfer_x[b])
            xticklabels.append(block_label[b])
        if post_baseline is not None:
            xticks.append(postbase_x)
            xticklabels.append("post")
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)
        axis.set_xlim(dome_x - 0.35, right_x)
        axis.set_ylabel(labels[metric_idx])

    ax1.set_xticklabels([])
    ax0.set_xlabel("Days")
    ax2.set_xlabel("Days")

    ax0.set_ylim(0, 1.02)
    ax0.set_yticks(numpy.arange(0, 1.2, 0.2))
    ax1.set_yticks(numpy.arange(0, 26, 5))
    ax2.set_yticks(numpy.arange(0, 10, 2))

    for y in numpy.linspace(0.1, 1, 9):
        ax0.axhline(y=y, color="0.9", linewidth=0.5, zorder=-1)
    for y in numpy.arange(5, 22, 5):
        ax1.axhline(y=y, color="0.9", linewidth=0.5, zorder=-1)
    for y in numpy.arange(2, 9, 2):
        ax2.axhline(y=y, color="0.9", linewidth=0.5, zorder=-1)

    ax0.annotate("A", xy=(-0.1, 1.005), xycoords="axes fraction", fontsize=fs, weight="bold")
    ax1.annotate("B", xy=(-0.3, 1.005), xycoords="axes fraction", fontsize=fs, weight="bold")
    ax2.annotate("C", xy=(-0.3, 1.005), xycoords="axes fraction", fontsize=fs, weight="bold")

    # Legend for the marker scheme, built only from conditions actually present
    # (replaces the per-point time/condition text annotations).
    present = set()
    has_uso = False
    for day_meta in meta_by_day:
        for meta in day_meta:
            present.add(condition_of_meta(meta))
            if (meta.get("stim") or "").lower() == "uso":
                has_uso = True

    def _handle(marker, label, open_marker=False, color="0"):
        return Line2D([0], [0], marker=marker, linestyle="None", markersize=markersize,
                      markerfacecolor="none" if open_marker else color,
                      markeredgecolor=color, label=label)

    handles = []
    if "learning" in present:
        handles.append(_handle("o", "Learning (shifted, LE)"))
    if "baseline" in present:
        handles.append(_handle("D", "Baseline (own HRTF, BL)", open_marker=True))
    if "mirrored" in present:
        handles.append(_handle("^", "Mirrored (untrained ear, RE)"))
    if has_uso:
        handles.append(_handle("s", "USO probe"))
    if dome_baseline is not None:
        handles.append(_handle("o", "Dome (real ear)", color="0.65"))
    if transfer_order:
        handles.append(Line2D([0], [0], marker="None", linestyle="None",
                              label="Final 2x2 (D main; B,C counterbalance)"))
    if post_baseline is not None:
        handles.append(_handle("D", "Post (own HRTF)", open_marker=True, color="0.65"))

    if handles:
        ax0.legend(handles=handles, frameon=False, fontsize=7, loc="lower right",
                   handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)

    hrir_names = []
    for day_meta in meta_by_day:
        for meta in day_meta:
            hrir_name = meta.get("hrir")
            if hrir_name and hrir_name not in hrir_names:
                hrir_names.append(hrir_name)

    hrir_label = hrir_names[1] if len(hrir_names) > 1 else (hrir_names[0] if hrir_names else None)

    if hrir_names:
        fig.suptitle(f"Subject {subject_id} | HRIR: {hrir_label}")
    else:
        fig.suptitle(f"Subject {subject_id}")

    plt.tight_layout(pad=1.08, h_pad=0.5)
    plt.show()
    return hrir_label, fig, axes


def parse_loc_key(key):
    """Parse keys like SK_13.02_14:08, SK_13.02_14.08 or SS_13.07_14-01_dome.

    The time separator varies across sessions (':', '.', '-') and newer keys
    carry a suffix after the time (e.g. '_dome', '_SS_shift_left'), so the
    time must not be end-anchored. Keys that don't parse sort last
    (datetime.max) — with '-' unhandled this silently applied to *all* keys
    of a session, leaving chronological order to dict insertion order.
    """
    try:
        return datetime.datetime.fromisoformat(key)
    except Exception:
        pass

    match = re.search(r"(\d{2})\.(\d{2})_(\d{2})[-:.](\d{2})", key)
    if match:
        day, month, hour, minute = match.groups()
        now = datetime.datetime.now()
        year = now.year
        if int(month) > now.month + 1:
            year -= 1
        return datetime.datetime(year, int(month), int(day), int(hour), int(minute))

    return datetime.datetime.max


def extract_day(key):
    """Return the DD.MM date substring from a localization key.

    Keys are inconsistently formatted across subjects/sessions — some are
    'SID_DD.MM_HH-MM_...', others 'SIDDD.MM_HH:MM' (no separating
    underscore), others have extra tokens before the date
    (e.g. 'AvS_KU100_loc_DD.MM_HH.MM'). A plain key.split('_')[1][-5:]
    picks up the wrong token (often a time, not a date) for the latter
    formats, which silently mis-groups sessions by day. Search for the
    DD.MM pattern directly instead, since it's the one token whose shape
    (two digits, a literal dot, two digits) is unambiguous regardless of
    surrounding formatting.
    """
    match = re.search(r"\d{2}\.\d{2}", key)
    if match:
        return match.group(0)
    return key.split("_")[1][-5:]


def key_time_str(key):
    """Return HH:MM from a localization key.

    Matches the time token following the DD.MM date rather than anchoring at
    end-of-string, since keys may end in a condition suffix
    (e.g. SS_13.07_14-01_dome). Accepts ':', '.' or '-' as separator.
    """
    match = re.search(r"\d{2}\.\d{2}_(\d{2})[-:.](\d{2})", key)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return ""


def extract_seq_meta(seq, fallback_name=""):
    """Extract only the fields needed for plotting."""
    settings = getattr(seq, "settings", {})
    if not isinstance(settings, dict):
        settings = {}

    return {
        "name": getattr(seq, "name", fallback_name),
        "stim": getattr(seq, "stim", None),
        "hrir": getattr(seq, "hrir", None),
        "ear": getattr(seq, "ear", None),
        "azimuth_range": settings.get("azimuth_range"),
        "elevation_range": settings.get("elevation_range"),
    }


def _is_analysable(seq):
    """True for a complete, plottable localization run.

    A run counts as analysable when slab reports it finished (n_remaining==-1)
    *and* its recorded data reshapes to whole [response, target] x [az, el]
    pairs. Aborted runs are padded to full length with empty trials, so the
    finished flag alone is not enough — e.g. FD_21.07_11-52 (1 response) and
    FD_22.07_10-35 (0 responses) are flagged unfinished, but the reshape guard
    also rejects any future finished-but-degenerate run. Note a *complete* run
    with no saved PNG on disk (e.g. FD_21.07_12-13) is still analysable — a
    missing plot file does not mean the test didn't run.
    """
    if not bool(getattr(seq, "finished", False)):
        return False
    data = getattr(seq, "data", None)
    if not data:
        return False
    try:
        arr = numpy.asarray(data, dtype=float).reshape(len(data), 2, 2)
    except (ValueError, TypeError):
        return False
    return bool(numpy.all(numpy.isfinite(arr)))


def condition_of_meta(meta):
    """Classify a test by its HRIR into one of three learning conditions.

    - 'mirrored' : mirrored HRTF, probes the untrained (right) ear -> RE
    - 'learning' : shifted/modified HRTF, the trained (left) ear -> LE
    - 'baseline' : unmodified own HRTF, the pre/post reference -> BL

    The previous logic labelled *everything* that wasn't mirrored as 'LE',
    so baseline (unmodified) runs were wrongly tagged LE. Baseline is neither
    a learning nor a mirrored condition and gets its own label/marker here.
    """
    hrir = (meta.get("hrir") or "").lower()
    if "mirrored" in hrir:
        return "mirrored"
    if "shift" in hrir or "modified" in hrir:
        return "learning"
    return "baseline"


def is_learning_point(meta):
    """True for the LE learning-trace points that should be connected.

    Only the shifted-HRTF learning condition forms the continuous trace;
    baseline and mirrored points are shown but left unconnected. USO probes
    are excluded even on a shifted HRTF."""
    return (condition_of_meta(meta) == "learning"
            and (meta.get("stim") or "").lower() != "uso")


def field_of_meta(meta):
    """Tested hemifield 'LF'/'RF' from the azimuth range, or None (midline)."""
    az = meta.get("azimuth_range")
    if az is None:
        return None
    try:
        a, b = float(az[0]), float(az[1])
    except Exception:
        return None
    mid = 0.5 * (a + b)
    if mid < 0:
        return "LF"
    if mid > 0:
        return "RF"
    return None


def trained_hemifield(metas):
    """The hemifield the subject was trained in = the modal field of the LE
    learning-trace tests. Used to place each final test in the 2x2 (Ear x Side)
    design relative to what was trained."""
    from collections import Counter
    fields = [field_of_meta(m) for m in metas if is_learning_point(m)]
    fields = [f for f in fields if f]
    return Counter(fields).most_common(1)[0][0] if fields else None


def transfer_block(meta, trained_field):
    """Classify a test in the final 2x2 (Ear x Side) transfer design, or None.

    A = trained ear, same side  -> the trained condition; forms the curve.
    B = trained ear, opposite side.
    C = untrained (mirrored) ear, same side.
    D = untrained (mirrored) ear, opposite side (main transfer condition).

    Ear = trained when not mirrored (the cue filter is always the trained ear's
    own filter; mirroring puts it on the other ear). Side is the raw tested
    hemifield vs the trained hemifield. Baseline (own-HRTF) and USO return None.
    """
    if (meta.get("stim") or "").lower() == "uso":
        return None
    cond = condition_of_meta(meta)
    if cond == "baseline":
        return None
    field = field_of_meta(meta)
    if field is None or trained_field is None:
        return None
    trained_ear = cond == "learning"          # not mirrored
    same_side = field == trained_field
    if trained_ear and same_side:
        return "A"
    if trained_ear and not same_side:
        return "B"
    if not trained_ear and same_side:
        return "C"
    return "D"


def marker_for_meta(meta):
    """Marker by condition. Baseline and modified are deliberately different."""
    stim = (meta.get("stim") or "").lower()
    if stim == "uso":
        return "s"
    return {"baseline": "D", "learning": "o", "mirrored": "^"}[condition_of_meta(meta)]


def flags_for_meta(meta):
    """Annotation label: USO + condition (BL/LE/RE) + field."""
    stim = (meta.get("stim") or "").lower()
    az = meta.get("azimuth_range")
    cond = condition_of_meta(meta)

    flags = []

    if stim == "uso":
        flags.append("USO")

    flags.append({"baseline": "BL", "learning": "LE", "mirrored": "RE"}[cond])

    if az is not None:
        try:
            a = float(az[0])
            b = float(az[1])
            if b <= 0 and a < 0:
                flags.append("LF")
            elif a >= 0 and b > 0:
                flags.append("RF")
            else:
                mid = 0.5 * (a + b)
                if mid < 0:
                    flags.append("LF")
                elif mid > 0:
                    flags.append("RF")
        except Exception:
            pass

    return " ".join(flags)


if __name__ == "__main__":
    hrir_name, fig, axes = learning_plot(subject_id, annotate_times=True)
    plt.savefig(paths.subject_plot_dir(subject_id) / f'learning_plot.svg')
    import slab
    h = slab.HRTF(paths.SOFA_DIR / subject_id / str(subject_id+'_notch.sofa'))
    h.plot_tf(h.cone_sources(0), ear='left')
    plt.title(f"{hrir_name}")
    # plt.savefig(paths.subject_plot_dir(subject_id) / f"{hrir_name}")