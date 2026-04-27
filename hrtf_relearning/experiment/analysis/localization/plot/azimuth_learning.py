import datetime
import re
from collections import OrderedDict

import numpy
from matplotlib import pyplot as plt

import hrtf_relearning as hr


subject_id = "JP"


def learning_plot(
    subject_id,
    *,
    last_day_width=0.75,
    other_day_width=0.22,
    annotate_times=True,
):
    """Plot learning metrics across days for one subject."""
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
        if not getattr(seq, "finished", False):
            continue
        day = key.split("_")[1][-5:]
        by_day.setdefault(day, []).append((key, seq))

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

    labels = ["Azimuth Gain", "RMSE (deg)", "SD (deg)"]
    days = numpy.arange(1, len(data_by_day) + 1)

    def x_positions(day_idx, n_points):
        center = float(day_idx + 1)
        if n_points <= 1:
            return numpy.array([center], dtype=float)
        width = last_day_width if day_idx == len(data_by_day) - 1 else other_day_width
        return numpy.linspace(center - width, center + width, n_points)

    for metric_idx, axis in enumerate(axes):
        metric_idx = metric_idx + 2
        for day_idx, day_data in enumerate(data_by_day):
            x = x_positions(day_idx, day_data.shape[0])
            y = day_data[:, metric_idx]

            axis.plot(x, y, c="0", zorder=1)

            for point_idx, meta in enumerate(meta_by_day[day_idx]):
                axis.plot(
                    [x[point_idx]],
                    [y[point_idx]],
                    marker=marker_for_meta(meta),
                    markersize=markersize,
                    color="0",
                    linestyle="None",
                    zorder=2,
                )

            if metric_idx == 2:
                for point_idx, meta in enumerate(meta_by_day[day_idx]):
                    parts = []
                    if annotate_times:
                        time_str = times_by_day[day_idx][point_idx]
                        if time_str:
                            parts.append(time_str)
                    flag = flags_for_meta(meta)
                    if flag:
                        parts.append(flag)
                    if not parts:
                        continue

                    axis.annotate(
                        "\n".join(parts),
                        (x[point_idx], y[point_idx]),
                        textcoords="offset points",
                        xytext=(3, 3),
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        color="0.3",
                    )

        for day_idx in range(len(data_by_day) - 1):
            x0 = x_positions(day_idx, data_by_day[day_idx].shape[0])[-1]
            y0 = data_by_day[day_idx][-1, metric_idx-2]
            x1 = x_positions(day_idx + 1, data_by_day[day_idx + 1].shape[0])[0]
            y1 = data_by_day[day_idx + 1][0, metric_idx-2]

            axis.plot(
                [x0, x1],
                [y0, y1],
                linestyle=":",
                color="0.5",
                linewidth=1,
                zorder=0,
            )

        axis.set_xticks(days)
        axis.set_xticklabels(days)
        axis.set_ylabel(labels[metric_idx-2])

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

    hrir_names = []
    for day_meta in meta_by_day:
        for meta in day_meta:
            hrir_name = meta.get("hrir")
            if hrir_name and hrir_name not in hrir_names:
                hrir_names.append(hrir_name)

    if hrir_names:
        fig.suptitle(f"Subject {subject_id} | HRIR: {hrir_names[1]}")
    else:
        fig.suptitle(f"Subject {subject_id}")

    plt.tight_layout(pad=1.08, h_pad=0.5)
    plt.show()
    return hrir_names[1], fig, axes


def parse_loc_key(key):
    """Parse keys like SK_13.02_14:08 or SK_13.02_14.08."""
    try:
        return datetime.datetime.fromisoformat(key)
    except Exception:
        pass

    match = re.search(r"(\d{2})\.(\d{2})_(\d{2})[:.](\d{2})", key)
    if match:
        day, month, hour, minute = match.groups()
        now = datetime.datetime.now()
        year = now.year
        if int(month) > now.month + 1:
            year -= 1
        return datetime.datetime(year, int(month), int(day), int(hour), int(minute))

    return datetime.datetime.max


def key_time_str(key):
    """Return HH:MM from a localization key."""
    match = re.search(r"(\d{2})[:.](\d{2})$", key)
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


def marker_for_meta(meta):
    """Marker by stimulus / mirrored state."""
    stim = (meta.get("stim") or "").lower()
    hrir = (meta.get("hrir") or "").lower()

    if stim == "uso":
        return "s"
    if "mirrored" in hrir:
        return "^"
    return "o"


def flags_for_meta(meta):
    """Annotation label: USO + ear + field."""
    stim = (meta.get("stim") or "").lower()
    hrir = (meta.get("hrir") or "").lower()
    az = meta.get("azimuth_range")

    flags = []

    if stim == "uso":
        flags.append("USO")

    flags.append("RE" if "mirrored" in hrir else "LE")

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

    import slab
    h = slab.HRTF(hr.PATH / 'data' / 'hrtf' / 'sofa' / str(subject_id+'_notch.sofa'))
    h.plot_tf(h.cone_sources(0), ear='left')
    plt.title(f"{hrir_name}")