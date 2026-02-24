import numpy
from matplotlib import pyplot as plt
from collections import OrderedDict
import hrtf_relearning as hr
import datetime
import re

subject_id = "RK"


def learning_plot(
    subject_id,
    *,
    last_day_width=0.75,      # --- CHANGED: more horizontal room for last day
    other_day_width=0.22,     # --- CHANGED: compact spacing for other days
    annotate_times=True,      # --- CHANGED
    annotate_az_range=True,   # --- CHANGED: mark with azimuth range
    annotate_flags=True,      # --- CHANGED: mark mirrored / uso
):
    """
    Plot single subject stimulus response pattern
    and draw SD RMS and EG indications
    """
    figsize = (17.5, 6.5)
    fig_width = figsize[0] / 2.54  # convert to inches
    fig_height = figsize[1] / 2.54
    dpi = 264
    fs = 8  # label fontsize
    markersize = 2
    lw = 0.7
    params = {
        "font.family": "Helvetica",
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.labelsize": fs,
        "boxplot.capprops.linewidth": lw,
        "lines.linewidth": lw,
        "ytick.direction": "in",
        "xtick.direction": "in",
        "ytick.major.size": 2,
        "xtick.major.size": 2,
        "axes.linewidth": lw,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
    plt.rcParams.update(params)

    # get data
    localization_dict = hr.Subject(subject_id).localization

    # correct keys
    old_key = "AvS_1_KU100_loc_18.12_15.04"
    new_key = "AvS_KU100_loc_18.12_15.04"
    if old_key in localization_dict:
        print(f"Renaming key {old_key} to {new_key}")
        localization_dict[new_key] = localization_dict.pop(old_key)

    # sort by day
    items = sorted(localization_dict.items(), key=lambda kv: parse_loc_key(kv[0]))
    by_day = OrderedDict()
    for k, seq in items:
        day = k.split("_")[0][-5:]  # "12.01"
        if getattr(seq, "finished", False):
            by_day.setdefault(day, []).append((k, seq))

    # pick all sequences
    data = []  # list over days, each element: (n_meas_that_day x n_metrics)
    times_by_day = []  # list over days, each element: list[str] length n_meas_that_day

    # --- CHANGED: also collect per-test metadata (stim/hrir/az_range) for marking
    meta_by_day = []  # list over days, each element: list[dict] length n_meas_that_day

    for day, loc_tests in by_day.items():
        _data = []
        _times = []
        _meta = []

        # ensure within-day order by time
        loc_tests_sorted = sorted(loc_tests, key=lambda x: parse_loc_key(x[0]))

        for k, seq in loc_tests_sorted:
            _data.append(hr.localization_accuracy(seq))  # -> (n_metrics,)
            _times.append(key_time_str(k))
            _meta.append(extract_seq_meta(seq, fallback_name=k))  # --- CHANGED

        data.append(numpy.vstack(_data))        # (n_meas, n_metrics)
        times_by_day.append(_times)             # list length n_meas
        meta_by_day.append(_meta)               # list length n_meas  # --- CHANGED

    # ----- plot ----- #
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True, dpi=dpi)
    ax0 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1)
    axes = fig.get_axes()

    labels = ["Elevation Gain", "RMSE (deg)", "SD (deg)"]
    days = numpy.arange(1, len(data) + 1)

    # --- CHANGED: helper to place points within each day
    def x_positions_for_day(day_index_0based: int, n_points: int) -> numpy.ndarray:
        """Return x positions centered on the day tick with controlled width."""
        day_center = float(day_index_0based + 1)
        if n_points <= 1:
            return numpy.array([day_center], dtype=float)

        width = other_day_width
        if day_index_0based == (len(data) - 1):  # last day
            width = last_day_width

        return numpy.linspace(day_center - width, day_center + width, n_points, dtype=float)

    for i, axis in enumerate(axes):
        label = labels[i]

        for day_idx, _data in enumerate(data):
            n = _data.shape[0]
            _x = x_positions_for_day(day_idx, n)  # --- CHANGED
            y = _data[:, i]

            # line within day
            axis.plot(_x, y, c="0", label=label, lw=lw, zorder=1)

            # --- CHANGED: markers encode flags (uso / mirrored)
            for j in range(n):
                m = meta_by_day[day_idx][j]
                marker = marker_for_meta(m)
                axis.plot(
                    [_x[j]],
                    [y[j]],
                    marker=marker,
                    markersize=markersize + 0.5,
                    color="0",
                    linestyle="None",
                    zorder=2,
                )

            # --- CHANGED: annotate timestamps + azimuth range + flags (only once, on first panel)
            if i == 0:
                for j in range(n):
                    t = times_by_day[day_idx][j] if annotate_times else ""
                    m = meta_by_day[day_idx][j]
                    az = m.get("azimuth_range", None)

                    parts = []
                    if t:
                        parts.append(t)
                    if annotate_az_range and az is not None:
                        parts.append(f"az {format_range(az)}")
                    if annotate_flags:
                        flag = flags_for_meta(m)
                        if flag:
                            parts.append(flag)

                    if not parts:
                        continue

                    axis.annotate(
                        "\n".join(parts),
                        (_x[j], y[j]),
                        textcoords="offset points",
                        xytext=(3, 3),
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        color="0.3",
                    )

        # dotted connections across day boundaries (last point of day -> first of next day)
        for day_idx in range(len(data) - 1):
            y0 = data[day_idx][-1, i]
            x0 = x_positions_for_day(day_idx, data[day_idx].shape[0])[-1]   # --- CHANGED
            y1 = data[day_idx + 1][0, i]
            x1 = x_positions_for_day(day_idx + 1, data[day_idx + 1].shape[0])[0]  # --- CHANGED

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
        axis.set_ylabel(label)

    axes[1].set_xticklabels([])
    axes[0].set_xlabel("Days")
    axes[2].set_xlabel("Days")

    axes[0].set_ylim(0, 1.02)
    axes[0].set_yticks(numpy.arange(0, 1.2, 0.2))

    ticklabels = [item.get_text() for item in axes[0].get_yticklabels()]
    ticklabels[0] = "0"
    ticklabels[-1] = "1"
    axes[0].set_yticklabels(ticklabels)

    axes[1].set_yticks(numpy.arange(0, 26, 5))
    axes[2].set_yticks(numpy.arange(0, 10, 2))

    # horizontal gridlines
    for y in numpy.linspace(0.1, 1, 9):
        axes[0].axhline(y=y, xmin=0, xmax=20, color="0.9", linewidth=0.5, zorder=-1)
    for y in numpy.arange(5, 22, 5):
        axes[1].axhline(y=y, xmin=0, xmax=20, color="0.9", linewidth=0.5, zorder=-1)
    for y in numpy.arange(2, 9, 2):
        axes[2].axhline(y=y, xmin=0, xmax=20, color="0.9", linewidth=0.5, zorder=-1)

    plt.tight_layout(pad=1.08, h_pad=0.5, w_pad=None, rect=None)

    # subplot labels
    axes[0].annotate("A", c="k", weight="bold", xycoords="axes fraction", xy=(-0.1, 1.005), fontsize=fs)
    axes[1].annotate("B", c="k", weight="bold", xycoords="axes fraction", xy=(-0.3, 1.005), fontsize=fs)
    axes[2].annotate("C", c="k", weight="bold", xycoords="axes fraction", xy=(-0.3, 1.005), fontsize=fs)

    fig.suptitle(f"Subject {subject_id}")
    plt.show()
    return fig, axes


# --- helpers ---
def parse_loc_key(key: str) -> datetime.datetime:
    """
    Parse localization dict keys and return a datetime that sorts correctly
    by day AND time.

    Supported anywhere in the key:
      - dd.mm_HH:MM
      - dd.mm_HH.MM
      - ISO (yyyy-mm-ddTHH:MM:SS)

    Unknown formats sort last.
    """
    # 1) ISO (future-proof)
    try:
        return datetime.datetime.fromisoformat(key)
    except Exception:
        pass

    # 2) dd.mm_HH[:.]MM anywhere in the string
    match = re.search(r"(\d{2})\.(\d{2})_(\d{2})[:.](\d{2})", key)
    if match:
        d, m, hh, mm = match.groups()
        now = datetime.datetime.now()

        year = now.year
        month = int(m)

        # Handle year rollover (Dec → Jan)
        if month > now.month + 1:
            year -= 1

        return datetime.datetime(year=year, month=month, day=int(d), hour=int(hh), minute=int(mm))

    # 3) Fallback → last
    return datetime.datetime.max


def key_time_str(k: str) -> str:
    # expects "..._dd.mm_HH.MM" or "..._dd.mm_HH:MM"
    # returns "HH:MM"
    parts = k.split("_")
    if len(parts) >= 2:
        t = parts[-1]
        t = t.replace(".", ":")
        m = re.search(r"(\d{2}:\d{2})", t)
        return m.group(1) if m else t
    return ""


# --- CHANGED: safely read Trialsequence attributes without loading anything extra
def extract_seq_meta(seq, fallback_name: str = "") -> dict:
    """
    Extract the attributes you care about from a Trialsequence-like object.
    Works even if some fields are missing.
    """
    meta = {}

    # common direct attributes
    meta["name"] = getattr(seq, "name", fallback_name)
    meta["stim"] = getattr(seq, "stim", None)
    meta["hrir"] = getattr(seq, "hrir", None)
    meta["ear"] = getattr(seq, "ear", None)

    # settings dict (where azimuth_range typically lives in your example)
    settings = getattr(seq, "settings", None)
    if isinstance(settings, dict):
        meta["azimuth_range"] = settings.get("azimuth_range", None)
        meta["elevation_range"] = settings.get("elevation_range", None)
    else:
        meta["azimuth_range"] = None
        meta["elevation_range"] = None

    return meta


# --- CHANGED
def marker_for_meta(meta: dict) -> str:
    """
    Encode 'uso' / mirrored / default in marker shape.
    (All in grayscale to match your current style.)
    """
    stim = (meta.get("stim") or "").lower()
    hrir = (meta.get("hrir") or "").lower()

    if stim == "uso":
        return "s"      # square for USO
    if "mirrored" in hrir:
        return "^"      # triangle for mirrored
    return "o"          # circle default


# --- CHANGED
def flags_for_meta(meta: dict) -> str:
    stim = (meta.get("stim") or "").lower()
    hrir = (meta.get("hrir") or "").lower()

    flags = []
    if stim == "uso":
        flags.append("USO")
    if "mirrored" in hrir:
        flags.append("mirr")
    return " ".join(flags)


# --- CHANGED
def format_range(rng) -> str:
    """
    Format (min,max) nicely even if it's a list/tuple/numpy array.
    """
    try:
        a = float(rng[0])
        b = float(rng[1])
        # drop trailing .0 when integer-ish
        def fmt(x):
            return str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x:g}"
        return f"[{fmt(a)},{fmt(b)}]"
    except Exception:
        return str(rng)


if __name__ == "__main__":
    learning_plot(subject_id)