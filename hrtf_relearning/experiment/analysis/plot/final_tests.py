import numpy
from matplotlib import pyplot as plt
import hrtf_relearning as hr
import datetime
import re

subject_id = "RK"

def normalize_az_range(az_range):
    """Return canonical (float, float) tuple or None."""
    if az_range is None:
        return None
    try:
        return (float(az_range[0]), float(az_range[1]))
    except Exception:
        return None


def switch_az_range_if_mirrored(az_range, is_mirrored):
    """
    If mirrored, relabel (-35,0) <-> (0,35) as requested.
    Only switches those exact canonical ranges.
    """
    az_range = normalize_az_range(az_range)
    if (not is_mirrored) or az_range is None:
        return az_range

    if az_range == (-35.0, 0.0):
        return (0.0, 35.0)
    if az_range == (0.0, 35.0):
        return (-35.0, 0.0)
    return az_range


def plot_last_four_matching(
    subject_id,
    *,
    metric_index=1,              # 0=EG, 1=RMSE, 2=SD (based on your labels)
    n_last=4,
    switch_mirrored_az=True,
    localization_dict=None,
    figsize=(7.5, 4.0),
    dpi=220,
):
    """
    Scatter plot ONLY the last `n_last` measurements (chronologically)
    that match any of these conditions:

      az (-35,0), not mirrored
      az (-35,0), mirrored
      az (0,35), not mirrored
      az (0,35), mirrored

    If switch_mirrored_az=True, mirrored sequences are re-labeled:
      (-35,0) -> (0,35) and (0,35) -> (-35,0)
    (This affects grouping/labels only.)
    """
    if localization_dict is None:
        localization_dict = hr.Subject(subject_id).localization

    allowed_az = {(-35.0, 0.0), (0.0, 35.0)}

    # Collect all matching tests with metadata + metric
    rows = []
    for k, seq in localization_dict.items():
        if not getattr(seq, "finished", False):
            continue

        dt = parse_loc_key(k)
        meta = extract_seq_meta(seq, fallback_name=k)

        is_mirrored = "mirrored" in (meta.get("hrir") or "").lower()
        az0 = normalize_az_range(meta.get("azimuth_range", None))
        if az0 is None:
            continue

        az_eff = switch_az_range_if_mirrored(az0, is_mirrored) if switch_mirrored_az else az0
        if az_eff not in allowed_az:
            continue  # not one of the two required az ranges

        # compute accuracy metrics for this test
        try:
            acc = hr.localization_accuracy(seq)  # (n_metrics,)
        except Exception:
            continue

        rows.append({
            "dt": dt,
            "key": k,
            "seq": seq,
            "meta": meta,
            "mirrored": is_mirrored,
            "az": az_eff,
            "acc": acc,
        })

    # Sort by time and keep only the last n that match requirements
    rows.sort(key=lambda r: r["dt"])
    rows = rows[-n_last:]

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    metric_labels = ["Elevation Gain", "RMSE (deg)", "SD (deg)"]
    ax.set_ylabel(metric_labels[metric_index])
    ax.set_title(
        f"{subject_id} – last {len(rows)} matching tests"
        + (" (mirrored az switched)" if switch_mirrored_az else "")
    )

    # x positions: categorical by condition, with jitter so points don’t overlap
    def condition_label(r):
        az = r["az"]
        az_str = "[-35,0]" if az == (-35.0, 0.0) else "[0,35]"
        mirr = "mirr" if r["mirrored"] else "orig"
        return f"az {az_str} / {mirr}"

    # Fixed order for x-axis
    cond_order = [
        "az [-35,0] / orig",
        "az [-35,0] / mirr",
        "az [0,35] / orig",
        "az [0,35] / mirr",
    ]
    cond_to_x = {c: i + 1 for i, c in enumerate(cond_order)}

    # jitter based on index in the (up to 4) selected points
    jitter = numpy.linspace(-0.10, 0.10, max(len(rows), 1))

    for i, r in enumerate(rows):
        x0 = cond_to_x.get(condition_label(r), 0.0)
        x = x0 + jitter[i]
        y = float(r["acc"][metric_index])

        marker = "D" if r["mirrored"] else "o"
        ax.scatter([x], [y], marker=marker)

        # annotate time + (optional) key short
        ax.annotate(
            key_time_str(r["key"]),
            (x, y),
            textcoords="offset points",
            xytext=(4, 3),
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    ax.set_xticks(list(cond_to_x.values()))
    ax.set_xticklabels(cond_order, rotation=20, ha="right")
    ax.grid(True, axis="y", linewidth=0.5, color="0.9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, ax, rows


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




localization_dict = hr.Subject(subject_id).localization

fig, ax, rows = plot_last_four_matching(
    subject_id,
    metric_index=1,         # RMSE
    n_last=4,
    switch_mirrored_az=True,
    localization_dict=localization_dict,
)