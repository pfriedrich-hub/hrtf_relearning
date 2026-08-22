import matplotlib
import matplotlib.patches
# Interactive backend so figures show on screen. This module is imported by the
# package __init__, which has already resolved the backend; the call here only
# matters when the module is imported on its own, and is a no-op otherwise.
from hrtf_relearning.utils.mpl_backend import use_interactive, use_headless
use_interactive()
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy
import scipy
import logging


def _preferred_font(candidates=('Helvetica', 'Arial', 'DejaVu Sans')):
    """First installed font from `candidates` (falls back to DejaVu Sans, which
    always ships with matplotlib). Setting font.family to a font that is actually
    present avoids matplotlib's noisy 'findfont: Font family 'Helvetica' not
    found' warning while still using Helvetica for publication figures on any
    machine where it is installed."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    return next((c for c in candidates if c in available), 'DejaVu Sans')


_FONT_FAMILY = _preferred_font()


def _safe_subplots(**kwargs):
    """plt.subplots that never blocks a figure from being saved.

    An interactive backend is in use so figures can be shown when a plot
    function is run on its own (e.g. Localization_AR.__main__ + plt.show()).
    When the same plot functions are called from inside Localization.run() via
    learning_transfer.py — i.e. after the pybinsim multiprocessing worker, the
    training subprocess and the pynput keyboard listener have run — creating a
    GUI figure can still raise, and because the savefig() call comes *after*
    figure creation the PNG would then never be written. Falling back to the
    file-only backend keeps the save.
    """
    try:
        return plt.subplots(**kwargs)
    except Exception as exc:
        logging.warning("interactive matplotlib backend failed (%s); "
                        "falling back to Agg so the figure can still be saved",
                        type(exc).__name__)
        use_headless()
        return plt.subplots(**kwargs)

def _no_usable_data(sequence):
    """True when a sequence has nothing that can be analysed or plotted.

    Beyond the never-started cases (this_n == -1, no data, no trial collected),
    this also catches runs aborted *part way*: Trialsequence pre-allocates one
    slot per trial and leaves the unrun ones as [], so sequence.data is ragged
    and numpy.asarray() raises "inhomogeneous shape" instead of the guard
    tripping. Such runs hold a handful of trials at most and are not worth
    plotting, so treat a partially filled data list as unusable.
    """
    data = getattr(sequence, "data", None)
    if not data or sequence.this_n == -1 or sequence.n_remaining == len(data):
        return True
    return any(not trial for trial in data)


def localization_accuracy(sequence):
    if _no_usable_data(sequence):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]

    #  elevation gain, rmse, response variability
    #  SD = scatter of responses around the gain-fit regression line (precision).
    #  Residuals around the fit remove the (1 - gain) * target term, so the SD is
    #  not confounded by gain (unlike per-sector target-aligned spread).
    try:
        elevation_gain, ele_intercept = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
        ele_resid = responses[:, 1] - (elevation_gain * targets[:, 1] + ele_intercept)
        ele_sd = float(numpy.std(ele_resid, ddof=1)) if len(ele_resid) > 1 else numpy.nan
    except ValueError:
        elevation_gain, ele_sd = 0, numpy.nan
    if not len(numpy.unique(targets[:, 0])) == 1:
        azimuth_gain, az_intercept = scipy.stats.linregress(targets[:, 0], responses[:, 0])[:2]
        az_resid = responses[:, 0] - (azimuth_gain * targets[:, 0] + az_intercept)
        az_sd = float(numpy.std(az_resid, ddof=1)) if len(az_resid) > 1 else numpy.nan
    else:
        azimuth_gain = None
        az_sd = numpy.nan
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    az_rmse, ele_rmse = rmse[0], rmse[1]
    return elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd


def _wrap_diff_deg(a, b):
    """Smallest signed difference a-b on a 360° circle, result in [-180, 180)."""
    return (numpy.asarray(a) - numpy.asarray(b) + 180.0) % 360.0 - 180.0


def is_degenerate(sequence, min_unique=2):
    """True when a finished run's responses never moved — a dead pointer.

    ``_no_usable_data`` catches runs that were never finished or are ragged; it
    cannot catch a run that completed with the SAME response on every trial.
    That happens when the head tracker stops updating: the pose is sampled
    once, every trial records it, and the run looks perfectly well formed.
    Scored, it yields elevation gain exactly 0.00 and residual SD exactly 0.00,
    which is a plausible-looking "learned nothing" rather than an obvious
    failure — IR's two dome blocks on 18.08 are both this, 21 trials each with
    the response frozen at (az 0.000, el 0.047).

    Deliberately NOT folded into ``_no_usable_data``: that would silently
    change what every existing figure and analysis includes. Call it explicitly
    where runs are selected.
    """
    if _no_usable_data(sequence):
        return False                       # already excluded by the other gate
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    responses = loc_data[:, 0]
    if responses.shape[0] < 2:
        return True
    return len(numpy.unique(responses, axis=0)) < int(min_unique)


def _interaural_polar(azimuth, elevation):
    """(lateral, polar) in degrees from vertical-polar (azimuth, elevation).

    The spectral elevation cue is constant along a cone of confusion, so the
    coordinate the manipulation acts on is the POLAR angle around the
    interaural axis, not elevation. On the midline the two coincide; they
    diverge as soon as azimuth spreads, which every ±35° sector block does.
    """
    azimuth = numpy.radians(numpy.asarray(azimuth, dtype=float))
    elevation = numpy.radians(numpy.asarray(elevation, dtype=float))
    x = numpy.cos(elevation) * numpy.cos(azimuth)
    y = numpy.cos(elevation) * numpy.sin(azimuth)
    z = numpy.sin(elevation)
    return numpy.degrees(numpy.arcsin(numpy.clip(y, -1.0, 1.0))), \
        numpy.degrees(numpy.arctan2(z, x))


def polar_error(sequence, quadrant_threshold=90.0):
    """Polar-angle accuracy of one localization run — the primary outcome.

    Elevation gain is a slope: it can look healthy while absolute accuracy is
    poor, and the fitted intercept absorbs any constant bias. The polar error
    is the quantity the experiment itself optimises — ``target_p`` weights the
    training targets by per-sector polar error — so report this first and use
    gain, RMSE and residual SD as secondary.

    Responses more than ``quadrant_threshold`` degrees away in polar angle are
    front-back / up-down confusions rather than imprecision, and averaging them
    in swamps everything else. They are split off and reported as a rate, and
    the LOCAL mean is taken over the rest (Middlebrooks' convention).

    Returns
    -------
    (local_mean, local_rmse, quadrant_rate) : tuple of float
        Mean absolute local polar error in degrees, its RMS, and the fraction
        of trials counted as quadrant errors. ``(nan, nan, nan)`` for a run with
        nothing analysable.
    """
    if _no_usable_data(sequence):
        return numpy.nan, numpy.nan, numpy.nan
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    responses, targets = loc_data[:, 0], loc_data[:, 1]

    _, target_polar = _interaural_polar(targets[:, 0], targets[:, 1])
    _, response_polar = _interaural_polar(responses[:, 0], responses[:, 1])
    error = _wrap_diff_deg(target_polar, response_polar)

    quadrant = numpy.abs(error) > float(quadrant_threshold)
    local = numpy.abs(error[~quadrant])
    if local.size == 0:
        return numpy.nan, numpy.nan, float(quadrant.mean())
    return (float(local.mean()), float(numpy.sqrt(numpy.mean(local ** 2))),
            float(quadrant.mean()))


def target_p(sequence, show=False, axis=None):
    """
    Compute per-sector error and target probabilities from a localization run.

    Returns
    -------
    response_errors : (N_sectors, 4) array
        columns: [sector_center_az, sector_center_el, polar_error, probability]
    """
    if not sequence:
        logging.debug('No sequence found')
        return None
    if not hasattr(sequence, "settings"):
        raise AttributeError("sequence must have a 'settings' dict.")
    settings = sequence.settings
    az_size, el_size = settings['sector_size']
    half_az, half_el = az_size / 2.0, el_size / 2.0
    centers = numpy.asarray(sequence.settings['sector_centers'], dtype=float)  # (N,2)
    # --- unpack data (targets = 2nd row, responses = 1st row) ---
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    # --- assign each target to exactly one sector (nearest center within box) ---
    # Compute deltas of each target to every center
    d_az = _wrap_diff_deg(targets[:, None, 0], centers[None, :, 0])  # (T,N)
    d_el = targets[:, None, 1] - centers[None, :, 1]                # (T,N)
    # inside rectangular box?
    inside = (numpy.abs(d_az) <= half_az) & (numpy.abs(d_el) <= half_el)  # (T,N)
    # If multiple sectors match (edge cases), pick the nearest; if none match,
    # pick the nearest anyway (prevents drops due to rounding).
    rect_dist = numpy.stack([numpy.abs(d_az) / half_az, numpy.abs(d_el) / half_el], axis=-1)  # (T,N,2)
    rect_dist = numpy.linalg.norm(rect_dist, axis=-1)  # normalized rectangular distance (T,N)
    # Prefer valid-inside sectors; if none, fall back to absolute nearest
    big = 1e6
    choice_inside = numpy.where(inside, rect_dist, big)
    idx_inside = numpy.argmin(choice_inside, axis=1)  # (T,)
    none_inside = ~inside.any(axis=1)
    if numpy.any(none_inside):
        # fall back to nearest by rect_dist for those
        idx_fallback = numpy.argmin(rect_dist[none_inside], axis=1)
        idx_inside[none_inside] = idx_fallback
    # Now we have, for each trial, the chosen sector index
    T = targets.shape[0]
    N = centers.shape[0]
    # --- aggregate per-sector errors ---
    response_errors = numpy.zeros((N, 3), dtype=float)
    # Prepare lists of trial indices per sector
    trials_per_sector = [[] for _ in range(N)]
    for t in range(T):
        s = int(idx_inside[t])
        trials_per_sector[s].append(t)
    # Compute RMSE in az/el, then use polar (el) RMSE as your training metric
    for s, center in enumerate(centers):
        idxs = trials_per_sector[s]
        if len(idxs) == 0:
            rmse_el = 0.0
        else:
            tgt_s = targets[idxs]        # (#,2)
            rsp_s = responses[idxs]      # (#,2)
            # Polar (elevation) error only, per your definition
            el_err = tgt_s[:, 1] - rsp_s[:, 1]
            rmse_el = float(numpy.sqrt(numpy.mean(el_err ** 2)))
        response_errors[s, :] = [center[0], center[1], rmse_el]
    # --- probabilities proportional to polar error (handle all-zero safely) ---
    pe = response_errors[:, 2]
    total = float(pe.sum())
    if total <= 0:
        probs = numpy.full_like(pe, 1.0 / len(pe))
    else:
        probs = pe / total
    response_errors = numpy.column_stack([response_errors, probs])  # (N,4)
    # --- optional heatmap ---
    if show:
        # make regular grids of unique az/el from centers
        az_vals = numpy.unique(centers[:, 0])
        el_vals = numpy.unique(centers[:, 1])
        # map each sector to its grid cell
        P = numpy.zeros((len(el_vals), len(az_vals)))
        for row in response_errors:
            az, el, _, p = row
            xi = numpy.where(az_vals == az)[0][0]
            yi = numpy.where(el_vals == el)[0][0]
            P[yi, xi] = p
        if axis is None:
            fig, axis = plt.subplots(figsize=(7, 5))
        mesh = axis.pcolormesh(az_vals, el_vals, P, shading='auto')
        cbar = plt.colorbar(mesh, ax=axis)
        cbar.set_label('Probability')
        # ticks aligned to sector edges
        axis.set_xlabel('Azimuth (°)')
        axis.set_ylabel('Elevation (°)')
        axis.set_xticks(az_vals)
        axis.set_yticks(el_vals)
        axis.grid(True, linestyle='--', linewidth=0.4)
        # optional: draw sector boxes lightly
        for (caz, cel) in centers:
            axis.add_patch(
                plt.Rectangle((caz - half_az, cel - half_el),
                              az_size, el_size, fill=False, linestyle='--', linewidth=0.6, alpha=0.7)
            )
        axis.set_title('Per-sector training probability (from polar RMSE)')
    return response_errors


def _azimuth_span(sequence):
    """Azimuth span (deg) of a localization sequence.

    Uses settings['azimuth_range'] if available, otherwise falls back to the
    range of unique target azimuths in the data.
    """
    az_range = getattr(sequence, 'settings', {}).get('azimuth_range', None)
    if az_range is not None:
        return float(max(az_range) - min(az_range))
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    target_az = loc_data[:, 1, 0]
    return float(target_az.max() - target_az.min())


def condition_tag(sequence):
    """One-line 'which condition is this' tag: listening ear, mirroring, hemifield.

    `sequence.name` is `<subject>_<date>_<hrir name>`, which does not say
    whether the block was monaural, whether the HRIR was mirrored, or which
    half of the field was sampled. In a design where ONE modified SOFA serves
    four different cells (trained/untrained ear x same/mirrored locations),
    two figures can therefore carry near-identical titles and be completely
    different conditions -- and a MIRRORED block's azimuth axis is in the
    mirrored frame, which reads as a real leftward or rightward bias if you
    forget.

    Everything is read off the stored sequence, so this also applies to blocks
    recorded before the tag existed. Returns '' if the sequence carries none of
    the attributes.

    Example: ``left ear (other: envelope)  |  MIRRORED hrir  |  left hemifield (-35, 0)``
    """
    parts = []

    ear = getattr(sequence, 'ear', None)
    if ear:
        other = getattr(sequence, 'other_ear', None)
        parts.append(f"{ear} ear" + (f" (other: {other})" if other else ""))
    elif hasattr(sequence, 'ear'):
        parts.append("binaural")

    if getattr(sequence, 'mirrored', False):
        parts.append("MIRRORED hrir")

    az_range = (getattr(sequence, 'settings', None) or {}).get('azimuth_range', None)
    if az_range is not None:
        lo, hi = float(min(az_range)), float(max(az_range))
        # Negative azimuth is the left side: the protocol sets the trained
        # hemifield to (-35, 0) for a left-trained ear and (0, 35) for a right.
        if lo < 0 < hi:
            where = "full field"
        elif hi <= 0:
            where = "left hemifield"
        else:
            where = "right hemifield"
        parts.append(f"{where} ({lo:g}, {hi:g})")

    return "  |  ".join(parts)


def plot_localization(sequence, report_stats=['elevation', 'azimuth'], axis=None, filepath=None):
    """
    Plots representative mean responses by aligning targets,
    connects them in a grid, and shows reference sector center lines across the field.
    Style matches publication: all 4 spines, ticks on all sides, light grey reference
    grid (lw=0.3), black response grid (lw=0.6), small filled dots.

    Skipped for (near-)midline tests: the response grid is only drawn when the
    sequence azimuth span exceeds 2 deg (e.g. not for azimuth_range=(-1, 1)).
    """
    if _no_usable_data(sequence):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    if _azimuth_span(sequence) <= 2:
        logging.info(f'{getattr(sequence, "name", "sequence")}: azimuth span <= 2 deg '
                     '(midline test) — skipping response grid plot.')
        return None

    fs = 8
    lw = 0.5
    plt.rcParams.update({
        'font.family': _FONT_FAMILY,
        'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
        'lines.linewidth': lw,
        'ytick.direction': 'in', 'xtick.direction': 'in',
        'ytick.major.size': 2, 'xtick.major.size': 2,
        'axes.linewidth': lw, 'axes.titlesize': fs,
    })

    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    sector_centers = sequence.settings['sector_centers']
    az_size, el_size = sequence.settings['sector_size']
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(sequence)

    mean_responses = []
    center_grid = {}
    for center in sector_centers:
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2
        in_sector = numpy.where((targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max))[0]
        if len(in_sector) == 0:
            continue
        response_shift = responses[in_sector] - targets[in_sector]
        mean_shift = numpy.mean(response_shift, axis=0)
        representative_response = center + mean_shift
        mean_responses.append(representative_response)
        center_grid[tuple(center)] = representative_response

    mean_responses = numpy.array(mean_responses)

    az_vals = sorted(set([c[0] for c in sector_centers]))
    el_vals = sorted(set([c[1] for c in sector_centers]))
    az_pad = az_size + 5
    el_pad = el_size + 5

    if axis is None:
        fig, ax = _safe_subplots(figsize=(6, 6), dpi=264)
    else:
        ax = axis
        fig = ax.get_figure()

    ax.set_aspect('equal')
    # All 4 spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
    # Ticks on all 4 sides, inward
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, width=lw, length=2)

    ax.set_xlim(min(az_vals) - az_pad, max(az_vals) + az_pad)
    ax.set_ylim(min(el_vals) - el_pad, max(el_vals) + el_pad)
    ax.set_xlabel("Response Azimuth (deg)")
    ax.set_ylabel("Response Elevation (deg)")

    # 3 ticks across each axis from the actual sector range
    az_ticks = numpy.linspace(min(az_vals), max(az_vals), 3).astype(int)
    el_ticks = numpy.linspace(min(el_vals), max(el_vals), 3).astype(int)
    ax.set_xticks(az_ticks)
    ax.set_yticks(el_ticks)

    title = sequence.name
    _cond = condition_tag(sequence)
    if _cond:
        title += f"\n{_cond}"
    if 'elevation' in report_stats:
        title += f"\nEG: {eg:.2f}, RMSE: {ele_rmse:.1f}°, SD: {ele_sd:.1f}°"
    if 'azimuth' in report_stats and ag:
        title += f"\nAG: {ag:.2f}, az RMSE: {az_rmse:.1f}°, az SD: {az_sd:.1f}°"
    ax.set_title(title, fontsize=fs)

    # Reference grid at sector centers — light grey, very thin
    for x in az_vals:
        ax.plot([x, x], [min(el_vals), max(el_vals)],
                color='0.6', linestyle='-', linewidth=0.3, zorder=-1)
    for y in el_vals:
        ax.plot([min(az_vals), max(az_vals)], [y, y],
                color='0.6', linestyle='-', linewidth=0.3, zorder=-1)

    # Mean response dots — small filled black
    ax.scatter(mean_responses[:, 0], mean_responses[:, 1],
               color='black', s=6, zorder=2)

    # Connect mean responses in grid layout — black, thin
    sector_lookup = {tuple(sc): center_grid[tuple(sc)]
                     for sc in sector_centers if tuple(sc) in center_grid}
    for el in el_vals:
        row = [sector_lookup[(az, el)] for az in az_vals if (az, el) in sector_lookup]
        if len(row) > 1:
            ax.plot([p[0] for p in row], [p[1] for p in row], 'k-', linewidth=0.6)
    for az in az_vals:
        col = [sector_lookup[(az, el)] for el in el_vals if (az, el) in sector_lookup]
        if len(col) > 1:
            ax.plot([p[0] for p in col], [p[1] for p in col], 'k-', linewidth=0.6)

    plt.tight_layout()
    if filepath:
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / f'{sequence.name}.png')

def plot_elevation_response(sequence, axis=None, add_fit=True, filepath=None, n_ticks=3):
    """
    Plot elevation responses against elevation targets.

    Per-target summaries (style from publication):
      - Light grey SD bar centred on each target elevation
      - Grey horizontal line at the mean response
      - Grey vertical line from mean response to target (RMSE indicator)
      - Black horizontal tick at the target elevation
    Individual trials shown as open grey circles.
    Black dashed EG regression line.
    Legend: EG / RMSE / SD.

    Parameters
    ----------
    sequence : object
        Localization sequence with .data, .this_n, .n_remaining, .settings, .name
    axis : matplotlib.axes.Axes or None
    add_fit : bool
        Draw the EG (elevation gain) regression line.
    filepath : Path or None
        Directory to save the figure.
    n_ticks : int
        Number of evenly spaced axis ticks across the sequence elevation
        range (default 3), matching the 2-D grid plot (plot_localization) so
        the elevation axes line up between the two figures.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if _no_usable_data(sequence):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

    fs = 8
    lw = 1.0
    plt.rcParams.update({
        'font.family': _FONT_FAMILY,
        'xtick.labelsize': fs, 'ytick.labelsize': fs, 'axes.labelsize': fs,
        'lines.linewidth': lw,
        'ytick.direction': 'in', 'xtick.direction': 'in',
        'ytick.major.size': 2, 'xtick.major.size': 2,
        'axes.linewidth': 0.5, 'axes.titlesize': fs,
        'axes.spines.right': False, 'axes.spines.top': False,
    })

    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets   = loc_data[:, 1]
    responses = loc_data[:, 0]
    targ_el = targets[:, 1]
    resp_el = responses[:, 1]

    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(sequence)

    if axis is None:
        fig, axis = _safe_subplots(figsize=(6, 6), dpi=264)
    else:
        fig = axis.get_figure()

    axis.set_aspect('equal', adjustable='box')

    el_targets = numpy.unique(targ_el)

    # Individual responses — open grey circles
    axis.scatter(targ_el, resp_el, s=10, edgecolor='0.5', facecolor='none',
                 linewidth=0.7, zorder=2)

    # EG regression line — black dashed
    if add_fit and len(el_targets) >= 2:
        slope, intercept, _, _, _ = scipy.stats.linregress(targ_el, resp_el)
        pad = (float(el_targets[-1]) - float(el_targets[0])) * 0.1
        x_line = numpy.array([float(el_targets[0]) - pad, float(el_targets[-1]) + pad])
        axis.plot(x_line, intercept + slope * x_line,
                  c='0', linewidth=lw, linestyle='--', zorder=4)

    # Evenly spaced ticks across the sequence elevation range, matching the
    # 2-D grid plot's tick style (plot_localization uses linspace(min,max,3)).
    # Fall back to the observed target range if it's missing from settings.
    el_range = getattr(sequence, 'settings', {}).get('elevation_range', None)
    if el_range is None:
        el_range = (float(targ_el.min()), float(targ_el.max()))
    lo, hi = float(min(el_range)), float(max(el_range))
    ticks = numpy.unique(numpy.linspace(lo, hi, n_ticks).astype(int))

    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    try:
        axis.get_xticklabels()[0].set_horizontalalignment('left')
        axis.get_xticklabels()[-1].set_horizontalalignment('right')
    except IndexError:
        pass

    axis.set_xlabel('Target Elevations (deg)')
    axis.set_ylabel('Response Elevations (deg)')

    _cond = condition_tag(sequence)
    title = (getattr(sequence, 'name', 'Localization')
             + (f"\n{_cond}" if _cond else "")
             + f"\nEG: {eg:.2f}, RMSE={ele_rmse:.1f}°, SD={ele_sd:.1f}°")
    axis.set_title(title, fontsize=fs)

    # Legend: EG only
    legend_handles = [
        Line2D([0], [0], color='0', linestyle='--', linewidth=lw, label='EG'),
    ]
    axis.legend(handles=legend_handles, frameon=False, loc='upper left',
                handleheight=0.25, labelspacing=0, fontsize=fs,
                bbox_to_anchor=(0, 0.95))

    plt.tight_layout()

    if filepath:
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / f'{sequence.name}_el_response.png')

    return fig


def learning_plot(subject_id, save=True, **kwargs):
    """Plot the learning curve (EG / RMSE / SD across days) for one subject.

    Thin wrapper around plot.elevation_learning.learning_plot so the curve can
    be produced by running this module directly with a subject ID. The import
    is deferred: this module is imported by the package __init__, and
    elevation_learning imports hrtf_relearning — importing it at module level
    would be circular.

    Parameters
    ----------
    subject_id : str
        Subject initials, e.g. 'SS'.
    save : bool
        Save the figure to subject_plot_dir(subject_id)/learning_plot.svg.
    **kwargs
        Passed to elevation_learning.learning_plot
        (last_day_width, other_day_width, annotate_times).
    """
    from hrtf_relearning.experiment.analysis.localization.plot.elevation_learning \
        import learning_plot as _learning_plot
    from hrtf_relearning.utils import paths
    hrir_name, fig, axes = _learning_plot(subject_id, **kwargs)
    if save:
        plot_dir = paths.subject_plot_dir(subject_id)
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / 'learning_plot.svg')
    return hrir_name, fig, axes


if __name__ == "__main__":
    import sys
    subject_id = sys.argv[1] if len(sys.argv) > 1 else 'SS'
    learning_plot(subject_id)
    plt.show()



