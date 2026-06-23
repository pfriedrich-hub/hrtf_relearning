"""Training trajectory analysis: metrics, block-level elevation learning, plots.

Analyses head-pose trajectories recorded during localization training and
visualizes how performance changes over the course of training.

Trial schema (see Training.py)::

    trial_idx        int, dense
    game_idx         int, which 90s game within the run
    trial_in_game    int, position within the game
    session_id       str, one id per training run ('day.month_hour.minute')
    game_start_time  float, wall clock when the game began
    t_start/t_end    float, wall clock trial bounds
    trial_duration   float, actual trial length (s)
    game_clock       float, cumulative time within the game (s)
    target           (yaw_deg, pitch_deg)
    pose_trace       [(t_unix, yaw_deg, pitch_deg), ...]
    score            0 miss / 1 hit / 2 fast hit
    reached_target   bool

Training hierarchy::

    trial            one target presentation
    game (90s)       trials sharing one (session_id, game_idx)
    block            consecutive games separated by a short rest (auto-detected)
    day              blocks separated by a long break (> ~1h)

Two analyses:
  * PoseAnalysis           — per-trial trajectory metrics (ballistic + corrective
                             decomposition) and single-trial / learning-curve plots.
  * BlockElevationAnalysis — elevation-plane accuracy across games and blocks
                             (the HRTF-relearning learning curve).

Run directly (``python -m ...training_analysis``) to analyze and plot one subject.
"""

from pathlib import Path

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# default gap thresholds (seconds) for auto-detecting training structure
BLOCK_GAP_S = 120.0     # rest between blocks
DAY_GAP_S = 3600.0      # break between days / runs


# ======================================================================
# per-trial trajectory analysis
# ======================================================================

class PoseAnalysis:
    """Trajectory analysis of head-pose traces recorded during training.

    Metrics follow the standard ballistic + corrective decomposition of a
    goal-directed movement: an initial fast feedforward ("ballistic") phase
    that gets the head roughly on target, followed by slower aurally guided
    corrective submovements that home in on the target.
    """

    def __init__(
        self,
        subject,
        target_radius_deg=3.0,
        ballistic_window_s=0.15,
        n_bootstrap=1000,
        onset_speed_deg_s=20.0,
        smooth_window_s=0.05,
        random_state=0,
    ):
        self.subject = subject
        self.target_radius = float(target_radius_deg)
        self.ballistic_window = float(ballistic_window_s)
        self.n_bootstrap = int(n_bootstrap)
        # speed threshold (deg/s) used to detect movement onset / submovements
        self.onset_speed = float(onset_speed_deg_s)
        # temporal window (s) for smoothing angle traces before differentiating
        self.smooth_window = float(smooth_window_s)
        self.rng = numpy.random.default_rng(random_state)

    # ---- helpers ----
    @staticmethod
    def _moving_average(x, win):
        """Centered moving average with edge padding. win is in samples (odd)."""
        win = int(win)
        if win < 3:
            return numpy.asarray(x, dtype=float)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        xp = numpy.pad(numpy.asarray(x, dtype=float), pad, mode="edge")
        kernel = numpy.ones(win) / win
        return numpy.convolve(xp, kernel, mode="valid")

    @staticmethod
    def _unwrap_deg(angle_deg):
        return numpy.rad2deg(
            numpy.unwrap(numpy.deg2rad(numpy.asarray(angle_deg, dtype=float)))
        )

    def _trace_to_arrays(self, pose_trace):
        if not pose_trace or len(pose_trace) < 2:
            return None

        t = numpy.asarray([s[0] for s in pose_trace], dtype=float)
        yaw = numpy.asarray([s[1] for s in pose_trace], dtype=float)
        pitch = numpy.asarray([s[2] for s in pose_trace], dtype=float)

        t_rel = t - t[0]

        yaw_unw = self._unwrap_deg(yaw)
        pitch_unw = self._unwrap_deg(pitch)

        # The head tracker samples irregularly; tiny dt values make the raw
        # numerical derivative explode. Floor dt at a fraction of the median
        # step and lightly smooth the angle traces before differentiating so
        # peak-speed / submovement metrics reflect movement, not sampling jitter.
        dt = numpy.gradient(t_rel)
        med_dt = numpy.median(dt[dt > 0]) if numpy.any(dt > 0) else numpy.finfo(float).eps
        dt = numpy.clip(dt, 0.25 * med_dt, None)

        win = max(3, int(round(self.smooth_window / med_dt)))
        yaw_s = self._moving_average(yaw_unw, win)
        pitch_s = self._moving_average(pitch_unw, win)

        vyaw = numpy.gradient(yaw_s) / dt
        vpitch = numpy.gradient(pitch_s) / dt
        vspeed = numpy.sqrt(vyaw**2 + vpitch**2)

        return dict(
            t=t_rel,
            yaw=yaw,
            pitch=pitch,
            yaw_unw=yaw_unw,
            pitch_unw=pitch_unw,
            vyaw=vyaw,
            vpitch=vpitch,
            vspeed=vspeed,
        )

    @staticmethod
    def _angular_error(gaze, target):
        da = (gaze[:, 0] - target[0] + 180) % 360 - 180
        de = gaze[:, 1] - target[1]
        return numpy.sqrt(da**2 + de**2)

    def _movement_onset_idx(self, t, vspeed):
        """First sample at which speed crosses the onset threshold and stays
        above it for at least 2 samples. Returns 0 if never clearly moving."""
        above = vspeed >= self.onset_speed
        if not above.any():
            return 0
        # require two consecutive samples above threshold to reject noise spikes
        sustained = above[:-1] & above[1:]
        idx = numpy.where(sustained)[0]
        return int(idx[0]) if len(idx) else int(numpy.argmax(above))

    def _count_submovements(self, t, vspeed):
        """Number of speed peaks above the onset threshold, separated by dips
        below half the threshold. A clean single-shot movement -> 1; each
        corrective homing-in movement adds one."""
        thresh = self.onset_speed
        low = thresh / 2.0
        count = 0
        armed = True  # ready to register a new peak once speed dips low again
        for v in vspeed:
            if armed and v >= thresh:
                count += 1
                armed = False
            elif not armed and v < low:
                armed = True
        return count

    # ---- per-trial metrics ----
    def analyze_trial(self, trial):
        data = self._trace_to_arrays(trial.get("pose_trace", []))
        if data is None:
            return None

        gaze = numpy.column_stack([data["yaw"], data["pitch"]])
        target = numpy.asarray(trial["target"], dtype=float)
        t = data["t"]
        vspeed = data["vspeed"]

        err = self._angular_error(gaze, target)

        # --- target acquisition ---
        hit_mask = err <= self.target_radius
        hit_idx = numpy.where(hit_mask)[0]
        t_first_hit = t[hit_idx[0]] if len(hit_idx) else numpy.nan
        reached_target = bool(len(hit_idx))

        # --- movement kinematics ---
        onset_idx = self._movement_onset_idx(t, vspeed)
        onset_latency = t[onset_idx]
        peak_idx = int(numpy.argmax(vspeed))
        peak_speed = vspeed[peak_idx]
        time_to_peak = t[peak_idx]
        movement_time = (t_first_hit - onset_latency) if reached_target else numpy.nan

        # --- accuracy ---
        initial_error = err[0]
        final_error = err[-1]
        min_error = err.min()

        # --- path geometry ---
        path_length = numpy.trapezoid(vspeed, t)
        # straight-line angular distance the head actually travelled (start->end)
        net_disp = numpy.sqrt(
            (data["yaw_unw"][-1] - data["yaw_unw"][0]) ** 2
            + (data["pitch_unw"][-1] - data["pitch_unw"][0]) ** 2
        )
        # tortuosity = actual path / straight line. 1.0 = perfectly direct.
        tortuosity = path_length / net_disp if net_disp > 1e-6 else numpy.nan

        # --- corrective behaviour ---
        n_submovements = self._count_submovements(t, vspeed)
        # settling time: last time the error drops to within radius and stays
        if reached_target:
            outside_after = numpy.where(~hit_mask)[0]
            last_outside = outside_after[-1] if len(outside_after) else -1
            settle_candidates = numpy.where(numpy.arange(len(err)) > last_outside)[0]
            settle_time = t[settle_candidates[0]] if len(settle_candidates) else t_first_hit
        else:
            settle_time = numpy.nan

        # --- ballistic phase (feedforward, before any correction) ---
        b_idx = numpy.where(t <= self.ballistic_window)[0]
        ballistic_error = err[b_idx[-1]] if len(b_idx) >= 2 else numpy.nan

        # --- elevation-plane (pitch) trajectory metrics ---
        # The spectral cues that HRTF relearning targets live in the elevation
        # axis, so we quantify the pitch trajectory specifically.
        ele = data["pitch"]
        target_ele = target[1]
        ele_err_signed = ele - target_ele
        ele_abs = numpy.abs(ele_err_signed)

        # time-averaged absolute elevation error over the whole search
        dur = t[-1] - t[0]
        if dur > 0:
            ele_mean_abs_error = numpy.trapezoid(ele_abs, t) / dur
        else:
            ele_mean_abs_error = ele_abs.mean()

        ele_initial_error = ele_abs[0]
        ele_min_error = ele_abs.min()
        ele_final_error = ele_abs[-1]
        ele_final_bias = ele_err_signed[-1]  # signed: + = above target

        # approach efficiency: useful progress toward target elevation divided
        # by total pitch excursion. 1.0 = moved straight to target elevation;
        # low = wandered / reversed in elevation.
        ele_path = numpy.sum(numpy.abs(numpy.diff(ele)))
        ele_progress = ele_initial_error - ele_min_error
        ele_efficiency = ele_progress / ele_path if ele_path > 1e-6 else numpy.nan

        # initial elevation-direction correctness: did the first sustained pitch
        # movement go toward the target elevation? Indexes up/down confusions.
        needed_sign = numpy.sign(target_ele - ele[0])
        dpitch = numpy.diff(ele)
        sig = dpitch[numpy.abs(dpitch) > 0.5]  # ignore sub-0.5deg jitter
        if needed_sign == 0 or len(sig) == 0:
            ele_initial_dir_correct = numpy.nan
        else:
            ele_initial_dir_correct = bool(numpy.sign(sig[0]) == needed_sign)

        return dict(
            session_id=trial["session_id"],
            trial_idx=trial["trial_idx"],

            # accuracy
            initial_error_deg=initial_error,
            ballistic_error_deg=ballistic_error,
            min_error_deg=min_error,
            final_error_deg=final_error,

            # timing
            onset_latency_s=onset_latency,
            time_to_peak_s=time_to_peak,
            t_first_hit=t_first_hit,
            movement_time_s=movement_time,
            settle_time_s=settle_time,

            # kinematics
            peak_speed_deg_s=peak_speed,
            mean_speed_deg_s=numpy.nanmean(vspeed),

            # path geometry / corrections
            path_length_deg=path_length,
            tortuosity=tortuosity,
            n_submovements=n_submovements,

            # elevation-plane trajectory accuracy
            target_ele_deg=target_ele,
            ele_mean_abs_error_deg=ele_mean_abs_error,
            ele_initial_error_deg=ele_initial_error,
            ele_min_error_deg=ele_min_error,
            ele_final_error_deg=ele_final_error,
            ele_final_bias_deg=ele_final_bias,
            ele_efficiency=ele_efficiency,
            ele_initial_dir_correct=ele_initial_dir_correct,

            # outcome
            reached_target=reached_target,
            success=bool(trial.get("score", 0) > 0),
        )

    def trial_dataframe(self):
        rows = []
        for tr in self.subject.trials:
            r = self.analyze_trial(tr)
            if r is not None:
                rows.append(r)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = self._with_session_order(df)
        return df

    # ---- session ordering ----
    @staticmethod
    def _parse_session_dt(sid):
        """Parse 'day.month_hour.minute' into a sortable tuple. Falls back to
        the raw string if the format is unexpected."""
        try:
            date_part, time_part = str(sid).split("_")
            day, month = (int(x) for x in date_part.split("."))
            hour, minute = (int(x) for x in time_part.split("."))
            return (month, day, hour, minute)
        except Exception:
            return (9999, 9999, 9999, str(sid))

    def _with_session_order(self, df):
        """Attach a chronological session_order index so plots/groupby respect
        real time rather than alphabetical session_id sorting."""
        sids = list(df["session_id"].unique())
        ordered = sorted(sids, key=self._parse_session_dt)
        order_map = {sid: i for i, sid in enumerate(ordered)}
        df = df.copy()
        df["session_order"] = df["session_id"].map(order_map)
        return df

    # ---- bootstrap ----
    def _bootstrap_ci(self, values, stat=numpy.mean, alpha=0.05):
        values = numpy.asarray(values)
        values = values[~numpy.isnan(values)]
        if len(values) == 0:
            return numpy.nan, numpy.nan

        samples = self.rng.choice(
            values, size=(self.n_bootstrap, len(values)), replace=True
        )
        stats = stat(samples, axis=1)
        lo = numpy.percentile(stats, 100 * alpha / 2)
        hi = numpy.percentile(stats, 100 * (1 - alpha / 2))
        return lo, hi

    # ---- per-session metrics ----
    def session_dataframe(self):
        df = self.trial_dataframe()
        if df.empty:
            return df

        sess = (
            df.groupby(["session_order", "session_id"])
              .agg(
                  n_trials=("trial_idx", "count"),
                  mean_initial_error_deg=("initial_error_deg", "mean"),
                  mean_ballistic_error_deg=("ballistic_error_deg", "mean"),
                  mean_final_error_deg=("final_error_deg", "mean"),
                  mean_t_first_hit=("t_first_hit", "mean"),
                  mean_onset_latency_s=("onset_latency_s", "mean"),
                  mean_peak_speed_deg_s=("peak_speed_deg_s", "mean"),
                  mean_path_length_deg=("path_length_deg", "mean"),
                  mean_tortuosity=("tortuosity", "mean"),
                  mean_n_submovements=("n_submovements", "mean"),
                  success_rate=("success", "mean"),
              )
              .reset_index()
              .sort_values("session_order")
              .reset_index(drop=True)
        )

        # learning slope of initial error across sessions
        x = numpy.arange(len(sess))
        if len(sess) >= 2:
            slope, _ = numpy.polyfit(x, sess["mean_initial_error_deg"], 1)
            sess["learning_slope"] = slope
        else:
            sess["learning_slope"] = numpy.nan

        # bootstrap CIs on initial error
        ci_lo, ci_hi = [], []
        for sid in sess["session_id"]:
            vals = df.loc[df["session_id"] == sid, "initial_error_deg"]
            lo, hi = self._bootstrap_ci(vals)
            ci_lo.append(lo)
            ci_hi.append(hi)
        sess["initial_error_ci_lo"] = ci_lo
        sess["initial_error_ci_hi"] = ci_hi

        return sess

    # ---- plotting: learning curves ----
    def plot_learning_curves(self, show=True):
        df = self.session_dataframe()
        if df.empty:
            print("No data to plot.")
            return None

        x = numpy.arange(len(df))
        fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

        def _line(a, col, ylabel, **kw):
            a.plot(x, df[col], marker="o", **kw)
            a.set_ylabel(ylabel)
            a.grid(True, alpha=0.3)

        # initial error + bootstrap CI
        ax[0, 0].plot(x, df["mean_initial_error_deg"], marker="o", color="C0")
        ax[0, 0].fill_between(
            x, df["initial_error_ci_lo"], df["initial_error_ci_hi"], alpha=0.25, color="C0"
        )
        ax[0, 0].set_ylabel("Initial error (deg)")
        ax[0, 0].grid(True, alpha=0.3)

        _line(ax[0, 1], "mean_final_error_deg", "Final error (deg)", color="C3")
        _line(ax[1, 0], "mean_onset_latency_s", "Onset latency (s)", color="C1")
        _line(ax[1, 1], "mean_peak_speed_deg_s", "Peak speed (deg/s)", color="C2")
        _line(ax[2, 0], "mean_tortuosity", "Path tortuosity", color="C4")
        _line(ax[2, 1], "mean_n_submovements", "# submovements", color="C5")

        for a in ax[-1, :]:
            a.set_xlabel("Session")
            a.set_xticks(x)
            a.set_xticklabels(df["session_id"], rotation=45, ha="right", fontsize=8)

        fig.suptitle(f"Learning curves — subject {getattr(self.subject, 'id', '')}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ---- plotting: single-trial trajectory + velocity ----
    def _find_trial(self, trial_idx):
        for tr in self.subject.trials:
            if tr.get("trial_idx") == trial_idx and tr.get("pose_trace"):
                return tr
        return None

    def plot_trial_pose(self, trial_idx, show=True):
        trial = self._find_trial(trial_idx)
        if trial is None:
            print(f"Trial {trial_idx} not found or has no pose trace.")
            return None

        data = self._trace_to_arrays(trial["pose_trace"])
        target = numpy.asarray(trial["target"], dtype=float)
        t, yaw, pitch, vspeed = data["t"], data["yaw"], data["pitch"], data["vspeed"]
        m = self.analyze_trial(trial)

        fig, (ax_traj, ax_vel) = plt.subplots(1, 2, figsize=(12, 5.5))

        # --- 2D trajectory, coloured by time ---
        pts = numpy.column_stack([yaw, pitch]).reshape(-1, 1, 2)
        segs = numpy.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=t[:-1], linewidth=2)
        ax_traj.add_collection(lc)
        ax_traj.scatter(yaw[0], pitch[0], c="k", s=60, zorder=5, label="start")
        ax_traj.scatter(yaw[-1], pitch[-1], marker="X", c="red", s=80, zorder=5, label="end")
        # target + acquisition radius
        circ = plt.Circle(target, self.target_radius, color="green", fill=False, lw=2)
        ax_traj.add_patch(circ)
        ax_traj.scatter(*target, marker="*", c="green", s=200, zorder=5, label="target")
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.set_xlabel("Yaw (deg)")
        ax_traj.set_ylabel("Pitch (deg)")
        ax_traj.set_title("Head trajectory (colour = time)")
        ax_traj.legend(loc="best", fontsize=8)
        ax_traj.grid(True, alpha=0.3)
        fig.colorbar(lc, ax=ax_traj, label="time (s)")

        # --- velocity profile ---
        ax_vel.plot(t, vspeed, color="C0", lw=1.5)
        ax_vel.axhline(self.onset_speed, color="grey", ls="--", lw=1, label="onset threshold")
        if not numpy.isnan(m["onset_latency_s"]):
            ax_vel.axvline(m["onset_latency_s"], color="C1", ls=":", label="onset")
        ax_vel.axvline(m["time_to_peak_s"], color="C2", ls=":", label="peak")
        if not numpy.isnan(m["t_first_hit"]):
            ax_vel.axvline(m["t_first_hit"], color="green", ls=":", label="first hit")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Angular speed (deg/s)")
        ax_vel.set_title(
            f"Velocity profile  |  {m['n_submovements']} submovement(s), "
            f"tortuosity={m['tortuosity']:.2f}"
        )
        ax_vel.legend(loc="best", fontsize=8)
        ax_vel.grid(True, alpha=0.3)

        fig.suptitle(
            f"Trial {trial_idx}  (session {trial['session_id']})  "
            f"target=({target[0]:.1f}, {target[1]:.1f})  reached={m['reached_target']}"
        )
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ---- plotting: trajectory small-multiples for one session ----
    def plot_session_trajectories(self, session_id=None, max_trials=12, show=True):
        trials = [
            tr for tr in self.subject.trials
            if tr.get("pose_trace") and (session_id is None or tr.get("session_id") == session_id)
        ]
        if not trials:
            print("No trials with pose traces for that session.")
            return None
        trials = trials[:max_trials]

        n = len(trials)
        ncol = min(4, n)
        nrow = int(numpy.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)

        for ax, trial in zip(axes.ravel(), trials):
            data = self._trace_to_arrays(trial["pose_trace"])
            target = numpy.asarray(trial["target"], dtype=float)
            yaw, pitch, t = data["yaw"], data["pitch"], data["t"]
            pts = numpy.column_stack([yaw, pitch]).reshape(-1, 1, 2)
            segs = numpy.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", array=t[:-1], linewidth=1.5)
            ax.add_collection(lc)
            ax.scatter(yaw[0], pitch[0], c="k", s=20, zorder=5)
            circ = plt.Circle(target, self.target_radius, color="green", fill=False, lw=1.5)
            ax.add_patch(circ)
            ax.scatter(*target, marker="*", c="green", s=80, zorder=5)
            ax.set_title(f"#{trial['trial_idx']}", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for ax in axes.ravel()[n:]:
            ax.axis("off")

        sid = session_id or "all sessions"
        fig.suptitle(f"Trajectories — subject {getattr(self.subject, 'id', '')}, {sid}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig


# ======================================================================
# training-structure segmentation
# ======================================================================

def _trial_start(trial):
    return trial["pose_trace"][0][0]


def _trial_end(trial):
    return trial["pose_trace"][-1][0]


def segment_games(trials):
    """Split trials into 90s games, time-ordered.

    Trials carry an explicit ``game_idx`` (with ``session_id`` to disambiguate
    across runs); games are runs of consecutive trials sharing the same
    (session_id, game_idx).

    Returns a list of games, each a time-ordered list of trial dicts.
    """
    trials = [t for t in trials if t.get("pose_trace")]
    trials = sorted(trials, key=_trial_start)
    if not trials:
        return []

    if not all("game_idx" in t for t in trials):
        raise ValueError(
            "segment_games requires every trial to have 'game_idx' "
            "(current recording schema)."
        )

    games, cur, prev_key = [], [], None
    for t in trials:
        key = (t.get("session_id"), t.get("game_idx"))
        if prev_key is not None and key != prev_key:
            games.append(cur)
            cur = []
        cur.append(t)
        prev_key = key
    if cur:
        games.append(cur)
    return games


def segment_blocks(games, block_gap_s=BLOCK_GAP_S, day_gap_s=DAY_GAP_S, verbose=True):
    """Group games into blocks and days using inter-game rest gaps.

    Returns a list of records, one per game::
        {game, block, day, n_trials, gap_before_s, t_start}
    Block and day indices restart appropriately. Prints detected boundaries.
    """
    records = []
    day, block = 0, 0
    prev_end = None
    for gi, g in enumerate(games):
        gap = (_trial_start(g[0]) - prev_end) if prev_end is not None else 0.0
        if prev_end is not None:
            if gap >= day_gap_s:
                day += 1
                block = 0
            elif gap >= block_gap_s:
                block += 1
        records.append(dict(
            game=gi,
            block=block,
            day=day,
            n_trials=len(g),
            gap_before_s=gap,
            t_start=_trial_start(g[0]),
        ))
        prev_end = _trial_end(g[-1])

    if verbose:
        print("Detected training structure:")
        print(" game | day | block | n_trials | gap_before")
        for r in records:
            mark = ""
            if r["gap_before_s"] >= day_gap_s:
                mark = "  <== NEW DAY"
            elif r["gap_before_s"] >= block_gap_s:
                mark = "  <== new block"
            print(f"  {r['game']+1:>3} |  {r['day']} |   {r['block']}   |   {r['n_trials']:>3}    "
                  f"| {r['gap_before_s']:8.1f}{mark}")
        n_days = max(r["day"] for r in records) + 1
        print(f"  -> {len(records)} games, {n_days} day(s)")
    return records


# ======================================================================
# block-level elevation learning
# ======================================================================

class BlockElevationAnalysis:
    """Elevation-plane learning across games and blocks for one subject."""

    def __init__(self, subject, block_gap_s=BLOCK_GAP_S, day_gap_s=DAY_GAP_S,
                 drop_first_day=False, **pose_kwargs):
        self.subject = subject
        self.pa = PoseAnalysis(subject, **pose_kwargs)
        self.block_gap_s = block_gap_s
        self.day_gap_s = day_gap_s
        self.drop_first_day = drop_first_day

    # ---- assemble per-trial frame tagged with game/block/day ----
    def trial_dataframe(self, verbose=True):
        games = segment_games(self.subject.trials)
        records = segment_blocks(games, self.block_gap_s, self.day_gap_s, verbose=verbose)

        rows = []
        for g, rec in zip(games, records):
            for t in g:
                m = self.pa.analyze_trial(t)
                if m is None:
                    continue
                m["game"] = rec["game"]
                m["block"] = rec["block"]
                m["day"] = rec["day"]
                rows.append(m)
        df = pd.DataFrame(rows)
        if not df.empty and self.drop_first_day:
            df = df[df["day"] > 0].reset_index(drop=True)
        return df

    # ---- aggregate ----
    _ELE_AGG = dict(
        n_trials=("trial_idx", "count"),
        ele_mean_abs_error=("ele_mean_abs_error_deg", "mean"),
        ele_final_error=("ele_final_error_deg", "mean"),
        ele_final_bias=("ele_final_bias_deg", "mean"),
        ele_efficiency=("ele_efficiency", "mean"),
        ele_dir_correct=("ele_initial_dir_correct", "mean"),
        success_rate=("success", "mean"),
    )

    def game_dataframe(self, verbose=False):
        df = self.trial_dataframe(verbose=verbose)
        if df.empty:
            return df
        g = (df.groupby(["day", "block", "game"]).agg(**self._ELE_AGG)
               .reset_index().sort_values("game").reset_index(drop=True))
        return g

    def block_dataframe(self, verbose=False):
        df = self.trial_dataframe(verbose=verbose)
        if df.empty:
            return df
        b = (df.groupby(["day", "block"]).agg(**self._ELE_AGG)
               .reset_index().sort_values(["day", "block"]).reset_index(drop=True))
        # bootstrap CI on the primary DV per block
        lo, hi = [], []
        for _, r in b.iterrows():
            vals = df.loc[(df["day"] == r["day"]) & (df["block"] == r["block"]),
                          "ele_mean_abs_error_deg"]
            l, h = self.pa._bootstrap_ci(vals)
            lo.append(l); hi.append(h)
        b["ele_mean_abs_error_ci_lo"] = lo
        b["ele_mean_abs_error_ci_hi"] = hi
        return b

    # ---- plotting ----
    def plot_elevation_learning(self, show=True, verbose=True):
        gdf = self.game_dataframe(verbose=verbose)
        if gdf.empty:
            print("No data to plot.")
            return None
        bdf = self.block_dataframe(verbose=False)

        # colour each game by its block
        blocks = sorted(gdf["block"].unique())
        cmap = plt.get_cmap("tab10")
        bcolor = {b: cmap(i % 10) for i, b in enumerate(blocks)}
        x = numpy.arange(len(gdf))

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # (0,0) primary DV per game, coloured by block
        for b in blocks:
            sel = gdf["block"] == b
            ax[0, 0].plot(x[sel], gdf.loc[sel, "ele_mean_abs_error"],
                          marker="o", color=bcolor[b], label=f"block {b}")
        ax[0, 0].set_ylabel("Mean abs. elevation error (deg)")
        ax[0, 0].set_title("Time-averaged elevation error per game")
        ax[0, 0].legend(fontsize=8)
        ax[0, 0].grid(True, alpha=0.3)

        # (0,1) elevation efficiency per game
        for b in blocks:
            sel = gdf["block"] == b
            ax[0, 1].plot(x[sel], gdf.loc[sel, "ele_efficiency"],
                          marker="o", color=bcolor[b])
        ax[0, 1].set_ylabel("Elevation approach efficiency")
        ax[0, 1].set_title("Directness of elevation search")
        ax[0, 1].grid(True, alpha=0.3)

        # (1,0) initial elevation-direction correctness (up/down confusions)
        for b in blocks:
            sel = gdf["block"] == b
            ax[1, 0].plot(x[sel], gdf.loc[sel, "ele_dir_correct"],
                          marker="o", color=bcolor[b])
        ax[1, 0].axhline(0.5, color="grey", ls="--", lw=1)
        ax[1, 0].set_ylabel("P(initial pitch move correct)")
        ax[1, 0].set_ylim(0, 1)
        ax[1, 0].set_xlabel("Game #")
        ax[1, 0].set_title("Initial elevation-direction correctness")
        ax[1, 0].grid(True, alpha=0.3)

        # (1,1) block-level summary of primary DV with bootstrap CI
        bx = numpy.arange(len(bdf))
        ax[1, 1].errorbar(
            bx, bdf["ele_mean_abs_error"],
            yerr=[bdf["ele_mean_abs_error"] - bdf["ele_mean_abs_error_ci_lo"],
                  bdf["ele_mean_abs_error_ci_hi"] - bdf["ele_mean_abs_error"]],
            marker="o", capsize=4, color="C3",
        )
        ax[1, 1].set_xticks(bx)
        ax[1, 1].set_xticklabels([f"d{int(r.day)}b{int(r.block)}"
                                  for r in bdf.itertuples()])
        ax[1, 1].set_ylabel("Mean abs. elevation error (deg)")
        ax[1, 1].set_xlabel("Block")
        ax[1, 1].set_title("Block means (95% bootstrap CI)")
        ax[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"Elevation learning — subject {getattr(self.subject, 'id', '')}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig


# ======================================================================
# multi-participant day x metric grid
# ======================================================================

# (column in game_dataframe, y-axis label, y-limits or None)
_GRID_METRICS = [
    ("ele_mean_abs_error", "Mean abs. elevation\nerror (deg)", None),
    ("ele_efficiency",     "Elevation approach\nefficiency",   None),
    ("ele_dir_correct",    "P(initial pitch\nmove correct)",   (0, 1)),
    ("success_rate",       "Success rate",                     (0, 1)),
]


def _per_game_by_day(subject, block_gap_s=BLOCK_GAP_S, day_gap_s=DAY_GAP_S,
                     drop_first_day=True, **pose_kwargs):
    """Per-game dataframe for one subject with display day number (1..N) and
    game-within-day index attached. Days are aggregated only (no blocks)."""
    bea = BlockElevationAnalysis(
        subject, block_gap_s=block_gap_s, day_gap_s=day_gap_s,
        drop_first_day=drop_first_day, **pose_kwargs,
    )
    g = bea.game_dataframe(verbose=False)
    if g.empty:
        return g
    day_codes = {d: i + 1 for i, d in enumerate(sorted(g["day"].unique()))}
    g = g.copy()
    g["day_n"] = g["day"].map(day_codes)
    g["game_in_day"] = g.groupby("day").cumcount() + 1
    return g


def plot_day_metric_grid(subjects, labels=None, metrics=None,
                         block_gap_s=BLOCK_GAP_S, day_gap_s=DAY_GAP_S,
                         drop_first_day=True, show=True, **pose_kwargs):
    """Grid of rows = metrics, columns = training days, with all participants
    overlaid. x within each panel = game number within that day; the dashed
    horizontal line is that participant's day mean.

    `subjects` is a list of subject objects (each with `.id` and `.trials`).
    """
    metrics = metrics or _GRID_METRICS
    if labels is None:
        labels = [getattr(s, "id", f"S{i}") for i, s in enumerate(subjects)]
    colors = {lab: f"C{i}" for i, lab in enumerate(labels)}

    frames, days = {}, set()
    for lab, subj in zip(labels, subjects):
        g = _per_game_by_day(subj, block_gap_s, day_gap_s, drop_first_day, **pose_kwargs)
        if not g.empty:
            frames[lab] = g
            days |= set(g["day_n"].unique())
    if not frames:
        print("No data to plot.")
        return None
    days = sorted(days)

    nrow, ncol = len(metrics), len(days)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.4 * nrow),
                             squeeze=False, sharex="col", sharey="row")

    for i, (col, ylabel, ylim) in enumerate(metrics):
        for j, day in enumerate(days):
            ax = axes[i][j]
            for lab, g in frames.items():
                sub = g[g["day_n"] == day]
                if sub.empty:
                    continue
                ax.plot(sub["game_in_day"], sub[col], marker="o", ms=4,
                        color=colors[lab], label=lab)
                ax.axhline(sub[col].mean(), color=colors[lab], ls="--", lw=1, alpha=0.5)
            if col == "ele_dir_correct":
                ax.axhline(0.5, color="grey", ls=":", lw=0.8)
            if i == 0:
                ax.set_title(f"Day {day}")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel("Game # within day")
            ax.grid(True, alpha=0.3)
        if ylim:
            axes[i][0].set_ylim(*ylim)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="upper right", ncol=len(uniq_l), fontsize=9)
    fig.suptitle("Training learning — day × metric (dashed = day mean)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if show:
        plt.show()
    return fig


def plot_group(subject_ids=("AH", "JS"), target_radius_deg=4.0, show=True, **kw):
    """Convenience: load several subjects and draw the day × metric grid."""
    from hrtf_relearning.experiment.Subject import Subject
    subjects = [Subject(sid) for sid in subject_ids]
    return plot_day_metric_grid(subjects, labels=list(subject_ids),
                                target_radius_deg=target_radius_deg, show=show, **kw)


# ======================================================================
# runner
# ======================================================================

def main(subject_id="JS", target_radius_deg=4.0, block_gap_s=BLOCK_GAP_S,
         day_gap_s=DAY_GAP_S, drop_first_day=True, export_dir="analysis_results"):
    """Load a subject, run both analyses, plot, and export CSVs."""
    import matplotlib
    matplotlib.use("tkagg")
    from hrtf_relearning.experiment.Subject import Subject

    subject = Subject(subject_id)
    print(f"Loaded subject '{subject_id}' with {len(subject.trials)} trial slots")
    if not subject.trials:
        raise RuntimeError("No trials found – nothing to analyze.")

    out_dir = Path(export_dir)
    out_dir.mkdir(exist_ok=True)

    # --- per-trial trajectory analysis ---
    pa = PoseAnalysis(subject, target_radius_deg=target_radius_deg)
    df_trials = pa.trial_dataframe()
    df_sessions = pa.session_dataframe()
    print("\nPer-trial metrics (head):")
    print(df_trials.head())
    df_trials.to_csv(out_dir / f"{subject_id}_trial_metrics.csv", index=False)
    df_sessions.to_csv(out_dir / f"{subject_id}_session_metrics.csv", index=False)
    pa.plot_learning_curves()

    # --- block-level elevation learning ---
    bea = BlockElevationAnalysis(
        subject, target_radius_deg=target_radius_deg, block_gap_s=block_gap_s,
        day_gap_s=day_gap_s, drop_first_day=drop_first_day,
    )
    df_block = bea.block_dataframe(verbose=True)
    print("\nPer-block elevation metrics:")
    print(df_block)
    bea.game_dataframe().to_csv(out_dir / f"{subject_id}_elevation_by_game.csv", index=False)
    df_block.to_csv(out_dir / f"{subject_id}_elevation_by_block.csv", index=False)
    bea.plot_elevation_learning()

    print(f"\nResults written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
