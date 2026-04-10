import numpy
import pandas as pd
import matplotlib.pyplot as plt


class PoseAnalysis:
    def __init__(
        self,
        subject,
        target_radius_deg=3.0,
        ballistic_window_s=0.15,
        n_bootstrap=1000,
        random_state=0,
    ):
        self.subject = subject
        self.target_radius = float(target_radius_deg)
        self.ballistic_window = float(ballistic_window_s)
        self.n_bootstrap = int(n_bootstrap)
        self.rng = numpy.random.default_rng(random_state)

    # ======================================================
    # helpers
    # ======================================================

    @staticmethod
    def _unwrap_deg(angle_deg):
        return numpy.rad2deg(
            numpy.unwrap(numpy.deg2rad(numpy.asarray(angle_deg, dtype=float)))
        )

    @classmethod
    def _trace_to_arrays(cls, pose_trace):
        if not pose_trace:
            return None

        t = numpy.asarray([s[0] for s in pose_trace], dtype=float)
        yaw = numpy.asarray([s[1] for s in pose_trace], dtype=float)
        pitch = numpy.asarray([s[2] for s in pose_trace], dtype=float)

        t_rel = t - t[0]

        yaw_unw = cls._unwrap_deg(yaw)
        pitch_unw = cls._unwrap_deg(pitch)

        dt = numpy.gradient(t_rel)
        dt = numpy.where(dt == 0.0, numpy.finfo(float).eps, dt)

        vyaw = numpy.gradient(yaw_unw) / dt
        vpitch = numpy.gradient(pitch_unw) / dt
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

    # ======================================================
    # per-trial metrics
    # ======================================================

    def analyze_trial(self, trial):
        data = self._trace_to_arrays(trial.get("pose_trace", []))
        if data is None:
            return None

        gaze = numpy.column_stack([data["yaw"], data["pitch"]])
        target = numpy.asarray(trial["target"], dtype=float)
        t = data["t"]

        err = self._angular_error(gaze, target)

        # --- main metrics ---
        hit_idx = numpy.where(err <= self.target_radius)[0]
        t_first_hit = t[hit_idx[0]] if len(hit_idx) else numpy.nan

        path_length = numpy.trapz(data["vspeed"], t)

        # --- ballistic phase ---
        b_idx = numpy.where(t <= self.ballistic_window)[0]
        if len(b_idx) >= 2:
            ballistic_error = err[b_idx[-1]]
        else:
            ballistic_error = numpy.nan

        return dict(
            session_id=trial["session_id"],
            trial_idx=trial["trial_idx"],

            # required metrics
            t_first_hit=t_first_hit,
            initial_error_deg=err[0],
            min_error_deg=err.min(),
            path_length_deg=path_length,
            mean_speed_deg_s=numpy.nanmean(data["vspeed"]),
            success=trial.get("score", 0) > 0,

            # new metrics
            ballistic_error_deg=ballistic_error,
        )

    def trial_dataframe(self):
        rows = []
        for tr in self.subject.trials:
            r = self.analyze_trial(tr)
            if r is not None:
                rows.append(r)
        return pd.DataFrame(rows)

    # ======================================================
    # bootstrap utilities
    # ======================================================

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

    # ======================================================
    # per-session metrics
    # ======================================================

    def session_dataframe(self):
        df = self.trial_dataframe()
        if df.empty:
            return df

        sess = (
            df.groupby("session_id")
              .agg(
                  n_trials=("trial_idx", "count"),
                  mean_t_first_hit=("t_first_hit", "mean"),
                  mean_initial_error_deg=("initial_error_deg", "mean"),
                  mean_ballistic_error_deg=("ballistic_error_deg", "mean"),
                  success_rate=("success", "mean"),
              )
              .reset_index()
        )

        # learning slope across sessions
        x = numpy.arange(len(sess))
        if len(sess) >= 2:
            slope, _ = numpy.polyfit(x, sess["mean_initial_error_deg"], 1)
            sess["learning_slope"] = slope
        else:
            sess["learning_slope"] = numpy.nan

        # bootstrap CIs
        ci_lo, ci_hi = [], []
        for sid in sess["session_id"]:
            vals = df.loc[df["session_id"] == sid, "initial_error_deg"]
            lo, hi = self._bootstrap_ci(vals)
            ci_lo.append(lo)
            ci_hi.append(hi)

        sess["initial_error_ci_lo"] = ci_lo
        sess["initial_error_ci_hi"] = ci_hi

        return sess

    # ======================================================
    # plotting
    # ======================================================

    def plot_learning_curves(self):
        df = self.session_dataframe()
        if df.empty:
            print("No data to plot.")
            return

        fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

        # time to target
        ax[0].plot(df["mean_t_first_hit"], marker="o")
        ax[0].set_ylabel("Mean t_first_hit (s)")
        ax[0].grid(True)

        # initial error + CI
        ax[1].plot(df["mean_initial_error_deg"], marker="o")
        ax[1].fill_between(
            range(len(df)),
            df["initial_error_ci_lo"],
            df["initial_error_ci_hi"],
            alpha=0.3,
        )
        ax[1].set_ylabel("Initial error (deg)")
        ax[1].grid(True)

        # success
        ax[2].plot(df["success_rate"], marker="o")
        ax[2].set_ylabel("Success rate")
        ax[2].set_ylim(0, 1)
        ax[2].grid(True)

        ax[2].set_xlabel("Session")
        plt.tight_layout()
        plt.show()
