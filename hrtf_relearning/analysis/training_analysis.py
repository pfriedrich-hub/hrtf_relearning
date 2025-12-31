"""
Example script to run gaze / pose trajectory analysis
after localization training.
"""

from pathlib import Path
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from hrtf_relearning.experiment.Subject import Subject
from hrtf_relearning.analysis.training import PoseAnalysis


# ------------------------------------------------------
# configuration
# ------------------------------------------------------

SUBJECT_ID = "AvS"

TARGET_RADIUS_DEG = 3.0        # must match training target_size
BALLISTIC_WINDOW_S = 0.15      # 150 ms feedforward window
N_BOOTSTRAP = 1000


# ------------------------------------------------------
# load subject
# ------------------------------------------------------

subject = Subject(SUBJECT_ID)

print(f"Loaded subject '{SUBJECT_ID}'")
print(f"Number of trials: {len(subject.trials)}")

if len(subject.trials) == 0:
    raise RuntimeError("No trials found – nothing to analyze.")


# ------------------------------------------------------
# run analysis
# ------------------------------------------------------

pa = PoseAnalysis(
    subject,
    target_radius_deg=TARGET_RADIUS_DEG,
    ballistic_window_s=BALLISTIC_WINDOW_S,
    n_bootstrap=N_BOOTSTRAP,
)

df_trials = pa.trial_dataframe()
df_sessions = pa.session_dataframe()

print("\nPer-trial metrics (head):")
print(df_trials.head())

print("\nPer-session metrics:")
print(df_sessions)


# ------------------------------------------------------
# plotting
# ------------------------------------------------------

pa.plot_learning_curves()

# optional: inspect individual trial
last_trial_idx = subject.trials[-1]["trial_idx"]
print(f"\nPlotting last trial (idx {last_trial_idx})")
pa.plot_trial_pose(last_trial_idx)


# ------------------------------------------------------
# export (optional, but recommended)
# ------------------------------------------------------

out_dir = Path("analysis_results")
out_dir.mkdir(exist_ok=True)

df_trials.to_csv(out_dir / f"{SUBJECT_ID}_trial_metrics.csv", index=False)
df_sessions.to_csv(out_dir / f"{SUBJECT_ID}_session_metrics.csv", index=False)

print(f"\nResults written to: {out_dir.resolve()}")
