"""Quick per-ear high-frequency hearing check over the DT990, Enter to respond.

Why: AM could use donor spectral detail on her LEFT ear only (2026-10-12/15).
The detail lives at ~5-12 kHz, so a right-ear high-frequency deficit would
explain it. This measures detection thresholds per ear and frequency in a few
minutes, with the same headphones and OS volume as the experiment.

Task: a pulsed tone (3 x 150 ms) comes at a random moment; press Enter as soon
as you hear it. Simple 1-up/1-down staircase per ear x frequency (converges on
the 50 % point), both ears interleaved at random within each frequency so
fatigue and attention hit them equally. Presses outside a response window are
counted as false alarms and reported.

Levels are dBFS at OS_VOLUME -- NOT calibrated dB HL. What this gives you is
the INTERAURAL difference (R - L) per frequency, and a comparison across people
on the same rig. The headphone's own channel mismatch is part of R - L, so run
it on a couple of people with normal hearing (yourself) to know the rig's
normal R - L before reading anything into one participant.

Run cell by cell.
"""
# %% config ---------------------------------------------------------------------
import datetime
import json
import random
import threading
import time

import numpy
from pynput import keyboard

from hrtf_relearning.utils import paths
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume

SUBJECT_ID  = "AM"
FREQUENCIES = (1000, 4000, 6000, 8000, 11000)   # Hz; 1 kHz is the low-frequency reference
OS_VOLUME   = 50          # same slider setting as the localization tests
START_DB    = -40.0       # dBFS peak, comfortably audible
STEP_DB     = (8.0, 4.0)  # before / after the 2nd reversal
N_REVERSALS = 6           # threshold = mean of the last 4 reversal levels
MAX_TRIALS  = 30          # per track, safety stop
FLOOR_DB, CEIL_DB = -110.0, -10.0
WINDOW_S    = 1.5         # response window from tone onset
GAP_S       = (1.0, 3.0)  # random silence before each tone


# %% helpers --------------------------------------------------------------------
import sounddevice as sd  # slab's playback backend; `pip install sounddevice` if missing

FS = int(sd.query_devices(kind="output")["default_samplerate"])


def pulsed_tone(freq, level_db, ear):
    """3 x 150 ms pulses, 100 ms gaps, 20 ms cosine ramps, one channel only."""
    n_on, n_off, n_ramp = int(.15 * FS), int(.10 * FS), int(.02 * FS)
    t = numpy.arange(n_on) / FS
    pulse = numpy.sin(2 * numpy.pi * freq * t)
    ramp = numpy.sin(numpy.linspace(0, numpy.pi / 2, n_ramp)) ** 2
    pulse[:n_ramp] *= ramp
    pulse[-n_ramp:] *= ramp[::-1]
    gap = numpy.zeros(n_off)
    mono = numpy.concatenate([pulse, gap, pulse, gap, pulse]) * 10 ** (level_db / 20)
    stereo = numpy.zeros((len(mono), 2), dtype=numpy.float32)
    stereo[:, 0 if ear == "left" else 1] = mono
    return stereo


class EnterLog:
    """Records the time of every Enter press in the background."""
    def __init__(self):
        self.presses, self._lock = [], threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.enter:
            with self._lock:
                self.presses.append(time.perf_counter())

    def since(self, t0):
        with self._lock:
            return [p for p in self.presses if p >= t0]

    def stop(self):
        self.listener.stop()


class Track:
    def __init__(self, ear, freq):
        self.ear, self.freq = ear, freq
        self.level, self.last_dir = START_DB, None
        self.reversals, self.trials = [], []

    @property
    def done(self):
        return len(self.reversals) >= N_REVERSALS or len(self.trials) >= MAX_TRIALS

    def update(self, heard):
        self.trials.append((self.level, heard))
        direction = -1 if heard else +1
        if self.last_dir is not None and direction != self.last_dir:
            self.reversals.append(self.level)
        self.last_dir = direction
        step = STEP_DB[0] if len(self.reversals) < 2 else STEP_DB[1]
        self.level = float(numpy.clip(self.level + direction * step, FLOOR_DB, CEIL_DB))

    @property
    def threshold(self):
        r = self.reversals[-4:]
        return float(numpy.mean(r)) if len(r) >= 2 else numpy.nan


def run_frequency(freq, log, stats):
    tracks = [Track("left", freq), Track("right", freq)]
    print(f"\n--- {freq} Hz ---")
    while not all(t.done for t in tracks):
        track = random.choice([t for t in tracks if not t.done])
        silence_start = time.perf_counter()
        time.sleep(random.uniform(*GAP_S))
        onset = time.perf_counter()
        stats["false_alarms"] += len([p for p in log.since(silence_start) if p < onset])
        sd.play(pulsed_tone(freq, track.level, track.ear), FS)
        time.sleep(WINDOW_S)
        heard = any(onset <= p <= onset + WINDOW_S for p in log.since(onset))
        stats["trials"] += 1
        track.update(heard)
    sd.wait()
    return {t.ear: dict(threshold_dbfs=t.threshold, reversals=t.reversals,
                        trials=t.trials) for t in tracks}


# %% run ------------------------------------------------------------------------
set_windows_volume(OS_VOLUME)
input("Headphones on (DT990, left cup on the LEFT ear). Press Enter whenever you hear "
      "a beep, in either ear, as fast as you can.\nPress Enter to start... ")
log, stats, results = EnterLog(), {"trials": 0, "false_alarms": 0}, {}
time.sleep(1.0)
try:
    for f in FREQUENCIES:
        results[f] = run_frequency(f, log, stats)
finally:
    log.stop()

print(f"\n{SUBJECT_ID}: thresholds in dBFS (lower = more sensitive), OS volume {OS_VOLUME}%")
print(f"{'Hz':>6} {'left':>8} {'right':>8} {'R-L':>6}")
for f, r in results.items():
    L, R = r["left"]["threshold_dbfs"], r["right"]["threshold_dbfs"]
    print(f"{f:>6} {L:8.1f} {R:8.1f} {R - L:+6.1f}")
print(f"false alarms: {stats['false_alarms']} over {stats['trials']} trials")

stamp = f"{datetime.datetime.now():%d.%m_%H-%M}"
out = paths.RESULTS_DIR / SUBJECT_ID / f"hearing_check_{stamp}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(subject=SUBJECT_ID, os_volume=OS_VOLUME, fs=FS,
                               stats=stats, results={str(k): v for k, v in results.items()}),
                          indent=1))
print(f"saved {out}")


# %% plot -----------------------------------------------------------------------
# Audiogram convention: right ear red circles, left ear blue crosses, worse
# hearing DOWN (y inverted, since a higher dBFS threshold = less sensitive).
import matplotlib.pyplot as plt

freqs = list(results)
L = numpy.array([results[f]["left"]["threshold_dbfs"] for f in freqs])
R = numpy.array([results[f]["right"]["threshold_dbfs"] for f in freqs])
fig, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(freqs, L, "x-", color="tab:blue", ms=9, mew=2, label="left")
ax1.plot(freqs, R, "o-", color="tab:red", ms=8, mfc="none", mew=2, label="right")
ax1.invert_yaxis()
ax1.set_ylabel("threshold (dBFS)  - better hearing up")
ax1.set_title(f"{SUBJECT_ID} hearing check {stamp}  |  OS volume {OS_VOLUME}%  |  "
              f"FA {stats['false_alarms']}/{stats['trials']}", fontsize=9)
ax1.legend(frameon=False)
ax1.set_xscale("log")
ax1.set_xticks(freqs)
ax1.set_xticklabels([f"{f / 1000:g}k" for f in freqs])
ax1.minorticks_off()
ax1.set_xlabel("frequency (Hz)")
ax1.grid(alpha=.3)
fig.tight_layout()
png = paths.subject_plot_dir(SUBJECT_ID) / f"hearing_check_{stamp}.png"
png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(png, dpi=150)
plt.show()
print(f"saved {png}")
