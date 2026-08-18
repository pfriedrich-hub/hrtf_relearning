"""
Dome loudspeaker training ("coin game").

Class-based port of the original `training_dome.py` script, brought in line with
the rest of the experiment package: `Subject` persistence, per-trial pose traces,
the shared target helpers, and the MetaMotion sensor used by `Localization_dome`
and `Training_AR`.

What is deliberately UNCHANGED
------------------------------
Stimulus playback on the processors is exactly the old circuit and the old tag
protocol (`play_buf_pulse.rcx` on both RX8s):

    source      1 -> pulse-train buffer, 0 -> goal-sound buffer
    data        pulse-train stimulus            (via freefield.set_signal_and_speaker)
    playbuflen  its length                      (via freefield.set_signal_and_speaker)
    chan        analog channel of the target speaker, 99 on the other processor
    interval    inter-pulse interval in ms, updated continuously from head pose
    goal_data   coin / coins / buzzer sound
    goal_len    its length
    goal_playback  read to wait for the goal sound to finish
    zBusA       start the pulse train      zBusB   goal sound / interrupt

The circuit itself lives in `experiment/training/rcx/` -- the freefield package
has no pulse circuit, so it is kept with the game that needs it.

What changed relative to the old script
---------------------------------------
* Class with a session -> game -> trial structure identical to `Training_AR`, so
  trials land in `subject.trials` in the schema `training_analysis` expects.
* Head pose from `misc.meta_motion` (BLE MetaMotion) instead of the freefield
  sensor, so the RP2/arduino circuit is no longer needed. The sensor is
  recalibrated at the start of every trial, as before.
* The full head trajectory of every trial is recorded (`pose_trace`).
* Targets come from the shared helpers in `training_helpers.training_targets`:
  sectors of the last matching localization run are weighted by response error,
  with uniform sampling as a fallback.
* `goal_len` is now written together with every `goal_data` (the old script wrote
  it once, so `coins.wav` played truncated to the length of `coin.wav`), and the
  goal sounds are converted to mono at the playback rate before they are written
  (`coin.wav` and `hi_score.wav` are 44.1 kHz stereo, so the old script sent an
  interleaved buffer to the mono goal tag -- half the sound, wrong pitch).
* A fresh pink noise token is generated for every trial, so a speaker cannot be
  recognised by its token. Playback stays uncalibrated (`equalize=False`), as in
  the old script, because no whole-dome equalization is currently measured --
  set `equalize=True` once one exists, so that per-speaker level differences
  stop being a non-spatial cue to the target.
* The old 3 degree azimuth "freebie" in the distance metric is gone; the target
  window is a plain circular-azimuth / linear-elevation distance, identical to
  the one used in AR, so the two conditions are comparable.
* Dome high scores are kept in `subject.highscore_dome`, separate from the AR
  `subject.highscore` shown on the training scoreboard -- the two conditions are
  not equally difficult, so one number for both would be meaningless.

Quick selection
---------------
Edit the config block below (or set the environment variables) and run:

    python -m hrtf_relearning.experiment.training.Training_Dome

`REGION` picks the target area -- 'dome' (all speakers) or 'midline' (the az 0
column) -- and `SETTINGS` overrides any game parameter. From a console or
protocol cell:

    from hrtf_relearning.experiment.misc.Subject import Subject
    from hrtf_relearning.experiment.training.Training_Dome import TrainingDome

    training = TrainingDome(Subject('AH'), region='midline',
                            settings={'trial_time': 8, 'game_time': 60})
    training.run()

Note on target probabilities: a midline localization run is a 'standard'
sequence without sectors, so midline training always samples uniformly. Whole
dome training uses the sector weights of the last matching sector-style run.
"""
import datetime
import logging
import os
import time
from pathlib import Path

import numpy
import slab
import freefield
from pynput import keyboard

from hrtf_relearning.experiment.misc import meta_motion
from hrtf_relearning.experiment.misc.Subject import Subject
from hrtf_relearning.experiment.training.training_helpers.pulse import distance_to_interval
from hrtf_relearning.experiment.training.training_helpers.training_targets import (
    find_last_matching_sequence, set_target_probabilistic)
from hrtf_relearning.utils import paths

# ==================== quick config ====================
SUBJECT_ID = os.environ.get('TRAINING_SUBJECT_ID', 'test')
REGION = os.environ.get('TRAINING_REGION', 'dome')      # 'dome' | 'midline'
SETTINGS = {}       # e.g. {'game_time': 60, 'trial_time': 8, 'target_size': 5}
# ======================================================

# TDT playback rate. The stimulus is synthesised with slab, whose module default
# is 8000 Hz -- see the same note in Localization_dome.
SAMPLERATE = 48828

# Pulse-train circuit. It is not part of the freefield package (which has no
# pulse circuit at all), so it is kept next to the game that needs it. The
# freefield rcx folder is still searched, in case the rig has a newer copy there.
RCX_PULSE = 'play_buf_pulse.rcx'
RCX_DIR = Path(__file__).resolve().parent / 'rcx'

# Marks the processors as configured for this game, so re-running the class in an
# open console does not re-initialize, but coming from a localization block
# (mode 'play_rec', i.e. play_buf.rcx without the pulse tags) does.
PROCESSOR_MODE = 'training_dome'

# Target areas. Elevation is capped at +-37.5 degrees: the +-50 degree midline
# speakers are outside the HRIR recording grid. (0, 0) is never a target -- it is
# the pose the sensor is calibrated to at trial start, so it would be an
# instant hit.
REGIONS = {
    'dome':    dict(azimuth_range=(-52.5, 52.5), elevation_range=(-37.5, 37.5), min_dist=30),
    'midline': dict(azimuth_range=(0, 0),        elevation_range=(-37.5, 37.5), min_dist=25),
}

DEFAULT_SETTINGS = dict(
    target_size=4,          # deg, radius of the target window
    target_time=0.5,        # s the pose must stay inside the window to score
    trial_time=10,          # s before a trial is given up
    game_time=90,           # s of playing time per game (trial time only)
    score_time=3,           # s below which a hit is worth 2 points instead of 1
    min_dist=None,          # deg between consecutive targets (None -> region default)
    # distance law -- see training_helpers.pulse
    pulse_map='ar',         # 'ar' (same mapping as Training_AR) | 'legacy'
    max_pulse_interval=None,  # ms at the far end (None -> mapping default)
    min_pulse_interval=None,  # ms closest to the target ('ar' mapping only)
    # No whole-dome equalization measured at the moment, so the pulse train goes
    # out uncalibrated, exactly as in the old script. Set True once a dome
    # calibration exists: per-speaker level differences are otherwise a
    # non-spatial cue to the target.
    equalize=False,
    stim_level=None,        # dB; None -> level of the raw pinknoise, as in the old script
    verbose=True,           # live head-pose readout in the console
)


def angular_distance(pose, target):
    """Distance in degrees between two (azimuth, elevation) pairs, azimuth circular."""
    d_az = (float(pose[0]) - float(target[0]) + 180.0) % 360.0 - 180.0
    d_el = float(pose[1]) - float(target[1])
    return float(numpy.hypot(d_az, d_el))


class _Sources:
    """Minimal stand-in for a `slab.HRTF` for the shared target helpers.

    They only ever touch `.sources.vertical_polar`, an (N, 3) array of
    azimuth (0..360), elevation and radius -- so the dome speakers can be fed
    through exactly the same target selection as the AR HRIR positions.
    """

    class _VP:
        def __init__(self, array):
            self.vertical_polar = array

    def __init__(self, speakers, radius=1.4):
        self.speakers = list(speakers)
        self.sources = self._VP(numpy.array(
            [[s.azimuth % 360, s.elevation, radius] for s in self.speakers], dtype=float))


class TrainingDome:
    """Sound localization training in the dome, with real loudspeakers.

    A pulse train plays from the target speaker; the interval shortens as the
    subject turns towards it and goes continuous inside the target window.
    Holding the pose there for `target_time` scores a coin (two coins if it took
    less than `score_time`). A game lasts `game_time` seconds of playing time;
    several games make a session.

    Parameters
    ----------
    subject : Subject
    region : str
        Key of `REGIONS`: 'dome' (all speakers) or 'midline' (the az 0 column).
    settings : dict, optional
        Overrides for `DEFAULT_SETTINGS` (target_size, target_time, trial_time,
        game_time, score_time, min_dist, pulse_map, equalize, ...).
    """

    def __init__(self, subject, region=REGION, settings=None):
        if region not in REGIONS:
            raise ValueError(f"region must be one of {list(REGIONS)}, got {region!r}")
        self.subject = subject
        self.region = region
        self.session_id = datetime.datetime.now().strftime('%d.%m_%H.%M')

        area = REGIONS[region]
        self.settings = dict(DEFAULT_SETTINGS, **area, **(settings or {}))
        if self.settings['min_dist'] is None:
            self.settings['min_dist'] = area['min_dist']
        # aliases the shared target helpers expect
        self.settings['az_range'] = self.settings['azimuth_range']
        self.settings['ele_range'] = self.settings['elevation_range']
        self.settings['region'] = region

        # see SAMPLERATE -- must happen before any stimulus is synthesised
        slab.set_default_samplerate(SAMPLERATE)
        # buffer a little longer than a trial, so the pulse train never runs out
        self.stim_duration = float(self.settings['trial_time']) + 1.0

        self.target = [0.0, 0.0]        # (az, el) of the current target
        self.motion_sensor = None
        self._interval_written = None
        self._sounds = {}

        self.speakers = self._select_speakers()
        self._sources = _Sources(self.speakers)
        self._check_min_dist()
        # sector weights from the last localization run covering this area
        self.sequence = find_last_matching_sequence(self.subject, self.settings)
        logging.info('Dome training | region %r | %d target speakers | az=%s el=%s',
                     region, len(self.speakers), self.settings['az_range'],
                     self.settings['ele_range'])

    # ------------------------------------------------------------ setup
    def _select_speakers(self):
        """Speakers of the dome table inside the region, minus (0, 0).

        Reads the table directly (rather than freefield.SPEAKERS) so the target
        set is known before the processors are initialized. Playback always goes
        through `speaker.index`, so the equalization loaded into the global
        speaker list on initialize() is the one that gets applied.
        """
        previous_setup = freefield.SETUP
        freefield.SETUP = 'dome'          # read_speaker_table() reads the global
        try:
            table = freefield.read_speaker_table()
        finally:
            freefield.SETUP = previous_setup
        az_lo, az_hi = sorted(self.settings['azimuth_range'])
        el_lo, el_hi = sorted(self.settings['elevation_range'])
        speakers = [s for s in table
                    if az_lo <= s.azimuth <= az_hi and el_lo <= s.elevation <= el_hi
                    and not (s.azimuth == 0 and s.elevation == 0)]
        if not speakers:
            raise ValueError(f'No dome speakers in region {self.region!r}.')
        return speakers

    def _check_min_dist(self):
        """Fail fast if a speaker has no partner far enough away.

        The uniform sampler loops until it finds a target at least `min_dist`
        from the previous one, so a min_dist that no pair of speakers satisfies
        would hang the session mid-game rather than here.
        """
        min_dist = self.settings['min_dist']
        positions = [(s.azimuth, s.elevation) for s in self.speakers]
        for p in positions:
            if not any(angular_distance(p, q) >= min_dist for q in positions):
                raise ValueError(
                    f'min_dist={min_dist} deg is too large for region {self.region!r}: '
                    f'speaker at {p} has no target that far away. Lower min_dist.')

    @staticmethod
    def _pulse_circuit():
        """Path of the pulse-train circuit: training/rcx first, then the freefield package."""
        candidates = [RCX_DIR / RCX_PULSE,
                      Path(freefield.DIR) / 'data' / 'rcx' / RCX_PULSE]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f'Could not find the pulse-train circuit {RCX_PULSE}. Looked in: '
            + ', '.join(str(c) for c in candidates))

    def _init_processors(self):
        if freefield.PROCESSORS.mode != PROCESSOR_MODE:
            circuit = self._pulse_circuit()
            freefield.initialize('dome', device=[['RX81', 'RX8', circuit],
                                                 ['RX82', 'RX8', circuit]],
                                 sensor_tracking=False)
            freefield.PROCESSORS.mode = PROCESSOR_MODE

    def _load_sounds(self):
        """Coin / double coin / buzzer / high score, played from the goal buffer.

        The goal tag is a single mono buffer, so stereo files have to be mixed
        down before they are written -- freefield flattens a 2-D array, which
        would send an interleaved stream to the processor.
        """
        levels = {'coin': 70, 'coins': 70, 'buzzer': 75, 'hi_score': 70}
        for name, level in levels.items():
            sound = slab.Sound(paths.SOUNDS_DIR / f'{name}.wav')
            if sound.n_channels > 1:
                sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
            if sound.samplerate != SAMPLERATE:
                sound = sound.resample(SAMPLERATE)
            sound.level = level
            self._sounds[name] = sound

    @staticmethod
    def _init_sensor():
        device = meta_motion.get_device()
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    # ------------------------------------------------------------ session
    def run(self, n_games=None):
        """Play games until Esc at the play-again prompt (or `n_games` are done)."""
        self._init_processors()
        self._load_sounds()
        self.motion_sensor = self._init_sensor()
        games_played = 0
        try:
            while n_games is None or games_played < n_games:
                games_played += 1
                print(f'\n=== {self.subject.id} | {self.region} | game {games_played} ===')
                print('Enter: start   |   Esc: quit')
                if self.wait_for_key() == 'esc':
                    games_played -= 1
                    break
                self.play_game(games_played)
                if n_games is not None and games_played >= n_games:
                    break
                print('\nEnter: play again   |   Esc: end session')
                if self.wait_for_key() == 'esc':
                    break
        except KeyboardInterrupt:
            print('\nInterrupted -- ending session.')
        finally:
            self._silence()
            try:
                self.motion_sensor.halt()
            except Exception:
                logging.exception('Could not disconnect the motion sensor')
            logging.info('Dome training session ended after %d game(s).', games_played)

    def play_game(self, game_idx):
        """One `game_time` game. Returns the total score."""
        game_timer, trial_in_game, total = 0.0, 0, 0
        game_start_wall = time.time()
        while game_timer < self.settings['game_time']:
            trial_in_game += 1
            self.set_target()
            # arm the speaker before the prompt: writing the stimulus takes a
            # moment, and it must not sit between calibration and playback
            speaker = self.speaker_at(self.target)
            self._arm_speaker(speaker)
            print(f"\nTrial {trial_in_game} | {self.settings['game_time'] - game_timer:.0f} s left"
                  f" | look at the center speaker and press Enter (Esc: end game)")
            if self.wait_for_key() == 'esc':
                break
            self.motion_sensor.calibrate()
            game_timer, score = self.play_trial(
                speaker=speaker, trial_idx=len(self.subject.trials), game_idx=game_idx,
                trial_in_game=trial_in_game, game_start_wall=game_start_wall,
                game_timer=game_timer)
            total += score
        self._game_over(game_idx, total)
        return total

    def _game_over(self, game_idx, total):
        highscore = int(getattr(self.subject, 'highscore_dome', 0))
        if total > highscore:
            self.subject.highscore_dome = int(total)
            self.subject.write()
            print(f'\nGame {game_idx} over | {total} points -- new high score!')
            self._play_goal_sound('hi_score')
        else:
            print(f'\nGame {game_idx} over | {total} points (high score {highscore})')
            self._play_goal_sound('buzzer')

    # ------------------------------------------------------------ trial
    def set_target(self):
        """Pick the next target speaker, at least `min_dist` from the current one.

        Sectors of the last matching localization run are weighted by response
        error; without such a run (or if the pick violates min_dist) the target
        is drawn uniformly from the region.
        """
        previous = tuple(self.target)
        if self.sequence is not None:
            set_target_probabilistic(self.target, self.settings, self.sequence, self._sources)
        # The uniform fallback inside the shared helper compares a wrapped
        # azimuth against an unwrapped one, so its min_dist check can pass
        # spuriously for negative azimuths. Verify here and resample if needed.
        if (self.sequence is None
                or (previous != (0.0, 0.0)
                    and angular_distance(self.target, previous) < self.settings['min_dist'])):
            self._set_target_uniform(previous)
        self.target = [float(self.target[0]), float(self.target[1])]
        logging.info('Target: azimuth %.1f, elevation %.1f', self.target[0], self.target[1])

    def _set_target_uniform(self, previous):
        candidates = [(s.azimuth, s.elevation) for s in self.speakers]
        if previous != (0.0, 0.0):
            candidates = [c for c in candidates
                          if angular_distance(c, previous) >= self.settings['min_dist']]
        choice = candidates[int(numpy.random.randint(len(candidates)))]
        self.target[:] = [float(choice[0]), float(choice[1])]

    def play_trial(self, speaker, trial_idx, game_idx, trial_in_game, game_start_wall, game_timer):
        """One target presentation, on the speaker armed by `_arm_speaker`.

        Returns (game_timer, score).
        """
        trace, score, count_down, on_target_since = [], 0, False, 0.0
        distance = angular_distance(self.motion_sensor.get_pose(), self.target)
        self._write_interval(distance)
        freefield.play(kind='zBusA', proc='all')     # start the pulse train

        t0 = time.time()
        while True:
            now = time.time()
            trial_timer = now - t0
            pose = self.motion_sensor.get_pose()
            trace.append((now, float(pose[0]), float(pose[1])))
            distance = angular_distance(pose, self.target)
            self._write_interval(distance)
            if self.settings['verbose']:
                print(f'head pose: azimuth {pose[0]:6.1f}, elevation {pose[1]:6.1f}'
                      f' | distance {distance:5.1f}', end='\r', flush=True)

            if distance <= self.settings['target_size']:
                if not count_down:
                    on_target_since, count_down = now, True
            else:
                count_down = False

            if count_down and now - on_target_since >= self.settings['target_time']:
                score = 2 if trial_timer <= self.settings['score_time'] else 1
                print(f'\nScore! {score}')
                self._play_goal_sound('coins' if score == 2 else 'coin')
                break
            if trial_timer >= self.settings['trial_time']:
                print('\nMiss.')
                freefield.play(kind='zBusB', proc='all')   # interrupt the pulse train
                break
            if game_timer + trial_timer >= self.settings['game_time']:
                print('\nTime up.')
                freefield.play(kind='zBusB', proc='all')   # interrupt the pulse train
                break
            time.sleep(0.02)

        t1 = time.time()
        game_timer += t1 - t0
        self._store_trial(dict(
            trial_idx=int(trial_idx),
            game_idx=int(game_idx),
            trial_in_game=int(trial_in_game),
            game_start_time=float(game_start_wall),
            session_id=self.session_id,
            t_start=float(t0),
            t_end=float(t1),
            trial_duration=float(t1 - t0),
            game_clock=float(game_timer),
            duration=float(game_timer),          # legacy alias (== game_clock)
            target=(float(self.target[0]), float(self.target[1])),
            pose_trace=trace,
            score=int(score),
            reached_target=bool(score > 0),
            # dome specific
            condition='dome',
            region=self.region,
            speaker=int(speaker.index),
            settings=dict(self.settings),
        ))
        return game_timer, score

    def _store_trial(self, trial):
        if 0 <= trial['trial_idx'] < len(self.subject.trials):
            self.subject.trials[trial['trial_idx']] = trial     # resumed run
        else:
            self.subject.trials.append(trial)
        self.subject.write()

    # ------------------------------------------------------------ processors
    def speaker_at(self, target):
        """Speaker of the region closest to (az, el); the target is one of them."""
        distances = [angular_distance(target, (s.azimuth, s.elevation)) for s in self.speakers]
        nearest = int(numpy.argmin(distances))
        if distances[nearest] > 0.1:
            raise ValueError(f'No dome speaker at {tuple(target)}.')
        return self.speakers[nearest]

    def _make_stim(self):
        stim = slab.Sound.pinknoise(duration=self.stim_duration)
        if self.settings['stim_level'] is not None:
            stim.level = self.settings['stim_level']
        return stim

    def _arm_speaker(self, speaker):
        """Load a fresh pulse-train stimulus and route it to the target speaker.

        A new noise token every trial, so the subject cannot learn a speaker by
        its token. set_signal_and_speaker writes data/playbuflen/chan exactly as
        every other dome experiment does; only `source` is specific to the pulse
        circuit. With `equalize=True` it raises if the speakers carry no
        equalization, rather than playing something uncalibrated unnoticed.
        """
        freefield.set_signal_and_speaker(signal=self._make_stim(), speaker=speaker.index,
                                        equalize=self.settings['equalize'])
        freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])
        self._interval_written = None

    def _write_interval(self, distance):
        """Update the inter-pulse interval on the processors (ms), if it changed."""
        interval = distance_to_interval(distance, self.settings)
        if self._interval_written is None or abs(interval - self._interval_written) >= 1:
            freefield.write(tag='interval', value=interval, processors=['RX81', 'RX82'])
            self._interval_written = interval

    def _play_goal_sound(self, name, timeout=5.0):
        sound = self._sounds[name]
        freefield.write(tag='goal_data', value=sound.data, processors=['RX81', 'RX82'])
        freefield.write(tag='goal_len', value=sound.n_samples, processors=['RX81', 'RX82'])
        freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])
        freefield.play(kind='zBusB', proc='all')
        deadline = time.time() + timeout
        while freefield.read('goal_playback', processor='RX81', n_samples=1):
            if time.time() > deadline:
                logging.warning('Goal sound %r did not finish within %.1f s.', name, timeout)
                break
            time.sleep(0.05)

    def _silence(self):
        """Stop the pulse train and leave the buffer switched to the goal sound."""
        try:
            freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])
            freefield.play(kind='zBusB', proc='all')
        except Exception:
            logging.debug('Could not silence the processors.', exc_info=True)

    # ------------------------------------------------------------ input
    @staticmethod
    def wait_for_key(keys=('enter', 'esc')):
        """Block until Enter or Esc; returns the name of the key that was pressed."""
        pressed = {}

        def on_press(key):
            if key == keyboard.Key.enter and 'enter' in keys:
                pressed['key'] = 'enter'
                return False
            if key == keyboard.Key.esc and 'esc' in keys:
                pressed['key'] = 'esc'
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        return pressed.get('key', 'enter')


# ==================== main ====================
if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    TrainingDome(Subject(SUBJECT_ID), region=REGION, settings=SETTINGS).run()
