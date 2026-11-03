SHE's original native from 03.11.2026 (morning), superseded 03.11.2026 ~12:30.

Fault: built at the fallback head radius 0.0875 m. Step 0's phase-slope fit hit the
0.055 m bound (residual 107 us), so usable_radius fell back. The phase estimator is
unreliable on human heads in this rig (it reproduces KEMAR, but on SHE the 200-1500 Hz
phase ITD shifts by up to +-200 us with window length while the 10 repeats agree to
2-5 us: late energy inside the 512-tap window, not noise). The xcorr cross-check is
clean: 0.07807 m, residual 28 us.

Rebuild (from the existing sweeps, no re-recording): same pipeline and arguments as
HRIR_Recording.py step 1 (ref_31.08, equalize_dome=False, hp 120 Hz, 10 recordings),
head_radius = 0.07807292615508385 (the xcorr cross-check in rec/SHE/head_radius_fit.json).
Control: the same build at 0.0875 reproduced this archived file to 4e-8 (peak 1.5).
Result: lateral ITDs -7 to -11% (e.g. +-33 deg: 287 -> 268 us); midline magnitudes
changed by <= 0.03 dB (< 1.5 kHz only, low-frequency extrapolation); midline phase ITD 0
on all rows. fit_from_sofa on the new file returns 0.07808 m.

The native-baseline localization block run in the VR lab on 03.11 before ~12:30 played
THIS (archived) file. Any donor composites built from it are also at 0.0875 m.
