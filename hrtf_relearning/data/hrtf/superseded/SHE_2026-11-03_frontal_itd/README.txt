SHE's native (noon rebuild) and active composite (AM right ear -> SHE left, env4),
superseded 03.11.2026 ~15:20.

Faults: (1) midline ITD zeroed on the 200-1500 Hz phase slope, which is corrupted on
human heads -> frontal xcorr ITD -57..+21 us, lateral ITD asymmetric (-284/+253 us at
+-33 deg); (2) the composite was staged before the noon native was pulled, so it was
expanded at the 0.0875 m fallback (-314/+287 us) and carried up to 0.61 dB midline ILD.

Rebuild: from existing sweeps with the pipeline as of commit 13bf512 (GCC-PHAT midline
ITD zeroing, midline ILD re-zeroed in the expansion), head radius 0.07807 m (xcorr).
Result: native and composite midline PHAT ITD 0, xcorr ITD within +-4 us, ILD 0.000 dB,
lateral +-269 us; donor spectral shape unchanged (<= 0.004 dB), per-ear level <= 0.32 dB.
The first archived native at 0.0875 m is in ../SHE_2026-11-03_fallback_radius/.

NOT rebuilt: the other screened composites in sofa/SHE (AGV-R, GM-R, MM-R, TE-R, VD).
Everything SHE ran before ~15:20 (native, screen, baselines) played the archived files.
