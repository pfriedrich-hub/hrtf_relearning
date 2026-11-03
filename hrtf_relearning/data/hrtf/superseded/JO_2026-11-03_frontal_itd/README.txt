JO's trained composite (LP left ear -> JO right, env4) from 02.11.2026, superseded 03.11.2026 ~13:50.

Fault: elevation-dependent frontal ITD. The pipeline zeroes each midline row on the
200-1500 Hz interaural phase slope, which on JO is off by up to +120 / -47 us per speaker
(late energy inside the 512-tap window; same failure as the head-radius phase fit). Zeroing
that wrong number shifted the real timing: xcorr ITD on the midline -111..-49 us below
el +4 and +43..+65 us above el +17 (raw sweeps: smooth -26..+6 us). Behaviour: every
headphone block's azimuth error tilts with elevation (~-0.2 deg/deg), mirrored blocks flip
the sign, dome blocks show none. Also built at the fallback radius 0.0875 m.

Rebuild (from existing sweeps, no re-recording; ONLY this composite was replaced):
native rebuilt in a scratch dir at head_radius 0.08064 m (xcorr cross-check in
rec/JO/head_radius_fit.json), and in both native and composite builds the frontal ITD was
zeroed with broadband cross-correlation wherever it disagreed with the phase slope by > 30 us
(32/38 estimates) -- the EM_2026-10-26 rule. Controls: old settings reproduced the lab JO.sofa
to 5e-8 and this composite to 3e-14 before the change. Result: midline xcorr ITD 0 on every
row; lateral ITD symmetric (+-280 us at +-33 deg, was -386/+220); magnitudes unchanged
(max 0.003 dB on the midline, 1.5-16 kHz).

NOT replaced: JO.sofa (native) and the other screened composites in sofa/JO still carry the
fault and the 0.0875 radius. Rebuild before using any of them.
