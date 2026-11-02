EM's original files from 26.10.2026, superseded 27.10.2026.

Fault: elevation rows 0, -4.17 and -8.33 deg (all three positions of dome speaker 23)
carried a frontal ITD of about +290 to +350 us instead of ~0. Cause: the interaural
phase-slope estimator (_interaural_delay_s, 200-1500 Hz) was pulled by a right-ear notch
near 1.3-1.5 kHz on those rows (interaural phase ~0 below 1.2 kHz, then spinning
270-290 deg at the band edge). Every composite built from the native inherited it.

Rebuild (from the existing sweeps, no re-recording): same pipeline, same reference
(ref_31.08), same head radius as the original (0.083206 m, the xcorr cross-check);
the only change is that wherever the phase-slope estimate disagreed with a band-limited
cross-correlation by > 30 us, the cross-correlation value was used. Magnitudes are
unchanged (0.000 dB). Old-path control reproduced the original EM.sofa to 1e-7 on the
measured median plane before the change was applied.

The composite (PA right ear -> EM left, env4) was rebuilt from the fixed native at the
same radius, with the frontal ITD re-zeroing held at zero (the native is already zeroed;
the composite keeps its phase). Midline QC: PASS.

NOT rebuilt: the other screened composites in sofa/EM (AGV, AH, AH-R, SW, TE-R) still
carry the fault. Rebuild before using any of them.
