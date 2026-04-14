"""
hrtf.record_mesm — MESM-based HRIR recording system.

Public API
----------
record_hrir_mesm   : full acquisition + processing pipeline
compute_mesm_params: compute MESM timing from reference measurement
MESMParams         : timing/sweep parameter container
MESMRecording      : raw binaural recording container
compute_ir_mesm    : deconvolution + windowing (DSP only, no hardware)
"""
from .record_hrir_mesm import record_hrir_mesm
from .sweep import compute_mesm_params, MESMParams
from .recordings import MESMRecording, ReferenceParams, record_mesm, record_reference
from .processing import compute_ir_mesm

__all__ = [
    "record_hrir_mesm",
    "compute_mesm_params",
    "MESMParams",
    "MESMRecording",
    "ReferenceParams",
    "record_mesm",
    "record_reference",
    "compute_ir_mesm",
]
