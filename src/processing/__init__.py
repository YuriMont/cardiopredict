"""Módulo de processamento de sinais, medições e laudos clínicos."""

from processing.measurements import clean_machine_measurements
from processing.reports import (
    classify_report_line,
    extract_core_diagnosis,
    normalize_report,
    normalize_reports,
)
from processing.signal import (
    interpolate_missing_and_validate,
    load_raw_record,
    preprocess_ecg_for_cnn,
    remove_baseline_wander_highpass,
    remove_high_freq_muscle_noise_lowpass,
    remove_powerline_interference_notch,
    reorder_leads,
    standardize_amplitude_units_to_mv,
)

__all__ = [
    "classify_report_line",
    "clean_machine_measurements",
    "extract_core_diagnosis",
    "interpolate_missing_and_validate",
    "load_raw_record",
    "normalize_report",
    "normalize_reports",
    "preprocess_ecg_for_cnn",
    "remove_baseline_wander_highpass",
    "remove_high_freq_muscle_noise_lowpass",
    "remove_powerline_interference_notch",
    "reorder_leads",
    "standardize_amplitude_units_to_mv",
]
