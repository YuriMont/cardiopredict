"""Módulo de compatibilidade retroativa para src/processing/signal.py."""

from core.constants import STANDARD_LEADS, UNIT_TO_MV_FACTOR
from processing.signal import (
    MAX_NAN_RATIO,
    MAX_PHYSICAL_MV,
    MIN_STD_MV,
    interpolate_missing_and_validate,
    load_raw_record,
    main,
    preprocess_ecg_for_cnn,
    remove_baseline_wander_highpass,
    remove_high_freq_muscle_noise_lowpass,
    remove_powerline_interference_notch,
    reorder_leads,
    standardize_amplitude_units_to_mv,
)

__all__ = [
    "MAX_NAN_RATIO",
    "MAX_PHYSICAL_MV",
    "MIN_STD_MV",
    "STANDARD_LEADS",
    "UNIT_TO_MV_FACTOR",
    "interpolate_missing_and_validate",
    "load_raw_record",
    "main",
    "preprocess_ecg_for_cnn",
    "remove_baseline_wander_highpass",
    "remove_high_freq_muscle_noise_lowpass",
    "remove_powerline_interference_notch",
    "reorder_leads",
    "standardize_amplitude_units_to_mv",
]

if __name__ == "__main__":
    main()
