"""Módulo de compatibilidade retroativa para src/processing/measurements.py."""

from core.constants import DERIVED_INTERVAL_LIMITS, ESSENTIAL_VENTRICULAR_COLS
from processing.measurements import clean_machine_measurements, main

__all__ = [
    "DERIVED_INTERVAL_LIMITS",
    "ESSENTIAL_VENTRICULAR_COLS",
    "clean_machine_measurements",
    "main",
]

if __name__ == "__main__":
    main()
