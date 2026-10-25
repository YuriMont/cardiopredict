"""Módulo de compatibilidade retroativa para src/data/audit.py."""

from core.constants import PHYSIOLOGICAL_LIMITS
from data.audit import (
    inspect_all_files,
    inspect_dataframe,
    inspect_measurements_outliers,
    main,
)

__all__ = [
    "PHYSIOLOGICAL_LIMITS",
    "inspect_all_files",
    "inspect_dataframe",
    "inspect_measurements_outliers",
    "main",
]

if __name__ == "__main__":
    main()
