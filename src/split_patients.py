"""Módulo de compatibilidade retroativa para src/data/split.py."""

from core.constants import NON_DIAGNOSTIC_PREFIXES, SUMMARY_STAMP_TEXTS
from data.split import (
    build_study_diagnostic_status,
    extract_exam_findings_and_status,
    main,
    split_by_patient,
    verify_zero_patient_leakage,
)

__all__ = [
    "NON_DIAGNOSTIC_PREFIXES",
    "SUMMARY_STAMP_TEXTS",
    "build_study_diagnostic_status",
    "extract_exam_findings_and_status",
    "main",
    "split_by_patient",
    "verify_zero_patient_leakage",
]

if __name__ == "__main__":
    main()
