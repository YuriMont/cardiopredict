"""Módulo de compatibilidade retroativa para src/processing/reports.py."""

from processing.reports import (
    LEADING_UNCERTAINTY_PREFIX,
    TEXT_REPLACEMENTS,
    UNCERTAINTY_PATTERN,
    classify_report_line,
    extract_core_diagnosis,
    main,
    normalize_report,
    normalize_reports,
)

__all__ = [
    "LEADING_UNCERTAINTY_PREFIX",
    "TEXT_REPLACEMENTS",
    "UNCERTAINTY_PATTERN",
    "classify_report_line",
    "extract_core_diagnosis",
    "main",
    "normalize_report",
    "normalize_reports",
]

if __name__ == "__main__":
    main()
