"""Módulo de dados, conexão DuckDB e particionamento."""

from data.audit import inspect_dataframe, inspect_measurements_outliers
from data.database import create_schema, get_connection, migrate_all
from data.reader import (
    DEFAULT_SUBJECT_ID,
    get_patient_records,
    get_records_dataframe,
    patients_records,
    records,
    subject_id,
)
from data.split import (
    build_study_diagnostic_status,
    extract_exam_findings_and_status,
    split_by_patient,
    verify_zero_patient_leakage,
)

__all__ = [
    "DEFAULT_SUBJECT_ID",
    "build_study_diagnostic_status",
    "create_schema",
    "extract_exam_findings_and_status",
    "get_connection",
    "get_patient_records",
    "get_records_dataframe",
    "inspect_dataframe",
    "inspect_measurements_outliers",
    "migrate_all",
    "patients_records",
    "records",
    "split_by_patient",
    "subject_id",
    "verify_zero_patient_leakage",
]
