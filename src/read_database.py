"""Módulo de compatibilidade retroativa para src/data/reader.py."""

from core.constants import DATA_DIR
from data.reader import (
    DEFAULT_SUBJECT_ID,
    get_patient_records,
    get_records_dataframe,
    patients_records,
    records,
    subject_id,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_SUBJECT_ID",
    "get_patient_records",
    "get_records_dataframe",
    "patients_records",
    "records",
    "subject_id",
]
