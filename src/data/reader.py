"""
Módulo de leitura e consulta rápida de pacientes e registros no DuckDB.
"""

import sys
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from core.constants import DATA_DIR
from data.database import get_connection

# Paciente padrão para demonstração e inspeção
DEFAULT_SUBJECT_ID = 10000032


def get_records_dataframe(source_dataset: str = "mimic") -> pd.DataFrame:
    """Carrega registros de exames para um dataset específico."""
    con = get_connection(read_only=True)
    df = con.execute(
        """
        SELECT 
            CAST(p.original_patient_id AS BIGINT) AS subject_id,
            CAST(r.original_exam_id AS BIGINT) AS study_id,
            r.original_exam_id AS file_name,
            r.ecg_time,
            r.waveform_relative_path AS path
        FROM ecg_records r
        JOIN patients p ON r.patient_uid = p.patient_uid
        WHERE r.source_dataset = ?
    """,
        [source_dataset],
    ).fetchdf()
    con.close()
    return df


def get_patient_records(patient_uid: str, con=None) -> pd.DataFrame:  # pyright: ignore[reportMissingParameterType]
    """Busca registros de um paciente específico pelo UID universal."""
    close_con = False
    if con is None:
        con = get_connection(read_only=True)
        close_con = True

    df = con.execute(
        "SELECT * FROM ecg_records WHERE patient_uid = ?", [patient_uid]
    ).fetchdf()

    if close_con:
        con.close()
    return df


# Variáveis compatíveis para importação direta
subject_id = DEFAULT_SUBJECT_ID
records = get_records_dataframe()
patients_records = records.loc[records["subject_id"] == subject_id]

__all__ = [
    "DATA_DIR",
    "DEFAULT_SUBJECT_ID",
    "get_patient_records",
    "get_records_dataframe",
    "patients_records",
    "records",
    "subject_id",
]
