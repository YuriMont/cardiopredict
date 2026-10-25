"""
Módulo de Visualização e Inspeção Gráfica de Exame Individual (WFDB).
"""

import sys
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import wfdb

from core.constants import DATA_DIR
from data.database import get_connection
from data.reader import patients_records, subject_id


def plot_single_patient_exam() -> None:
    """Carrega e plota as 12 derivações e laudos de um paciente de exemplo."""
    if patients_records.empty:
        print("Nenhum exame encontrado para o paciente.")
        return

    exam = patients_records.iloc[0]
    raw_path = Path(exam["path"])
    record_path = (
        raw_path
        if raw_path.exists() or str(raw_path).startswith("data")
        else DATA_DIR / raw_path
    )

    if record_path.suffix in {".hea", ".dat"}:
        record_path = record_path.with_suffix("")

    record = wfdb.rdrecord(str(record_path))
    if isinstance(record, wfdb.MultiRecord):
        record = record.multi_to_single(physical=True)

    if record.p_signal is not None:
        print(f"Identificador do exame: {exam['study_id']}")
        print(f"Identificador do paciente: {exam['subject_id']}")
        print(f"Frequência da amostragem: {record.fs} Hz")
        print(f"Formato do sinal: {record.p_signal.shape}")
        print(f"Derivações: {record.sig_name}")
    else:
        print("Erro ao carregar sinal")
        return

    con = get_connection(read_only=True)
    patient_ecg = con.execute(
        """
        SELECT m.*, r.ecg_time, r.patient_uid, r.original_exam_id AS study_id
        FROM ecg_measurements m
        JOIN ecg_records r ON m.ecg_uid = r.ecg_uid
        WHERE r.patient_uid = ?
        """,
        [f"mimic_{subject_id}"],
    ).fetchdf()

    print(patient_ecg[["patient_uid", "study_id", "ecg_time"]].to_string(index=False))

    labels_df = con.execute(
        """
        SELECT raw_statement, clean_statement, diagnostic_class, is_uncertain
        FROM ecg_labels
        WHERE ecg_uid = ?
        """,
        [f"mimic_{exam['study_id']}"],
    ).fetchdf()

    print(f"\nLaudos do exame {exam['study_id']}:")
    print(labels_df)
    con.close()

    wfdb.plot_wfdb(
        record=record,
        title=f"ECG - Study {exam['study_id']}",
        figsize=(16, 10),
        ecg_grids="all",
    )
    plt.show()


def main() -> None:
    plot_single_patient_exam()


if __name__ == "__main__":
    main()
