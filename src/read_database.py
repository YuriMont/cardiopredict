"""
Módulo base para configuração do diretório de dados (MIMIC-IV-ECG) e
consulta inicial de registros de exames (`record_list.csv`) por paciente (`subject_id`).
"""

from pathlib import Path

import pandas as pd

# Diretório raiz onde estão armazenados os CSVs e os sinais WFDB (.hea/.dat) da base MIMIC-IV-ECG.
DATA_DIR = Path("data")

# Paciente de exemplo para inspeção e visualização individual das 12 derivações.
subject_id = 10000032

records = pd.read_csv(DATA_DIR / "record_list.csv")
patients_records = records.loc[records["subject_id"] == subject_id]


def main() -> None:
    print("Colunas: ")
    print(records.columns.to_list())
    print(patients_records)
    print(f"Total: {len(patients_records)}")


if __name__ == "__main__":
    main()
