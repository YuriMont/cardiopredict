"""
Módulo de Geração e Carga Relacional de Rótulos Multilabel no DuckDB.

Cria e popula a tabela `ecg_processed_labels` consolidando as 5 superclasses
e 25 subclasses clínicas com pesos de incerteza por classe para os 821 mil exames.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb
import pandas as pd

from core.constants import DEFAULT_DB_PATH, TARGET_CLASSES
from processing.ontology import map_exam_statements


def create_processed_labels_table(con: duckdb.DuckDBPyConnection) -> None:
    """Cria a tabela ecg_processed_labels com colunas para as 30 classes e 30 pesos."""
    col_defs: list[str] = [
        "ecg_uid VARCHAR PRIMARY KEY",
        "patient_uid VARCHAR NOT NULL",
        "source_dataset VARCHAR NOT NULL",
        "split VARCHAR NOT NULL",
        "is_signal_valid BOOLEAN NOT NULL DEFAULT TRUE",
        "hdf5_index INTEGER",
        "is_supervised_target BOOLEAN NOT NULL DEFAULT TRUE",
    ]

    # 30 colunas de rótulo binário (label_<classe>)
    for cls in TARGET_CLASSES:
        col_defs.append(f"label_{cls.lower()} INTEGER NOT NULL DEFAULT 0")

    # 30 colunas de peso de confiança (weight_<classe>)
    for cls in TARGET_CLASSES:
        col_defs.append(f"weight_{cls.lower()} DOUBLE NOT NULL DEFAULT 1.0")

    cols_sql = ",\n        ".join(col_defs)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS ecg_processed_labels (
            {cols_sql},
            FOREIGN KEY (ecg_uid) REFERENCES ecg_records(ecg_uid)
        );
    """)

    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_processed_split ON ecg_processed_labels (split);"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_processed_supervised ON ecg_processed_labels (is_supervised_target);"
    )


def build_processed_labels(
    db_path: Path | str | None = None,
    batch_size: int = 50000,
    limit: int | None = None,
) -> int:
    """Extrai os laudos das tabelas relacionais, harmoniza as classes e insere no DuckDB."""
    target_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    con = duckdb.connect(str(target_path))

    create_processed_labels_table(con)

    # Verifica se a tabela já possui dados
    res = con.execute("SELECT count(*) FROM ecg_processed_labels").fetchone()
    existing_count = int(res[0]) if res is not None else 0
    if existing_count > 0:
        print(
            f"Tabela ecg_processed_labels já contém {existing_count:,} registros. Recriando..."
        )
        con.execute("DELETE FROM ecg_processed_labels;")

    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    query = f"""
        SELECT 
            l.ecg_uid, 
            r.patient_uid, 
            r.source_dataset, 
            r.split,
            list(l.clean_statement) as clean_stmts,
            list(l.raw_statement) as raw_stmts
        FROM ecg_labels l
        JOIN ecg_records r ON l.ecg_uid = r.ecg_uid
        GROUP BY l.ecg_uid, r.patient_uid, r.source_dataset, r.split
        ORDER BY r.source_dataset DESC, l.ecg_uid
        {limit_clause}
    """

    print("Executando agregação dos laudos no DuckDB...")
    df_all = con.execute(query).fetchdf()
    total_records = len(df_all)
    print(f"Total de registros a processar: {total_records:,}")

    total_inserted = 0
    start_time = time.time()
    n_batches = (total_records + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        df_slice = df_all.iloc[start_idx:end_idx]

        batch_data: list[dict[str, int | float | str | bool | None]] = []

        for row in df_slice.itertuples():
            source_dataset = str(row.source_dataset)
            is_ptbxl = source_dataset.lower() == "ptbxl"
            stmts: list[str] = row.raw_stmts if is_ptbxl else row.clean_stmts

            lbls, wghts = map_exam_statements(stmts, is_ptbxl=is_ptbxl)

            has_positive = any(val == 1 for val in lbls.values())

            record_dict: dict[str, int | float | str | bool | None] = {
                "ecg_uid": str(row.ecg_uid),
                "patient_uid": str(row.patient_uid),
                "source_dataset": source_dataset,
                "split": str(row.split),
                "is_signal_valid": True,
                "hdf5_index": None,
                "is_supervised_target": bool(has_positive),
            }

            for cls in TARGET_CLASSES:
                record_dict[f"label_{cls.lower()}"] = int(lbls[cls])

            for cls in TARGET_CLASSES:
                record_dict[f"weight_{cls.lower()}"] = float(wghts[cls])

            batch_data.append(record_dict)

        df_batch = pd.DataFrame(batch_data)
        con.register("df_batch", df_batch)
        con.execute("INSERT INTO ecg_processed_labels BY NAME SELECT * FROM df_batch;")
        con.unregister("df_batch")

        total_inserted += len(df_batch)
        elapsed = time.time() - start_time
        rate = total_inserted / elapsed if elapsed > 0 else 0.0
        print(
            f"Lote {batch_idx + 1:02d}/{n_batches:02d}: {total_inserted:,}/{total_records:,} registros ({rate:,.0f} reg/s)"
        )

    con.close()
    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera a tabela ecg_processed_labels no DuckDB"
    )
    parser.add_argument(
        "--db-path", type=str, default=None, help="Caminho do arquivo DuckDB"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Tamanho do lote para processamento",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de registros para execução parcial/teste",
    )
    args = parser.parse_args()

    count = build_processed_labels(
        db_path=args.db_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"Sucesso: {count:,} registros persistidos em ecg_processed_labels.")


if __name__ == "__main__":
    main()
