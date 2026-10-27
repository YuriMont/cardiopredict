"""
Módulo de Banco de Dados Relacional Embutido (DuckDB) para o CardioPredict.

Gerencia a modelagem e migração dos metadados das bases MIMIC-IV-ECG e PTB-XL,
unificando pacientes, registros de exames (com splits por paciente para evitar data leakage),
medições fisiológicas e diagnósticos multilabel em tabelas relacionais indexadas.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb
import pandas as pd

from core.constants import DATA_DIR, DEFAULT_DB_PATH, MIMIC_DIR, PTBXL_DIR


def get_connection(
    db_path: Path | str | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Retorna uma conexão com o banco DuckDB."""
    target_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target_path), read_only=read_only)


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Cria a estrutura de tabelas relacionais e índices do projeto."""
    # 1. Tabela unificada de Pacientes
    con.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_uid VARCHAR PRIMARY KEY,
            source_dataset VARCHAR NOT NULL,
            original_patient_id VARCHAR NOT NULL,
            age DOUBLE,
            sex VARCHAR
        );
    """)

    # 2. Tabela unificada de Registros de ECG (com caminhos para os arquivos WFDB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ecg_records (
            ecg_uid VARCHAR PRIMARY KEY,
            patient_uid VARCHAR NOT NULL,
            source_dataset VARCHAR NOT NULL,
            original_exam_id VARCHAR NOT NULL,
            ecg_time TIMESTAMP,
            sampling_rate INTEGER NOT NULL DEFAULT 500,
            duration_samples INTEGER NOT NULL DEFAULT 5000,
            duration_seconds DOUBLE NOT NULL DEFAULT 10.0,
            strat_fold INTEGER NOT NULL,
            split VARCHAR NOT NULL,
            waveform_relative_path VARCHAR NOT NULL,
            FOREIGN KEY (patient_uid) REFERENCES patients(patient_uid)
        );
    """)

    # 3. Tabela unificada de Medições Fisiológicas Validadas
    con.execute("""
        CREATE TABLE IF NOT EXISTS ecg_measurements (
            ecg_uid VARCHAR PRIMARY KEY,
            rr_interval DOUBLE,
            heart_rate DOUBLE,
            p_onset DOUBLE,
            p_end DOUBLE,
            qrs_onset DOUBLE,
            qrs_end DOUBLE,
            t_end DOUBLE,
            p_axis DOUBLE,
            qrs_axis DOUBLE,
            t_axis DOUBLE,
            pr_interval DOUBLE,
            qrs_duration DOUBLE,
            qt_interval DOUBLE,
            FOREIGN KEY (ecg_uid) REFERENCES ecg_records(ecg_uid)
        );
    """)

    # 4. Tabela unificada de Rótulos / Diagnósticos Multilabel
    con.execute("""
        CREATE TABLE IF NOT EXISTS ecg_labels (
            id BIGINT PRIMARY KEY,
            ecg_uid VARCHAR NOT NULL,
            source_dataset VARCHAR NOT NULL,
            raw_statement VARCHAR NOT NULL,
            clean_statement VARCHAR NOT NULL,
            diagnostic_class VARCHAR,
            diagnostic_subclass VARCHAR,
            statement_category VARCHAR,
            is_uncertain BOOLEAN NOT NULL DEFAULT FALSE,
            likelihood DOUBLE NOT NULL DEFAULT 100.0,
            FOREIGN KEY (ecg_uid) REFERENCES ecg_records(ecg_uid)
        );
    """)

    # 5. Tabela de Ontologia SCP-ECG (Referência PTB-XL)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ptbxl_scp_statements (
            scp_code VARCHAR PRIMARY KEY,
            description VARCHAR,
            diagnostic DOUBLE,
            form DOUBLE,
            rhythm DOUBLE,
            diagnostic_class VARCHAR,
            diagnostic_subclass VARCHAR,
            statement_category VARCHAR,
            scp_ecg_statement_description VARCHAR,
            aha_code BIGINT,
            aecg_refid VARCHAR,
            cdisc_code VARCHAR,
            dicom_code VARCHAR
        );
    """)

    # Criação de índices para consultas ultra-rápidas
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_records_patient ON ecg_records(patient_uid);
        CREATE INDEX IF NOT EXISTS idx_records_split ON ecg_records(split);
        CREATE INDEX IF NOT EXISTS idx_records_dataset ON ecg_records(source_dataset);
        CREATE INDEX IF NOT EXISTS idx_labels_ecg ON ecg_labels(ecg_uid);
        CREATE INDEX IF NOT EXISTS idx_labels_class ON ecg_labels(diagnostic_class);
    """)


def migrate_ptbxl(
    con: duckdb.DuckDBPyConnection,
    ptbxl_dir: Path = PTBXL_DIR,
) -> dict[str, int]:
    """Migra os metadados e laudos do PTB-XL para o DuckDB."""
    db_csv = ptbxl_dir / "ptbxl_database.csv"
    scp_csv = ptbxl_dir / "scp_statements.csv"

    if not db_csv.exists() or not scp_csv.exists():
        raise FileNotFoundError(f"Arquivos PTB-XL não encontrados em {ptbxl_dir}")

    # 1. Carregar tabela de declarações SCP-ECG
    con.execute(f"""
        INSERT OR REPLACE INTO ptbxl_scp_statements
        SELECT 
            column00 AS scp_code,
            description,
            diagnostic,
            form,
            rhythm,
            diagnostic_class,
            diagnostic_subclass,
            "Statement Category" AS statement_category,
            "SCP-ECG Statement Description" AS scp_ecg_statement_description,
            "AHA code" AS aha_code,
            "aECG REFID" AS aecg_refid,
            "CDISC Code" AS cdisc_code,
            "DICOM Code" AS dicom_code
        FROM read_csv_auto('{scp_csv}')
    """)

    # 2. Carregar pacientes únicos do PTB-XL
    con.execute(f"""
        INSERT OR REPLACE INTO patients
        SELECT DISTINCT
            CASE 
                WHEN patient_id IS NOT NULL THEN 'ptbxl_' || CAST(CAST(patient_id AS BIGINT) AS VARCHAR)
                ELSE 'ptbxl_anon_' || CAST(ecg_id AS VARCHAR)
            END AS patient_uid,
            'ptbxl' AS source_dataset,
            COALESCE(CAST(CAST(patient_id AS BIGINT) AS VARCHAR), CAST(ecg_id AS VARCHAR)) AS original_patient_id,
            age,
            CASE 
                WHEN sex = 0 THEN 'F' 
                WHEN sex = 1 THEN 'M' 
                ELSE NULL 
            END AS sex
        FROM read_csv_auto('{db_csv}')
    """)

    # 3. Carregar registros de exames do PTB-XL (records500)
    con.execute(f"""
        INSERT OR REPLACE INTO ecg_records
        SELECT
            'ptbxl_' || CAST(ecg_id AS VARCHAR) AS ecg_uid,
            CASE 
                WHEN patient_id IS NOT NULL THEN 'ptbxl_' || CAST(CAST(patient_id AS BIGINT) AS VARCHAR)
                ELSE 'ptbxl_anon_' || CAST(ecg_id AS VARCHAR)
            END AS patient_uid,
            'ptbxl' AS source_dataset,
            CAST(ecg_id AS VARCHAR) AS original_exam_id,
            TRY_CAST(recording_date AS TIMESTAMP) AS ecg_time,
            500 AS sampling_rate,
            5000 AS duration_samples,
            10.0 AS duration_seconds,
            CAST(strat_fold AS INTEGER) AS strat_fold,
            CASE 
                WHEN strat_fold <= 8 THEN 'train'
                WHEN strat_fold = 9 THEN 'val'
                ELSE 'test'
            END AS split,
            'data/ptb-xl/' || filename_hr AS waveform_relative_path
        FROM read_csv_auto('{db_csv}')
    """)

    # 4. Processar e inserir rótulos multilabel (scp_codes)
    scp_ref = pd.read_csv(scp_csv, index_col=0)
    ptb_df = pd.read_csv(db_csv, usecols=["ecg_id", "scp_codes"])

    labels_data: list[dict[str, Any]] = []
    res_max = con.execute("SELECT COALESCE(MAX(id), 0) FROM ecg_labels").fetchall()
    current_max_id: int = int(res_max[0][0]) if res_max else 0

    label_counter = current_max_id + 1
    for row in ptb_df.itertuples(index=False):
        row_tuple: tuple[Any, ...] = tuple(row)
        ecg_id, scp_codes = row_tuple[0], row_tuple[1]
        ecg_uid = f"ptbxl_{ecg_id}"
        codes_dict: dict[str, float] = ast.literal_eval(str(scp_codes))

        for code, prob in codes_dict.items():
            info = scp_ref.loc[code] if code in scp_ref.index else None
            diag_class = (
                str(info["diagnostic_class"])
                if info is not None and bool(pd.notna(info["diagnostic_class"]))
                else None
            )
            diag_subclass = (
                str(info["diagnostic_subclass"])
                if info is not None and bool(pd.notna(info["diagnostic_subclass"]))
                else None
            )
            desc = (
                str(info["description"])
                if info is not None and bool(pd.notna(info["description"]))
                else code
            )
            cat = (
                str(info["Statement Category"])
                if info is not None and bool(pd.notna(info["Statement Category"]))
                else None
            )

            is_uncertain = 0.0 < float(prob) < 100.0

            labels_data.append(
                {
                    "id": label_counter,
                    "ecg_uid": ecg_uid,
                    "source_dataset": "ptbxl",
                    "raw_statement": str(code),
                    "clean_statement": desc,
                    "diagnostic_class": diag_class,
                    "diagnostic_subclass": diag_subclass,
                    "statement_category": cat,
                    "is_uncertain": is_uncertain,
                    "likelihood": float(prob),
                }
            )
            label_counter += 1

    labels_df = pd.DataFrame(labels_data)
    _ = labels_df  # Utilizado pelo engine do DuckDB no comando SQL abaixo
    con.execute("INSERT INTO ecg_labels SELECT * FROM labels_df")

    counts = {
        "ptbxl_patients": int(
            con.execute(
                "SELECT COUNT(*) FROM patients WHERE source_dataset = 'ptbxl'"
            ).fetchall()[0][0]
        ),
        "ptbxl_records": int(
            con.execute(
                "SELECT COUNT(*) FROM ecg_records WHERE source_dataset = 'ptbxl'"
            ).fetchall()[0][0]
        ),
        "ptbxl_labels": int(
            con.execute(
                "SELECT COUNT(*) FROM ecg_labels WHERE source_dataset = 'ptbxl'"
            ).fetchall()[0][0]
        ),
    }
    return counts


def migrate_mimic(
    con: duckdb.DuckDBPyConnection,
    mimic_dir: Path = MIMIC_DIR,
) -> dict[str, int]:
    """Migra os registros, medições limpas e laudos do MIMIC-IV-ECG para o DuckDB."""
    record_list_csv = mimic_dir / "record_list.csv"
    measurements_csv = mimic_dir / "machine_measurements.csv"

    if not record_list_csv.exists() or not measurements_csv.exists():
        raise FileNotFoundError(f"Arquivos MIMIC-IV-ECG não encontrados em {mimic_dir}")

    # 1. Carregar Pacientes Únicos do MIMIC
    con.execute(f"""
        INSERT OR REPLACE INTO patients (patient_uid, source_dataset, original_patient_id)
        SELECT DISTINCT
            'mimic_' || CAST(subject_id AS VARCHAR) AS patient_uid,
            'mimic' AS source_dataset,
            CAST(subject_id AS VARCHAR) AS original_patient_id
        FROM read_csv_auto('{record_list_csv}')
    """)

    # 2. Carregar Registros de ECG do MIMIC (com split estratificado por paciente: zero data leakage)
    con.execute(f"""
        INSERT OR REPLACE INTO ecg_records
        SELECT
            'mimic_' || CAST(study_id AS VARCHAR) AS ecg_uid,
            'mimic_' || CAST(subject_id AS VARCHAR) AS patient_uid,
            'mimic' AS source_dataset,
            CAST(study_id AS VARCHAR) AS original_exam_id,
            TRY_CAST(ecg_time AS TIMESTAMP) AS ecg_time,
            500 AS sampling_rate,
            5000 AS duration_samples,
            10.0 AS duration_seconds,
            (subject_id % 10) + 1 AS strat_fold,
            CASE 
                WHEN (subject_id % 10) < 8 THEN 'train'
                WHEN (subject_id % 10) = 8 THEN 'val'
                ELSE 'test'
            END AS split,
            'data/mimic-iv-ecg/1.0/' || path AS waveform_relative_path
        FROM read_csv_auto('{record_list_csv}')
    """)

    # 3. Carregar Medições Fisiológicas Validadas (com limites clínicos aplicados)
    con.execute(f"""
        INSERT OR REPLACE INTO ecg_measurements
        SELECT
            'mimic_' || CAST(study_id AS VARCHAR) AS ecg_uid,
            CASE WHEN rr_interval BETWEEN 200 AND 3000 THEN CAST(rr_interval AS DOUBLE) ELSE NULL END AS rr_interval,
            CASE WHEN rr_interval BETWEEN 200 AND 3000 THEN 60000.0 / CAST(rr_interval AS DOUBLE) ELSE NULL END AS heart_rate,
            CASE WHEN p_onset BETWEEN 1 AND 1000 THEN CAST(p_onset AS DOUBLE) ELSE NULL END AS p_onset,
            CASE WHEN p_end BETWEEN 1 AND 1200 THEN CAST(p_end AS DOUBLE) ELSE NULL END AS p_end,
            CASE WHEN qrs_onset BETWEEN 1 AND 1500 THEN CAST(qrs_onset AS DOUBLE) ELSE NULL END AS qrs_onset,
            CASE WHEN qrs_end BETWEEN 1 AND 1800 THEN CAST(qrs_end AS DOUBLE) ELSE NULL END AS qrs_end,
            CASE WHEN t_end BETWEEN 100 AND 2500 THEN CAST(t_end AS DOUBLE) ELSE NULL END AS t_end,
            CASE WHEN p_axis BETWEEN -180 AND 360 THEN CAST(p_axis AS DOUBLE) ELSE NULL END AS p_axis,
            CASE WHEN qrs_axis BETWEEN -180 AND 360 THEN CAST(qrs_axis AS DOUBLE) ELSE NULL END AS qrs_axis,
            CASE WHEN t_axis BETWEEN -180 AND 360 THEN CAST(t_axis AS DOUBLE) ELSE NULL END AS t_axis,
            CASE WHEN (qrs_onset - p_onset) BETWEEN 60 AND 450 THEN CAST(qrs_onset - p_onset AS DOUBLE) ELSE NULL END AS pr_interval,
            CASE WHEN (qrs_end - qrs_onset) BETWEEN 40 AND 300 THEN CAST(qrs_end - qrs_onset AS DOUBLE) ELSE NULL END AS qrs_duration,
            CASE WHEN (t_end - qrs_onset) BETWEEN 200 AND 700 THEN CAST(t_end - qrs_onset AS DOUBLE) ELSE NULL END AS qt_interval
        FROM read_csv_auto('{measurements_csv}')
    """)

    # 4. Despivar e Classificar os Laudos Multilabel do MIMIC (report_0..report_17)
    res_max = con.execute("SELECT COALESCE(MAX(id), 0) FROM ecg_labels").fetchall()
    current_max_id: int = int(res_max[0][0]) if res_max else 0

    con.execute(f"""
        INSERT INTO ecg_labels
        WITH unpivoted AS (
            UNPIVOT (
                SELECT 
                    study_id,
                    report_0, report_1, report_2, report_3, report_4, report_5,
                    report_6, report_7, report_8, report_9, report_10, report_11,
                    report_12, report_13, report_14, report_15, report_16, report_17
                FROM read_csv_auto('{measurements_csv}')
            )
            ON COLUMNS('^report_')
            INTO NAME col VALUE raw_text
        ),
        cleaned AS (
            SELECT
                'mimic_' || CAST(study_id AS VARCHAR) AS ecg_uid,
                'mimic' AS source_dataset,
                TRIM(raw_text) AS raw_statement,
                LOWER(TRIM(raw_text)) AS clean_statement,
                regexp_matches(
                    LOWER(raw_text),
                    '(^|\\\\W)(possible|possibly|probable|probably|consider|cannot rule out|can not rule out|may be|maybe|suggests|suggestive of|questionable|equivocal|rule out)($|\\\\W)'
                ) AS is_uncertain,
                CASE 
                    WHEN LOWER(raw_text) LIKE '%infarct%' THEN 'MI'
                    WHEN LOWER(raw_text) LIKE '%block%' OR LOWER(raw_text) LIKE '%conduction%' THEN 'CD'
                    WHEN LOWER(raw_text) LIKE '%hypertrophy%' OR LOWER(raw_text) LIKE '%enlargement%' THEN 'HYP'
                    WHEN LOWER(raw_text) LIKE '%st elevation%' OR LOWER(raw_text) LIKE '%st depression%' OR LOWER(raw_text) LIKE '%t wave%' OR LOWER(raw_text) LIKE '%ischemi%' THEN 'STTC'
                    WHEN (LOWER(raw_text) LIKE '%normal%' OR LOWER(raw_text) LIKE '%sinus rhythm%') 
                         AND LOWER(raw_text) NOT LIKE '%abnormal%' THEN 'NORM'
                    ELSE 'OTHER'
                END AS diagnostic_class
            FROM unpivoted
            WHERE raw_text IS NOT NULL 
              AND TRIM(raw_text) != ''
              AND LOWER(raw_text) NOT LIKE 'age not entered%'
              AND LOWER(raw_text) NOT LIKE 'lead(s) unsuitable%'
              AND LOWER(raw_text) NOT LIKE 'recording unsuitable%'
              AND LOWER(raw_text) NOT LIKE 'data quality%'
        )
        SELECT
            {current_max_id} + ROW_NUMBER() OVER () AS id,
            ecg_uid,
            source_dataset,
            raw_statement,
            clean_statement,
            diagnostic_class,
            NULL AS diagnostic_subclass,
            'report_finding' AS statement_category,
            is_uncertain,
            CASE WHEN is_uncertain THEN 50.0 ELSE 100.0 END AS likelihood
        FROM cleaned
    """)

    counts = {
        "mimic_patients": int(
            con.execute(
                "SELECT COUNT(*) FROM patients WHERE source_dataset = 'mimic'"
            ).fetchall()[0][0]
        ),
        "mimic_records": int(
            con.execute(
                "SELECT COUNT(*) FROM ecg_records WHERE source_dataset = 'mimic'"
            ).fetchall()[0][0]
        ),
        "mimic_measurements": int(
            con.execute("SELECT COUNT(*) FROM ecg_measurements").fetchall()[0][0]
        ),
        "mimic_labels": int(
            con.execute(
                "SELECT COUNT(*) FROM ecg_labels WHERE source_dataset = 'mimic'"
            ).fetchall()[0][0]
        ),
    }
    return counts


def migrate_all(
    db_path: Path | str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Executa a migração completa do PTB-XL e MIMIC-IV-ECG para o banco DuckDB."""
    base_data = data_dir if data_dir is not None else DATA_DIR
    con = get_connection(db_path=db_path)

    print(f"--> Inicializando esquema relacional em {db_path or DEFAULT_DB_PATH}...")
    create_schema(con)

    print(f"--> Migrando PTB-XL de {base_data / 'ptb-xl'}...")
    ptb_counts = migrate_ptbxl(con, ptbxl_dir=base_data / "ptb-xl")
    print(f"    PTB-XL concluído: {ptb_counts}")

    print(f"--> Migrando MIMIC-IV-ECG de {base_data / 'mimic-iv-ecg' / '1.0'}...")
    mimic_counts = migrate_mimic(con, mimic_dir=base_data / "mimic-iv-ecg" / "1.0")
    print(f"    MIMIC-IV-ECG concluído: {mimic_counts}")

    total_patients = int(con.execute("SELECT COUNT(*) FROM patients").fetchall()[0][0])
    total_records = int(
        con.execute("SELECT COUNT(*) FROM ecg_records").fetchall()[0][0]
    )
    total_labels = int(con.execute("SELECT COUNT(*) FROM ecg_labels").fetchall()[0][0])

    summary = {
        "total_patients": total_patients,
        "total_records": total_records,
        "total_labels": total_labels,
        **ptb_counts,
        **mimic_counts,
    }
    con.close()
    return summary


def query_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Retorna uma visão agregada dos registros por base de dados e divisão de treino/val/teste."""
    return con.execute("""
        SELECT 
            source_dataset,
            split,
            COUNT(*) AS n_records,
            COUNT(DISTINCT patient_uid) AS n_patients
        FROM ecg_records
        GROUP BY source_dataset, split
        ORDER BY source_dataset, split
    """).fetchdf()


def main() -> None:
    """Ponto de entrada para execução da migração via terminal."""
    print("=" * 70)
    print("CardioPredict - Migração de Dados para DuckDB")
    print("=" * 70)
    summary = migrate_all()
    print("\nResumo Final da Migração:")
    for key, val in summary.items():
        print(f"  - {key}: {val:,}")

    con = get_connection(read_only=True)
    splits_df = query_summary(con)
    print("\nDistribuição de Splits:")
    print(splits_df)
    con.close()


if __name__ == "__main__":
    main()
