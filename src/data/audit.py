"""
Módulo de Auditoria e Qualidade de Dados das Bases de ECG (MIMIC-IV-ECG e PTB-XL).
"""

import sys
from pathlib import Path
from typing import Any

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from core.constants import DATA_DIR, MIMIC_DIR, PHYSIOLOGICAL_LIMITS, PTBXL_DIR


def inspect_dataframe(
    df: pd.DataFrame,
    name: str,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Imprime diagnóstico descritivo do dataframe e retorna tabela de qualidade."""
    print(f"\n{'=' * 70}")
    print(f"ARQUIVO: {name}")
    print(f"{'=' * 70}")

    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Dimensões: {rows:,} linhas x {cols} colunas")
    print(f"Uso de memória: {memory_mb:.2f} MB")

    summary_records: list[dict[str, Any]] = []
    for col, series in df.items():
        null_count = int(series.isna().sum())
        null_pct = (null_count / rows * 100) if rows > 0 else 0.0

        empty_str_count = 0
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            empty_str_count = int(series.dropna().astype(str).str.strip().eq("").sum())

        summary_records.append(
            {
                "coluna": col,
                "tipo": str(series.dtype),
                "nulos": null_count,
                "nulos_%": round(null_pct, 2),
                "strings_vazias": empty_str_count,
                "unicos": int(series.nunique(dropna=True)),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    print("\n--- Tipos de Dados, Valores Ausentes e Cardinalidade ---")
    print(summary_df.to_string(index=False))

    print("\n--- Verificação de Duplicados ---")
    exact_dups = int(df.duplicated().sum())
    print(f"Linhas inteiras duplicadas: {exact_dups:,}")

    if key_columns:
        valid_keys = [k for k in key_columns if k in df.columns]
        if valid_keys:
            key_dups = int(df.duplicated(subset=valid_keys).sum())
            print(f"Duplicados na chave {valid_keys}: {key_dups:,}")

    return summary_df


def inspect_measurements_outliers(df: pd.DataFrame) -> None:
    """Verifica e quantifica valores biologicamente implausíveis ou sentinelas de hardware."""
    available_cols = [c for c in PHYSIOLOGICAL_LIMITS if c in df.columns]
    if not available_cols:
        return

    print("\n--- Estatísticas Descritivas das Medições Numéricas ---")
    numeric_df = df[available_cols].apply(pd.to_numeric, errors="coerce")
    print(numeric_df.describe().T[["count", "mean", "std", "min", "50%", "max"]])

    print("\n--- Auditoria de Violações dos Limites Fisiológicos ---")
    outlier_records: list[dict[str, Any]] = []
    total_rows = len(df)

    for col in available_cols:
        min_limit, max_limit = PHYSIOLOGICAL_LIMITS[col]
        series = pd.Series(numeric_df[col]).dropna()
        below = int((series < min_limit).sum())
        above = int((series > max_limit).sum())
        total_out = below + above
        pct = (total_out / total_rows * 100) if total_rows > 0 else 0.0

        outlier_records.append(
            {
                "coluna": col,
                "faixa_valida": f"[{min_limit}, {max_limit}]",
                "abaixo_min": below,
                "acima_max": above,
                "total_invalidos": total_out,
                "invalidos_%": round(pct, 2),
            }
        )

    outlier_df = pd.DataFrame(outlier_records)
    print(outlier_df.to_string(index=False))


def inspect_all_files(base_dir: Path = DATA_DIR) -> None:
    """Executa varredura e auditoria nos arquivos CSV de ambas as bases."""
    # 1. MIMIC-IV-ECG
    mimic_records_csv = MIMIC_DIR / "record_list.csv"
    if mimic_records_csv.exists():
        df_rec = pd.read_csv(mimic_records_csv)
        inspect_dataframe(df_rec, "mimic_record_list.csv", key_columns=["study_id"])

    mimic_meas_csv = MIMIC_DIR / "machine_measurements.csv"
    if mimic_meas_csv.exists():
        df_meas = pd.read_csv(mimic_meas_csv, low_memory=False)
        inspect_dataframe(
            df_meas, "mimic_machine_measurements.csv", key_columns=["study_id"]
        )
        inspect_measurements_outliers(df_meas)

    # 2. PTB-XL
    ptbxl_csv = PTBXL_DIR / "ptbxl_database.csv"
    if ptbxl_csv.exists():
        df_ptb = pd.read_csv(ptbxl_csv)
        inspect_dataframe(df_ptb, "ptbxl_database.csv", key_columns=["ecg_id"])


def main() -> None:
    inspect_all_files()


if __name__ == "__main__":
    main()
