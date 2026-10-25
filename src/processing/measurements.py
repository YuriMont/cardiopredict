"""
Módulo de Limpeza e Validação Fisiológica de Medições Numéricas de ECG.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from core.constants import (
    DATA_DIR,
    DERIVED_INTERVAL_LIMITS,
    ESSENTIAL_VENTRICULAR_COLS,
    MIMIC_DIR,
    PHYSIOLOGICAL_LIMITS,
)


def clean_machine_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa, valida e prepara as medições numéricas do ECG para uso clínico e em redes neurais."""
    report_cols = [c for c in df.columns if c.startswith("report_")]
    df_clean = df.drop(columns=report_cols).copy()

    # 1. Substituir códigos de erro e valores fora dos limites humanos por NaN
    for col, (min_val, max_val) in PHYSIOLOGICAL_LIMITS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(float)
            invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            df_clean.loc[invalid_mask, col] = np.nan

    # 2. Calcular variáveis clínicas derivadas (Engenharia de Atributos)
    df_clean["heart_rate_bpm"] = (60000.0 / df_clean["rr_interval"]).round(2)
    df_clean["p_duration"] = df_clean["p_end"] - df_clean["p_onset"]
    df_clean["pr_interval"] = df_clean["qrs_onset"] - df_clean["p_onset"]
    df_clean["qrs_duration"] = df_clean["qrs_end"] - df_clean["qrs_onset"]
    df_clean["qt_interval"] = df_clean["t_end"] - df_clean["qrs_onset"]

    # 3. Validar limites fisiológicos das durações derivadas
    for col, (min_val, max_val) in DERIVED_INTERVAL_LIMITS.items():
        if col in df_clean.columns:
            invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            df_clean.loc[invalid_mask, col] = np.nan

    # 4. Avaliar qualidade técnica do exame
    missing_ventricular = df_clean[ESSENTIAL_VENTRICULAR_COLS].isna().sum(axis=1)
    df_clean["is_technically_valid"] = (missing_ventricular == 0).astype(int)

    return df_clean


def main() -> None:
    csv_path = MIMIC_DIR / "machine_measurements.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "machine_measurements.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=5000)
        cleaned = clean_machine_measurements(df)
        print("Amostra de medições limpas:")
        print(
            cleaned[
                [
                    "heart_rate_bpm",
                    "qrs_duration",
                    "qt_interval",
                    "is_technically_valid",
                ]
            ].head(10)
        )


if __name__ == "__main__":
    main()
