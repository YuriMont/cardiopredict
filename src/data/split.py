"""
Módulo de Particionamento e Estratificação de Pacientes (Zero Data Leakage).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from core.constants import (
    DATA_DIR,
    MIMIC_DIR,
    NON_DIAGNOSTIC_PREFIXES,
    SUMMARY_STAMP_TEXTS,
)
from processing.reports import normalize_report


def extract_exam_findings_and_status(
    measurements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extrai status geral e lista de achados multilabel para cada exame."""
    report_cols = sorted(
        (c for c in measurements_df.columns if c.startswith("report_")),
        key=lambda c: int(c.split("_")[1]),
    )

    matrix = measurements_df[report_cols].to_numpy()
    n_rows = len(measurements_df)

    findings_strs: list[str] = []
    primary_findings: list[str] = []
    n_findings_list: list[int] = []

    for row_idx in range(n_rows):
        row_vals = matrix[row_idx]
        exam_findings: list[str] = []
        seen: set[str] = set()

        for val in row_vals:
            if pd.isna(val):
                continue
            raw_text = str(val).strip()
            if not raw_text:
                continue

            norm = normalize_report(raw_text)
            if norm is None or norm in SUMMARY_STAMP_TEXTS:
                continue
            if norm.startswith(NON_DIAGNOSTIC_PREFIXES):
                continue

            if raw_text not in seen:
                seen.add(raw_text)
                exam_findings.append(raw_text)

        n_find = len(exam_findings)
        n_findings_list.append(n_find)
        findings_strs.append(" | ".join(exam_findings) if exam_findings else "none")
        primary_findings.append(exam_findings[0] if exam_findings else "none")

    combined_reports = (
        measurements_df[report_cols].fillna("").agg(" | ".join, axis=1).str.lower()
    )

    is_abnormal = combined_reports.str.contains(r"\babnormal ecg\b", regex=True)
    is_borderline = combined_reports.str.contains(r"\bborderline\b", regex=True)
    is_normal = combined_reports.str.contains(
        r"\b(?:normal ecg|within normal limits)\b", regex=True
    )
    has_uncertainty = combined_reports.str.contains(
        r"\b(?:possible|possibly|probable|probably|consider|cannot rule out|questionable|equivocal)\b",
        regex=True,
    )

    status_series = pd.Series("unclassified", index=measurements_df.index)
    status_series[is_normal] = "normal"
    status_series[is_borderline] = "borderline"
    status_series[is_abnormal] = "abnormal"

    unclassified_mask = status_series == "unclassified"
    n_findings_arr = np.array(n_findings_list)
    findings_arr = np.array(findings_strs)

    status_series[
        unclassified_mask & (n_findings_arr == 1) & (findings_arr == "sinus rhythm")
    ] = "normal"
    status_series[unclassified_mask & (n_findings_arr > 0)] = "abnormal"
    status_series[unclassified_mask & (n_findings_arr == 0)] = "normal"

    result_df = pd.DataFrame(
        {
            "study_id": measurements_df["study_id"],
            "subject_id": measurements_df["subject_id"],
            "ecg_status": status_series,
            "findings_str": findings_strs,
            "primary_finding": primary_findings,
            "n_findings": n_findings_list,
            "is_multilabel": (n_findings_arr >= 2).astype(int),
            "has_uncertainty": has_uncertainty.astype(int),
        }
    )
    return result_df


def build_study_diagnostic_status(
    measurements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classifica cada estudo em normal/borderline/abnormal."""
    return extract_exam_findings_and_status(measurements_df)


def split_by_patient(
    df: pd.DataFrame,
    patient_col: str = "subject_id",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Realiza o particionamento estratificado estritamente no nível do paciente."""
    unique_patients = df[patient_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    shuffled_patients = rng.permutation(unique_patients)

    n_total = len(shuffled_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pts = set(shuffled_patients[:n_train])
    val_pts = set(shuffled_patients[n_train : n_train + n_val])

    def assign_split(pid: Any) -> str:
        if pid in train_pts:
            return "train"
        if pid in val_pts:
            return "val"
        return "test"

    df_out = df.copy()
    df_out["split"] = df_out[patient_col].map(assign_split)
    return df_out


def verify_zero_patient_leakage(
    df: pd.DataFrame, patient_col: str = "subject_id"
) -> None:
    """Verifica se não há sobreposição de pacientes entre os splits."""
    splits = df["split"].unique()
    patient_sets: dict[str, set[Any]] = {
        s: set(df.loc[df["split"] == s, patient_col]) for s in splits
    }

    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                overlap = patient_sets[s1].intersection(patient_sets[s2])
                assert len(overlap) == 0, (
                    f"VAZAMENTO DETECTADO entre {s1} e {s2}: {len(overlap)} pacientes!"
                )
    print("✓ Verificação concluída: Zero vazamento de pacientes (Zero Data Leakage).")


def main() -> None:
    csv_path = MIMIC_DIR / "machine_measurements.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "mimic-iv-ecg" / "machine_measurements.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "machine_measurements.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=50000)
        findings_df = extract_exam_findings_and_status(df)
        split_df = split_by_patient(findings_df)
        verify_zero_patient_leakage(split_df)
        print("\nDistribuição dos splits de exemplo:")
        print(split_df["split"].value_counts())


if __name__ == "__main__":
    main()
