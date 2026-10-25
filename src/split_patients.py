import numpy as np
import pandas as pd

from normalize_report import normalize_report
from read_database import DATA_DIR

# Frases que são apenas carimbos de resumo geral do exame (e não achados/diagnósticos específicos para a CNN multilabel).
SUMMARY_STAMP_TEXTS = {
    "normal ecg",
    "normal ecg except for rate",
    "within normal limits",
    "borderline ecg",
    "abnormal ecg",
}

# Avisos técnicos do aparelho que não são patologias do coração.
NON_DIAGNOSTIC_PREFIXES = (
    "age not entered",
    "lead(s) unsuitable",
    "recording unsuitable",
    "data quality",
)


def extract_exam_findings_and_status(
    measurements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada exame (`study_id`), extrai e separa:
    1. `ecg_status`: O resumo geral ('normal', 'borderline', 'abnormal', 'unclassified').
    2. `findings_list` e `findings_str`: A lista de diagnósticos/achados específicos (Findings)
       daquele exame (ex.: 'sinus bradycardia | first degree av block | left axis deviation'),
       já unificados pelo `core_diagnosis` para uso em predição MULTILABEL na CNN.
    3. `n_findings`: Quantidade de diagnósticos simultâneos no mesmo exame (Cardinalidade Multilabel).
    4. `is_multilabel`: 1 se o exame possui 2 ou mais diagnósticos simultâneos, 0 caso contrário.

    COMO TRATAMOS EXAMES COM MAIS DE UM DIAGNÓSTICO (MULTILABEL):
    - Um mesmo ECG frequentemente apresenta múltiplos achados simultâneos (ex.: ritmo + bloqueio + isquemia).
    - Em vez de escolher apenas o primeiro (`report_0`) e perder os demais, preservamos todos os
      achados em `findings_list` (separados por ' | ' em `findings_str`).
    - Para treinar a CNN no Keras, cada `finding` vira uma coluna binária independente (0 ou 1) via
      `sklearn.preprocessing.MultiLabelBinarizer`, usando ativação `sigmoid` e `binary_crossentropy`.
    """
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

            # Preserva exatamente o texto original colocado na base de dados (machine_measurements.csv)
            if raw_text not in seen:
                seen.add(raw_text)
                exam_findings.append(raw_text)

        n_find = len(exam_findings)
        n_findings_list.append(n_find)
        findings_strs.append(" | ".join(exam_findings) if exam_findings else "none")
        primary_findings.append(exam_findings[0] if exam_findings else "none")

    # Concatena todas as linhas de report_0..report_17 para identificar o status geral e a incerteza
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

    # Nos aparelhos antigos (~105 mil exames) que não imprimiam o carimbo final 'Normal/Abnormal ECG',
    # inferimos 'normal' se o único achado for 'sinus rhythm', ou 'abnormal' se houver outros achados patológicos.
    findings_series = pd.Series(findings_strs, index=measurements_df.index)
    inferred_normal = (
        (~is_abnormal)
        & (~is_borderline)
        & (~is_normal)
        & (findings_series.str.lower().str.rstrip(".") == "sinus rhythm")
    )
    inferred_abnormal = (
        (~is_abnormal)
        & (~is_borderline)
        & (~is_normal)
        & (~inferred_normal)
        & (findings_series != "none")
    )

    status = np.select(
        [
            is_abnormal | inferred_abnormal,
            is_borderline,
            is_normal | inferred_normal,
        ],
        ["abnormal", "borderline", "normal"],
        default="unclassified",
    )

    target_soft = np.select(
        [status == "abnormal", status == "borderline", status == "normal"],
        [1.0, 0.5, 0.0],
        default=np.nan,
    )

    sample_weight = np.where(
        (status == "borderline") | has_uncertainty,
        0.5,
        1.0,
    )

    n_findings_arr = np.asarray(n_findings_list, dtype=np.int64)

    return pd.DataFrame(
        {
            "subject_id": measurements_df["subject_id"],
            "study_id": measurements_df["study_id"],
            "ecg_status": status,
            "primary_finding": primary_findings,
            "findings_str": findings_strs,
            "n_findings": n_findings_arr,
            "is_multilabel": (n_findings_arr >= 2).astype(int),
            "is_borderline_study": is_borderline.astype(int),
            "has_uncertainty_study": has_uncertainty.astype(int),
            "target_soft": target_soft,
            "sample_weight": sample_weight,
        }
    )


def build_study_diagnostic_status(
    measurements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Alias compatível que retorna o status diagnóstico e os findings multilabel por exame."""
    return extract_exam_findings_and_status(measurements_df)


def split_by_patient(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Realiza a divisão Treino / Validação / Teste agrupada rigorosamente por PACIENTE (`subject_id`).
    Garante que 100% dos exames de um paciente fiquem exclusivamente em um único conjunto.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            "As proporções train_size + val_size + test_size devem somar 1.0"
        )

    result_df = df.copy()
    unique_patients = np.unique(result_df["subject_id"].to_numpy(dtype=np.int64))
    n_patients = len(unique_patients)

    rng = np.random.default_rng(random_state)
    shuffled_patients = rng.permutation(unique_patients)

    n_train = int(n_patients * train_size)
    n_val = int(n_patients * val_size)

    train_patients = shuffled_patients[:n_train].tolist()
    val_patients = shuffled_patients[n_train : n_train + n_val].tolist()

    is_train = result_df["subject_id"].isin(train_patients)
    is_val = result_df["subject_id"].isin(val_patients)

    result_df["split"] = np.select(
        [is_train, is_val],
        ["train", "val"],
        default="test",
    )
    return result_df


def verify_zero_patient_leakage(df: pd.DataFrame) -> None:
    """Audita e comprova matematicamente que a interseção de pacientes entre os splits é ZERO."""
    train_patients = set(df.loc[df["split"] == "train", "subject_id"])
    val_patients = set(df.loc[df["split"] == "val", "subject_id"])
    test_patients = set(df.loc[df["split"] == "test", "subject_id"])

    overlap_train_val = len(train_patients & val_patients)
    overlap_train_test = len(train_patients & test_patients)
    overlap_val_test = len(val_patients & test_patients)

    print(f"\n{'=' * 75}")
    print("AUDITORIA DE SEPARAÇÃO POR PACIENTES E MULTILABEL FINDINGS")
    print(f"{'=' * 75}")
    print(f" Interseção de pacientes (Treino ∩ Validação): {overlap_train_val}")
    print(f" Interseção de pacientes (Treino ∩ Teste):     {overlap_train_test}")
    print(f" Interseção de pacientes (Validação ∩ Teste):  {overlap_val_test}")

    summary_rows = []
    total_exams = len(df)
    total_patients = len(set(df["subject_id"]))

    for split_name in ["train", "val", "test"]:
        sub = df.loc[df["split"] == split_name]
        n_exams = len(sub)
        n_patients = len(set(sub["subject_id"]))
        multilabel_pct = float((sub["is_multilabel"] == 1).mean() * 100)
        summary_rows.append(
            {
                "split": split_name,
                "pacientes_unicos": n_patients,
                "pacientes_%": round(n_patients / total_patients * 100, 2),
                "exames_totais": n_exams,
                "exames_%": round(n_exams / total_exams * 100, 2),
                "normal": int((sub["ecg_status"] == "normal").sum()),
                "borderline": int((sub["ecg_status"] == "borderline").sum()),
                "abnormal": int((sub["ecg_status"] == "abnormal").sum()),
                "multilabel_>=2_findings_%": round(multilabel_pct, 2),
            }
        )

    print("\n--- Resumo da Divisão por Pacientes, Classes e Multilabel ---")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main() -> None:
    print("Carregando machine_measurements.csv e record_list.csv...")
    measurements = pd.read_csv(DATA_DIR / "machine_measurements.csv", low_memory=False)
    records = pd.read_csv(DATA_DIR / "record_list.csv", low_memory=False)

    status_df = build_study_diagnostic_status(measurements)
    dataset_df = records.merge(
        status_df.drop(columns=["subject_id"]),
        on="study_id",
        how="inner",
    )

    split_df = split_by_patient(
        dataset_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    verify_zero_patient_leakage(split_df)

    output_path = DATA_DIR / "patient_splits.csv"
    split_df.to_csv(output_path, index=False)
    print(f"\nTabela de splits e findings multilabel salva em: {output_path}")


if __name__ == "__main__":
    main()
