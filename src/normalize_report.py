import re
from typing import Any

import pandas as pd

from read_database import DATA_DIR

# Padrão regex com os termos clínicos de incerteza diagnóstica usados nos laudos do MIMIC-IV-ECG.
# Motivo: Preservar e sinalizar a incerteza evita tratar suspeitas ("possible infarct")
# como diagnósticos definitivos (100% de certeza), reduzindo ruído de rótulo na CNN.
UNCERTAINTY_PATTERN = re.compile(
    r"\b("
    r"possible|possibly|probable|probably|consider|cannot rule out|"
    r"can not rule out|may be|maybe|suggests|suggestive of|"
    r"questionable|equivocal|rule out"
    r")\b"
)

# Padrão para remover apenas o prefixo de incerteza ao extrair o conceito clínico central (core_diagnosis),
# sem perder o texto original completo em `report_text`.
LEADING_UNCERTAINTY_PREFIX = re.compile(
    r"^(?:possible|possibly|probable|probably|consider|questionable|equivocal)\s+"
)

# Padronizações de sinônimos ortográficos frequentes nos laudos de ECG.
TEXT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:summary|ekg|ecg)\s*:\s*"), ""),
    (re.compile(r"\b1st\s+degree\b"), "first degree"),
    (re.compile(r"\b2nd\s+degree\b"), "second degree"),
    (re.compile(r"\b3rd\s+degree\b"), "third degree"),
    (re.compile(r"\ba-v\b"), "av"),
]


def normalize_report(value: Any) -> str | None:
    """
    Limpa e padroniza o texto de uma linha de laudo do ECG.
    - Converte para minúsculas e remove espaços duplicados.
    - Remove pontuações finais soltas (ex.: unifica 'sinus rhythm.' e 'sinus rhythm').
    - Remove prefixos redundantes ('summary: borderline ecg' -> 'borderline ecg').
    - Preserva integralmente os termos de incerteza ('possible', 'probable', 'borderline').
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" *=-\t.,;:")

    for pattern, replacement in TEXT_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = re.sub(r"\s+", " ", text).strip(" *=-\t.,;:")
    return text or None


def classify_report_line(text: str) -> tuple[str, int, int, float]:
    """
    Analisa uma frase de laudo normalizada para:
    1. Preservar e sinalizar incerteza (`is_uncertain = 1` quando há dúvida diagnóstica).
    2. Identificar e tratar explicitamente casos limítrofes (`is_borderline = 1` para 'borderline ecg').
    3. Definir a categoria da frase (`diagnostic_category`: 'normal', 'borderline', 'abnormal' ou 'finding').
    4. Atribuir um peso de confiança (`confidence_weight`) para uso como `sample_weight` no Keras/sklearn.
    """
    is_uncertain = int(bool(UNCERTAINTY_PATTERN.search(text)))
    is_borderline = int("borderline" in text)

    if text in {"normal ecg", "normal ecg except for rate", "within normal limits"}:
        category = "normal"
    elif text == "borderline ecg" or is_borderline:
        category = "borderline"
    elif text in {"abnormal ecg"}:
        category = "abnormal"
    else:
        category = "finding"

    confidence_weight = 0.5 if (is_uncertain or is_borderline) else 1.0
    return category, is_uncertain, is_borderline, confidence_weight


def extract_core_diagnosis(text: str) -> str:
    """
    Extrai o achado central sem o modificador inicial (ex.: 'possible left atrial abnormality'
    -> 'left atrial abnormality'), permitindo agrupar a patologia mantendo a flag `is_uncertain`.
    """
    return LEADING_UNCERTAINTY_PREFIX.sub("", text).strip()


def normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra e normaliza todas as colunas report_0..report_17 de machine_measurements.csv
    preservando incerteza e tratando casos 'borderline ecg'.
    """
    reports_cols = sorted(
        (col for col in df.columns if col.startswith("report_")),
        key=lambda col: int(col.split("_")[1]),
    )

    records: list[dict[str, Any]] = []

    subject_ids = df["subject_id"].to_numpy()
    study_ids = df["study_id"].to_numpy()
    report_matrix = df[reports_cols].to_numpy()

    for row_idx in range(len(df)):
        subj_id = int(subject_ids[row_idx])
        stud_id = int(study_ids[row_idx])
        row_vals = report_matrix[row_idx]

        for col_idx, col_name in enumerate(reports_cols):
            text = normalize_report(row_vals[col_idx])
            if text is None:
                continue

            category, is_uncertain, is_borderline, weight = classify_report_line(text)
            core_diag = extract_core_diagnosis(text)

            records.append(
                {
                    "subject_id": subj_id,
                    "study_id": stud_id,
                    "report_column": col_name,
                    "report_text": text,
                    "core_diagnosis": core_diag,
                    "diagnostic_category": category,
                    "is_uncertain": is_uncertain,
                    "is_borderline": is_borderline,
                    "confidence_weight": weight,
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "subject_id",
            "study_id",
            "report_column",
            "report_text",
            "core_diagnosis",
            "diagnostic_category",
            "is_uncertain",
            "is_borderline",
            "confidence_weight",
        ],
    )


def build_unique_reports_catalog(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa todos os laudos únicos encontrados na base inteira com sua contagem,
    percentual, categoria diagnóstica, flag de incerteza e peso de confiança.
    """
    total_lines = len(normalized_df)
    counts = (
        normalized_df.value_counts(
            [
                "report_text",
                "core_diagnosis",
                "diagnostic_category",
                "is_uncertain",
                "is_borderline",
                "confidence_weight",
            ]
        )
        .reset_index(name="ocorrencias")
        .sort_values("ocorrencias", ascending=False)
    )
    counts["percentual_%"] = (counts["ocorrencias"] / total_lines * 100).round(4)
    return counts


def print_all_reports_summary(
    normalized_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> None:
    """Imprime no terminal o resumo detalhado de todas as colunas report_0..report_17 e os laudos."""
    print(f"\n{'=' * 80}")
    print("1. COLUNAS DE REPORT FILTRADAS (report_0 até report_17)")
    print(f"{'=' * 80}")
    col_counts = (
        normalized_df["report_column"]
        .value_counts()
        .rename_axis("report_column")
        .reset_index(name="laudos_preenchidos")
    )
    col_counts["ordem"] = col_counts["report_column"].map(
        lambda c: int(str(c).split("_")[1])
    )
    col_counts = col_counts.sort_values("ordem").drop(columns=["ordem"])
    print(col_counts.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(
        "2. RESUMO POR CATEGORIA DIAGNÓSTICA (NORMAL / BORDERLINE / ABNORMAL / FINDING)"
    )
    print(f"{'=' * 80}")
    cat_counts = (
        normalized_df["diagnostic_category"]
        .value_counts()
        .rename_axis("diagnostic_category")
        .reset_index(name="ocorrencias")
    )
    cat_counts["percentual_%"] = (
        cat_counts["ocorrencias"] / len(normalized_df) * 100
    ).round(2)
    print(cat_counts.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(
        f"3. CATÁLOGO DE LAUDOS ÚNICOS ({len(catalog_df):,} LAUDOS DISTINTOS EM {len(normalized_df):,} OCORRÊNCIAS)"
    )
    print(f"{'=' * 80}")
    print("\n--- Top 50 Laudos Mais Frequentes na Base ---")
    print(
        catalog_df[
            [
                "report_text",
                "diagnostic_category",
                "is_uncertain",
                "is_borderline",
                "ocorrencias",
                "percentual_%",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


def main() -> None:
    measurements = pd.read_csv(
        DATA_DIR / "machine_measurements.csv",
        low_memory=False,
    )

    normalized = normalize_reports(measurements)
    catalog = build_unique_reports_catalog(normalized)

    output_path = DATA_DIR / "normalized_reports.csv"
    catalog_path = DATA_DIR / "unique_reports_catalog.csv"

    normalized.to_csv(output_path, index=False)
    catalog.to_csv(catalog_path, index=False)

    print_all_reports_summary(normalized, catalog)
    print(f"\nLaudos normalizados salvos em: {output_path}")
    print(
        f"Catálogo completo com todos os {len(catalog):,} laudos únicos salvo em: {catalog_path}"
    )


if __name__ == "__main__":
    main()
