"""
Módulo de Normalização e Extração de Achados Clínicos em Laudos de ECG.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from core.constants import DATA_DIR, MIMIC_DIR

# Padrão regex com os termos clínicos de incerteza diagnóstica usados nos laudos
UNCERTAINTY_PATTERN = re.compile(
    r"\b("
    r"possible|possibly|probable|probably|consider|cannot rule out|"
    r"can not rule out|may be|maybe|suggests|suggestive of|"
    r"questionable|equivocal|rule out"
    r")\b"
)

# Padrão para remover apenas o prefixo de incerteza ao extrair o conceito clínico central
LEADING_UNCERTAINTY_PREFIX = re.compile(
    r"^(?:possible|possibly|probable|probably|consider|questionable|equivocal)\s+"
)

# Padronizações de sinônimos ortográficos frequentes nos laudos de ECG
TEXT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:summary|ekg|ecg)\s*:\s*"), ""),
    (re.compile(r"\b1st\s+degree\b"), "first degree"),
    (re.compile(r"\b2nd\s+degree\b"), "second degree"),
    (re.compile(r"\b3rd\s+degree\b"), "third degree"),
    (re.compile(r"\ba-v\b"), "av"),
]


def normalize_report(value: Any) -> str | None:
    """Limpa e padroniza o texto de uma linha de laudo do ECG."""
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
    """Analisa uma frase de laudo para categoria, incerteza e peso."""
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
    """Extrai o achado central sem o modificador de incerteza inicial."""
    return LEADING_UNCERTAINTY_PREFIX.sub("", text).strip()


def normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Processa todas as colunas report_0..report_17 de um dataframe."""
    report_cols = [c for c in df.columns if c.startswith("report_")]
    records_list: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        study_id = row.get("study_id")
        for col_name in report_cols:
            val = row[col_name]
            cleaned = normalize_report(val)
            if cleaned:
                category, is_unc, is_bord, weight = classify_report_line(cleaned)
                core = extract_core_diagnosis(cleaned)
                records_list.append(
                    {
                        "study_id": study_id,
                        "report_slot": col_name,
                        "report_text": cleaned,
                        "core_diagnosis": core,
                        "diagnostic_category": category,
                        "is_uncertain": is_unc,
                        "is_borderline": is_bord,
                        "confidence_weight": weight,
                    }
                )

    return pd.DataFrame(records_list)


def main() -> None:
    csv_path = MIMIC_DIR / "machine_measurements.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "machine_measurements.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=5000)
        norm_df = normalize_reports(df)
        print("Amostra de laudos normalizados:")
        print(norm_df.head(10))


if __name__ == "__main__":
    main()
