"""
Módulo de Ontologia e Harmonização Clínica Multilabel (MIMIC-IV-ECG & PTB-XL).

Mapeia laudos médicos textuais e códigos SCP-ECG em uma matriz multilabel de
5 superclasses e 25 subclasses clínicas críticas de emergência, com
propagação hierárquica e pesos de incerteza diagnóstica.
"""

from __future__ import annotations

import re
from typing import Any

from core.constants import SUBCLASSES, SUPERCLASSES, TARGET_CLASSES

# =============================================================================
# 1. Padrões de Incerteza Diagnóstica
# =============================================================================
UNCERTAINTY_PATTERN: re.Pattern[str] = re.compile(
    r"\b("
    r"possible|possibly|probable|probably|consider|cannot rule out|"
    r"can not rule out|may be|maybe|suggests|suggestive of|"
    r"questionable|equivocal|borderline|rule out"
    r")\b",
    re.IGNORECASE,
)

# =============================================================================
# 2. Mapeamento Ontológico PTB-XL (SCP-ECG -> TARGET_CLASSES)
# =============================================================================
# Mapeia código SCP-ECG para lista de classes alvo ativadas
PTBXL_SCP_TO_TARGETS: dict[str, list[str]] = {
    # Superclasses diretas
    "NORM": ["NORM"],
    # Ritmos
    "AFIB": ["AFIB"],
    "AFLT": ["AFLT"],
    "SR": ["SR"],
    "SBRAD": ["SBRAD"],
    "STACH": ["STACH"],
    "PVC": ["PVC"],
    "PAC": ["PAC"],
    # Infarto do Miocárdio (MI) e Localizações Anatômicas
    "AMI": ["AMI", "MI"],
    "ASMI": ["AMI", "MI"],
    "ALMI": ["AMI", "MI"],
    "INJAS": ["AMI", "MI", "STE"],
    "INJAL": ["AMI", "MI", "STE"],
    "INJLA": ["AMI", "MI", "STE"],
    "IMI": ["IMI", "MI"],
    "ILMI": ["IMI", "MI"],
    "IPLMI": ["IMI", "MI"],
    "IPMI": ["IMI", "MI"],
    "INJIN": ["IMI", "MI", "STE"],
    "INJIL": ["IMI", "MI", "STE"],
    "LMI": ["LMI", "MI"],
    "PMI": ["MI"],
    # Distúrbios de Condução (CD) e Bloqueios
    "CLBBB": ["CLBBB", "CD"],
    "ILBBB": ["CLBBB", "CD"],
    "CRBBB": ["CRBBB", "CD"],
    "IRBBB": ["IRBBB", "CD"],
    "LAFB": ["LAFB", "CD"],
    "LPFB": ["LAFB", "CD"],
    "1AVB": ["1AVB", "CD"],
    "2AVB": ["2AVB", "CD"],
    "3AVB": ["3AVB", "CD"],
    "IVCD": ["IVCD", "CD"],
    "WPW": ["CD"],
    # Hipertrofias e Sobrecargas (HYP)
    "LVH": ["LVH", "HYP"],
    "VCLVH": ["LVH", "HYP"],
    "RVH": ["RVH", "HYP"],
    "LAO/LAE": ["HYP"],
    "RAO/RAE": ["HYP"],
    "SEHYP": ["HYP"],
    # Alterações de ST/T e Isquemia (STTC)
    "ISCAN": ["STTC_ISCA", "STTC"],
    "ISCAS": ["STTC_ISCA", "STTC"],
    "ISCAL": ["STTC_ISCA", "STTC"],
    "ISCLA": ["STTC_ISCA", "STTC"],
    "ISCIN": ["STTC_ISCI", "STTC"],
    "ISCIL": ["STTC_ISCI", "STTC"],
    "ISC_": ["STTC"],
    "NST_": ["STTC"],
    "STD_": ["STD", "STTC"],
    "STE_": ["STE", "STTC"],
    "TAB_": ["STTC"],
    "INVT": ["STTC"],
    "DIG": ["STTC"],
    "LNGQT": ["STTC"],
    "ANEUR": ["STTC"],
    "EL": ["STTC"],
}

# =============================================================================
# 3. Regras Léxicas Determinísticas para Laudos MIMIC-IV-ECG
# =============================================================================
# Lista de tuplas: (expressão regular compilada, lista de targets ativados)
MIMIC_LEXICAL_RULES: list[tuple[re.Pattern[str], list[str]]] = [
    # Normalidade
    (
        re.compile(
            r"\b(normal ecg|within normal limits|normal ecg except for rate)\b",
            re.IGNORECASE,
        ),
        ["NORM"],
    ),
    # Ritmos
    (
        re.compile(r"\b(atrial fibrillation|afib)\b", re.IGNORECASE),
        ["AFIB"],
    ),
    (
        re.compile(r"\b(atrial flutter)\b", re.IGNORECASE),
        ["AFLT"],
    ),
    (
        re.compile(r"\b(sinus rhythm)\b", re.IGNORECASE),
        ["SR"],
    ),
    (
        re.compile(r"\b(sinus bradycardia)\b", re.IGNORECASE),
        ["SBRAD", "SR"],
    ),
    (
        re.compile(r"\b(sinus tachycardia)\b", re.IGNORECASE),
        ["STACH", "SR"],
    ),
    (
        re.compile(
            r"\b(premature ventricular|ventricular premature|pvc|ventricular ectopic)\b",
            re.IGNORECASE,
        ),
        ["PVC"],
    ),
    (
        re.compile(
            r"\b(premature atrial|atrial premature|pac|atrial ectopic)\b",
            re.IGNORECASE,
        ),
        ["PAC"],
    ),
    # Infarto Agudo com Supra de ST (STEMI)
    (
        re.compile(
            r"\b(st elevation myocardial infarction|stemi|acute st elevation|acute infarct)\b",
            re.IGNORECASE,
        ),
        ["STEMI", "STE", "MI"],
    ),
    # Infarto Anterior / Anterosseptal / Septal (AMI)
    (
        re.compile(
            r"\b(anterior infarct|anteroseptal infarct|anterolateral infarct|septal infarct|anterior transmural infarct)\b",
            re.IGNORECASE,
        ),
        ["AMI", "MI"],
    ),
    # Infarto Inferior / Inferoposterior / Inferolateral (IMI)
    (
        re.compile(
            r"\b(inferior infarct|inferolateral infarct|inferoposterior infarct)\b",
            re.IGNORECASE,
        ),
        ["IMI", "MI"],
    ),
    # Infarto Lateral / Posterolateral (LMI)
    (
        re.compile(
            r"\b(lateral infarct|posterior infarct|posterolateral infarct)\b",
            re.IGNORECASE,
        ),
        ["LMI", "MI"],
    ),
    # Infarto Inespecífico
    (
        re.compile(
            r"\b(myocardial infarction|infarct - age undetermined|old infarct|infarct)\b",
            re.IGNORECASE,
        ),
        ["MI"],
    ),
    # Bloqueio de Ramo Esquerdo (CLBBB)
    (
        re.compile(
            r"\b(left bundle branch block|complete left bundle branch block|clbbb|lbbb)\b",
            re.IGNORECASE,
        ),
        ["CLBBB", "CD"],
    ),
    # Bloqueio de Ramo Direito Completo (CRBBB)
    (
        re.compile(
            r"\b(right bundle branch block|complete right bundle branch block|crbbb|rbbb)\b",
            re.IGNORECASE,
        ),
        ["CRBBB", "CD"],
    ),
    # Bloqueio de Ramo Direito Incompleto (IRBBB)
    (
        re.compile(
            r"\b(incomplete right bundle branch block|irbbb|incomplete rbbb)\b",
            re.IGNORECASE,
        ),
        ["IRBBB", "CD"],
    ),
    # Bloqueio Fascicular / Divisional Anterossuperior (LAFB)
    (
        re.compile(
            r"\b(left anterior fascicular block|left anterior hemiblock|lafb)\b",
            re.IGNORECASE,
        ),
        ["LAFB", "CD"],
    ),
    # Bloqueio Atrioventricular de 1º Grau (1AVB)
    (
        re.compile(
            r"\b(first degree av block|1st degree a-v block|1st degree av block|first degree a-v block|prolonged pr interval)\b",
            re.IGNORECASE,
        ),
        ["1AVB", "CD"],
    ),
    # Bloqueio Atrioventricular de 2º Grau (2AVB)
    (
        re.compile(
            r"\b(second degree av block|2nd degree a-v block|mobitz type|mobitz i|mobitz ii|wenckebach)\b",
            re.IGNORECASE,
        ),
        ["2AVB", "CD"],
    ),
    # Bloqueio Atrioventricular Total / 3º Grau (3AVB)
    (
        re.compile(
            r"\b(third degree av block|3rd degree a-v block|complete heart block|complete av block|bavt)\b",
            re.IGNORECASE,
        ),
        ["3AVB", "CD"],
    ),
    # Atraso Inespecífico de Condução Intraventricular (IVCD)
    (
        re.compile(
            r"\b(iv conduction defect|intraventricular conduction delay|ivcd|intraventricular block)\b",
            re.IGNORECASE,
        ),
        ["IVCD", "CD"],
    ),
    # Hipertrofia Ventricular Esquerda (LVH)
    (
        re.compile(
            r"\b(left ventricular hypertrophy|lvh|voltage criteria for lvh)\b",
            re.IGNORECASE,
        ),
        ["LVH", "HYP"],
    ),
    # Hipertrofia Ventricular Direita (RVH)
    (
        re.compile(
            r"\b(right ventricular hypertrophy|rvh)\b",
            re.IGNORECASE,
        ),
        ["RVH", "HYP"],
    ),
    # Sobrecargas Atriais / Inespecíficas (HYP)
    (
        re.compile(
            r"\b(left atrial abnormality|left atrial enlargement|right atrial abnormality|right atrial enlargement|atrial enlargement)\b",
            re.IGNORECASE,
        ),
        ["HYP"],
    ),
    # Alterações Isquêmicas Anteriores / Anterosseptais de ST/T
    (
        re.compile(
            r"\b(anterior.*ischemia|anteroseptal.*ischemia|anterior t wave changes|anteroseptal t wave changes)\b",
            re.IGNORECASE,
        ),
        ["STTC_ISCA", "STTC"],
    ),
    # Alterações Isquêmicas Inferiores de ST/T
    (
        re.compile(
            r"\b(inferior.*ischemia|inferior t wave changes)\b",
            re.IGNORECASE,
        ),
        ["STTC_ISCI", "STTC"],
    ),
    # Supradesnivelamento do Segmento ST (STE)
    (
        re.compile(
            r"\b(st elevation|st segment elevation|early repolarization)\b",
            re.IGNORECASE,
        ),
        ["STE", "STTC"],
    ),
    # Infradesnivelamento do Segmento ST (STD)
    (
        re.compile(
            r"\b(st depression|st segment depression)\b",
            re.IGNORECASE,
        ),
        ["STD", "STTC"],
    ),
    # Alterações Inespecíficas de Onda T e Segmento ST
    (
        re.compile(
            r"\b(t wave abnormality|t wave changes|st-t changes|nonspecific st-t|st and/or t wave changes)\b",
            re.IGNORECASE,
        ),
        ["STTC"],
    ),
]


# =============================================================================
# 4. Função Principal de Harmonização de Laudos
# =============================================================================
def map_exam_statements(
    statements: list[str],
    is_ptbxl: bool = False,
) -> tuple[dict[str, int], dict[str, float]]:
    """Harmoniza uma lista de declarações/códigos clínicos em alvos multilabel e pesos.

    Retorna:
        - labels: dicionário {classe: 0 ou 1} para as 30 classes alvo.
        - weights: dicionário {classe: float} com peso de confiança (1.0, 0.5).
    """
    labels: dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    weights: dict[str, float] = {c: 1.0 for c in TARGET_CLASSES}

    # Rastreia quais classes foram explicitamente ativadas
    activated_targets: set[str] = set()

    for stmt in statements:
        if not stmt or not isinstance(stmt, str):
            continue
        cleaned = stmt.strip()
        if not cleaned:
            continue

        is_uncertain = bool(UNCERTAINTY_PATTERN.search(cleaned))
        stmt_weight = 0.5 if is_uncertain else 1.0

        if is_ptbxl:
            code = cleaned.upper()
            if code in PTBXL_SCP_TO_TARGETS:
                for target in PTBXL_SCP_TO_TARGETS[code]:
                    labels[target] = 1
                    if target not in activated_targets:
                        weights[target] = stmt_weight
                        activated_targets.add(target)
                    else:
                        weights[target] = max(weights[target], stmt_weight)
        else:
            # Varredura léxica em regras MIMIC
            for pattern, targets in MIMIC_LEXICAL_RULES:
                if pattern.search(cleaned):
                    for target in targets:
                        labels[target] = 1
                        if target not in activated_targets:
                            weights[target] = stmt_weight
                            activated_targets.add(target)
                        else:
                            weights[target] = max(weights[target], stmt_weight)

    # -------------------------------------------------------------------------
    # 5. Regras de Propagação Hierárquica e Consistência Clínica
    # -------------------------------------------------------------------------
    # Subclasses de Infarto -> Superclasse MI
    if any(labels[c] == 1 for c in ("AMI", "IMI", "LMI", "STEMI")):
        labels["MI"] = 1
        weights["MI"] = max(
            weights[c] for c in ("AMI", "IMI", "LMI", "STEMI") if labels[c] == 1
        )

    # Subclasses de Bloqueio/Condução -> Superclasse CD
    cd_children = (
        "CLBBB",
        "CRBBB",
        "IRBBB",
        "LAFB",
        "1AVB",
        "2AVB",
        "3AVB",
        "IVCD",
    )
    if any(labels[c] == 1 for c in cd_children):
        labels["CD"] = 1
        weights["CD"] = max(weights[c] for c in cd_children if labels[c] == 1)

    # Subclasses de Sobrecarga -> Superclasse HYP
    if labels["LVH"] == 1 or labels["RVH"] == 1:
        labels["HYP"] = 1
        weights["HYP"] = max(weights[c] for c in ("LVH", "RVH") if labels[c] == 1)

    # Subclasses de Isquemia/Repolarização -> Superclasse STTC
    sttc_children = ("STTC_ISCA", "STTC_ISCI", "STD", "STE")
    if any(labels[c] == 1 for c in sttc_children):
        labels["STTC"] = 1
        weights["STTC"] = max(weights[c] for c in sttc_children if labels[c] == 1)

    # -------------------------------------------------------------------------
    # 6. Regras de Normalidade e Exclusão Mútua
    # -------------------------------------------------------------------------
    pathology_superclasses = ("MI", "STTC", "CD", "HYP")
    has_any_pathology = any(labels[p] == 1 for p in pathology_superclasses)

    if has_any_pathology:
        # Se há qualquer patologia clínica, NORM é estritamente 0
        labels["NORM"] = 0
        weights["NORM"] = 1.0
    elif labels["NORM"] == 1:
        # NORM ativado e sem patologias
        labels["NORM"] = 1

    return labels, weights


def extract_targets_for_record(
    raw_statements: list[str] | str | Any,
    source_dataset: str,
) -> tuple[dict[str, int], dict[str, float]]:
    """Função utilitária que recebe strings ou listas e despacha para o motor de harmonização."""
    if isinstance(raw_statements, str):
        # Pode ser representação de lista ou texto com ponto e vírgula
        raw_list = [s.strip() for s in raw_statements.split(";") if s.strip()]
    elif isinstance(raw_statements, (list, tuple)):
        raw_list = [str(s).strip() for s in raw_statements if s]
    else:
        raw_list = []

    is_ptbxl = source_dataset.lower() == "ptbxl"
    return map_exam_statements(raw_list, is_ptbxl=is_ptbxl)


__all__ = [
    "MIMIC_LEXICAL_RULES",
    "PTBXL_SCP_TO_TARGETS",
    "SUBCLASSES",
    "SUPERCLASSES",
    "TARGET_CLASSES",
    "UNCERTAINTY_PATTERN",
    "extract_targets_for_record",
    "map_exam_statements",
]
