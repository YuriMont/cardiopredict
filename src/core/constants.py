"""
Constantes centrais do sistema CardioPredict:
- Caminhos para arquivos e bancos de dados locais
- Ordem anatômica e conversões das derivações de ECG
- Limites fisiológicos humanos de intervalos cardíacos
- Termos de resumo e exclusão técnica de laudos
"""

from pathlib import Path

# =============================================================================
# 1. Caminhos do Sistema
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "cardiopredict.duckdb"
_mimic_candidate = DATA_DIR / "mimic-iv-ecg"
MIMIC_DIR = (
    _mimic_candidate / "1.0"
    if (_mimic_candidate / "1.0").exists()
    else _mimic_candidate
)
PTBXL_DIR = DATA_DIR / "ptb-xl"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =============================================================================
# 2. Convenções e Padronização de Sinal (ECG 12 Derivações)
# =============================================================================
# Ordem anatômica universal das 12 derivações
STANDARD_LEADS: list[str] = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# Fatores multiplicativos de conversão para unidade física milivolt (mV)
UNIT_TO_MV_FACTOR: dict[str, float] = {
    "mv": 1.0,
    "millivolt": 1.0,
    "millivolts": 1.0,
    "uv": 1e-3,
    "μv": 1e-3,
    "microvolt": 1e-3,
    "microvolts": 1e-3,
    "v": 1e3,
    "volt": 1e3,
    "volts": 1e3,
}

# Limites de qualidade e integridade física de sinal
FLATLINE_STD_MIN_MV: float = 1e-4
MAX_ALLOWED_NAN_RATIO: float = 0.05
ABERRANT_MAX_AMPLITUDE_MV: float = 15.0

# =============================================================================
# 3. Taxonomia e Classes Alvo Multilabel (5 Superclasses + 25 Subclasses)
# =============================================================================
SUPERCLASSES: tuple[str, ...] = (
    "NORM",  # Normal ECG
    "MI",  # Myocardial Infarction
    "STTC",  # ST/T Change
    "CD",  # Conduction Disturbance
    "HYP",  # Hypertrophy
)

SUBCLASSES: tuple[str, ...] = (
    "AFIB",  # Atrial Fibrillation
    "AFLT",  # Atrial Flutter
    "SR",  # Sinus Rhythm
    "SBRAD",  # Sinus Bradycardia
    "STACH",  # Sinus Tachycardia
    "PVC",  # Premature Ventricular Contractions
    "PAC",  # Premature Atrial Contractions
    "AMI",  # Anterior / Anteroseptal MI
    "IMI",  # Inferior MI
    "LMI",  # Lateral MI
    "STEMI",  # ST-Elevation MI (Acute urgency marker)
    "CLBBB",  # Complete Left Bundle Branch Block
    "CRBBB",  # Complete Right Bundle Branch Block
    "IRBBB",  # Incomplete Right Bundle Branch Block
    "LAFB",  # Left Anterior Fascicular Block
    "1AVB",  # 1st Degree AV Block
    "2AVB",  # 2nd Degree AV Block (Mobitz I / II)
    "3AVB",  # Complete / 3rd Degree AV Block
    "IVCD",  # Intraventricular Conduction Delay
    "LVH",  # Left Ventricular Hypertrophy
    "RVH",  # Right Ventricular Hypertrophy
    "STTC_ISCA",  # Anterior Ischemic ST/T changes
    "STTC_ISCI",  # Inferior Ischemic ST/T changes
    "STD",  # ST Depression
    "STE",  # ST Elevation
)

TARGET_CLASSES: tuple[str, ...] = SUPERCLASSES + SUBCLASSES

# =============================================================================
# 3. Limites Fisiológicos Humanos (Diretrizes AHA/ACCF/HRS)
# =============================================================================
# Valores fora destas faixas indicam ruído extremo ou códigos sentinela de hardware (ex: 29999, 32767).
PHYSIOLOGICAL_LIMITS: dict[str, tuple[int, int]] = {
    "rr_interval": (200, 3000),  # 20 bpm a 300 bpm
    "p_onset": (1, 1000),
    "p_end": (1, 1200),
    "qrs_onset": (1, 1500),
    "qrs_end": (1, 1800),
    "t_end": (100, 2500),
    "p_axis": (-180, 360),
    "qrs_axis": (-180, 360),
    "t_axis": (-180, 360),
}

# Limites fisiológicos para intervalos cardíacos derivados (em ms)
DERIVED_INTERVAL_LIMITS: dict[str, tuple[int, int]] = {
    "p_duration": (30, 200),
    "pr_interval": (60, 450),
    "qrs_duration": (40, 300),
    "qt_interval": (200, 700),
}

# Colunas essenciais do ciclo ventricular para auditoria de sinal
ESSENTIAL_VENTRICULAR_COLS: list[str] = [
    "rr_interval",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "qrs_axis",
    "t_axis",
    "qrs_duration",
    "qt_interval",
]

# =============================================================================
# 4. Textos de Resumo e Prefixos Não-Diagnósticos
# =============================================================================
SUMMARY_STAMP_TEXTS: set[str] = {
    "normal ecg",
    "normal ecg except for rate",
    "within normal limits",
    "borderline ecg",
    "abnormal ecg",
}

NON_DIAGNOSTIC_PREFIXES: tuple[str, ...] = (
    "age not entered",
    "lead(s) unsuitable",
    "recording unsuitable",
    "data quality",
)
