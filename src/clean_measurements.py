import numpy as np
import pandas as pd

from inspect_database import PHYSIOLOGICAL_LIMITS
from read_database import DATA_DIR

# Limites fisiológicos humanos para os intervalos clínicos derivados (em ms).
# Mesmo que onset e end estejam individualmente no limite, a diferença entre eles
# precisa respeitar a velocidade de condução elétrica do coração humano.
DERIVED_INTERVAL_LIMITS: dict[str, tuple[int, int]] = {
    # Duração da onda P (despolarização atrial): 30 ms a 200 ms
    "p_duration": (30, 200),
    # Intervalo PR (condução átrio-ventricular): 60 ms a 450 ms
    "pr_interval": (60, 450),
    # Duração do complexo QRS (despolarização ventricular): 40 ms a 300 ms
    "qrs_duration": (40, 300),
    # Intervalo QT (despolarização + repolarização ventricular): 200 ms a 700 ms
    "qt_interval": (200, 700),
}

# Colunas essenciais do ciclo ventricular (QRS-T, intervalo RR e suas durações).
# Se alguma dessas falhar (< 0,8% da base), indica provável falha técnica ou eletrodo solto.
ESSENTIAL_VENTRICULAR_COLS = [
    "rr_interval",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "qrs_axis",
    "t_axis",
    "qrs_duration",
    "qt_interval",
]


def clean_machine_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa, valida e prepara as medições numéricas do ECG para uso em Redes Neurais (CNN)."""
    # Remove apenas as colunas brutas de texto (report_0..report_17), pois já foram
    # processadas separadamente em normalized_reports.csv.
    report_cols = [c for c in df.columns if c.startswith("report_")]
    df_clean = df.drop(columns=report_cols).copy()

    # =========================================================================
    # PASSO 1: Substituir códigos de erro e valores fora dos limites humanos por NaN
    # Motivo (CNN): Valores sentinela (ex.: 29999, 32767, -32768) distorcem a média,
    # o desvio padrão (StandardScaler) e explodem os gradientes da rede neural.
    # =========================================================================
    for col, (min_val, max_val) in PHYSIOLOGICAL_LIMITS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(float)
            invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            df_clean.loc[invalid_mask, col] = np.nan

    # =========================================================================
    # PASSO 2: Calcular variáveis clínicas derivadas (Engenharia de Atributos)
    # Motivo: Diferentes modelos de eletrocardiógrafos (cart_id) usam pontos zero
    # distintos (ex.: p_onset=40ms vs p_onset=365ms), mas as durações relativas
    # (PR, QRS, QT, duração P e BPM) têm escala única em qualquer equipamento.
    # =========================================================================
    df_clean["heart_rate_bpm"] = (60000.0 / df_clean["rr_interval"]).round(2)
    df_clean["p_duration"] = df_clean["p_end"] - df_clean["p_onset"]
    df_clean["pr_interval"] = df_clean["qrs_onset"] - df_clean["p_onset"]
    df_clean["qrs_duration"] = df_clean["qrs_end"] - df_clean["qrs_onset"]
    df_clean["qt_interval"] = df_clean["t_end"] - df_clean["qrs_onset"]

    # =========================================================================
    # PASSO 3: Validar limites fisiológicos das durações derivadas
    # Motivo: Captura casos onde onset e end pareciam válidos isoladamente, mas sua
    # subtração gerou durações negativas ou impossíveis (ex.: PR de 3 ms ou QRS de 20 ms).
    # =========================================================================
    for col, (min_val, max_val) in DERIVED_INTERVAL_LIMITS.items():
        invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
        df_clean.loc[invalid_mask, col] = np.nan

    # Se o intervalo PR for inválido, o p_onset foi detectado incorretamente pelo aparelho
    invalid_pr = df_clean["pr_interval"].isna()
    df_clean.loc[invalid_pr, ["p_onset", "p_end", "p_duration"]] = np.nan

    # =========================================================================
    # PASSO 4: Remover exames com falha nas medições ventriculares essenciais (< 0,8%)
    # Motivo (CNN): Exames sem intervalo RR, complexo QRS ou onda T válidos geralmente
    # possuem ruído severo ou falha de captação nos eletrodos, prejudicando o treino da CNN.
    # =========================================================================
    df_clean = df_clean.dropna(subset=ESSENTIAL_VENTRICULAR_COLS).copy()

    # =========================================================================
    # PASSO 5: Criar variável binária indicadora de ausência da Onda P (0 ou 1)
    # Motivo Clínico e de IA: Não podemos excluir os exames sem onda P (~16% da base), pois
    # excluiríamos os pacientes com Fibrilação Atrial. A flag `p_wave_missing = 1`
    # informa explicitamente à rede neural que a onda P não existia naquele exame.
    # Nota: Usamos p_onset e p_axis (e não p_end), pois ~107 mil exames de aparelhos
    # antigos têm onda P normal mas não exportam a coluna p_end.
    # =========================================================================
    df_clean["p_wave_missing"] = (
        df_clean["p_onset"].isna() | df_clean["p_axis"].isna()
    ).astype(int)

    # =========================================================================
    # PASSO 6: Imputação consistente das variáveis da Onda P (Zero NaN para a CNN)
    # Motivo (CNN): Redes Neurais não aceitam NaN. Primeiro imputamos p_onset, p_axis,
    # p_duration e pr_interval com a mediana; depois reconstruímos p_end ausentes como
    # (p_onset + p_duration) para respeitar o alinhamento de cada aparelho.
    # =========================================================================
    for col in ["p_onset", "p_axis", "p_duration", "pr_interval"]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Onde p_end estava ausente (ou inválido), reconstrói a partir de (p_onset + p_duration)
    missing_p_end = df_clean["p_end"].isna()
    df_clean.loc[missing_p_end, "p_end"] = (
        df_clean.loc[missing_p_end, "p_onset"]
        + df_clean.loc[missing_p_end, "p_duration"]
    )

    return df_clean


def main() -> None:
    input_path = DATA_DIR / "machine_measurements.csv"
    output_path = DATA_DIR / "cleaned_machine_measurements.csv"

    print(f"Carregando {input_path}...")
    raw_df = pd.read_csv(input_path, low_memory=False)
    print(f"Total de exames originais: {len(raw_df):,}")

    cleaned_df = clean_machine_measurements(raw_df)

    removed_count = len(raw_df) - len(cleaned_df)
    removed_pct = (removed_count / len(raw_df)) * 100
    p_missing_count = int((cleaned_df["p_wave_missing"] == 1).sum())
    p_missing_pct = (p_missing_count / len(cleaned_df)) * 100

    print(f"Exames válidos mantidos: {len(cleaned_df):,}")
    print(
        f"Exames descartados por falha ventricular/ruído: {removed_count:,} ({removed_pct:.2f}%)"
    )
    print(
        f"Exames preservados com flag p_wave_missing=1: {p_missing_count:,} ({p_missing_pct:.2f}%)"
    )
    print(
        f"Total de valores NaN restantes no dataset: {int(cleaned_df.isna().sum().sum())}"
    )

    cleaned_df.to_csv(output_path, index=False)
    print(f"\nArquivo salvo com sucesso em: {output_path}")


if __name__ == "__main__":
    main()
