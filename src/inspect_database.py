"""
Módulo de Auditoria e Qualidade de Dados da Base MIMIC-IV-ECG.

Fundamentação Médica dos Limites Fisiológicos Humanos (`PHYSIOLOGICAL_LIMITS`):
- Em eletrocardiógrafos hospitalares (padrão ISO/IEEE 11073-91064 SCP-ECG), quando uma onda
  cardíaca não pode ser medida (ex.: ausência fisiológica de onda P na Fibrilação Atrial ou
  ruído de eletrodo), o hardware grava constantes inteiras de 16 bits como códigos sentinela
  (`29999`, `32767`, `65535`, `-32768`, `999` ou `0`) em vez de `NULL`.
- Os intervalos abaixo seguem as diretrizes conjuntas AHA/ACCF/HRS (Mason et al., 2007;
  Kligfield et al., 2007; Surawicz et al., 2009), cobrindo 100% da fisiologia humana viável
  (de bradicardias extremas de 20 bpm até taquicardias de 300 bpm) e barrando códigos de erro.
"""

import pandas as pd

from read_database import DATA_DIR

# Limites fisiológicos humanos para cada variável numérica do ECG: (mínimo, máximo)
# Valores fora dessas faixas (ex.: 0, 999, 29999, 32767, -32768) indicam falha de leitura ou código de erro do aparelho.
PHYSIOLOGICAL_LIMITS: dict[str, tuple[int, int]] = {
    # Intervalo entre batimentos (R-R): 200 ms (300 bpm, taquicardia extrema) a 3000 ms (20 bpm, bradicardia extrema)
    "rr_interval": (200, 3000),
    # Início da onda P: 1 ms a 1000 ms dentro do ciclo cardíaco
    "p_onset": (1, 1000),
    # Fim da onda P: 1 ms a 1200 ms dentro do ciclo cardíaco
    "p_end": (1, 1200),
    # Início do complexo QRS: 1 ms a 1500 ms dentro do ciclo cardíaco
    "qrs_onset": (1, 1500),
    # Fim do complexo QRS: 1 ms a 1800 ms dentro do ciclo cardíaco
    "qrs_end": (1, 1800),
    # Fim da onda T: 100 ms a 2500 ms dentro do ciclo cardíaco
    "t_end": (100, 2500),
    # Eixo elétrico da onda P: ângulo válido em graus entre -180° e +360°
    "p_axis": (-180, 360),
    # Eixo elétrico do complexo QRS: ângulo válido em graus entre -180° e +360°
    "qrs_axis": (-180, 360),
    # Eixo elétrico da onda T: ângulo válido em graus entre -180° e +360°
    "t_axis": (-180, 360),
}


def inspect_dataframe(
    df: pd.DataFrame,
    name: str,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print(f"ARQUIVO: {name}")
    print(f"{'=' * 70}")

    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Dimensões: {rows:,} linhas x {cols} colunas")
    print(f"Uso de memória: {memory_mb:.2f} MB")

    # Resumo por coluna: tipo de dado, nulos, vazios e valores únicos
    summary_records = []
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

    # Verificação de duplicados
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
    available_cols = [c for c in PHYSIOLOGICAL_LIMITS if c in df.columns]
    if not available_cols:
        return

    print("\n--- Estatísticas Descritivas das Medições Numéricas ---")
    stats = df[available_cols].describe().T
    print(stats.to_string())

    print("\n--- Valores Fora dos Limites Humanos / Fisiológicos ---")
    outlier_records = []
    total_rows = len(df)
    for col in available_cols:
        min_limit, max_limit = PHYSIOLOGICAL_LIMITS[col]
        series = df[col].dropna()
        below_min = int((series < min_limit).sum())
        above_max = int((series > max_limit).sum())
        total_invalid = below_min + above_max
        invalid_pct = (total_invalid / total_rows * 100) if total_rows > 0 else 0.0

        outlier_records.append(
            {
                "coluna": col,
                "limite_humano": f"[{min_limit}, {max_limit}]",
                "abaixo_min": below_min,
                "acima_max": above_max,
                "total_invalido": total_invalid,
                "invalido_%": round(invalid_pct, 2),
            }
        )

    print(pd.DataFrame(outlier_records).to_string(index=False))


def check_referential_integrity(
    records_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
) -> None:
    print(f"\n{'=' * 70}")
    print("CONSISTÊNCIA ENTRE TABELAS (record_list vs machine_measurements)")
    print(f"{'=' * 70}")

    rec_studies = set(records_df["study_id"].dropna())
    meas_studies = set(measurements_df["study_id"].dropna())

    only_in_records = len(rec_studies - meas_studies)
    only_in_measurements = len(meas_studies - rec_studies)

    print(f"Total de study_id únicos em record_list: {len(rec_studies):,}")
    print(f"Total de study_id únicos em machine_measurements: {len(meas_studies):,}")
    print(f"Exames em record_list sem medições: {only_in_records:,}")
    print(
        f"Exames em machine_measurements sem registro em record_list: {only_in_measurements:,}"
    )


def main() -> None:
    datasets: list[tuple[str, list[str]]] = [
        ("record_list.csv", ["study_id"]),
        ("machine_measurements.csv", ["study_id"]),
        ("normalized_reports.csv", ["study_id", "report_column"]),
        ("waveform_note_links.csv", ["study_id"]),
    ]

    loaded_dfs: dict[str, pd.DataFrame] = {}

    for filename, keys in datasets:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"\n[AVISO] Arquivo não encontrado: {file_path}")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        loaded_dfs[filename] = df
        inspect_dataframe(df, filename, key_columns=keys)

        if filename == "machine_measurements.csv":
            inspect_measurements_outliers(df)

    if "record_list.csv" in loaded_dfs and "machine_measurements.csv" in loaded_dfs:
        check_referential_integrity(
            loaded_dfs["record_list.csv"],
            loaded_dfs["machine_measurements.csv"],
        )


if __name__ == "__main__":
    main()
