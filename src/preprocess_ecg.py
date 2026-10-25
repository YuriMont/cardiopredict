from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from read_database import DATA_DIR, patients_records

# Ordem padrão universal das 12 derivações para garantir que cada canal (0 a 11)
# da CNN (Keras Conv1D) receba sempre a mesma derivação anatômica.
STANDARD_LEADS = [
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

# Fatores de conversão de unidade física para padronizar tudo em milivolts (mV).
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

# Limite físico máximo de amplitude do ECG em milivolts (mV).
MAX_PHYSICAL_MV = 10.0

# Fração máxima tolerável de valores NaN em uma derivação (5% de 5000 amostras = 250 pontos).
MAX_NAN_RATIO = 0.05

# Desvio padrão mínimo em mV para considerar que o eletrodo não está mudo (flatline).
MIN_STD_MV = 1e-4


def load_raw_record(record_rel_path: str | Path) -> wfdb.Record:
    """Carrega o arquivo binário (.hea/.dat) do exame de ECG via WFDB em unidades físicas."""
    full_path = DATA_DIR / record_rel_path
    if full_path.suffix in {".hea", ".dat"}:
        full_path = full_path.with_suffix("")

    record = wfdb.rdrecord(str(full_path), physical=True)
    if isinstance(record, wfdb.MultiRecord):
        record = record.multi_to_single(physical=True)
    return record


def standardize_amplitude_units_to_mv(
    signal: np.ndarray,
    units: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """
    PASSO 1A: Confere e padroniza as unidades de escala de amplitude para milivolts (mV),
    mantendo a MESMA escala física real entre todos os 12 canais e entre pacientes.

    Motivo Clínico e de IA:
    - Diagnósticos como 'Low QRS voltages' (~60 mil exames na base) e 'Left Ventricular
      Hypertrophy' (~31 mil exames) dependem diretamente da altura real da onda em mV.
    - Se dividíssemos cada canal pelo seu próprio desvio padrão (Z-score por canal),
      uma derivação de baixa voltagem (0.1 mV) ficaria do mesmo tamanho que uma hipertrofia
      (4.0 mV), destruindo a relação de amplitude entre os 12 canais!
    - Por isso, convertemos qualquer canal em uV ou V para mV e mantemos a escala em mV.
    """
    standardized = signal.astype(np.float32, copy=True)
    n_leads = standardized.shape[1]

    if not units:
        return standardized, ["mV"] * n_leads

    resolved_units: list[str] = []
    for idx in range(min(n_leads, len(units))):
        raw_unit = str(units[idx]).strip().lower()
        factor = UNIT_TO_MV_FACTOR.get(raw_unit, 1.0)
        if factor != 1.0:
            standardized[:, idx] *= factor
        resolved_units.append("mV")

    return standardized, resolved_units


def reorder_leads(signal: np.ndarray, sig_name: list[str]) -> np.ndarray:
    """
    PASSO 1B: Padroniza a ordem das 12 derivações para STANDARD_LEADS.
    Motivo: Nos arquivos brutos do MIMIC-IV-ECG, 'aVF' vem antes de 'aVL'.
    """
    if sig_name == STANDARD_LEADS:
        return signal

    indices = [sig_name.index(lead) for lead in STANDARD_LEADS]
    return signal[:, indices]


def interpolate_missing_and_validate(signal: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    PASSO 2: Verifica a integridade da onda e interpola pequenos trechos NaN (< 5%).
    Descarta exames com derivações em linha reta (flatline) ou excesso de NaNs.
    """
    cleaned = signal.copy()
    _, n_leads = cleaned.shape

    for lead_idx in range(n_leads):
        lead_sig = cleaned[:, lead_idx]
        nan_mask = np.isnan(lead_sig)
        nan_ratio = float(nan_mask.mean())

        if nan_ratio > MAX_NAN_RATIO:
            return cleaned, False

        if nan_mask.any():
            valid_idx = np.flatnonzero(~nan_mask)
            missing_idx = np.flatnonzero(nan_mask)
            if len(valid_idx) == 0:
                return cleaned, False
            lead_sig[nan_mask] = np.interp(missing_idx, valid_idx, lead_sig[valid_idx])
            cleaned[:, lead_idx] = lead_sig

        if float(np.std(cleaned[:, lead_idx])) < MIN_STD_MV:
            return cleaned, False

    return cleaned, True


def remove_baseline_wander_highpass(
    signal: np.ndarray,
    fs: float = 500.0,
    cutoff_hz: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    FILTRO 1 — Passa-Alta Butterworth (0.5 Hz):
    Identifica e remove a flutuação da linha de base (Baseline Wander) causada pela
    respiração do paciente e movimentação lenta do tórax (< 0.5 Hz).
    """
    nyquist = 0.5 * fs
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return np.asarray(sosfiltfilt(sos, signal, axis=0), dtype=np.float32)


def remove_powerline_interference_notch(
    signal: np.ndarray,
    fs: float = 500.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    FILTRO 2 — Rejeita-Faixa (Notch Filter em 50 Hz e 60 Hz):
    Identifica e remove especificamente a interferência eletromagnética da rede elétrica.
    Na base MIMIC-IV-ECG existem exames com filtro de 60 Hz, exames de 50 Hz e mais de
    120 mil exames sem filtro de rede elétrica especificado.
    """
    out = signal
    for freq_hz in (50.0, 60.0):
        if freq_hz < 0.5 * fs:
            b, a = iirnotch(w0=freq_hz, Q=quality_factor, fs=fs)
            sos = tf2sos(b, a)
            out = np.asarray(sosfiltfilt(sos, out, axis=0), dtype=np.float32)
    return out


def remove_high_freq_muscle_noise_lowpass(
    signal: np.ndarray,
    fs: float = 500.0,
    cutoff_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    FILTRO 3 — Passa-Baixa Butterworth (40 Hz):
    Identifica e remove ruído muscular de alta frequência (eletromiografia / EMG,
    tremores musculares do paciente) acima de 40 Hz, preservando a escala em mV.
    """
    nyquist = 0.5 * fs
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return np.asarray(sosfiltfilt(sos, signal, axis=0), dtype=np.float32)


def preprocess_ecg_for_cnn(
    record: wfdb.Record,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """
    Executa o pipeline completo de pré-processamento do ECG para Keras Conv1D,
    mantendo todas as 12 derivações na MESMA unidade e escala física (mV).

    Retorna:
    - cnn_tensor: matriz (5000, 12) em milivolts (mV) para Keras Conv1D (channels_last)
    - steps_dict: dicionário com cada etapa do filtro para inspeção e plotagem
    - is_valid: booleano indicando se o sinal passou no controle de qualidade
    """
    if record.p_signal is None or record.sig_name is None:
        empty = np.zeros((5000, 12), dtype=np.float32)
        return empty, {}, False

    fs_val = float(record.fs or 500.0)

    # Passo 1: Padronizar unidade física para mV e reordenar as 12 derivações
    step1_mv, _ = standardize_amplitude_units_to_mv(record.p_signal, record.units)
    step1_reordered = reorder_leads(step1_mv, list(record.sig_name))

    # Passo 2: Interpolação de NaNs isolados e validação contra linha reta (flatline)
    step2_interpolated, is_valid = interpolate_missing_and_validate(step1_reordered)
    if not is_valid:
        empty = np.zeros((step1_reordered.shape[0], 12), dtype=np.float32)
        return empty, {"raw_mv": step1_reordered}, False

    # Passo 3: Corte de picos mecânicos extremos (> ±10 mV) mantendo a escala mV
    step3_clipped = np.clip(step2_interpolated, -MAX_PHYSICAL_MV, MAX_PHYSICAL_MV)

    # Passo 4A (Filtro 1): Remoção de oscilação da linha de base (Passa-Alta 0.5 Hz)
    step4a_no_baseline = remove_baseline_wander_highpass(step3_clipped, fs=fs_val)

    # Passo 4B (Filtro 2): Remoção de interferência da rede elétrica (Notch 50/60 Hz)
    step4b_no_powerline = remove_powerline_interference_notch(
        step4a_no_baseline, fs=fs_val
    )

    # Passo 4C (Filtro 3): Remoção de ruído muscular EMG (Passa-Baixa 40 Hz)
    cnn_tensor_mv = remove_high_freq_muscle_noise_lowpass(
        step4b_no_powerline, fs=fs_val
    )

    # Cálculo dos ruídos isolados (diferença subtraída por cada filtro) para auditoria
    baseline_noise = step3_clipped - step4a_no_baseline
    powerline_and_emg_noise = step4a_no_baseline - cnn_tensor_mv

    steps = {
        "raw_mv": step1_reordered,
        "no_baseline_mv": step4a_no_baseline,
        "no_powerline_mv": step4b_no_powerline,
        "clean_mv": cnn_tensor_mv,
        "baseline_noise_mv": baseline_noise,
        "high_freq_noise_mv": powerline_and_emg_noise,
    }
    return cnn_tensor_mv, steps, True


def inspect_ecg_sample(sample_size: int = 200) -> pd.DataFrame:
    """Inspeciona unidades de amplitude, derivações e anomalias em uma amostra de exames."""
    hea_files = sorted(DATA_DIR.joinpath("files").rglob("*.hea"))[:sample_size]
    print(f"\n{'=' * 75}")
    print(f"1. AUDITORIA DE UNIDADES DE ESCALA E SINAIS ({len(hea_files)} EXAMES)")
    print(f"{'=' * 75}")

    records_stats = []
    units_found: set[tuple[str, ...]] = set()
    lead_orders: set[tuple[str, ...]] = set()

    for hea_path in hea_files:
        rec = wfdb.rdrecord(str(hea_path.with_suffix("")), physical=True)
        if isinstance(rec, wfdb.MultiRecord):
            rec = rec.multi_to_single(physical=True)

        sig = rec.p_signal
        if sig is None:
            continue

        units_found.add(tuple(rec.units or []))
        lead_orders.add(tuple(rec.sig_name or []))
        sig_mv, _ = standardize_amplitude_units_to_mv(sig, rec.units)

        nan_count = int(np.isnan(sig_mv).sum())
        clean_sig = np.nan_to_num(sig_mv, nan=0.0)
        min_std = float(np.std(clean_sig, axis=0).min())
        max_abs_mv = float(np.abs(clean_sig).max())

        records_stats.append(
            {
                "study_id": hea_path.stem,
                "fs_hz": int(rec.fs or 500),
                "nan_count": nan_count,
                "has_flatline": min_std < MIN_STD_MV,
                "min_mv": float(np.nanmin(sig_mv)),
                "mean_mv": float(np.nanmean(sig_mv)),
                "std_mv": float(np.nanstd(sig_mv)),
                "max_mv": float(np.nanmax(sig_mv)),
                "max_abs_mv": max_abs_mv,
            }
        )

    stats_df = pd.DataFrame(records_stats)
    print(f"Unidades originais nos cabeçalhos (.hea): {list(units_found)}")
    print(
        "Unidade padronizada em todos os 12 canais: ['mV' x 12] (mesma escala física)"
    )
    print(f"Ordem original nos arquivos: {list(lead_orders)}")
    print(f"Ordem padronizada para CNN:  {STANDARD_LEADS}")

    print("\n--- DESCRIBE DOS SINAIS NA ESCALA FÍSICA (mV) ---")
    desc_cols = ["nan_count", "min_mv", "mean_mv", "std_mv", "max_mv", "max_abs_mv"]
    print(stats_df[desc_cols].describe().T.to_string())

    return stats_df


def describe_single_ecg_steps(steps: dict[str, np.ndarray], study_id: int) -> None:
    """Mostra o .describe() comparando o sinal bruto e o sinal limpo na MESMA escala (mV)."""
    print(f"\n{'=' * 75}")
    print(f"2. COMPARATIVO .DESCRIBE() NA MESMA ESCALA (mV) - EXAME {study_id}")
    print(f"{'=' * 75}")

    raw_df = pd.DataFrame(steps["raw_mv"], columns=STANDARD_LEADS)
    clean_df = pd.DataFrame(steps["clean_mv"], columns=STANDARD_LEADS)
    base_noise_df = pd.DataFrame(steps["baseline_noise_mv"], columns=STANDARD_LEADS)
    hf_noise_df = pd.DataFrame(steps["high_freq_noise_mv"], columns=STANDARD_LEADS)

    print("\n[A] SINAL BRUTO (mV) — Média fora de zero pela respiração:")
    print(
        raw_df.describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string()
    )

    print(
        "\n[B] SINAL LIMPO FINAL (mV) — Linha de base em 0.0 mV e escala real de amplitude preservada:"
    )
    print(
        clean_df.describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string()
    )

    print("\n[C] ENERGIA (STD em mV) DO RUÍDO IDENTIFICADO E REMOVIDO EM CADA CANAL:")
    noise_summary = pd.DataFrame(
        {
            "ruido_respiracao_std_mv": base_noise_df.std().round(4),
            "ruido_eletrico_emg_std_mv": hf_noise_df.std().round(4),
            "sinal_limpo_std_mv": clean_df.std().round(4),
        }
    )
    print(noise_summary.to_string())


def plot_all_12_leads_and_filters(
    steps: dict[str, np.ndarray],
    fs: float,
    study_id: int,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """
    Plota as 12 derivações (I..V6) na MESMA escala vertical física (mV),
    comparando o Sinal Bruto com o Sinal Limpo para que seja possível visualizar
    todos os 12 canais e a proporção real de amplitude entre eles.
    """
    n_samples = steps["raw_mv"].shape[0]
    time_sec = np.arange(n_samples) / fs

    # Determina o limite vertical global em mV para manter TODAS as 12 derivações na mesma escala
    global_max_mv = float(
        max(
            np.abs(steps["raw_mv"]).max(),
            np.abs(steps["clean_mv"]).max(),
        )
    )
    y_limit = max(1.5, round(global_max_mv * 1.1, 1))

    fig, axes = plt.subplots(6, 2, figsize=(18, 14), sharex=True, sharey=True)
    fig.suptitle(
        f"ECG 12 Derivações na Mesma Escala Física (±{y_limit} mV) — Study {study_id}\n"
        "Vermelho Claro: Sinal Bruto | Azul Escuro: Sinal Limpo (Passa-Alta 0.5Hz + Notch 50/60Hz + Passa-Baixa 40Hz)",
        fontsize=13,
        fontweight="bold",
    )

    for idx, lead_name in enumerate(STANDARD_LEADS):
        row = idx % 6
        col = idx // 6
        ax = axes[row, col]

        ax.plot(
            time_sec,
            steps["raw_mv"][:, idx],
            color="#ff9896",
            linewidth=0.8,
            alpha=0.8,
            label="Bruto (mV)",
        )
        ax.plot(
            time_sec,
            steps["clean_mv"][:, idx],
            color="#1f77b4",
            linewidth=1.0,
            label="Limpo (mV)",
        )
        ax.set_title(f"Derivação {lead_name}", fontsize=10, fontweight="bold")
        ax.set_ylim(-y_limit, y_limit)
        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.3)

        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=8)
        if row == 5:
            ax.set_xlabel("Tempo (segundos)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"\nGráfico das 12 derivações na mesma escala salvo em: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    inspect_ecg_sample(sample_size=200)

    exam = patients_records.iloc[0]
    study_id = int(exam["study_id"])
    record = load_raw_record(exam["path"])

    cnn_tensor_mv, steps, is_valid = preprocess_ecg_for_cnn(record)

    print(
        f"\nStatus de validação do exame {study_id}: {'VÁLIDO' if is_valid else 'INVÁLIDO'}"
    )
    print(
        f"Formato final para Keras Conv1D (timesteps, channels): {cnn_tensor_mv.shape} (Unidade: mV)"
    )

    if is_valid:
        describe_single_ecg_steps(steps, study_id=study_id)
        plot_all_12_leads_and_filters(
            steps=steps,
            fs=float(record.fs or 500.0),
            study_id=study_id,
            save_path=DATA_DIR / f"ecg_preprocessing_{study_id}.png",
            show_plot=True,
        )


if __name__ == "__main__":
    main()
