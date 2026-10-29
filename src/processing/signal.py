"""
Módulo de Pré-processamento e Filtragem de Sinais de ECG (12 Derivações).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import wfdb
from scipy.signal import butter, iirnotch, resample, sosfiltfilt, tf2sos

from core.constants import DATA_DIR, STANDARD_LEADS, UNIT_TO_MV_FACTOR

# Limite físico máximo de amplitude do ECG em milivolts (mV)
MAX_PHYSICAL_MV = 10.0

# Fração máxima tolerável de valores NaN em uma derivação (5% de 5000 amostras = 250 pontos)
MAX_NAN_RATIO = 0.05

# Desvio padrão mínimo em mV para considerar que o eletrodo não está mudo (flatline)
MIN_STD_MV = 1e-4


def load_raw_record(record_rel_path: str | Path) -> wfdb.Record:
    """Carrega o arquivo binário (.hea/.dat) do exame de ECG via WFDB em unidades físicas."""
    p = Path(record_rel_path)
    full_path = p if (p.exists() or str(p).startswith("data")) else DATA_DIR / p
    if full_path.suffix in {".hea", ".dat"}:
        full_path = full_path.with_suffix("")

    if not full_path.with_suffix(".hea").exists():
        path_str = str(full_path)
        if "/1.0/" in path_str:
            alt = Path(path_str.replace("/1.0/", "/"))
            if alt.with_suffix(".hea").exists():
                full_path = alt
        elif "/mimic-iv-ecg/" in path_str:
            alt = Path(path_str.replace("/mimic-iv-ecg/", "/mimic-iv-ecg/1.0/"))
            if alt.with_suffix(".hea").exists():
                full_path = alt

    record = wfdb.rdrecord(str(full_path), physical=True)
    if isinstance(record, wfdb.MultiRecord):
        record = record.multi_to_single(physical=True)
    return record


def standardize_amplitude_units_to_mv(
    signal: np.ndarray,
    units: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Padroniza as unidades de escala de amplitude para milivolts (mV)."""
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
    """Padroniza a ordem das 12 derivações para STANDARD_LEADS (insensível a maiúsculas/minúsculas)."""
    if sig_name == STANDARD_LEADS:
        return signal

    # Normalização para maiúsculas para suportar 'AVR', 'aVR', 'avr' (ex.: PTB-XL vs MIMIC)
    clean_names = [str(s).strip().upper() for s in sig_name]
    target_names = [lead.upper() for lead in STANDARD_LEADS]

    indices: list[int] = []
    for target in target_names:
        if target not in clean_names:
            raise ValueError(
                f"Derivação obrigatória '{target}' não encontrada no exame (disponíveis: {sig_name})"
            )
        indices.append(clean_names.index(target))

    return signal[:, indices]


def interpolate_missing_and_validate(
    signal: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Verifica a integridade da onda e interpola pequenos trechos não finitos (NaN/Inf) (< 5%)."""
    cleaned = signal.copy()
    _, n_leads = cleaned.shape

    for lead_idx in range(n_leads):
        lead_sig = cleaned[:, lead_idx]
        invalid_mask = ~np.isfinite(lead_sig)
        invalid_ratio = float(invalid_mask.mean())

        if invalid_ratio > MAX_NAN_RATIO:
            return cleaned, False

        if invalid_mask.any():
            valid_idx = np.flatnonzero(~invalid_mask)
            missing_idx = np.flatnonzero(invalid_mask)
            if len(valid_idx) == 0:
                return cleaned, False
            lead_sig[invalid_mask] = np.interp(
                missing_idx, valid_idx, lead_sig[valid_idx]
            )
            cleaned[:, lead_idx] = lead_sig

        if float(np.std(cleaned[:, lead_idx])) < MIN_STD_MV:
            return cleaned, False

    return cleaned, True


def resample_ecg_signal(
    signal: np.ndarray,
    orig_fs: float,
    target_fs: float = 500.0,
    target_samples: int = 5000,
) -> np.ndarray:
    """Reamostra o sinal multicanal de ECG para target_fs e target_samples (padrão 500 Hz, 10s = 5000 amostras).

    Garante fidelidade morfológica sem truncamento destrutivo ou zero-padding que simula assistolia.
    """
    n_samples, _ = signal.shape
    if orig_fs == target_fs and n_samples == target_samples:
        return signal.astype(np.float32, copy=False)

    # Reamostragem via interpolação de Fourier (scipy.signal.resample)
    resampled = resample(signal, target_samples, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def remove_baseline_wander_highpass(
    signal: np.ndarray,
    fs: float = 500.0,
    cutoff_hz: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Passa-Alta Butterworth (0.5 Hz): remove oscilação da linha de base."""
    nyquist = 0.5 * fs
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return np.asarray(sosfiltfilt(sos, signal, axis=0), dtype=np.float32)


def remove_powerline_interference_notch(
    signal: np.ndarray,
    fs: float = 500.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Rejeita-Faixa (Notch Filter em 50 Hz e 60 Hz): remove interferência elétrica."""
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
    cutoff_hz: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """Passa-Baixa Butterworth (45 Hz): remove ruído muscular EMG."""
    nyquist = 0.5 * fs
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return np.asarray(sosfiltfilt(sos, signal, axis=0), dtype=np.float32)


def preprocess_ecg_for_cnn(
    record: wfdb.Record,
    target_fs: float = 500.0,
    target_samples: int = 5000,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """Executa o pipeline completo de pré-processamento do ECG em mV."""
    if record.p_signal is None or record.sig_name is None:
        empty = np.zeros((target_samples, 12), dtype=np.float32)
        return empty, {}, False

    fs_val = float(record.fs or target_fs)

    # 1. Padronizar unidade física para mV e reordenar derivações
    step1_mv, _ = standardize_amplitude_units_to_mv(record.p_signal, record.units)
    step1_reordered = reorder_leads(step1_mv, list(record.sig_name))

    # 2. Interpolação de valores não finitos (NaN/Inf) e validação de qualidade (flatline/incompletude)
    step2_interpolated, is_valid = interpolate_missing_and_validate(step1_reordered)
    if not is_valid:
        empty = np.zeros((target_samples, 12), dtype=np.float32)
        return empty, {"raw_mv": step1_reordered}, False

    # 3. Corte de picos mecânicos extremos (> ±10 mV) antes de reamostrar
    step3_clipped = np.clip(step2_interpolated, -MAX_PHYSICAL_MV, MAX_PHYSICAL_MV)

    # 4. Reamostragem fisiológica dinâmica para target_fs e target_samples (500 Hz / 5000 pontos)
    step4_resampled = resample_ecg_signal(
        step3_clipped,
        orig_fs=fs_val,
        target_fs=target_fs,
        target_samples=target_samples,
    )

    # 5. Filtragem digital zero-phase na grade padronizada de 500 Hz
    step5a_no_baseline = remove_baseline_wander_highpass(step4_resampled, fs=target_fs)
    step5b_no_powerline = remove_powerline_interference_notch(
        step5a_no_baseline, fs=target_fs
    )
    cnn_tensor_mv = remove_high_freq_muscle_noise_lowpass(
        step5b_no_powerline, fs=target_fs, cutoff_hz=45.0
    )

    steps = {
        "raw_mv": step1_reordered,
        "resampled_mv": step4_resampled,
        "no_baseline_mv": step5a_no_baseline,
        "no_powerline_mv": step5b_no_powerline,
        "clean_mv": cnn_tensor_mv,
        "baseline_noise_mv": step4_resampled - step5a_no_baseline,
        "high_freq_noise_mv": step5a_no_baseline - cnn_tensor_mv,
    }
    return cnn_tensor_mv, steps, True


def main() -> None:
    from data.reader import patients_records

    if not patients_records.empty:
        exam = patients_records.iloc[0]
        rec = load_raw_record(exam["path"])
        cnn_tensor, _, is_valid = preprocess_ecg_for_cnn(rec)
        print("Exame processado:")
        print(f"  - Válido: {is_valid}")
        print(f"  - Shape do tensor: {cnn_tensor.shape}")
        print(f"  - Min / Max (mV): {cnn_tensor.min():.2f} / {cnn_tensor.max():.2f}")


if __name__ == "__main__":
    main()
