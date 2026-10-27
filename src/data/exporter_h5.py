"""
Módulo de Extração, Filtragem e Persistência Física de Sinais em HDF5.

Processa exames brutos (.hea/.dat) das bases MIMIC-IV-ECG e PTB-XL,
aplica o pipeline fisiológico de pré-processamento e grava os tensores
e rótulos sincronizados nos contêineres HDF5 particionados:
`train.h5`, `val.h5` e `test.h5`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb
import h5py
import numpy as np

from core.constants import (
    ABERRANT_MAX_AMPLITUDE_MV,
    DATA_DIR,
    DEFAULT_DB_PATH,
    PROCESSED_DATA_DIR,
    TARGET_CLASSES,
)
from processing.signal import (
    load_raw_record,
    preprocess_ecg_for_cnn,
)


def process_single_exam(
    exam_info: tuple[str, str, list[float], list[float]],
) -> tuple[str, np.ndarray | None, list[float], list[float], bool]:
    """Processa um único exame físico de ECG.

    Retorna:
        (ecg_uid, tensor_ou_none, labels, weights, is_valid)
    """
    ecg_uid, rel_path, labels, weights = exam_info

    # 1. Checagem rápida de existência do arquivo .hea antes de chamar WFDB
    p = Path(rel_path)
    full_path = p if (p.exists() or str(p).startswith("data")) else DATA_DIR / p
    if full_path.suffix in {".hea", ".dat"}:
        full_path = full_path.with_suffix("")
    hea_path = full_path.with_suffix(".hea")
    if not hea_path.exists():
        return ecg_uid, None, labels, weights, False

    try:
        record = load_raw_record(full_path)
        cnn_tensor, _, is_valid = preprocess_ecg_for_cnn(record)

        if not is_valid:
            return ecg_uid, None, labels, weights, False

        # Garante exatas 5.000 amostras (10 segundos a 500 Hz)
        n_samples = cnn_tensor.shape[0]
        if n_samples < 5000:
            padded = np.zeros((5000, 12), dtype=np.float32)
            padded[:n_samples, :] = cnn_tensor
            cnn_tensor = padded
        elif n_samples > 5000:
            cnn_tensor = cnn_tensor[:5000, :]

        # Validação de amplitude máxima aberrante (> 15 mV)
        max_abs = float(np.max(np.abs(cnn_tensor)))
        if max_abs > ABERRANT_MAX_AMPLITUDE_MV or np.isnan(max_abs):
            return ecg_uid, None, labels, weights, False

        return ecg_uid, cnn_tensor.astype(np.float32), labels, weights, True

    except (ValueError, OSError, RuntimeError, KeyError, IndexError):
        return ecg_uid, None, labels, weights, False


def init_or_open_h5(
    h5_path: Path,
    chunk_size: int = 128,
) -> h5py.File:
    """Inicializa ou abre contêiner HDF5 particionado com datasets dinâmicos."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(h5_path), "a")

    str_dtype = h5py.string_dtype(encoding="ascii")
    n_classes = len(TARGET_CLASSES)

    if "signals" not in f:
        f.create_dataset(
            "signals",
            shape=(0, 5000, 12),
            maxshape=(None, 5000, 12),
            dtype=np.float32,
            chunks=(chunk_size, 5000, 12),
            compression="lzf",
        )
    if "labels" not in f:
        f.create_dataset(
            "labels",
            shape=(0, n_classes),
            maxshape=(None, n_classes),
            dtype=np.float32,
            chunks=(chunk_size, n_classes),
        )
    if "weights" not in f:
        f.create_dataset(
            "weights",
            shape=(0, n_classes),
            maxshape=(None, n_classes),
            dtype=np.float32,
            chunks=(chunk_size, n_classes),
        )
    if "ecg_uids" not in f:
        f.create_dataset(
            "ecg_uids",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(chunk_size,),
        )

    return f


def append_batch_to_h5(
    h5_file: h5py.File,
    signals_batch: list[np.ndarray],
    labels_batch: list[list[float]],
    weights_batch: list[list[float]],
    uids_batch: list[str],
) -> int:
    """Acrescenta um bloco de exames validados ao arquivo HDF5."""
    if not signals_batch:
        return 0

    batch_len = len(signals_batch)
    ds_sig = h5_file["signals"]
    ds_lbl = h5_file["labels"]
    ds_wgh = h5_file["weights"]
    ds_uid = h5_file["ecg_uids"]

    assert isinstance(ds_sig, h5py.Dataset)
    assert isinstance(ds_lbl, h5py.Dataset)
    assert isinstance(ds_wgh, h5py.Dataset)
    assert isinstance(ds_uid, h5py.Dataset)

    current_len = ds_sig.shape[0]
    new_len = current_len + batch_len

    ds_sig.resize(new_len, axis=0)
    ds_lbl.resize(new_len, axis=0)
    ds_wgh.resize(new_len, axis=0)
    ds_uid.resize(new_len, axis=0)

    ds_sig[current_len:new_len] = np.stack(signals_batch, axis=0)
    ds_lbl[current_len:new_len] = np.array(labels_batch, dtype=np.float32)
    ds_wgh[current_len:new_len] = np.array(weights_batch, dtype=np.float32)
    ds_uid[current_len:new_len] = uids_batch

    return current_len


def export_split_to_h5(
    split_name: str,
    db_path: Path | str | None = None,
    output_dir: Path | None = None,
    batch_size: int = 512,
    limit: int | None = None,
) -> dict[str, int]:
    """Processa e exporta todos os registros de uma partição para arquivo HDF5."""
    target_db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    out_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / f"{split_name}.h5"
    if h5_path.exists():
        h5_path.unlink()  # Garante recriação limpa

    h5_file = init_or_open_h5(h5_path)

    con = duckdb.connect(str(target_db), read_only=False)

    limit_sql = f"LIMIT {limit}" if limit is not None else ""

    # Constrói listas de colunas de label e weight ordenadas
    label_cols = [f"p.label_{cls.lower()}" for cls in TARGET_CLASSES]
    weight_cols = [f"p.weight_{cls.lower()}" for cls in TARGET_CLASSES]
    cols_query = ", ".join(label_cols + weight_cols)

    query = f"""
        SELECT 
            r.ecg_uid, 
            r.waveform_relative_path,
            {cols_query}
        FROM ecg_records r
        JOIN ecg_processed_labels p ON r.ecg_uid = p.ecg_uid
        WHERE r.split = '{split_name}'
          AND p.is_supervised_target = TRUE
        ORDER BY r.ecg_uid
        {limit_sql}
    """

    cursor = con.cursor()
    cursor.execute(query)

    n_classes = len(TARGET_CLASSES)
    total_processed = 0
    total_valid = 0
    total_rejected = 0

    valid_updates: list[tuple[int, str]] = []  # (hdf5_index, ecg_uid)
    invalid_updates: list[str] = []  # ecg_uids

    start_time = time.time()

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        signals_to_write: list[np.ndarray] = []
        labels_to_write: list[list[float]] = []
        weights_to_write: list[list[float]] = []
        uids_to_write: list[str] = []

        for row in rows:
            ecg_uid = str(row[0])
            rel_path = str(row[1])
            labels = [float(v) for v in row[2 : 2 + n_classes]]
            weights = [float(v) for v in row[2 + n_classes : 2 + 2 * n_classes]]

            uid, tensor, lbl, wgt, is_valid = process_single_exam(
                (ecg_uid, rel_path, labels, weights)
            )
            total_processed += 1

            if is_valid and tensor is not None:
                signals_to_write.append(tensor)
                labels_to_write.append(lbl)
                weights_to_write.append(wgt)
                uids_to_write.append(uid)
            else:
                total_rejected += 1
                invalid_updates.append(uid)

        # Gravação no HDF5
        if signals_to_write:
            start_pos = append_batch_to_h5(
                h5_file,
                signals_to_write,
                labels_to_write,
                weights_to_write,
                uids_to_write,
            )
            for idx_offset, uid in enumerate(uids_to_write):
                valid_updates.append((start_pos + idx_offset, uid))
            total_valid += len(signals_to_write)

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0.0
        print(
            f"[{split_name.upper()}] Processados: {total_processed:,} | "
            f"Válidos: {total_valid:,} | Rejeitados: {total_rejected:,} ({rate:,.0f} exames/s)"
        )

    h5_file.close()

    # Atualiza o DuckDB com os índices do HDF5 e status físico do sinal
    print(f"Atualizando índices de {split_name} no DuckDB...")
    if valid_updates:
        con.executemany(
            "UPDATE ecg_processed_labels SET hdf5_index = ?, is_signal_valid = TRUE WHERE ecg_uid = ?;",
            valid_updates,
        )
    if invalid_updates:
        # Marca exames descartados
        con.executemany(
            "UPDATE ecg_processed_labels SET hdf5_index = NULL, is_signal_valid = FALSE WHERE ecg_uid = ?;",
            [(uid,) for uid in invalid_updates],
        )

    con.close()
    return {
        "processed": total_processed,
        "valid": total_valid,
        "rejected": total_rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta sinais de ECG filtrados e alvos para HDF5"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Partição a exportar",
    )
    parser.add_argument("--db-path", type=str, default=None, help="Caminho do DuckDB")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Diretório de saída HDF5"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Tamanho do lote em memória"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de exames para teste rápido",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [str(args.split)]

    for sp in splits:
        print(f"\n==================== EXPORTANDO SPLIT: {sp} ====================")
        res = export_split_to_h5(
            split_name=sp,
            db_path=args.db_path,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(
            f"Concluído {sp}: {res['valid']:,} salvos no HDF5 ({res['rejected']:,} rejeitados)."
        )


if __name__ == "__main__":
    main()
