"""
Módulo de Carregamento e Interface PyTorch Dataset para o CardioPredict.

Implementa `CardioECGDataset` para consumo de alto desempenho dos contêineres HDF5
particionados (`train.h5`, `val.h5`, `test.h5`) com zero desserialização redundante
e total compatibilidade com multi-workers no PyTorch DataLoader.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core.constants import (
    PROCESSED_DATA_DIR,
    SUBCLASSES,
    SUPERCLASSES,
    TARGET_CLASSES,
)


class CardioECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset PyTorch indexado em HDF5 para sinais de ECG de 12 derivações.

    Retorna por amostra:
        (signal_tensor, label_tensor, weight_tensor)
        - signal_tensor: torch.Tensor de float32 em shape (12, 5000) [Canais, Amostras]
        - label_tensor: torch.Tensor de float32 em shape (30,)
        - weight_tensor: torch.Tensor de float32 em shape (30,)
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: Path | str | None = None,
        standardize: bool = False,
        standardize_mode: str = "per_lead",
        transform: (
            Callable[
                [tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        self.split: str = split.lower()
        base_dir = (
            Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
        )
        self.h5_path: Path = base_dir / f"{self.split}.h5"

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"Contêiner HDF5 não encontrado para a partição '{self.split}': {self.h5_path}. "
                f"Execute src/data/exporter_h5.py para gerar os dados."
            )

        self.standardize = standardize
        self.standardize_mode = standardize_mode
        self.transform = transform
        self._h5_file: h5py.File | None = None
        self._pos_weights: torch.Tensor | None = None

        # Lê o tamanho do dataset e metadados abrindo e fechando conexão inicial
        with h5py.File(str(self.h5_path), "r") as f:
            ds_sig = f["signals"]
            assert isinstance(ds_sig, h5py.Dataset)
            self._length: int = int(ds_sig.shape[0])
            if "pos_weights" in f.attrs:
                self._pos_weights = torch.from_numpy(
                    np.asarray(f.attrs["pos_weights"], dtype=np.float32)
                )

    def _get_h5(self) -> h5py.File:
        """Abertura tardia (lazy) por processo filho, garantindo compatibilidade com DataLoader multiprocessing."""
        if self._h5_file is None:
            self._h5_file = h5py.File(str(self.h5_path), "r")
        return self._h5_file

    @property
    def pos_weights(self) -> torch.Tensor | None:
        """Pesos de classes positivas para balanceamento em BCEWithLogitsLoss."""
        return self._pos_weights

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"Índice {idx} fora dos limites do dataset (tamanho: {self._length})"
            )

        h5 = self._get_h5()
        ds_sig = h5["signals"]
        ds_lbl = h5["labels"]
        ds_wgh = h5["weights"]

        assert isinstance(ds_sig, h5py.Dataset)
        assert isinstance(ds_lbl, h5py.Dataset)
        assert isinstance(ds_wgh, h5py.Dataset)

        # Carrega a amostra da fatia HDF5: sinal em (5000, 12)
        raw_signal: np.ndarray = ds_sig[idx]
        raw_label: np.ndarray = ds_lbl[idx]
        raw_weight: np.ndarray = ds_wgh[idx]

        # Converte para Tensores PyTorch
        # Transpõe o sinal de (5000, 12) para (12, 5000) [Channels-first para Conv1D/SSM]
        signal_tensor = torch.from_numpy(raw_signal).float().permute(1, 0).contiguous()
        label_tensor = torch.from_numpy(raw_label).float()
        weight_tensor = torch.from_numpy(raw_weight).float()

        # Normalização opcional Z-score por derivação ou global
        if self.standardize:
            if self.standardize_mode == "per_lead":
                mean = signal_tensor.mean(dim=-1, keepdim=True)
                std = signal_tensor.std(dim=-1, keepdim=True).clamp_min(1e-6)
                signal_tensor = (signal_tensor - mean) / std
            elif self.standardize_mode == "global":
                mean = signal_tensor.mean()
                std = signal_tensor.std().clamp_min(1e-6)
                signal_tensor = (signal_tensor - mean) / std

        item = (signal_tensor, label_tensor, weight_tensor)
        if self.transform is not None:
            item = self.transform(item)

        return item

    def get_uid(self, idx: int) -> str:
        """Retorna o identificador ecg_uid da amostra."""
        h5 = self._get_h5()
        ds_uid = h5["ecg_uids"]
        assert isinstance(ds_uid, h5py.Dataset)
        val = ds_uid[idx]
        return val.decode("ascii") if isinstance(val, bytes) else str(val)

    def close(self) -> None:
        """Fecha o handle HDF5 aberto se existir."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()

    @property
    def target_classes(self) -> tuple[str, ...]:
        return TARGET_CLASSES

    @property
    def superclasses(self) -> tuple[str, ...]:
        return SUPERCLASSES

    @property
    def subclasses(self) -> tuple[str, ...]:
        return SUBCLASSES


__all__ = ["CardioECGDataset"]
