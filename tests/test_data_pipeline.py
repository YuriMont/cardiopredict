"""
Suíte de Testes Automatizados para o Pipeline de Dados e Harmonização Multilabel.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Garante resolução do diretório raiz `src` mesmo quando executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

from core.constants import (
    DEFAULT_DB_PATH,
    PROCESSED_DATA_DIR,
    SUBCLASSES,
    SUPERCLASSES,
    TARGET_CLASSES,
)
from data.dataset import CardioECGDataset
from processing.measurements import clean_machine_measurements
from processing.ontology import map_exam_statements


class TestDataPipeline(unittest.TestCase):
    """Testes automatizados de ontologia, DuckDB e persistência HDF5."""

    def test_taxonomy_constants(self) -> None:
        """Valida a integridade da taxonomia de 5 superclasses e 25 subclasses."""
        self.assertEqual(len(SUPERCLASSES), 5)
        self.assertEqual(len(SUBCLASSES), 25)
        self.assertEqual(len(TARGET_CLASSES), 30)
        self.assertIn("NORM", SUPERCLASSES)
        self.assertIn("MI", SUPERCLASSES)
        self.assertIn("AFIB", SUBCLASSES)
        self.assertIn("AMI", SUBCLASSES)
        self.assertIn("CLBBB", SUBCLASSES)

    def test_ontology_ptbxl_mapping(self) -> None:
        """Valida o mapeamento de códigos SCP-ECG do PTB-XL."""
        labels, weights = map_exam_statements(["IMI", "SR"], is_ptbxl=True)
        self.assertEqual(labels["IMI"], 1)
        self.assertEqual(labels["MI"], 1)  # Propagação hierárquica
        self.assertEqual(labels["SR"], 1)
        self.assertEqual(labels["NORM"], 0)  # Exclusão mútua por ter MI ativo
        self.assertEqual(weights["IMI"], 1.0)
        self.assertEqual(weights["MI"], 1.0)

        # Teste de Normalidade pura
        labels_norm, _ = map_exam_statements(["NORM", "SR"], is_ptbxl=True)
        self.assertEqual(labels_norm["NORM"], 1)
        self.assertEqual(labels_norm["SR"], 1)
        self.assertEqual(labels_norm["MI"], 0)
        self.assertEqual(labels_norm["CD"], 0)

    def test_ontology_mimic_lexical_mapping_and_uncertainty(self) -> None:
        """Valida a extração léxica e detecção de incerteza em laudos do MIMIC."""
        statements = [
            "sinus rhythm",
            "possible anterior infarct - age undetermined",
            "left ventricular hypertrophy",
        ]
        labels, weights = map_exam_statements(statements, is_ptbxl=False)

        self.assertEqual(labels["SR"], 1)
        self.assertEqual(labels["AMI"], 1)
        self.assertEqual(labels["MI"], 1)
        self.assertEqual(labels["LVH"], 1)
        self.assertEqual(labels["HYP"], 1)
        self.assertEqual(labels["NORM"], 0)

        # AMI e MI devem refletir a incerteza de 'possible' (peso 0.5)
        self.assertEqual(weights["AMI"], 0.5)
        self.assertEqual(weights["MI"], 0.5)
        # LVH foi categórico (peso 1.0)
        self.assertEqual(weights["LVH"], 1.0)

    def test_ontology_hierarchical_conduction_blocks(self) -> None:
        """Valida que bloqueios ativam a superclasse CD."""
        labels, _ = map_exam_statements(
            ["complete left bundle branch block", "first degree av block"],
            is_ptbxl=False,
        )
        self.assertEqual(labels["CLBBB"], 1)
        self.assertEqual(labels["1AVB"], 1)
        self.assertEqual(labels["CD"], 1)

    def test_duckdb_processed_labels_table(self) -> None:
        """Valida que a tabela ecg_processed_labels existe e possui as colunas obrigatórias."""
        if not DEFAULT_DB_PATH.exists():
            return

        con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
        row = con.execute("SELECT count(*) FROM ecg_processed_labels").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        count = row[0]
        self.assertGreater(
            count, 0, "A tabela ecg_processed_labels não deve estar vazia"
        )

        cols = [
            col[0] for col in con.execute("DESCRIBE ecg_processed_labels").fetchall()
        ]
        self.assertIn("ecg_uid", cols)
        self.assertIn("split", cols)
        self.assertIn("is_supervised_target", cols)
        self.assertIn("label_norm", cols)
        self.assertIn("label_mi", cols)
        self.assertIn("weight_ami", cols)
        self.assertIn("weight_clbbb", cols)
        con.close()

    def test_h5_export_structure(self) -> None:
        """Valida a estrutura física do contêiner HDF5 gerado."""
        h5_path = PROCESSED_DATA_DIR / "test.h5"
        if not h5_path.exists():
            return

        with h5py.File(str(h5_path), "r") as f:
            self.assertIn("signals", f)
            self.assertIn("labels", f)
            self.assertIn("weights", f)
            self.assertIn("ecg_uids", f)

            ds_sig = f["signals"]
            ds_lbl = f["labels"]
            ds_wgh = f["weights"]
            ds_uid = f["ecg_uids"]

            assert isinstance(ds_sig, h5py.Dataset)
            assert isinstance(ds_lbl, h5py.Dataset)
            assert isinstance(ds_wgh, h5py.Dataset)
            assert isinstance(ds_uid, h5py.Dataset)

            n_records = ds_sig.shape[0]
            self.assertEqual(ds_lbl.shape[0], n_records)
            self.assertEqual(ds_wgh.shape[0], n_records)
            self.assertEqual(ds_uid.shape[0], n_records)

            if n_records > 0:
                self.assertEqual(ds_sig.shape[1:], (5000, 12))
                self.assertEqual(ds_lbl.shape[1], 30)
                self.assertEqual(ds_wgh.shape[1], 30)
                self.assertEqual(ds_sig.dtype, np.float32)
                self.assertEqual(ds_lbl.dtype, np.float32)

    def test_clean_measurements_smoke(self) -> None:
        """Smoke test para o módulo de medições fisiológicas."""
        df_dummy = pd.DataFrame(
            {
                "study_id": [1],
                "rr_interval": [800.0],
                "p_onset": [40.0],
                "p_end": [120.0],
                "qrs_onset": [200.0],
                "qrs_end": [290.0],
                "t_end": [600.0],
                "qrs_axis": [30.0],
                "t_axis": [40.0],
            }
        )
        cleaned = clean_machine_measurements(df_dummy)
        self.assertIn("heart_rate_bpm", cleaned.columns)
        self.assertEqual(cleaned["heart_rate_bpm"].iloc[0], 75.0)
        self.assertEqual(cleaned["is_technically_valid"].iloc[0], 1)

    def test_pytorch_dataset_loading(self) -> None:
        """Valida a leitura do CardioECGDataset a partir do contêiner HDF5."""
        h5_path = PROCESSED_DATA_DIR / "test.h5"
        if not h5_path.exists():
            return

        ds = CardioECGDataset(split="test")
        self.assertGreater(len(ds), 0)
        sig, lbl, wgt = ds[0]
        self.assertEqual(sig.shape, (12, 5000))
        self.assertEqual(lbl.shape, (30,))
        self.assertEqual(wgt.shape, (30,))
        self.assertEqual(sig.dtype, torch.float32)
        self.assertEqual(lbl.dtype, torch.float32)
        self.assertEqual(wgt.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
