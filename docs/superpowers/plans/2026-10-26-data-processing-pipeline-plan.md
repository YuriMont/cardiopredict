# Plano de Implementação: Pipeline de Tratamento de Dados e Harmonização Multilabel

- **Data**: 2026-10-26
- **Status**: Aprovado para Execução
- **Documento de Design Relacionado**: [`docs/superpowers/specs/2026-10-26-data-processing-pipeline-design.md`](./docs/superpowers/specs/2026-10-26-data-processing-pipeline-design.md)

---

## 1. Visão Geral da Execução

Este plano detalha a construção e validação da esteira completa de engenharia de dados e biossinais para as bases **MIMIC-IV-ECG** e **PTB-XL**, dividida em 5 fases sequenciais com verificação contínua de tipos (`pyright`), estilo (`ruff`) e integridade clínica:

```
[Fase 1: Constantes e Taxonomia] ──> [Fase 2: Motor Ontológico de Laudos]
                                                    │
                                                    ▼
[Fase 4: Exporter HDF5 Multiprocesso] <── [Fase 3: Tabela DuckDB ecg_processed_labels]
                 │
                 ▼
[Fase 5: PyTorch CardioECGDataset & Suíte de Testes Integrada]
```

---

## 2. Tarefas Detalhadas por Fase

### Fase 1: Atualização de Constantes e Definições de Classes
* **Arquivo alvo**: [`src/core/constants.py`](./src/core/constants.py)
* **Ações**:
  1. Definir lista imutável das 5 superclasses: `SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")`.
  2. Definir lista imutável das 25 subclasses: `SUBCLASSES = ("AFIB", "AFLT", "SR", "SBRAD", "STACH", "PVC", "PAC", "AMI", "IMI", "LMI", "STEMI", "CLBBB", "CRBBB", "IRBBB", "LAFB", "1AVB", "2AVB", "3AVB", "IVCD", "LVH", "RVH", "STTC_ISCA", "STTC_ISCI", "STD", "STE")`.
  3. Definir `TARGET_CLASSES = SUPERCLASSES + SUBCLASSES` (total 30 classes ordenadas).
  4. Definir diretório padrão de artefatos processados: `PROCESSED_DATA_DIR = DATA_DIR / "processed"`.
  5. Adicionar limiares físicos de qualidade de sinal:
     * `FLATLINE_STD_MIN_MV = 1e-4`
     * `MAX_ALLOWED_NAN_RATIO = 0.05`
     * `ABERRANT_MAX_AMPLITUDE_MV = 15.0`
* **Validação**: `ruff check src/core/constants.py` e `pyright src/core/constants.py`.

---

### Fase 2: Motor Ontológico de Harmonização de Laudos
* **Arquivo alvo**: [`src/processing/ontology.py`](./src/processing/ontology.py)
* **Ações**:
  1. Criar dicionário de correspondência de códigos SCP-ECG do PTB-XL (`PTBXL_SCP_TO_TARGETS`):
     * Mapeamento de códigos como `NORM`, `AMI`, `IMI`, `LMI`, `CLBBB`, `CRBBB`, `AFIB`, `LVH`, etc. para os respectivos alvos.
  2. Criar tabela canônica de expressões regulares e termos léxicos do MIMIC-IV-ECG (`MIMIC_REPORT_RULES`):
     * Mapear variações ortográficas e diagnósticos combinados:
       * Infarto Anterior/Anterosseptal $\rightarrow$ `AMI` (e propaga `MI = 1`)
       * Infarto Inferior $\rightarrow$ `IMI` (e propaga `MI = 1`)
       * Infarto Lateral $\rightarrow$ `LMI` (e propaga `MI = 1`)
       * Supradesnivelamento agudo de ST / STEMI $\rightarrow$ `STEMI` e `STE`
       * Fibrilação e Flutter Atrial $\rightarrow$ `AFIB` e `AFLT`
       * Bloqueios de ramo $\rightarrow$ `CLBBB`, `CRBBB`, `IRBBB`, `LAFB`
       * Bloqueios atrioventriculares $\rightarrow$ `1AVB`, `2AVB`, `3AVB`
       * Sobrecargas ventriculares $\rightarrow$ `LVH`, `RVH`
  3. Implementar função pura `map_exam_statements(statements: list[str], is_ptbxl: bool) -> tuple[dict[str, int], dict[str, float]]`:
     * Devolve vetor binário das 30 classes e vetor de pesos `weight` (`0.5` para incerteza, `1.0` para categórico).
     * Aplica propagação hierárquica automática (subclasses $\rightarrow$ superclasses).
     * Aplica regra de exclusão mútua (`NORM` vs patologias ativas).
* **Validação**: Testes unitários com casos clínicos reais do MIMIC e PTB-XL.

---

### Fase 3: Geração e Carga Relacional no DuckDB (`ecg_processed_labels`)
* **Arquivo alvo**: [`src/data/labels.py`](./src/data/labels.py)
* **Ações**:
  1. Criar tabela relacional no DuckDB:
     ```sql
     CREATE TABLE IF NOT EXISTS ecg_processed_labels (
         ecg_uid VARCHAR PRIMARY KEY,
         patient_uid VARCHAR NOT NULL,
         source_dataset VARCHAR NOT NULL,
         split VARCHAR NOT NULL,
         is_signal_valid BOOLEAN DEFAULT TRUE,
         hdf5_index INTEGER,
         is_supervised_target BOOLEAN NOT NULL DEFAULT TRUE,
         -- 30 colunas de rótulo binário (label_<classe> INTEGER)
         -- 30 colunas de peso de confiança (weight_<classe> DOUBLE)
         FOREIGN KEY (ecg_uid) REFERENCES ecg_records(ecg_uid)
     );
     ```
  2. Implementar migração em lotes (*batch processing*) consultando `ecg_records` e `ecg_labels` já consolidados no DuckDB.
  3. Criar índices colunares em `(split, is_supervised_target)` para consultas instantâneas.
* **Validação**: Execução do script gerador e verificação das contagens de classes no DuckDB.

---

### Fase 4: Pipeline Multiprocesso de Sinais e Persistência HDF5
* **Arquivo alvo**: [`src/data/exporter_h5.py`](./src/data/exporter_h5.py)
* **Ações**:
  1. Estruturar processamento paralelo com `multiprocessing.Pool`:
     * Ler caminhos dos arquivos `.hea`/`.dat` via WFDB.
     * Aplicar filtro passa-faixa (0.5–45 Hz Butterworth) e notch (50/60 Hz).
     * Reordenar derivações para `STANDARD_LEADS` e padronizar amplitude para mV.
     * Validar critérios de descarte (flatline $std < 10^{-4}\text{ mV}$, $>5\%$ NaNs, saturação $>15\text{ mV}$).
  2. Gravar os contêineres HDF5 por partição (`data/processed/train.h5`, `val.h5`, `test.h5`):
     * `signals`: shape `(N, 5000, 12)`, `float32`, chunk size `(128, 5000, 12)` com compressão `lzf`.
     * `labels`: shape `(N, 30)`, `float32`.
     * `weights`: shape `(N, 30)`, `float32`.
     * `ecg_uids`: shape `(N,)`, string ASCII.
  3. Atualizar no DuckDB os campos `is_signal_valid` e `hdf5_index` correspondentes a cada exame processado.
  4. Suportar modo teste com flag `--limit` para validação rápida de fatias antes do processamento completo.
* **Validação**: Auditoria de integridade do arquivo HDF5 gerado e checagem de alinhamento com DuckDB.

---

### Fase 5: PyTorch Dataset e Suíte de Testes Automatizada
* **Arquivos alvos**:
  * [`src/data/dataset.py`](./src/data/dataset.py)
  * [`tests/test_data_pipeline.py`](./tests/test_data_pipeline.py)
* **Ações**:
  1. Implementar classe `CardioECGDataset(torch.utils.data.Dataset)`:
     * Métodos `__len__` e `__getitem__` com leitura $O(1)$ do HDF5.
     * Suporte seguro a multi-workers do `torch.utils.data.DataLoader` (abertura tardia do descritor de arquivo HDF5 por processo filho do worker).
     * Retorno limpo de tupla: `(signal_tensor: Tensor[12, 5000], label_tensor: Tensor[30], weight_tensor: Tensor[30])`.
  2. Implementar bateria de testes unitários e de integração:
     * Teste da ontologia clínica e propagação de classes.
     * Teste dos filtros de qualidade física de sinal.
     * Teste de leitura e concorrência do `CardioECGDataset`.
* **Validação**: Execução completa de `pytest`, `ruff check .`, `ruff format --check .` e `pyright`.

---

## 3. Critérios de Conclusão e Entrega

1. **Protocolo AGENTS.md Obrigatório**:
   * `ruff check .` $\rightarrow$ 0 erros.
   * `ruff format --check .` $\rightarrow$ 0 erros.
   * `pyright` $\rightarrow$ 0 erros.
2. **Qualidade dos Dados**:
   * 100% dos exames válidos persistidos no HDF5 alinhados com DuckDB sem nenhum caso de índice desencontrado.
   * Zero data leakage entre partições (`train`, `val`, `test`).
