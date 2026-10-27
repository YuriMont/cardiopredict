# Design Doc: Pipeline de Tratamento de Dados e Harmonização Multilabel (MIMIC-IV-ECG & PTB-XL)

- **Data**: 2026-10-26
- **Status**: Revisado após Builder Check (Aguardando Aprovação do Usuário)
- **Autor**: CardioPredict AI Team

---

## 1. Contexto e Motivação (Intent & The Why)

O CardioPredict AI Server é uma plataforma de inteligência artificial de apoio à decisão clínica para pronto-atendimento cardiológico e telemedicina. O sistema realiza predição multilabel em exames de Eletrocardiograma (ECG) de 12 derivações.

A fundação de dados do projeto conta com 821.834 registros provenientes de duas bases hospitalares transcontinentais (MIMIC-IV-ECG e PTB-XL) unificadas no banco colunar embutido DuckDB (`data/cardiopredict.duckdb`) com particionamento estrito por paciente (80% Treino, 10% Validação e 10% Teste).

Para que as redes neurais profundas (CNN 1D, ResNet-1D, Mamba e Transformers) sejam treinadas com máxima acurácia e sem gargalos de I/O em disco, faz-se necessário construir a esteira completa de:
1. **Harmonização Ontológica Multilabel Hierárquica**: Mapear os laudos textuais do MIMIC-IV-ECG e as anotações SCP-ECG do PTB-XL para uma matriz unificada de 5 superclasses e 25 patologias específicas críticas de emergência.
2. **Processamento e Persistência Física de Biossinais**: Aplicar a cadeia de qualidade fisiológica de sinal (0.5–45 Hz, 500 Hz, ordenação padronizada e mV) e persistir os sinais em contêineres HDF5 estruturados por partição.

---

## 2. Objetivos, Não-Objetivos e Anti-Objetivos

### 2.1. Objetivos (Goals)
* Construir um mapeador ontológico determinístico entre os relatórios textuais do MIMIC (`report_0`..`report_17`) e os diagnósticos SCP-ECG do PTB-XL.
* Produzir uma tabela consolidada no DuckDB (`ecg_processed_labels`) contendo o vetor binário (0 ou 1) para 5 superclasses e 25 classes específicas, acompanhado de vetor de pesos de confiança clínica (`weight_<classe>`).
* Implementar esteira de processamento de sinais em lote com paralelismo (`multiprocessing`), consumindo os arquivos `.hea`/`.dat` e aplicando validação de qualidade, interpolação de falhas leves (< 5% NaNs), filtros passa-faixa (0.5–45 Hz) e notch (50/60 Hz).
* Persistir os traçados filtrados em três contêineres HDF5 particionados (`train.h5`, `val.h5`, `test.h5`) em `data/processed/`, armazenando em sincronia os sinais `(N, 5000, 12)`, os rótulos `(N, 30)`, os pesos `(N, 30)` e os identificadores `ecg_uids`.
* Fornecer um `Dataset` PyTorch nativo (`CardioECGDataset`) que realize leitura indexada $O(1)$ diretamente dos HDF5 com zero desserialização redundante.

### 2.2. Não-Objetivos (Non-Goals)
* Não contemplar retreinamento de arquiteturas neurais nesta etapa (o escopo encerra com os dados prontos para o `DataLoader`).
* Não modificar as partições existentes no DuckDB (os splits por paciente já calculados são imutáveis para garantir reproducibilidade).
* Não descartar exames sem laudo do banco relacional, apenas marcá-los como não rotulados na tabela relacional e excluí-los da matriz de treinamento supervisionado.

### 2.3. Anti-Objetivos (Anti-Goals)
* **Vazamento de dados (Data Leakage)**: Jamais misturar registros de um mesmo paciente entre partições de treino, validação ou teste.
* **Gargalo de I/O de Pequenos Arquivos**: Não gerar centenas de milhares de arquivos soltos em disco (como `.npy` individuais), evitando degradação de performance do sistema de arquivos.
* **Redução Redundante Multiclasse**: Não converter diagnósticos simultâneos em classes mutuamente exclusivas via Softmax; manter modelagem puramente Multilabel (`Sigmoid` + `BCEWithLogitsLoss`).

---

## 3. Restrições Técnicas e Regulatórias

1. **Padrão Anatômico e Dimensionalidade**: 
   * Ordem estrita das 12 derivações: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
   * Dimensão exata do tensor de entrada: `(5000, 12)` por exame (10 segundos a 500 Hz).
   * Escala de amplitude padronizada em milivolts (mV).
2. **Critérios de Rejeição de Sinal Ruim**:
   * Eletrodo mudo / flatline: desvio padrão da derivação $< 10^{-4}\text{ mV}$.
   * Amostras nulas: $> 5\%$ de valores NaN na derivação resulta em descarte do exame.
   * Amplitudes físicas aberrantes: valores absolutos $> 15\text{ mV}$ após filtragem.
3. **Padrões de Código e Tipagem**:
   * Conformidade obrigatória com `ruff check .` e `pyright` (zero erros).
   * Tipagem estrita em todas as funções públicas.

---

## 4. Abordagens Consideradas e Escolha de Arquitetura

### Abordagem 1: HDF5 por Partição + Tabela Relacional DuckDB (Escolhida)
* **Descrição**: Sinais persistidos em 3 contêineres HDF5 compactados (`train.h5`, `val.h5`, `test.h5`) organizados em chunks eficientes. Metadados e alvos binários multilabel salvos em tabela colunar no DuckDB.
* **Vantagens**: Acesso aleatório ultrarrápido; compressão reduz uso em disco para ~70–80 GB; sem sobrecarga de inodes; padrão ouro para biossinais.
* **Por que foi escolhida**: Melhor equilíbrio entre eficiência de disco, taxa de transferência de I/O para treinamento e aderência ao ambiente NVMe local.

### Abordagem 2: Arquivos Individuais `.npy` por Exame (Descartada)
* **Motivo de Descarte**: Geraria 820.000 arquivos individuais, degradando severamente a tabela de arquivos do sistema operacional e ocupando mais de 190 GB sem compressão.

### Abordagem 3: Chunks Colunares em Zarr / Parquet (Descartada)
* **Motivo de Descarte**: O shuffle aleatório por amostra no PyTorch DataLoader com chunks colunares requer descompactação de blocos inteiros em memória, gerando overhead desnecessário comparado ao chunking 1D do HDF5.

---

## 5. Especificação Detalhada da Implementação

### 5.1. Taxonomia e Harmonização Multilabel (5 Superclasses + 25 Subclasses Específicas)

#### 5.1.1. As 5 Superclasses SCP-ECG
1. `NORM`: Eletrocardiograma dentro dos limites normais.
2. `MI`: Infarto do Miocárdio (agudo, prévio ou indeterminado).
3. `STTC`: Alterações de Repolarização Ventricular / Isquemia (Segmento ST e Onda T).
4. `CD`: Distúrbios de Condução Cardíaca e Bloqueios.
5. `HYP`: Sobrecarga / Hipertrofia Ventricular ou Atrial.

#### 5.1.2. As 25 Classes Específicas de Emergência
1. `AFIB`: Fibrilação Atrial.
2. `AFLT`: Flutter Atrial.
3. `SR`: Ritmo Sinusal Normal.
4. `SBRAD`: Bradicardia Sinusal.
5. `STACH`: Taquicardia Sinusal.
6. `PVC`: Extrassístoles Ventriculares Prematuras.
7. `PAC`: Extrassístoles Atriais Prematuras.
8. `AMI`: Infarto de Parede Anterior / Anterosseptal (agudo, prévio ou idade indeterminada).
9. `IMI`: Infarto de Parede Inferior (agudo, prévio ou idade indeterminada).
10. `LMI`: Infarto de Parede Lateral (agudo, prévio ou idade indeterminada).
11. `STEMI`: Infarto com Supradesnivelamento do Segmento ST (Sinalizador Estrito de Urgência Aguda).
12. `CLBBB`: Bloqueio Completo de Ramo Esquerdo (BRE).
13. `CRBBB`: Bloqueio Completo de Ramo Direito (BRD).
14. `IRBBB`: Bloqueio Incompleto de Ramo Direito.
15. `LAFB`: Bloqueio Divisional Anterossuperior Esquerdo.
16. `1AVB`: Bloqueio Atrioventricular de 1º Grau.
17. `2AVB`: Bloqueio Atrioventricular de 2º Grau (Mobitz I / II).
18. `3AVB`: Bloqueio Atrioventricular Total (BAVT / 3º Grau).
19. `IVCD`: Atraso Inespecífico de Condução Intraventricular.
20. `LVH`: Sobrecarga Ventricular Esquerda (SVE).
21. `RVH`: Sobrecarga Ventricular Direita (SVD).
22. `STTC_ISCA`: Isquemia Anteroseptal / Inversão de T Isquêmica.
23. `STTC_ISCI`: Isquemia Inferior.
24. `STD`: Infradesnivelamento do Segmento ST.
25. `STE`: Supradesnivelamento inespecífico de ST / Repolarização Precoce.

#### 5.1.3. Regra de Incerteza Diagnóstica e Pesos por Classe
* A ponderação de incerteza clínica é gerada individualmente para cada uma das 30 classes (`weight_<classe>`):
  * **1.0**: Diagnóstico categórico e assertivo no laudo.
  * **0.5**: Laudo com termos de incerteza diagnóstica identificados por regex (`possible`, `borderline`, `cannot rule out`, `suggests`, `equivocal`).
  * **1.0 (ou 0.0 para perda mascarada)**: Quando o diagnóstico é negativo ausente.
* Essa matriz de pesos alimenta o argumento `weight` da `torch.nn.BCEWithLogitsLoss`, permitindo que incertezas em um achado (ex.: SVE duvidosa) não penalizem nem contaminem diagnósticos categóricos (ex.: IAM em evolução).

#### 5.1.4. Regras de Propagação Hierárquica e Consistência Clínica
* **Propagação Ascendente**: A ativação positiva de uma subclasse diagnóstica acarreta automaticamente a ativação da sua superclasse no vetor de alvos:
  * `AMI = 1`, `IMI = 1`, `LMI = 1` ou `STEMI = 1` $\implies$ `MI = 1`.
  * `CLBBB = 1`, `CRBBB = 1`, `IRBBB = 1`, `LAFB = 1`, `1AVB = 1`, `2AVB = 1`, `3AVB = 1` ou `IVCD = 1` $\implies$ `CD = 1`.
  * `LVH = 1` ou `RVH = 1` $\implies$ `HYP = 1`.
  * `STTC_ISCA = 1`, `STTC_ISCI = 1`, `STD = 1` ou `STE = 1` $\implies$ `STTC = 1`.
* **Classes de Ritmo e Normalidade**:
  * Classes de ritmo (`AFIB`, `AFLT`, `SR`, `SBRAD`, `STACH`, `PVC`, `PAC`) operam como descritores de ritmo independientes.
  * `SR = 1` só induz `NORM = 1` caso não exista nenhuma outra patologia ativa no exame.
  * `NORM = 1` e qualquer uma das 4 superclasses patológicas (`MI`, `STTC`, `CD`, `HYP`) são mutuamente exclusivas na supervisão (se uma patologia estiver presente, `NORM` é forçado a 0).

---

## 6. Pipeline de Processamento e Persistência Física dos Sinais

```
┌─────────────────────────┐
│  Arquivos WFDB Brutos   │  (.hea / .dat de MIMIC e PTB-XL)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Cadeia de Filtragem   │  - Reamostragem/garantia de 500 Hz (5.000 amostras)
│   e Padronização Fisiol.│  - Reordenação para I, II, III, aVR, aVL, aVF, V1..V6
└────────────┬────────────┘  - Normalização para mV
             │               - Passa-faixa Butterworth 0.5–45 Hz + Notch 50/60 Hz
             ▼
┌─────────────────────────┐
│  Filtro de Integridade  │  - Rejeição de Flatline (std < 1e-4 mV)
│    (Quality Gate)       │  - Rejeição de NaNs (> 5%) ou interpolação linear
└────────────┬────────────┘  - Rejeição de saturação (> 15 mV)
             │
             ▼
┌─────────────────────────┐
│  Contêineres HDF5       │  - train.h5  (partição 'train')
│  (Sinais + Rótulos +    │  - val.h5    (partição 'val')
│   Pesos sincronizados)  │  - test.h5   (partição 'test')
└─────────────────────────┘
```

### 6.1. Sincronização Absoluta DuckDB $\leftrightarrow$ HDF5
* Cada arquivo HDF5 (`train.h5`, `val.h5`, `test.h5`) contém internamente 4 datasets alinhados pelo mesmo índice numérico:
  1. `signals`: float32 em shape `(N, 5000, 12)`.
  2. `labels`: float32 em shape `(N, 30)` (5 superclasses + 25 subclasses).
  3. `weights`: float32 em shape `(N, 30)` (pesos de confiança por classe).
  4. `ecg_uids`: string ASCII em shape `(N,)`.
* No DuckDB, a tabela `ecg_processed_labels` armazena:
  * `ecg_uid VARCHAR PRIMARY KEY`
  * `split VARCHAR NOT NULL`
  * `is_signal_valid BOOLEAN NOT NULL`
  * `hdf5_index INTEGER` (índice posicional dentro do HDF5 da partição correspondente, ou `NULL` se o sinal foi descartado na qualidade)
  * As 30 colunas de targets e metadados de auditoria.
* Essa arquitetura elimina qualquer risco de desalinhamento posicional entre sinais e rótulos durante o treinamento concorrente com múltiplos processos (*workers* do DataLoader).

---

## 7. Estrutura de Arquivos e Módulos Criados / Alterados

1. [`src/core/constants.py`](./src/core/constants.py):
   * Definição de `SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]`.
   * Definição de `SUBCLASSES` (as 25 patologias específicas).
   * Definição de `TARGET_CLASSES = SUPERCLASSES + SUBCLASSES` (30 classes).
   * Definição do diretório de dados processados `PROCESSED_DATA_DIR = DATA_DIR / "processed"`.
2. [`src/processing/ontology.py`](./src/processing/ontology.py):
   * Mapeamento léxico de relatórios textuais do MIMIC (`report_0`..`report_17`).
   * Mapeamento de códigos SCP-ECG do PTB-XL.
   * Regras determinísticas de propagação hierárquica e cálculo de pesos de confiança.
3. [`src/data/labels.py`](./src/data/labels.py):
   * Criação do schema relacional no DuckDB (`ecg_processed_labels`).
   * Extração e povoamento dos rótulos multilabel para os 821 mil exames.
4. [`src/data/exporter_h5.py`](./src/data/exporter_h5.py):
   * Pipeline paralelo de extração de sinais WFDB, filtragem e escrita nos 3 arquivos HDF5 particionados.
   * Atualização dos ponteiros `hdf5_index` e flags `is_signal_valid` no DuckDB.
5. [`src/data/dataset.py`](./src/data/dataset.py):
   * Classe PyTorch `CardioECGDataset(torch.utils.data.Dataset)` com suporte a amostragem indexada em memória mapeada, cache inteligente e devolução direta de tensores `(signal, label, weight)`.

---

## 8. Decisões do Construtor (Builder's Calls)

* **Chunking HDF5**: Chunks de `(128, 5000, 12)` com compressão rápida `lzf` para minimizar latência de leitura randômica no NVMe.
* **Workers de Processamento**: `os.cpu_count() - 1` para maximizar o throughput da CPU sem travar o sistema.
* **Exames Sem Diagnóstico / Não-Alvo**: Exames cujo sinal é tecnicamente perfeito mas cujos laudos não mencionam nenhuma das 30 condições monitoradas e também não contêm atestado de normalidade são excluídos da matriz de treino supervisionado para evitar contaminação por falsos negativos silenciosos.
