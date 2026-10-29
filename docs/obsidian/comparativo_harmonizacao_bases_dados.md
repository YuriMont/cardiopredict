# Comparativo das Bases de Dados e Estratégia de Harmonização 🫀

Este documento detalha as diferenças estruturais, clínicas, elétricas e biomédicas entre as duas bases de dados adotadas no **CardioPredict** (**MIMIC-IV-ECG** e **PTB-XL**) e explica a engenharia empregada para harmonizá-las em um pipeline analítico unificado.

---

## 1. Tabela Comparativa Detalhada

| Dimensão | 🇺🇸 MIMIC-IV-ECG | 🇩🇪 PTB-XL |
| :--- | :--- | :--- |
| **Origem Geográfica** | Beth Israel Deaconess Medical Center (Boston, Massachusetts, EUA) | Physikalisch-Technische Bundesanstalt & centros clínicos associados (Alemanha) |
| **Volume de Dados** | **Massivo**: ~800.000 exames de ~160.000 pacientes | **Controlado**: 21.837 exames de 18.885 pacientes |
| **Contexto Clínico** | Pacientes de emergência, UTI cardiológica e internação de alta complexidade | Pacientes ambulatoriais, enfermaria geral e triagens preventivas |
| **Padrão de Rotulagem** | **Texto Livre não-estruturado**: laudos em inglês ditados por cardiologistas com jargões e incertezas | **Códigos Formais Padronizados**: vocabulário estruturado europeu **SCP-ECG** (71 códigos) |
| **Rede Elétrica Local** | **60 Hz** (padrão dos Estados Unidos) | **50 Hz** (padrão da Europa) |
| **Hardware / Fabricante** | GE Healthcare (sistemas Marquette MAC 5500 / MUSE) | Schiller AG (eletrocardiógrafos Schiller AT-104 PC) |
| **Escala e Unidades** | Gravado em microvolts ($\mu\text{V}$) ou ganho LSB do conversor A/D da GE | Gravado em milivolts ($\text{mV}$) ou resolução LSB Schiller |
| **Frequência de Amostragem** | Nativo e fixo a **500 Hz** (5.000 amostras em 10 segundos) | Versão dual do PhysioNet: **100 Hz** (`records100`) e **500 Hz** (`records500`) |
| **Nomenclatura das Derivações** | Padrão misto (ex.: `aVR`, `aVL`, `aVF`) | Padrão maiúsculo (ex.: `AVR`, `AVL`, `AVF`) |
| **Particionamento Original** | Inexistente; múltiplos exames seriados por paciente internado | Estratificado oficial em 10 dobras (`strat_fold 1..10`) de referência acadêmica |

---

## 2. As Diferenças e Como Foram/Serão Trabalhadas

### 2.1. Desafio 1: Laudos em Texto Livre vs Códigos Estruturados SCP-ECG
* **O Problema**: Redes neurais supervisionadas exigem tensores binários homogêneos. O MIMIC contém laudos humanos com termos coloquiais (*"cannot rule out acute inferior infarction"*, *"borderline LVH"*), enquanto o PTB-XL adota códigos categóricos (`IMI`, `LVH`, `AFIB`).
* **Tratamento Implementado ([`src/processing/ontology.py`](../../src/processing/ontology.py))**:
  1. **Motor Léxico com Detecção de Incerteza (MIMIC)**: Mapeamento determinístico via regex que reconhece termos qualificadores de dúvida (`possible`, `probable`, `suggests`, `cannot rule out`). Quando a incerteza é detectada, o rótulo é ativado como $1$, mas seu peso de confiança é reduzido para $0,5$ no tensor de perda.
  2. **Motor Ontológico SCP-ECG (PTB-XL)**: Dicionário relacional que mapeia as declarações SCP para a taxonomia padrão.
  3. **Espaço Comum de 30 Classes Alvo**: Convergência para 5 Superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`) e 25 Subclasses críticas, com propagação hierárquica automática (ex.: qualquer bloqueio ativa `CD`).

---

### 2.2. Desafio 2: Interferência da Rede Elétrica Transcontinental (50 Hz vs 60 Hz)
* **O Problema**: A corrente alternada dos hospitais irradia ondas senoidais captadas pelos cabos dos eletrodos. Se deixado intacto, o ruído residual de 60 Hz marca os dados americanos e o de 50 Hz marca os europeus, induzindo o modelo a **Shortcut Learning** (aprender a origem do hospital em vez da cardiologia).
* **Tratamento Implementado ([`src/processing/signal.py`](../../src/processing/signal.py))**:
  * **Filtro Notch Duplo Comutativo**: Aplicação em cascata de dois filtros rejeita-faixa IIR de alta seletividade ($Q=30$) centrados em $50\text{ Hz}$ ($\Delta f \approx 1,67\text{ Hz}$) e $60\text{ Hz}$ ($\Delta f \approx 2,0\text{ Hz}$).
  * **Fase Zero (`sosfiltfilt`)**: Filtragem bidirecional em Second-Order Sections (SOS) que anula qualquer distorção de fase, conservando rigorosamente a duração de $PR$, $QRS$ e $QT$.
  * **Passa-Faixa Butterworth (0,5 a 45 Hz)**: Corta a oscilação da linha de base e ruídos musculares acima de 45 Hz, atuando em conjunto com o Notch duplo.

---

### 2.3. Desafio 3: Calibração de Amplitude e Ganho dos Sensores ($\mu\text{V}$ vs mV vs LSB)
* **O Problema**: Sensores com ganhos analógicos desiguais geram amplitudes numéricas sem correspondência biomédica direta.
* **Tratamento Implementado ([`src/processing/signal.py`](../../src/processing/signal.py) e [`src/data/dataset.py`](../../src/data/dataset.py))**:
  1. **Conversão Absoluta para mV**: Leitura de metadados WFDB (`record.units`) e multiplicação matricial por fatores padronizados (`UNIT_TO_MV_FACTOR`), garantindo que $1,0$ na matriz HDF5 signifique exatamente $1,0\text{ mV}$ no miocárdio.
  2. **Normalização Z-Score Opcional por Derivação**: Para camadas de Self-Attention (Transformer) e SSM (Mamba), o `CardioECGDataset` oferece normalização $\frac{x_c - \mu_c}{\sigma_c + 10^{-6}}$ no eixo do tempo, estabilizando os gradientes sem apagar a morfologia relativa dos complexos.

---

### 2.4. Desafio 4: Nomenclatura e Ordem dos Canais / Derivações
* **O Problema**: No PTB-XL os canais são informados em maiúsculas (`AVR, AVL, AVF`), enquanto o MIMIC adota convenções mistas (`aVR, aVL, aVF`). Se as colunas de uma matriz $(5000, 12)$ forem invertidas, a rede interpretará o ventrículo esquerdo como se fosse o ventrículo direito.
* **Tratamento Implementado ([`src/processing/signal.py`](../../src/processing/signal.py))**:
  * A função `reorder_leads` normaliza todas as strings para maiúsculas e reordena os canais para a sequência canônica dos 12 eixos:
    $$\text{STANDARD\_LEADS} = [\text{I}, \text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V1}, \text{V2}, \text{V3}, \text{V4}, \text{V5}, \text{V6}]$$

---

### 2.5. Desafio 5: Taxas de Amostragem e Dualidade do PTB-XL (100 Hz vs 500 Hz)
* **O Problema**: O PTB-XL oferece versões a 100 Hz (`records100`) e 500 Hz (`records500`) dos mesmos exames. Utilizar 100 Hz achata o pico $R$, e importar ambas as versões duplicaria 100% dos exames.
* **Tratamento Implementado ([`src/data/database.py`](../../src/data/database.py) e [`src/processing/signal.py`](../../src/processing/signal.py))**:
  * O catálogo mestre no DuckDB seleciona exclusivamente a pasta `records500/` (`filename_hr` a 500 Hz), ignorando a versão subamostrada e evitando duplicações redundantes.
  * O pipeline incorpora a função `resample_ecg_signal`, que utiliza interpolação de Fourier para reamostrar dinamicamente para 5.000 pontos qualquer traçado futuro proveniente de fontes com frequências divergentes (ex.: 100 Hz ou 250 Hz).

---

### 2.6. Desafio 6: Desbalanceamento de Volume e Prevenção de Vazamento Clínico
* **O Problema**: O MIMIC é cerca de 38 vezes maior que o PTB-XL e contém exames repetidos de pacientes internados.
* **Tratamento Implementado ([`src/data/split.py`](../../src/data/split.py) e [`src/data/exporter_h5.py`](../../src/data/exporter_h5.py))**:
  * **Particionamento por Paciente (Zero Leakage)**: A divisão 80% treino / 10% validação / 10% teste é feita estritamente no nível do identificador do paciente (`subject_id` e `patient_id`), adotando a dobra 10 do PTB-XL como partição canônica de teste internacional.
  * **Pesos Positivos no HDF5**: O contêiner de persistência armazena o vetor de pesos de balanceamento `pos_weights` para parametrização direta da função de custo `BCEWithLogitsLoss`.

---

## 3. Composição Quantitativa das Partições (Treino, Validação e Teste)

### 3.1. É Aleatório?
**Não, o particionamento não é aleatório.** Ele é **estratificado, determinístico e agrupado estritamente no nível do paciente** (*patient-level split*), garantindo que nenhum sinal de um mesmo indivíduo atravesse as fronteiras entre treino e teste:
- **No PTB-XL**: Segue a divisão oficial em 10 dobras estratificadas (`strat_fold 1..10`), desenhada pelo consórcio alemão com base na co-ocorrência de patologias raras. As dobras 1 a 8 compõem o Treino (80%), a dobra 9 a Validação (10%) e a dobra 10 o Teste (10%).
- **No MIMIC-IV-ECG**: Realizado por hash determinístico do identificador do paciente (`(subject_id % 10)`), onde restos 0 a 7 compõem o Treino (80%), resto 8 a Validação (10%) e resto 9 o Teste (10%).

### 3.2. Proporção Exata de Cada Base nas Partições

Como ambas as bases foram divididas individualmente na proporção 80% / 10% / 10% por paciente, todas as três partições mantêm rigorosamente a mesma proporção natural das fontes de dados:

| Partição | Exames MIMIC | Exames PTB-XL | **Total de Exames** | Pacientes MIMIC | Pacientes PTB-XL | **Total Pacientes** | % MIMIC | % PTB-XL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Treino (`train`)** | 640.271 | 17.418 | **657.689** | 129.005 | 15.023 | **144.028** | **97,35%** | **2,65%** |
| **Validação (`val`)** | 79.514 | 2.183 | **81.697** | 16.134 | 1.942 | **18.076** | **97,33%** | **2,67%** |
| **Teste (`test`)** | 80.250 | 2.198 | **82.448** | 16.213 | 1.904 | **18.117** | **97,33%** | **2,67%** |
| **Total Geral** | **800.035** | **21.799** | **821.834** | **161.352** | **18.869** | **180.221** | **97,35%** | **2,65%** |

### 3.3. Como Essa Disparidade de Volume (97% vs 3%) é Conduzida nos Modelos?

1. **No Treinamento**:
   - Os 657.689 exames são embaralhados nos contêineres HDF5, permitindo que as redes neurais aprendam simultaneamente com a complexidade e volume massivo de emergência do MIMIC e a estrutura terminológica formal do PTB-XL.
   - O desbalanceamento intrínseco de classes é compensado diretamente na loss `BCEWithLogitsLoss(pos_weight=pos_weights)` pré-computada no contêiner `train.h5`.
2. **Na Avaliação de Teste (Estratégia de Benchmark Duplo)**:
   - **Teste Global Transcontinental (82.448 exames)**: Avalia a acurácia, Macro AUROC e Sensibilidade no cenário misto global de ambiente hospitalar generalizado.
   - **Teste Isolado no PTB-XL / Dobra 10 (2.198 exames)**: Avalia o modelo exclusivamente na partição de referência internacional do PTB-XL, viabilizando comparação direta e fidedigna com os modelos do estado da arte (SOTA) publicados na literatura acadêmica mundial (PhysioNet Challenge).

