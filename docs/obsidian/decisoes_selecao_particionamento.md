# Decisões de Seleção e Particionamento de Dados (Prevenção de Vazamento Clínico)

Este documento explica de forma clara as decisões de negócio e de engenharia adotadas na seleção, particionamento e auditoria dos exames de ECG (módulos [`src/data/database.py`](./src/data/database.py) e [`src/data/split.py`](./src/data/split.py)).

---

## 1. A Regra de Ouro: Divisão por Paciente e NUNCA por Exame (*Zero Data Leakage*)

### 1.1. O que é o Vazamento de Dados (*Data Leakage*) no ECG?
Em ambientes hospitalares, pacientes graves realizam múltiplos eletrocardiogramas ao longo de sua internação (por exemplo, um paciente em UTI pode fazer 5 exames por dia durante uma semana, acumulando dezenas de traçados).
- Cada coração humano possui características anatômicas únicas: a orientação espacial do coração no tórax, o padrão de repolarização, a condução elétrica residual e a impedância torácica agem quase como uma **"impressão digital"**.
- Se fizéssemos uma divisão aleatória por exame (*random split*), o Exame A do Paciente X cairia no conjunto de Treino, enquanto o Exame B do mesmo Paciente X cairia no conjunto de Teste.

### 1.2. A "Ilusão dos 99% de Acurácia"
- Se houver vazamento por paciente, a rede neural profunda memoriza a "morfologia individual" daquele tórax/coração específico em vez de aprender as regras patológicas universais da doença.
- O resultado é trágico: o modelo atinge 99% de acurácia no teste laboratorial, mas ao ser implementado em um hospital real diante de um paciente novo, seu desempenho despenca para 60% (colapso de generalização).
- **Nossa Decisão:** Todo o particionamento é agrupado pelo identificador do paciente (`subject_id` no MIMIC e `patient_id` no PTB-XL). Se um paciente está no Treino, **100% dos seus exames passados, presentes e futuros estão confinados exclusivamente ao Treino**.

---

## 2. A Escolha da Proporção 80% / 10% / 10% e Composição das Bases

A divisão segue a proporção 80% / 10% / 10% agrupada estritamente por paciente. Como ambas as bases foram particionadas individualmente por essa proporção, a distribuição em cada partição é homogênea e determinística:

### 2.1. Tabela de Distribuição Oficial no DuckDB

| Partição | Exames MIMIC | Exames PTB-XL | **Total de Exames** | Pacientes MIMIC | Pacientes PTB-XL | **Total Pacientes** | % MIMIC | % PTB-XL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Treino (`train`)** | 640.271 | 17.418 | **657.689** | 129.005 | 15.023 | **144.028** | **97,35%** | **2,65%** |
| **Validação (`val`)** | 79.514 | 2.183 | **81.697** | 16.134 | 1.942 | **18.076** | **97,33%** | **2,67%** |
| **Teste (`test`)** | 80.250 | 2.198 | **82.448** | 16.213 | 1.904 | **18.117** | **97,33%** | **2,67%** |
| **Total Global** | **800.035** | **21.799** | **821.834** | **161.352** | **18.869** | **180.221** | **97,35%** | **2,65%** |

- **Conjunto de Treino (80% ~ 657 mil exames):** Garante a máxima exposição das redes convolucionais e transformers a padrões raros, morfologias atípicas de bloqueios e ruídos diversos.
- **Conjunto de Validação (10% ~ 81 mil exames):** Usado durante o ciclo de treinamento para *Early Stopping* (interromper o treino antes que o modelo decore os dados) e calibração de hiperparâmetros (taxa de aprendizado, dropout, threshold de classificação).
- **Conjunto de Teste Independente (10% ~ 82 mil exames):** Fica completamente trancado e intocado durante todo o desenvolvimento. Ele simula o "mundo real" e é avaliado apenas para auditar o desempenho final do produto para conformidade clínica e regulatória (ANVISA/FDA).

### 2.2. Por que Não é Aleatório?
- **PTB-XL**: Mapeado diretamente pela divisão oficial em 10 dobras estratificadas (`strat_fold 1..10`), preservando a dobra 10 para comparação de estado da arte (*benchmark* internacional).
- **MIMIC-IV-ECG**: Mapeado por hash determinístico do identificador único do paciente `(subject_id % 10)`: restos 0 a 7 no Treino, resto 8 na Validação e resto 9 no Teste. Zero risco de vazamento de pacientes com exames seriados.

---

## 3. Validação Bi-Hospitalar: Combate à "Síndrome do Hospital Único"

### 3.1. O Problema
Um dos maiores gargalos da Inteligência Artificial médica global é o viés institucional:
- Algoritmos treinados apenas no Hospital X aprendem o ruído elétrico da rede daquele prédio, os filtros proprietários do fabricante dos aparelhos daquele hospital e as características demográficas locais.
- Ao levar o software para outra instituição, o modelo erra com frequência inaceitável.

### 3.2. A Solução do CardioPredict
Integramos propositalmente duas das maiores bases abertas da história da cardiologia:
1. **MIMIC-IV-ECG (EUA):** Base hospitalar norte-americana de atendimento terciário e emergência intensiva (Beth Israel Deaconess Medical Center, Boston). Rede elétrica de 60 Hz, aparelhos comerciais GE Marquette e laudos em texto livre em língua inglesa.
2. **PTB-XL (Alemanha):** Base clínica europeia multicêntrica de 12 derivações (Physikalisch-Technische Bundesanstalt). Rede elétrica de 50 Hz, aparelhos Schiller e diagnósticos estruturados segundo o padrão europeu SCP-ECG.

Treinar o sistema para lidar simultaneamente com variações geográficas, marcas distintas de eletrocardiógrafos e ruídos elétricos transcontinentais confere ao CardioPredict **robustez e interoperabilidade global**.

---

## 4. O que é o `strat_fold` do PTB-XL?

- **Origem Científica:** Os criadores do consórcio PTB-XL estruturaram previamente um particionamento estratificado de 10 dobras (*10-fold cross validation*) baseado na co-ocorrência dos diagnósticos médicos.
- **Aderência ao Padrão da Literatura:**
  - As dobras `1 a 8` correspondem à partição de **Treino (80%)**.
  - A dobra `9` corresponde à **Validação (10%)**.
  - A dobra `10` é o conjunto oficial de **Teste (10%)** adotado como referência nos maiores artigos científicos do mundo (como o PhysioNet/Computing in Cardiology Challenge).
- Respeitar esse padrão permite comparar diretamente os resultados do CardioPredict com o estado da arte acadêmico internacional.

---

## 5. Por que os Exames de 100 Hz (`records100`) do PTB-XL são Ignorados no Banco?

### 5.1. A Dualidade de Resolução no PTB-XL (`filename_lr` vs `filename_hr`)
No consórcio do **PTB-XL** (PhysioNet), os autores disponibilizaram o banco de dados com duas colunas de arquivos para os mesmos exames:
- `filename_lr` (pasta `records100/`): Sinais subamostrados a **100 Hz** (1.000 amostras em 10 segundos), gerados pelos autores exclusivamente para permitir testes preliminares rápidos em máquinas com pouca memória RAM.
- `filename_hr` (pasta `records500/`): Sinais gravados na taxa nativa de **500 Hz** (5.000 amostras em 10 segundos).

> [!IMPORTANT]
> Os arquivos em `records100/` e `records500/` **pertencem exatamente aos mesmos 21.837 exames dos mesmos 18.885 pacientes**. Não se tratam de exames adicionais ou de pacientes novos.

### 5.2. O Risco de Duplicação de Dados (*Redundant Duplication*)
Se o pipeline de migração ingerisse tanto a coluna `filename_lr` quanto a `filename_hr`, haveria:
1. **Duplicação Artificial da Base**: O número de registros do PTB-XL dobraria artificialmente de 21.837 para 43.674 exames idênticos, criando viés de sobre-representação do PTB-XL em relação ao MIMIC-IV-ECG.
2. **Confusão de Identidade de Exame**: O mesmo laudo médico seria atribuído a dois traçados do mesmo momento fisiológico, distorcendo métricas de sensibilidade e especificidade.

### 5.3. A Decisão Arquitetural do CardioPredict
- No módulo de migração relacional ([`src/data/database.py`](./src/data/database.py)), vinculamos exclusivamente a coluna de alta resolução:
  ```sql
  'data/ptb-xl/' || filename_hr AS waveform_relative_path
  ```
- **Fidelidade Clínica:** A 500 Hz preservam-se os harmônicos rápidos do complexo QRS (entalhes de bloqueio de ramo e microvoltagens) que a amostragem a 100 Hz inevitavelmente achata.
- **Isomorfismo com o MIMIC-IV-ECG:** Como o MIMIC-IV-ECG opera nativamente a 500 Hz, o uso de `records500/` padroniza ambas as bases de dados na mesma taxa física nativa.
- **Papel da Reamostragem Dinâmica:** A função `resample_ecg_signal` implementada em [`src/processing/signal.py`](./src/processing/signal.py) permanece como uma blindagem: se no futuro um usuário dispuser apenas da versão leve `records100/` (ou de dados de monitores a 250 Hz), o sistema reamostrará via Fourier de forma contínua para 500 Hz, eliminando qualquer risco de assistolia artificial por *zero-padding*.
