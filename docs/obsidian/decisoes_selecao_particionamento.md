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

## 2. A Escolha da Proporção 80% / 10% / 10% (Treino, Validação e Teste)

- **Conjunto de Treino (80% ~ 657 mil exames):** Garante a máxima exposição das redes convolucionais e transformers a padrões raros, morfologias atípicas de bloqueios e ruídos diversos.
- **Conjunto de Validação (10% ~ 81 mil exames):** Usado durante o ciclo de treinamento para *Early Stopping* (interromper o treino antes que o modelo decore os dados) e calibração de hiperparâmetros (taxa de aprendizado, dropout, threshold de classificação).
- **Conjunto de Teste Independente (10% ~ 82 mil exames):** Fica completamente trancado e intocado durante todo o desenvolvimento. Ele simula o "mundo real" e é avaliado apenas para auditar o desempenho final do produto para conformidade clínica e regulatória (ANVISA/FDA).

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
