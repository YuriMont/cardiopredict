# Decisões de Arquitetura e Engenharia de Dados: CNN 1D Multilabel (MIMIC-IV-ECG e PTB-XL)

Este documento descreve as decisões técnicas, metodológicas e clínicas adotadas para a construção de um modelo de aprendizado profundo (CNN 1D) multilabel utilizando em conjunto as bases de dados MIMIC-IV-ECG e PTB-XL, detalhando as motivações por trás de cada escolha.

---

## 1. Harmonização e Pré-Processamento dos Sinais Fisiológicos

### 1.1. Padronização da Taxa de Amostragem (500 Hz)
* **Decisão**: Fixar a frequência de amostragem universal em 500 Hz para todos os traçados (10 segundos correspondendo a 5.000 amostras por derivação). Utilizar os sinais de alta resolução da pasta `records500` da PTB-XL e a taxa nativa de 500 Hz da MIMIC-IV-ECG.
* **Motivo**: 
  * A literatura cardiológica e as diretrizes internacionais exigem amostragem de pelo menos 500 Hz para capturar com precisão detalhes morfológicos finos de alta frequência, como a morfologia do complexo QRS (entalhes, potenciais tardios e espículas de marcapasso).
  * Reduções para 100 Hz diminuem o custo computacional, mas causam perda irreversível de resolução temporal em arritmias complexas e infartos com pequenas alterações de condução.

### 1.2. Ordem Universal e Padronizada das 12 Derivações
* **Decisão**: Fixar rigidamente a ordem dos canais no tensor de entrada: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 (12 canais).
* **Motivo**: 
  * Redes neurais convolucionais 1D operam associando filtros específicos a posições específicas de canais de entrada. Caso a ordem dos eletrodos varie entre exames ou bases, os filtros convolucionais aprenderiam correlações espaciais e vetoriais incorretas.
  * Essa sequência reflete a leitura padrão dos aparelhos hospitalares e agrupa naturalmente as derivações dos membros (plano frontal) e pré-cordiais (plano horizontal).

### 1.3. Escala Física Homogênea em Milivolts (mV)
* **Decisão**: Converter previamente todos os registros para a unidade física de milivolts (mV) antes da normalização estocástica.
* **Motivo**: 
  * Os arquivos de cabeçalho WFDB (.hea) de ambas as bases podem registrar sinais com ganhos e conversores analógico-digitais (ADC) distintos, ou mesmo em microvolts ($\mu$V). 
  * Sem essa equalização inicial para uma escala física real, dois sinais idênticos gerariam amplitudes numéricas com ordens de grandeza distintas, desestabilizando o treinamento.

### 1.4. Filtragem de Ruído e Interferência de Rede Elétrica
* **Decisão**: Aplicação de filtro passa-faixa com frequência de corte inferior em 0,5 Hz e superior em 45 Hz (ou combinação de filtro passa-alta em 0,5 Hz com filtro passa-baixa em 45 Hz).
* **Motivo**:
  * **Linha de base respiratória (< 0,5 Hz)**: O movimento torácico da respiração e o suor do paciente geram desvios lentos de voltagem que distorcem as medições dos segmentos ST e TP, essenciais para detecção de isquemia. O corte em 0,5 Hz remove essas flutuações sem distorcer o segmento ST.
  * **Interferência de rede elétrica (50 Hz vs 60 Hz)**: A base PTB-XL foi coletada na Alemanha (frequência de rede de 50 Hz), enquanto a base MIMIC-IV-ECG foi coletada nos Estados Unidos (frequência de rede de 60 Hz). Um filtro com corte em 45 Hz elimina eficazmente os ruídos das duas redes sem a necessidade de múltiplos filtros Notch sintonizados separadamente.

### 1.5. Normalização Z-Score por Sinal / Derivação
* **Decisão**: Normalização de média zero e variância unitária (Z-score) calculada individualmente para cada traçado.
* **Motivo**: 
  * Diferenças anatômicas dos pacientes (como espessura da parede torácica, índice de massa corporal e condutividade da pele) criam variações globais de impedância que alteram a amplitude absoluta dos sinais, mesmo com corações sadios.
  * A normalização por exame foca o aprendizado da rede na forma da onda (morfologia relativa) e não em variações absolutas de ganho, prevenindo explosão ou esvanecimento de gradientes.

---

## 2. Harmonização da Ontologia e Espaço de Rótulos (Multilabel)

### 2.1. Formulação do Problema como Multilabel
* **Decisão**: Modelar o problema como classificação multilabel pura (cada exame possui um vetor binário de $C$ posições, onde múltiplas classes podem ser verdadeiras simultaneamente), descartando formulações multiclasse exclusivas (Softmax).
* **Motivo**: 
  * Na prática médica, a comorbidade cardiológica é a regra, não a exceção. Um único exame frequentemente apresenta ritmo anormal (ex.: fibrilação atrial), distúrbio de condução intraventricular (ex.: bloqueio de ramo direito) e alteração isquêmica (ex.: infarto antigo inferior).
  * Forçar a escolha de um único diagnóstico primário descartaria achados clínicos vitais e introduziria inconsistência de rotulagem.

### 2.2. Unificação de Ontologias (SCP-ECG vs Laudos Textuais do MIMIC)
* **Decisão**: Estabelecer um vocabulário central comum de classes clínicas (como as superclasses diagnósticas padrão ou as patologias de maior impacto clínico) e mapear ambas as bases para esse vocabulário unificado.
* **Motivo**: 
  * A PTB-XL adota códigos estruturados SCP-ECG com regras bem definidas. O MIMIC-IV-ECG fornece laudos em linguagem natural extraídos por aparelhos diagnósticos.
  * Sem um dicionário padronizado que alinhe termos do texto com as declarações SCP correspondentes, seria impossível treinar ou validar o mesmo preditor em ambas as fontes de dados.

### 2.3. Tratamento Explícito da Incerteza Diagnóstica
* **Decisão**: Isolar e filtrar prefixos de incerteza clínica (como "possível", "sugere", "suspeita de") nos laudos do MIMIC antes de gerar rótulos positivos definitivos.
* **Motivo**: 
  * Rotular achados incertos como positivos com 100% de certeza insere ruído de rótulo (label noise) de alta intensidade, penalizando a capacidade do modelo de discriminar padrões limpos. 
  * Preservar essa distinção permite descartar exames duvidosos ou tratá-los com suavização de rótulos (label smoothing).

---

## 3. Estratégia de Particionamento e Prevenção de Vazamento (Data Leakage)

### 3.1. Divisão Estrita no Nível de Paciente (Patient-Level Split)
* **Decisão**: A partição em subconjuntos de Treino, Validação e Teste deve ser realizada obrigatoriamente agrupada pelo identificador único do paciente (`subject_id` no MIMIC e `patient_id` no PTB-XL).
* **Motivo**: 
  * Um mesmo paciente internado ou em acompanhamento pode realizar múltiplos exames de ECG ao longo de dias ou semanas. Traçados do mesmo coração compartilham assinaturas morfológicas únicas (vetores elétricos intrínsecos, formato basal de onda T).
  * Se exames do mesmo indivíduo estivessem no treino e no teste, a rede aprenderia a identificar a identidade anatômica do paciente, inflando artificialmente as métricas de desempenho e mascarando uma incapacidade real de generalização para novos pacientes.

### 3.2. Estratégia de Validação Entre Centros (Cross-Center Evaluation)
* **Decisão**: Priorizar uma abordagem onde uma base (ex.: PTB-XL) é utilizada para o desenvolvimento inicial e a outra (ex.: MIMIC-IV-ECG) atua como teste externo independente (ou vice-versa), além do treino conjunto multi-hospitalar.
* **Motivo**: 
  * Em inteligência artificial clínica, modelos que atingem alto desempenho dentro de um único hospital frequentemente sofrem queda drástica de acurácia em outros centros devido a diferenças de demografia, protocolos de eletrodos e fabricantes de eletrocardiógrafos.
  * A validação cruzada entre PTB-XL (Alemanha) e MIMIC (Estados Unidos) é considerada o padrão de ouro para comprovar a robustez e generalização da rede em nível internacional.

---

## 4. Escolha da Arquitetura da Rede Neural (CNN 1D)

### 4.1. Convolução 1D Diretamente nos Sinais Brutos vs Abordagens 2D (Espectrogramas)
* **Decisão**: Utilizar redes neurais convolucionais unidimensionais (1D CNN / ResNet 1D) operando diretamente na série temporal de voltagem por derivação, em vez de transformar os sinais em imagens ou espectrogramas 2D.
* **Motivo**: 
  * Sinais de ECG contêm informações diagnósticas críticas concentradas na fase e na polaridade das ondas elétricas (por exemplo, inversão de onda T, supradesnivelamento do segmento ST em milímetros e direção do vetor elétrico). 
  * A conversão para representações de frequência ou imagens 2D descarta relações temporais finas e polaridades essenciais de voltagem, além de aumentar drasticamente o custo computacional e de memória.

### 4.2. Adoção de Conexões Residuais (Skip Connections / ResNet-1D)
* **Decisão**: Estruturar a CNN 1D com blocos residuais profundos.
* **Motivo**: 
  * Sinais longos (5.000 pontos) exigem que a rede possua um campo receptivo (receptive field) amplo o suficiente para captar eventos de baixa frequência (como bradicardia e intervalo QT de centenas de milissegundos) e, ao mesmo tempo, detalhes de alta frequência (complexo QRS de 80 milissegundos).
  * Redes convolucionais profundas sem conexões de atalho sofrem degradação de gradiente durante a retropropagação, impedindo o aprendizado simultâneo de escalas temporais curtas e longas.

### 4.3. Pooling Global Adaptativo na Camada Final
* **Decisão**: Substituir camadas totalmente conectadas densas e volumosas por uma camada de pooling médio global (`Global Average Pooling 1D`) antes da projeção final dos diagnósticos.
* **Motivo**: 
  * Reduz massivamente o número de parâmetros treináveis da rede neural, diminuindo o risco de sobreajuste (overfitting).
  * Torna o modelo invariante a deslocamentos temporais do batimento (o complexo QRS não precisa acontecer exatamente no mesmo segundo em cada gravação para ser reconhecido).

---

## 5. Função de Perda e Calibração Multilabel

### 5.1. Entropia Cruzada Binária com Logits e Pesos Positivos
* **Decisão**: Utilizar Entropia Cruzada Binária por classe (`BCEWithLogitsLoss`), associada a um fator de ponderação de classes positivas (`pos_weight`).
* **Motivo**: 
  * Como a tarefa é multilabel, cada classe é tratada como uma decisão binária de Bernoulli independente. O uso da saída em logits brutos melhora a estabilidade numérica contra overflow/underflow de ponto flutuante.
  * As bases de ECG apresentam severo desbalanceamento de classes (por exemplo, ritmo sinusal normal e alterações comuns de repolarização ocorrem aos milhares, enquanto bloqueios atrioventriculares avançados são raros). A ponderação com `pos_weight` penaliza com maior rigor os falsos negativos nas patologias graves e raras.

### 5.2. Otimização de Limiares de Decisão Específicos por Classe
* **Decisão**: Evitar a fixação cega do limiar de probabilidade em 0,5 para todas as patologias; calibrar o limiar de ativação de cada classe individualmente no conjunto de validação para maximizar o F1-Score daquela doença.
* **Motivo**: 
  * Em doenças com baixa prevalência populacional, a probabilidade prevista pela rede naturalmente tende a valores médios menores. Fixar o corte em 0,5 levaria a taxas inaceitáveis de falsos negativos.
  * Limiares dinâmicos por patologia permitem equilibrar a sensibilidade e especificidade de acordo com a gravidade clínica de cada diagnóstico.

---

## 6. Métricas de Avaliação Recomendadas

* **Macro AUROC (Área sob a Curva ROC)**: Avalia a capacidade global de ordenação e discriminação do modelo para cada classe, sendo robusta a variações de prevalência.
* **Macro AUPRC (Área sob a Curva Precisão-Recall)**: Métrica mandatória para problemas médicos desbalanceados, pois reflete o impacto real dos falsos positivos em classes raras.
* **Macro F1-Score**: Mede a média harmônica entre precisão e sensibilidade após a calibração dos limiares de decisão.

