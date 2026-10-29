# Decisões de Processamento de Sinal e Filtragem Digital de ECG

Este documento explica as decisões tomadas na engenharia de sinal e filtragem digital das 12 derivações do ECG (módulos [`src/processing/signal.py`](./src/processing/signal.py) e [`src/data/exporter_h5.py`](./src/data/exporter_h5.py)).

---

## 1. Por que Taxa de Amostragem de 500 Hz (5.000 Amostras em 10 Segundos)?

- **Origem Física:** Um exame padrão de eletrocardiografia de repouso tem duração de 10 segundos. A $500\text{ Hz}$, são capturadas 500 medições de voltagem por segundo em cada eletrodo, totalizando $5.000$ pontos por derivação.
- **Teorema de Nyquist-Shannon:** Para capturar fielmente uma componente de frequência sem distorção (efeito de *aliasing*), a taxa de amostragem deve ser de pelo menos o dobro da frequência máxima de interesse. Como as diretrizes da American Heart Association (AHA) estabelecem que o complexo QRS e as espículas de condução contêm harmônicos clínicos relevantes até 150–250 Hz, $500\text{ Hz}$ é a frequência mínima exigida para preservação morfológica fidedigna.
- **Por que não 100 Hz:** Bases que utilizam 100 Hz achatam picos de onda R rápida, transformando complexos pontiagudos em curvas arredondadas e impedindo a detecção precisa de entalhes e microvoltagens.

### 1.1. Tratamento do PTB-XL: Por que os Arquivos de 100 Hz (`records100`) são Ignorados?
- No banco de dados PTB-XL, existem duas colunas de sinal para os mesmos 21.837 exames: `filename_lr` (pasta `records100/` a 100 Hz) e `filename_hr` (pasta `records500/` a 500 Hz).
- Não são pacientes ou exames novos, mas sim **versões subamostradas do mesmo evento clínico**.
- Se ambas as pastas fossem importadas, os 21.837 exames seriam duplicados, causando redundância e enviesando o treino.
- Portanto, o banco DuckDB vincula compulsoriamente os arquivos de 500 Hz (`filename_hr`), padronizando com a frequência nativa do MIMIC-IV-ECG (500 Hz).
- Caso dados externos ou versões de 100 Hz sejam fornecidos pontualmente, a função `resample_ecg_signal` reamostra dinamicamente via Fourier para 500 Hz.

---

## 2. A Escolha do Filtro Passa-Faixa: 0,5 Hz a 45 Hz

Um sinal de ECG bruto captado na pele do paciente vem contaminado por múltiplos ruídos do ambiente hospitalar:

### 2.1. O Corte Inferior em 0,5 Hz (Filtro Passa-Alta)
- **O que elimina:** A oscilação lenta da linha de base (*baseline wander*), causada pelos movimentos respiratórios do tórax do paciente e transpiração na pele.
- **Por que não cortar acima de 0,5 Hz (ex.: 1,0 Hz ou 2,0 Hz):** O segmento ST — elemento crucial para diagnosticar Infarto Agudo do Miocárdio — possui energia em frequências muito baixas (em torno de 0,1 a 0,5 Hz). Um filtro agressivo demais distorceria a altura do segmento ST, criando "falsos infartos" ou apagando supradesnivelamentos reais. 0,5 Hz é o equilíbrio ideal entre estabilidade de linha de base e preservação isquêmica.

### 2.2. O Corte Superior em 45 Hz (Filtro Passa-Baixa)
- **O que elimina:** O ruído eletromiográfico (EMG), gerado pela contração dos músculos esqueléticos (tremores do paciente por frio, dor ou ansiedade).
- **Por que 45 Hz:** A atividade elétrica do músculo esquelético se concentra predominantemente acima de 40–50 Hz. Cortar em 45 Hz "limpa" a espessura do traçado sem apagar a ponta afiada das ondas R do coração.

---

## 3. Eliminação da Interferência Transcontinental: O Filtro Notch Duplo (50 Hz e 60 Hz)

### 3.1. O Fenômeno Físico da Interferência da Rede Elétrica (*Powerline Interference*)
A corrente alternada (AC) que alimenta as tomadas, aparelhos hospitalares e a iluminação emite um campo eletromagnético oscilatório permanente.
- O corpo humano do paciente e os fios condutores dos eletrodos agem como **antenas receptoras** por acoplamento capacitivo e indutivo.
- Esse campo induz no sinal de biopotencial cardíaco uma onda senoidal parasita contínua de alta amplitude que borra a espessura da linha do ECG e degrada a detecção precisa de ondas $P$ de baixa voltagem.
- **Divergência Geográfica Fundamental:**
  - Nos **Estados Unidos** (origem do **MIMIC-IV-ECG**), a rede elétrica opera a **60 Hz**.
  - Na **Europa** (origem do **PTB-XL**), a rede elétrica opera a **50 Hz**.

### 3.2. O Desafio de Machine Learning: Prevenção de *Shortcut Learning* (Efeito Clever Hans)
Em redes neurais convolucionais profundas (CNN 1D) e arquiteturas baseadas em Atenção (Transformers 1D), a presença de ruídos residuais correlacionados à fonte de dados gera um vício de aprendizado catastrófico:
- **O Atalho Ilusório (*Shortcut*):** Se o modelo detectar resquícios da componente harmônica de 60 Hz no sinal, ele deduz imediatamente que o exame provém do hospital norte-americano (Beth Israel), onde a taxa basal de pacientes críticos de UTI coronariana é muito alta. Se detectar 50 Hz, deduz que o exame provém do consórcio alemão, com predomínio de triagens ambulatoriais.
- O otimizador de gradiente descendente tende a minimizar a perda utilizando a frequência da tomada como atalho espúrio para estimar a probabilidade pré-teste da patologia, **em vez de aprender a ler os complexos $P\text{-}QRS\text{-}T$ e os desníveis do segmento $ST$**.
- Ao levar esse modelo para um hospital no Brasil ou no Japão (onde há regiões convivendo com 50 Hz e 60 Hz), o algoritmo falha gravemente porque a correlação espúria desaparece.

### 3.3. Como Tratamos: O Filtro Notch Duplo Comutativo com $Q=30$ e Fase Zero

Para neutralizar completamente a assinatura geográfica dos dados antes que os tensores entrem nas redes neurais, implementamos em [`src/processing/signal.py`](./src/processing/signal.py):

#### 1. Fator de Qualidade Elevado ($Q=30$) para Preservação Fisiológica
O filtro Notch (rejeita-faixa estreito) é parametrizado pelo fator de qualidade $Q$:
$$Q = \frac{f_0}{\Delta f} \implies \Delta f = \frac{f_0}{Q}$$
- Para $f_0 = 50\text{ Hz}$: $\Delta f = \frac{50}{30} \approx 1,67\text{ Hz}$ (rejeita com altíssima seletividade apenas entre 49,16 Hz e 50,84 Hz).
- Para $f_0 = 60\text{ Hz}$: $\Delta f = \frac{60}{30} = 2,0\text{ Hz}$ (rejeita apenas entre 59,0 Hz e 61,0 Hz).
- **Vantagem Clínica:** Mais de 99% do espectro de frequências fisiológicas adjacentes permanece completamente intacto, eliminando o ruído da tomada sem arredondar as espículas pontiagudas do complexo QRS.

#### 2. Propriedade Comutativa dos Filtros Lineares (LTI)
Os filtros aplicados são Lineares e Invariantes no Tempo (LTI). Em representação no domínio $z$, a função de transferência total em cascata é comutativa:
$$H_{\text{total}}(z) = H_{50}(z) \cdot H_{60}(z) = H_{60}(z) \cdot H_{50}(z)$$
O pipeline não precisa adivinhar se o exame veio de Boston, Berlim ou São Paulo: ambos os filtros são aplicados compulsoriamente a todos os traçados, garantindo que qualquer assinatura elétrica transcontinental seja uniformemente erradicada.

#### 3. Filtragem Bidirecional de Fase Zero (`sosfiltfilt`)
Filtros recursivos tradicionais (IIR) provocam distorções de fase dependentes da frequência, o que deslocaria temporalmente as ondas do ECG (podendo simular falsos bloqueios atrioventriculares por alargamento artificial do intervalo PR).
- O uso de representação em Seções de Segunda Ordem (SOS) com `sosfiltfilt` aplica o filtro na sequência temporal direta e em seguida na sequência reversa:
  $$\theta_{\text{total}}(\omega) = \theta(\omega) - \theta(\omega) = 0$$
- O cancelamento matemático exato do atraso de fase preserva a geometria, a amplitude e a sincronia milimétrica de todas as 12 derivações.

#### 4. Ação Conjunta com o Passa-Faixa Butterworth (0,5 a 45 Hz)
O filtro passa-baixa de 45 Hz e o filtro Notch duplo trabalham em sinergia complementar: o passa-baixa atenua a maior parte da energia de alta frequência acima de 45 Hz, enquanto o Notch duplo elimina qualquer resíduo pontual de altíssima energia exatamente em 50 Hz e 60 Hz que pudesse passar pela faixa de transição do filtro Butterworth.

---

## 4. Consistência Anatômica Universal: A Ordem das 12 Derivações

- **A Sequência Padrão:** `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
- **Por que a ordem importa para a Rede Neural:**
  - Redes Neurais Convolucionais 1D operam analisando canais paralelos ou convoluções espaciais transversais.
  - Se um arquivo de entrada fornecer as derivações como `[V1, I, II, aVR...]` e outro como `[I, II, III...]`, a rede neural interpretará a atividade precordial anterior como se fosse a atividade do membro superior, gerando confusão diagnóstica completa.
  - O sistema reordena compulsoriamente os canais de qualquer fabricante para a sequência canônica dos 12 eixos antes de submetê-los aos modelos.

---

## 5. Padronização de Escala Física para Milivolts (mV)

- **O Problema dos Conversores A/D:** Diferentes fabricantes de hardware digitalizam o sinal com resoluções distintas. Um aparelho grava em microvolts ($\mu\text{V}$, onde 1 mV = 1.000 unidades inteiras), enquanto outro armazena em LSB (*least significant bit* com ganho de 200 unidades por mV).
- **Nossa Decisão:** O pipeline inspeciona os metadados do cabeçalho do exame (`record.units`) e multiplica os fatores de escala para que a matriz final armazenada em HDF5 esteja **rigorosamente calibrada em milivolts físicos (mV)**. Um complexo QRS de 1,5 mV terá o valor numérico exato `1.50` no tensor de entrada da rede.

---

## 6. Reamostragem Fisiológica Dinâmica (Resampling via Fourier)

### 6.1. O Problema das Taxas Heterogêneas entre Bancos de Dados
- Na literatura biomédica, exames de ECG são arquivados em taxas de amostragem distintas: o **PTB-XL** disponibiliza versões em $100\text{ Hz}$ e $500\text{ Hz}$, monitores de UTI costumam operar a $250\text{ Hz}$, e eletrocardiógrafos de alta resolução a $1.000\text{ Hz}$.
- **O risco do preenchimento ingênuo (*Zero-Padding* e Truncamento):**
  - Se um sinal de $100\text{ Hz}$ (1.000 amostras em 10 segundos) for completado com zeros para atingir 5.000 pontos, são injetados **8 segundos de linha reta (assistolia artificial)**, destruindo a frequência cardíaca média e a coerência rítmica.
  - Se um sinal de $1.000\text{ Hz}$ (10.000 amostras) for apenas truncado em 5.000 amostras, o modelo receberá apenas **os primeiros 5 segundos** do exame, descartando metade da observação clínica.

### 6.2. A Decisão da Reamostragem Dinâmica
- O pipeline detecta `orig_fs` no cabeçalho do exame e executa `resample_ecg_signal` (baseado em interpolação trigonométrica de Fourier via `scipy.signal.resample`).
- Isso mapeia de forma contínua e suave o sinal para **exatamente 5.000 pontos a 500 Hz**, preservando a integral de energia e as relações temporais entre os complexos $P\text{-}QRS\text{-}T$ independentemente da frequência original do hardware.
- A reamostragem é executada **antes** da filtragem digital, garantindo que o filtro Notch de 50/60 Hz e o passa-baixa de 45 Hz operem sempre dentro da zona segura de Nyquist ($f_{\text{Nyquist}} = 250\text{ Hz}$).

---

## 7. Integridade de Sinal e Blindagem contra Infinitos (`~np.isfinite`)

### 7.1. O Fenômeno da Saturação do Amplificador Analógico
- Durante a aquisição em ambiente clínico, eletrodos soltos ou descolamento transitório geram sobrecargas de tensão que saturam os conversores A/D nos limites extremos da escala binária, produzindo valores infinitos (`+inf`, `-inf`) ou `NaN`.
- Filtros digitais recursivos IIR (como o Butterworth) operam via realimentação: a presença de um único ponto infinito nos coeficientes de entrada corrompe irreversivelmente toda a cadeia de convolução subsequente, transformando todo o traçado em um vetor nulo de `NaN`s.

### 7.2. A Decisão de Blindagem Pré-Filtro
- Em vez de checar apenas `np.isnan`, a função `interpolate_missing_and_validate` inspeciona a máscara universal `~np.isfinite(x)`.
- Se a fração não finita for menor ou igual a $5\%$ ($250$ pontos em 10 segundos), o trecho é interpolado linearmente a partir dos pontos válidos vizinhos.
- Se ultrapassar $5\%$ ou se o desvio padrão da derivação for inferior ao limiar de silêncio elétrico ($< 10^{-4}\text{ mV}$), o exame é **imediatamente rejeitado e auditado como inválido**.

---

## 8. Normalização Z-Score por Derivação no PyTorch Dataset

### 8.1. A Disparidade de Amplitude Anatômica entre Eixos
- No corpo humano, as derivações precordiais ($V_1$ a $V_6$) estão muito próximas do miocárdio ventricular, registrando potenciais elétricos que chegam a $2,5\text{ mV}$ ou mais. Em contraste, derivações de membros distais como $aVR$ e $aVL$ costumam registrar amplitudes reduzidas ($0,3\text{ a }0,6\text{ mV}$).
- Enquanto o contêiner HDF5 armazena os valores em **milivolts físicos absolutos** (garantindo calibração médica), redes neurais profundas (especialmente blocos de Atenção do Transformer e operadores contínuos do Mamba SSM) sofrem instabilidade numérica de gradientes quando alimentadas com canais em escalas muito díspares.

### 8.2. A Decisão de Padronização no `CardioECGDataset`
- Implementou-se no [`CardioECGDataset`](./src/data/dataset.py) o parâmetro `standardize=True` com modo `per_lead` (padrão) e `global`.
- No modo `per_lead`, cada derivação $c \in \{1..12\}$ é centralizada e escalonada independentemente no eixo temporal:
  $$\hat{x}_{c, t} = \frac{x_{c, t} - \mu_c}{\sigma_c + 10^{-6}}$$
- Isso equaliza o fluxo de gradientes entre todas as derivações, mantendo a geometria local e morfologia de ondas intactas.

---

## 9. Mitigação do Desbalanceamento Multilabel (`pos_weights`)

### 9.1. O Desafio Epidemiológico
- Diagnósticos rotineiros como Ritmo Sinusal (`SR`) e ECG Normal (`NORM`) abrangem mais de $40\text{--}50\%$ do banco de dados, enquanto emergências clínicas severas como Bloqueio Atrioventricular de 3º Grau (`3AVB`) e Flutter Atrial (`AFLT`) representam menos de $1\%$ dos casos.
- Treinar uma função `BCEWithLogitsLoss` clássica sem ponderação induz o modelo a predizer sempre zero para as classes raras, obtendo alta acurácia global artificial mas acurácia diagnóstica nula nas emergências.

### 9.2. A Decisão de Pré-Cálculo de Pesos no HDF5
- O módulo [`src/data/exporter_h5.py`](./src/data/exporter_h5.py) calcula a razão de balanceamento para cada uma das 30 classes na partição:
  $$\text{pos\_weight}_c = \frac{N_{\text{neg}, c}}{\max(N_{\text{pos}, c}, 1)}$$
- Esse vetor é gravado como metadado direto no cabeçalho do arquivo HDF5 (`f.attrs["pos_weights"]`) e disponibilizado na propriedade `dataset.pos_weights`, permitindo que os scripts de treino inicializem instantaneamente a função de perda ponderada sem custo de varredura prévia do disco.
