# Decisões de Processamento de Sinal e Filtragem Digital de ECG

Este documento explica as decisões tomadas na engenharia de sinal e filtragem digital das 12 derivações do ECG (módulos [`src/processing/signal.py`](./src/processing/signal.py) e [`src/data/exporter_h5.py`](./src/data/exporter_h5.py)).

---

## 1. Por que Taxa de Amostragem de 500 Hz (5.000 Amostras em 10 Segundos)?

- **Origem Física:** Um exame padrão de eletrocardiografia de repouso tem duração de 10 segundos. A $500\text{ Hz}$, são capturadas 500 medições de voltagem por segundo em cada eletrodo, totalizando $5.000$ pontos por derivação.
- **Teorema de Nyquist-Shannon:** Para capturar fielmente uma componente de frequência sem distorção (efeito de *aliasing*), a taxa de amostragem deve ser de pelo menos o dobro da frequência máxima de interesse. Como as diretrizes da American Heart Association (AHA) estabelecem que o complexo QRS e as espículas de condução contêm harmônicos clínicos relevantes até 150–250 Hz, $500\text{ Hz}$ é a frequência mínima exigida para preservação morfológica fidedigna.
- **Por que não 100 Hz:** Bases que utilizam 100 Hz achatam picos de onda R rápida, transformando complexos pontiagudos em curvas arredondadas e impedindo a detecção precisa de entalhes e microvoltagens.

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

## 3. O Filtro Notch Duplo: 50 Hz e 60 Hz

### 3.1. O que é a Interferência da Rede Elétrica (*Powerline Interference*)?
A fiação elétrica das paredes dos hospitais emite um campo eletromagnético contínuo que é absorvido pelos cabos condutores do eletrocardiógrafo, gerando uma onda senoidal persistente que borra todo o traçado do ECG.

### 3.2. A Decisão do Filtro Rejeita-Faixa Duplo
- Nos Estados Unidos (onde foi coletada a base MIMIC-IV-ECG), a rede elétrica opera a **60 Hz**.
- Na Europa (onde foi coletada a base PTB-XL), a rede elétrica opera a **50 Hz**.
- **Nossa Decisão:** Em vez de assumir a origem do exame, o CardioPredict aplica um filtro Notch duplo (rejeitando estreitamente bandas ao redor de 50 Hz e 60 Hz com fator de qualidade $Q=30$). Isso garante que qualquer traçado do mundo seja limpo sem perda de informação fisiológica circundante.

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
