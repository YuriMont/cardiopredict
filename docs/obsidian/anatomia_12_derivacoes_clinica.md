# Anatomia Clínica das 12 Derivações: Guia Visual para Desenvolvedores e Leigos

Este documento explica de forma intuitiva como um exame de Eletrocardiograma de 12 derivações funciona na prática médica e como o modelo de Deep Learning aprende a "enxergar" o coração em três dimensões.

---

## 1. O Mito dos "12 Eletrodos": 10 Eletrodos que Criam 12 Visões

Uma das confusões mais comuns de pessoas leigas é imaginar que um ECG de 12 derivações utiliza 12 fios colados ao corpo do paciente.
- Na realidade, o técnico de enfermagem aplica apenas **10 eletrodos físicos**:
  - **4 Eletrodos Periféricos:** Colocados nos quatro membros (braço direito, braço esquerdo, perna direita e perna esquerda).
  - **6 Eletrodos Precordiais:** Colocados sobre o tórax do paciente, ao redor do coração (posições anatômicas de V1 a V6).

### 1.1. O que é uma "Derivação"?
Uma **derivação** não é um fio elétrico, mas sim uma **câmera fotográfica imaginária**. Cada derivação registra a diferença de potencial elétrico entre pontos anatômicos, capturando a mesma atividade elétrica do coração a partir de um ângulo espacial diferente.

---

## 2. O Mapeamento das 12 "Câmeras" e as Paredes do Coração

O coração humano é uma bomba muscular tridimensional alimentada por três grandes artérias coronárias. Quando uma dessas artérias entope (Infarto), apenas a parede suprida por ela sofre isquemia. É comparando as 12 derivações que os cardiologistas (e as redes neurais) localizam exatamente onde está o problema:

| Derivações | Paredes do Coração Monitoradas | Artéria Coronária Suspeita | Diagnóstico Clínico no CardioPredict |
| :--- | :--- | :--- | :--- |
| **DII, DIII, aVF** | **Parede Inferior** (base do coração, apoiada sobre o diafragma) | Artéria Coronária Direita (CD) | `IMI` (Infarto Inferior) / `STTC_ISCI` |
| **V1, V2** | **Parede Septal** (septo interventricular que divide os dois ventrículos) | Artéria Descendente Anterior (DA) - ramos septais | `AMI` (Infarto Anterior/Septal) |
| **V3, V4** | **Parede Anterior** (frente do ventrículo esquerdo) | Artéria Descendente Anterior (DA) | `AMI` (Infarto Anterior) / `STEMI` |
| **DI, aVL, V5, V6** | **Parede Lateral** (lado esquerdo superior e inferior) | Artéria Circunflexa (CX) ou Ramos Diagonais | `LMI` (Infarto Lateral) |
| **aVR** | **Cavidade Interna / Ventrículo Direito** | Tronco da Coronária Esquerda (TCE) se houver supra | Indicador de gravidade extrema |

---

## 3. Por que a Ordem dos Canais Deve ser Rigorosamente Padronizada?

No módulo [`src/core/constants.py`](src/core/constants.py), definimos uma lista imutável:
```python
STANDARD_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
```

### 3.1. O Conceito Clínico de "Derivações Contíguas"
Na medicina diagnóstica de infarto, uma alteração em apenas uma derivação isolada pode ser apenas ruído ou artefato de eletrodo. Os protocolos médicos internacionais (como o protocolo de dor torácica da AHA) exigem que a elevação do segmento ST esteja presente em **pelo menos duas derivações contíguas** (que enxergam a mesma parede, como V1 e V2, ou DII e DIII).

### 3.2. Como a Convolução 1D Aprende essa Relação
Ao padronizar os canais de entrada da rede neural como uma matriz `(12, 5000)`, os filtros convolucionais 1D e blocos residuais do modelo aprendem a comparar os padrões de vizinhança elétrica:
- O modelo aprende que se `V2` e `V3` apresentarem supradesnivelamento simultâneo com onda T hiperaguda, trata-se de uma oclusão crítica da artéria descendente anterior, disparando as classes `AMI = 1` e `STEMI = 1`.
- Se as derivações entrassem embaralhadas, essa correlação espacial seria perdida.
