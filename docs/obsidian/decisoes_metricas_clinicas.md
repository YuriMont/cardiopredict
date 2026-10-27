# Decisões de Métricas Clínicas: Por que Acurácia é uma Armadilha na Medicina

Este documento explica de forma clara por que o CardioPredict rejeita a métrica padrão de "Acurácia Global" e prioriza a **Sensibilidade (Recall) $\ge 95\%$**, **Macro AUROC** e **Macro AUPRC** (conforme estabelecido na Seção 4 do [PRD](./PRD.md)).

---

## 1. O "Paradoxo da Acurácia" no Eletrocardiograma

Imagine um pronto-socorro onde chegam 1.000 pacientes com queixa de mal-estar.
- Estatisticamente, apenas 20 desses pacientes (2%) estão sofrendo um Infarto Agudo com Supra de ST (`STEMI`).
- Os outros 980 pacientes (98%) têm outras condições (refluxo, ansiedade, dores musculares ou exames normais).

### 1.1. O Modelo "Inútil", mas com 98% de Acurácia
Se um engenheiro desavisado treinar um modelo que simplesmente responda **"Zero / Sem Infarto" para 100% dos exames**, esse modelo alcançará matematicamente:
$$\text{Acurácia} = \frac{980\text{ acertos}}{1.000\text{ exames}} = 98\%$$

Em uma apresentação corporativa leiga, "98% de acurácia" soa excelente. Na medicina de emergência real, esse sistema é um desastre assassino: **ele deixou morrer 100% dos 20 pacientes infartados sem emitir nenhum alarme**.

---

## 2. A Assimetria do Erro Médico: Falso Negativo vs. Falso Positivo

Na inteligência artificial clínica, os dois tipos de erro têm pesos morais, éticos e jurídicos completamente desiguais:

### 2.1. O Falso Negativo (Custo Catastrófico / Risco de Morte)
- **O que é:** O paciente está sofrendo um Infarto Agudo (`STEMI`), mas a IA diz que o exame está "Normal" (`0`).
- **Consequência:** O paciente é liberado da triagem do pronto-socorro para a sala de espera ou mandado para casa, evoluindo para parada cardiorrespiratória (PCR), choque cardiogênico ou óbito por omissão médica (*malpractice*).

### 2.2. O Falso Positivo (Custo Operacional Aceitável)
- **O que é:** O paciente tem apenas uma alteração benigna na repolarização, mas a IA emite um alerta de suspeita de infarto (`1`).
- **Consequência:** O médico plantonista é chamado para revisar o traçado, solicita dosagem de enzimas cardíacas no sangue (troponina) e constata que foi um alarme falso. O paciente não corre risco de vida; o único custo foi o tempo de revisão médica.

---

## 3. As Métricas Reais do CardioPredict

Para refletir a ética da medicina de emergência, o CardioPredict é avaliado pelas seguintes métricas clínicas obrigatórias:

### 3.1. Sensibilidade / Recall ($\ge 95\%$ em Condições Críticas)
$$\text{Sensibilidade} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos} + \text{Falsos Negativos}}$$
- Mede: *"De todos os pacientes que realmente estavam infartando, quantos a IA conseguiu capturar?"*
- A meta de $\ge 95\%$ garante que menos de 5 em cada 100 casos graves passem despercebidos pelo sistema.

### 3.2. Macro AUROC (*Area Under the ROC Curve*) $\ge 0,90$
- Avalia a capacidade do modelo de discriminar entre pacientes doentes e saudáveis em todos os limiares de decisão possíveis.
- O cálculo é **Macro** (calculado individualmente para cada uma das 30 patologias e depois obtida a média aritmética), impedindo que classes frequentes (como ritmo sinusal) mascarem o desempenho ruim em patologias raras.

### 3.3. Macro AUPRC (*Area Under the Precision-Recall Curve*)
- Para doenças raras com baixíssima prevalência (como Bloqueio AV de 3º Grau, presente em menos de 0,5% da população), a curva ROC pode ser artificialmente otimista.
- A curva Precision-Recall (AUPRC) foca exclusivamente nos acertos da classe positiva minoritária, sendo a métrica mais severa e realista da literatura médica moderna.
