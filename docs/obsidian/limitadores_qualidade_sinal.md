# Limitadores de Integridade Física do Sinal (Flatline, NaNs e Picos)

Este documento explica os critérios de descarte e tolerância para garantir a integridade biofísica dos sinais de ECG brutos (módulo [`src/processing/signal.py`](./src/processing/signal.py)).

---

## 1. Eletrodo Mudo / Flatline (Desvio Padrão < 10⁻⁴ mV = 0,1 μV)

- **Origem Física:** Em um ECG humano real, mesmo em derivações com baixa voltagem ou em repouso absoluto, a atividade miocárdica e o ruído térmico natural da interface eletrodo-pele produzem oscilações na faixa de 10 a 50 μV. A amplitude normal do QRS varia entre 0,5 e 3,0 mV.

- **O que significa < 0,1 μV:** Indica matematicamente uma linha reta absoluta (*flatline digital*). Isso acontece quando o eletrodo se soltou do tórax do paciente, o cabo foi rompido ou o conversor analógico-digital (ADC) travou naquele canal.

- **Impacto na Rede Neural:** Convoluções 1D aplicadas sobre uma derivação completamente muda geram ativações nulas. Se a rede aprender a classificar com base em derivações fantasmas/mortas, ela falhará criticamente em produção ao tentar diagnosticar derivações interdependentes (por exemplo, analisar infarto anterior sem as precordiais V1–V4).

---

## 2. Tolerância Máxima de NaNs (≤ 5% das amostras)

- **Origem no Sinal:** Cada derivação possui exatamente 5.000 amostras (10 segundos × 500 Hz).

  - 5% representa 250 pontos, o que equivale a 0,5 segundo contínuo (aproximadamente a duração de um ciclo cardíaco inteiro de um batimento normal).

- **Por que esse limite:**

  - Para falhas pequenas (< 5%, perdas de 10 a 50 ms por instabilidade de transmissão), a interpolação linear reconstitui a curva sem distorcer o traçado.

  - Se interpolássemos trechos superiores a 0,5 segundo, estaríamos "inventando" matematicamente complexos QRS e ondas T que não foram captados, gerando falsos diagnósticos. Por isso, perdas > 5% exigem o descarte ético do exame por integridade insuficiente.

---

## 3. Amplitudes Físicas Aberrantes (> 15 mV)

- **Origem Fisiológica (Normas AHA / IEC 60601-2-25):**

  - A faixa diagnóstica humana máxima de voltagem do ECG é de -5 mV a +5 mV.

  - Mesmo em casos extremos de Hipertrofia Ventricular Esquerda severa (critérios de Sokolow-Lyon com ondas S gigantes em V1 e R gigantes em V5/V6), a amplitude do QRS raramente ultrapassa 5 a 6 mV.

- **O que é > 15 mV:** Qualquer leitura que ultrapasse 15 mV — especialmente após a filtragem passa-faixa (0,5–45 Hz) já ter eliminado a oscilação da linha de base — é um artefato não biológico violento: choque estático, desconexão brusca de polo, disparo de bisturi elétrico ou eletrochoque.

- **Impacto na Rede Neural:** Manter picos espúrios de 15 a 100 mV desestabiliza a normalização por lote (*batch normalization*) e causa explosão de gradientes (*gradient explosion*) durante o treinamento, degradando o aprendizado das características morfológicas reais.