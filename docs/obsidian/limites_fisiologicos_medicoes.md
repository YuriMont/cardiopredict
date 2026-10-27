# Decisões de Limitadores Fisiológicos e Códigos de Hardware (Medições de ECG)

Este documento explica de forma didática as decisões tomadas na limpeza das medições numéricas automáticas do ECG (armazenadas em `ecg_measurements` e no módulo [`src/processing/measurements.py`](./src/processing/measurements.py)).

---

## 1. A Armadilha dos "Códigos Sentinela" dos Eletrocardiógrafos (29999, 32767, 65535)

- **O que são:** Em bases reais como o MIMIC-IV-ECG (com dados gerados por máquinas comerciais GE Marquette / Philips), colunas como início de onda P (`p_onset`), fim de onda P (`p_end`) ou eixos elétricos frequentemente registram números absurdos como `29999`, `32767`, `61440` ou `65535`.
- **Origem Técnica:** Na engenharia de software embarcado dos aparelhos de ECG, esses valores astronômicos são **códigos de erro de hardware** (*sentinel error codes*) gravados em registradores de 16 bits. Significam simplesmente: *"a máquina tentou encontrar a onda P, mas não conseguiu localizá-la no sinal"*.
- **Por que é uma decisão crítica de limpeza:** Se um desenvolvedor leigo tratar esses números como dados fisiológicos e calcular a média, o "início médio da onda P" saltará de 40 ms para mais de 4.700 ms (uma aberração que destruiria qualquer modelo de Machine Learning). O pipeline converte obrigatoriamente qualquer valor acima do limite biológico máximo em `NaN` (nulo ausente).

---

## 2. Limite do Intervalo R-R: 200 ms a 3.000 ms

- **Origem Fisiológica:** O intervalo R-R é o tempo decorrido entre dois picos de contração ventricular (dois batimentos cardíacos sucessivos).
  - A frequência cardíaca em batimentos por minuto (bpm) é dada pela fórmula física: $\text{FC} = \frac{60.000}{\text{R-R (em ms)}}$.
- **Limite Mínimo (200 ms $\implies$ 300 bpm):** A frequência cardíaca máxima extrema documentada em humanos (durante taquicardias ventriculares fulminantes ou flutter ventricular) não sustenta intervalos R-R inferiores a 200 ms sem colapso hemodinâmico. Valores abaixo de 200 ms representam ruído de disparo duplo do detector de picos (*double counting* de onda T gigante ou artefato de alta frequência).
- **Limite Máximo (3.000 ms $\implies$ 20 bpm):** Um intervalo superior a 3 segundos representa uma pausa sinusal crítica ou assistolia temporária. Quando ultrapassa 3.000 ms em traçados com múltiplos batimentos, decorre de falha do algoritmo da máquina em detectar complexos QRS de baixa amplitude.

---

## 3. Limite do Complexo QRS: 40 ms a 300 ms

- **Origem Fisiológica:** O complexo QRS representa a despolarização elétrica dos ventrículos (o momento em que os ventrículos se preparam para bombear o sangue para o corpo e pulmões).
  - Em um coração humano adulto saudável, a duração normal do QRS é estreita: **60 ms a 100 ms**.
  - Em pacientes graves com Bloqueio de Ramo Esquerdo (BRE), Bloqueio de Ramo Direito (BRD) ou em ritmos conduzidos por marca-passo artificial, a despolarização é lenta e o QRS se alarga, podendo atingir **120 ms a 200 ms**.
- **Por que o teto é 300 ms:** Mesmo nas cardiomiopatias dilatadas mais severas ou sob efeito de intoxicação por antiarrítmicos, um QRS não excede 300 ms. Leituras acima de 300 ms ocorrem quando a máquina confunde a onda T subsequente como parte do complexo QRS.
- **Por que o piso é 40 ms:** O miocárdio ventricular possui uma massa muscular que requer um tempo físico mínimo para condução eletroquímica. Uma despolarização inferior a 40 ms é fisicamente incompatível com a biologia humana e representa ruído espúrio de espícula de marca-passo ou estática.

---

## 4. Limite do Intervalo PR: 60 ms a 450 ms

- **Origem Fisiológica:** O intervalo PR mede o tempo desde o início da despolarização dos átrios até o início da despolarização dos ventrículos. Ele reflete o atraso fisiológico benéfico que o nó atrioventricular (nó AV) introduz para permitir que os ventrículos se encham de sangue antes de contraírem.
  - Normal: **120 ms a 200 ms**.
  - Bloqueio AV de 1º Grau (atraso na condução): **> 200 ms**, podendo atingir **300 ms a 400 ms**.
  - Síndromes de Pré-Excitação (ex.: Wolff-Parkinson-White, onde existe uma via anômala que pula o nó AV): **< 120 ms**, podendo chegar a **80 ms**.
- **Por que valores fora de [60, 450] ms são descartados:** Um PR $< 60\text{ ms}$ significa que a onda P se sobrepôs ou foi fundida com o QRS, enquanto $> 450\text{ ms}$ indica dissociação atrioventricular completa, onde o algoritmo associou arbitrariamente uma onda P de um batimento anterior a um complexo QRS independente.

---

## 5. Limite do Intervalo QT: 200 ms a 700 ms

- **Origem Fisiológica:** O intervalo QT engloba toda a atividade elétrica ventricular, desde o início da contração (QRS) até o final do relaxamento/repolarização celular (fim da onda T).
- **Por que o teto de 700 ms é vital:** A síndrome do QT longo congênita ou induzida por fármacos pode alargar o QT para faixas de risco (480 ms a 600 ms), criando instabilidade elétrica extrema propensa à arritmia fatal *Torsades de Pointes*. Valores acima de 700 ms registrados pela máquina decorrem da fusão incorreta da onda U proeminente com a onda T.
- **Por que o piso de 200 ms:** As bombas iônicas de potássio e cálcio das células cardíacas possuem tempos de recarga biofísicos mínimos. Um ventrículo não consegue se despolarizar e repolarizar em menos de 200 ms.

---

## 6. O Critério de Validação Técnica Global (`is_technically_valid`)

- **Definição:** Uma flag booleana binária (`1` para válido, `0` para inválido) calculada pelo sistema.
- **Critério:** Um exame só recebe `is_technically_valid = 1` se apresentar medições consistentes e preenchidas em todas as variáveis ventriculares fundamentais (`rr_interval`, `qrs_onset`, `qrs_end`, `t_end`, `qrs_axis`, `t_axis`, `qrs_duration` e `qt_interval`).
- **Finalidade:** Permite que filtros de telemedicina e modelos de machine learning tabular descartem instantaneamente exames com pane de leitura do eletrocardiógrafo antes de submetê-los à esteira clínica.
