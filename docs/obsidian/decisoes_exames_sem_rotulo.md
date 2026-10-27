# Decisões sobre Exames Sem Diagnóstico Alvo: O que é `is_supervised_target`?

Este documento explica como o CardioPredict trata exames cujos laudos médicos não mencionam nenhuma das 30 condições monitoradas, evitando a contaminação do aprendizado por falsos negativos silenciosos (módulo [`src/data/labels.py`](src/data/labels.py)).

---

## 1. O Problema dos Laudos Inespecíficos em Bases Clínicas Reais

Na base MIMIC-IV-ECG, existem milhares de exames cujos laudos automáticos contêm apenas observações técnicas ou frases inconclusivas, por exemplo:
- *"Lead(s) unsuitable for analysis"* (Derivação imprópria para análise);
- *"Pacemaker rhythm - no further analysis"* (Ritmo de marca-passo artificial sem laudo adicional);
- *"Poor R wave progression - probable normal variant"* (Progressão lenta de R - provável variante da normalidade);
- *"Technically difficult recording"* (Registro tecnicamente difícil).

Nesses exames, o médico ou o software do aparelho não atestou que o paciente tem um infarto, mas **também não atestou que o exame é estritamente normal**.

---

## 2. A Armadilha do "Vetor de Zeros" (`[0, 0, ..., 0]`)

O que aconteceria se treinássemos a rede neural atribuindo `0` para todas as 30 patologias nesses exames inconclusivos?
- O modelo receberia um traçado com ritmo de marca-passo ou com eletrodo solto e aprenderia que aquilo representa a **ausência categórica** de infarto, ausência de bloqueios e ausência de sobrecargas.
- Isso injeta "gradientes mentirosos" no treinamento: o modelo é punido se detectar qualquer alteração elétrica legítima naquele exame, aprendendo a ignorar anomalias sutis.

---

## 3. Por que NÃO Considerar esses Exames como "Normais" (`NORM = 1`)?

- Um exame só recebe `NORM = 1` se o laudo médico contiver uma declaração explícita de normalidade (como *"Normal ECG"*, *"Within normal limits"* ou *"Normal ECG except for rate"*) e **nenhuma outra patologia ativa**.
- Um paciente com marca-passo elétrico implantado no ventrículo não possui um eletrocardiograma normal; forçar `NORM = 1` degradaria a confiabilidade clínica do rótulo de normalidade.

---

## 4. A Solução Adotada: A Flag `is_supervised_target`

Na tabela [`ecg_processed_labels`](./src/data/labels.py), adicionamos uma coluna booleana:
```sql
is_supervised_target BOOLEAN NOT NULL DEFAULT TRUE
```

### 4.1. Como Funciona a Regra:
1. **`is_supervised_target = TRUE` (794.567 exames ~ 96,7% do banco):**
   * O exame possui ao menos um diagnóstico afirmativo entre as 30 classes monitoradas (seja uma patologia como Fibrilação Atrial, IAM ou um atestado formal de normalidade `NORM`).
   * **Esses são os exames exportados para os arquivos HDF5 e usados no treinamento supervisionado.**
2. **`is_supervised_target = FALSE` (27.266 exames ~ 3,3% do banco):**
   * O exame não tem nenhuma das 30 condições ativadas (laudo técnico, inconclusivo ou ruído puro).
   * **Decisão:** Ele permanece preservado no DuckDB para fins de rastreabilidade, telemedicina e auditoria histórica hospitalar, mas é **excluído da matriz de treino das redes neurais**.

Essa decisão garante que **100% dos dados que entram na função de perda da rede neural possuam supervisão médica confiável e verificada**.
