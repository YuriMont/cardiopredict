# Decisões de Modelagem Multilabel, Taxonomia Clínica e Incerteza Diagnóstica

Este documento detalha as decisões conceituais e clínicas por trás da matriz de diagnóstico do CardioPredict (módulo [`src/processing/ontology.py`](./src/processing/ontology.py)).

---

## 1. Por que Multilabel e NUNCA Multiclasse (A Falácia do "Diagnóstico Único")

### 1.1. O que é Multiclasse (*Softmax* / Exclusão Mútua)?
Em problemas clássicos de visão computacional (ex.: classificar se uma imagem é um "gato", "cachorro" ou "carro"), as classes são mutuamente exclusivas: uma imagem não pode ser um gato e um carro ao mesmo tempo. A função matemática utilizada na camada final da rede é a `Softmax`, onde a soma das probabilidades de todas as classes é forçada a somar $100\%$.

### 1.2. O Erro Fatal de Usar Multiclasse na Medicina
No atendimento real de pronto-socorro, pacientes graves raramente possuem apenas uma doença isolada.
- Um senhor hipertenso de 72 anos pode dar entrada com:
  1. **Fibrilação Atrial** (arritmia de base);
  2. **Bloqueio de Ramo Esquerdo** (defeito elétrico crônico);
  3. **Infarto Agudo do Miocárdio Anterior** (emergência de vida ou morte em andamento).
- Se o modelo fosse multiclasse, ele seria forçado a escolher apenas um rótulo. Se ele respondesse "Fibrilação Atrial", a equipe médica poderia não receber o alerta de Infarto Agudo, levando o paciente a óbito por omissão diagnóstica (*malpractice*).
- **Nossa Decisão:** Cada patologia é avaliada de forma completamente independente por meio de neurônios de saída com ativação `Sigmoid` e função de perda `BCEWithLogitsLoss`. O exame pode ter zero, um, dois ou cinco diagnósticos ativados simultaneamente.

---

## 2. A Escolha da Taxonomia Hierárquica: 5 Superclasses + 25 Subclasses

Em vez de criar uma lista desorganizada de diagnósticos, estruturamos a decisão clínica em duas camadas interconectadas:

### 2.1. Nível 1: As 5 Grandes Superclasses Clínicas (Consórcio SCP-ECG)
Respondem à pergunta de triagem rápida: *"Qual é a categoria fisiopatológica do problema?"*
1. `NORM`: Exame normal (sem doenças).
2. `MI`: Infarto do Miocárdio (músculo cardíaco isquêmico / necrosado).
3. `STTC`: Alterações de Repolarização Ventricular e Isquemia (Segmento ST e Onda T).
4. `CD`: Distúrbios e Atrasos de Condução Elétrica (Bloqueios de Ramo e Atrioventriculares).
5. `HYP`: Sobrecarga / Hipertrofia do tecido muscular ventricular ou atrial.

### 2.2. Nível 2: As 25 Patologias Específicas de Emergência
Respondem à pergunta clínica detalhada: *"Qual é o diagnóstico exato e a conduta a tomar?"*
- **Ritmos Críticos:** Fibrilação Atrial (`AFIB`), Flutter Atrial (`AFLT`), Extrassístoles Ventriculares (`PVC`), Taquicardia e Bradicardia Sinusais.
- **Localização do Infarto:** Parede Anterior/Anterosseptal (`AMI`), Parede Inferior (`IMI`), Parede Lateral (`LMI`).
- **Alerta Vermelho de Cateterismo Imediato:** Infarto com Supradesnivelamento de ST (`STEMI`).
- **Condução e Bloqueios Graves:** Bloqueio de Ramo Esquerdo (`CLBBB`), Bloqueio de Ramo Direito (`CRBBB`), Bloqueio AV Total de 3º Grau (`3AVB`).
- **Sobrecargas:** Hipertrofia Ventricular Esquerda (`LVH`), Hipertrofia Ventricular Direita (`RVH`).

---

## 3. Decisão de Escopo Clínico de Infarto: Parede Anatômica vs. Emergência Aguda

### 3.1. A Realidade dos Laudos Médicos
Ao analisar os laudos hospitalares do MIMIC-IV-ECG, constatamos que mais de 75 mil exames descrevem infartos com localização precisa da parede afetada, porém com cronicidade indeterminada ou cicatriz prévia (ex.: *"inferior infarct - age undetermined"* ou *"probable old anterior infarct"*). Menções textuais estritas com as palavras "agudo em evolução" aparecem em menos de 10% desses casos.

### 3.2. A Decisão Adotada
- **As classes anatômicas (`AMI`, `IMI`, `LMI`):** Abrangem todos os infartos que acometem aquele território vascular específico (sejam agudos, prévios ou indeterminados). Isso permite que o cardiologista saiba exatamente qual parede do ventrículo sofreu perda de tecido miocárdico.
- **A classe dedicada `STEMI`:** Atua como o sinalizador estrito de urgência hiperaguda com supradesnivelamento do segmento ST. É a presença de `STEMI = 1` que aciona os protocolos de abertura imediata de sala de hemodinâmica (*Fast-Track IAM*).

---

## 4. O Manejo da Incerteza Diagnóstica: O Peso 0,5

### 4.1. O Problema da Linguagem Médica Humana
Médicos frequentemente utilizam termos de ponderação ao laudarem exames limítrofes:
- *"Possible inferior infarct"* (Possível infarto inferior);
- *"Borderline left ventricular hypertrophy"* (Sobrecarga ventricular esquerda limítrofe);
- *"Cannot rule out anterior ischemia"* (Não se pode descartar isquemia anterior).

### 4.2. Por que atribuir peso `0.5`?
- **Se descartássemos o exame:** Perderíamos mais de 150 mil exames reais valiosos, reduzindo drasticamente o repertório do modelo.
- **Se tratássemos como certeza absoluta (`1.0`):** O modelo seria forçado a aprender que um traçado sutil e ambíguo é um caso patente e incontestável de infarto, gerando uma taxa inaceitável de alarmes falsos em prontos-socorros.
- **Nossa Decisão:** O diagnóstico é ativado como positivo (`1`), mas na função de cálculo de erro (`BCEWithLogitsLoss`), o peso atribuído àquela patologia específica recebe `0.5`. O modelo é incentivado a reconhecer a suspeita clínica sem sofrer uma penalização desproporcional.
