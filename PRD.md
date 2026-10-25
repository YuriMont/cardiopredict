# PRD - Documento de Requisitos de Produto e Decisões de Negócio

**Produto**: CardioPredict AI Server  
**Domínio**: Inteligência Artificial Biomédica / Apoio à Decisão Clínica em Cardiologia  
**Público-Alvo**: Prontos-socorros, Unidades Coronarianas (UCO), Centros de Telemedicina e Clínicas Cardiológicas  

---

## 1. Visão Executiva e Problema de Negócio

### 1.1. O Problema
As doenças cardiovasculares são a principal causa de mortalidade global. No atendimento de emergência e na triagem clínica, o Eletrocardiograma (ECG) de 12 derivações é o exame de primeira linha para identificação de condições fatais, como o Infarto Agudo do Miocárdio (IAM) e arritmias graves. No entanto, o fluxo clínico tradicional enfrenta gargalos críticos:
* **Escassez de Especialistas em Tempo Real**: Plantões de pronto-socorro frequentemente não contam com cardiologistas presenciais 24/7 para emissão de laudo imediato.
* **Sobrecarga Cognitiva e Comorbidades**: Pacientes raramente apresentam apenas uma patologia isolada; a coexistência de arritmias, bloqueios e isquemias simultâneas retarda o diagnóstico.
* **Sistemas Proprietários Obsoletos**: Aparelhos de ECG tradicionais possuem algoritmos analíticos rígidos, com taxas elevadas de falso-positivo para infarto e baixa capacidade de adaptação a novos perfis demográficos.

### 1.2. A Proposta de Valor do CardioPredict
O CardioPredict AI Server é uma plataforma de inteligência artificial multilabel que atua como **segunda opinião clínica em tempo real**. O sistema analisa sinais brutos de ECG de 12 derivações em menos de 1 segundo, detectando simultaneamente múltiplas patologias com alta sensibilidade e fornecendo explicações morfológicas para a equipe médica.

---

## 2. Decisões Estratégicas de Negócio

### 2.1. Abordagem Multilabel como Requisito Central
* **Decisão de Negócio**: O produto é concebido desde a base como **Multilabel** (predição simultânea de múltiplos diagnósticos por exame), descartando classificadores multiclasse de diagnóstico único.
* **Justificativa de Negócio**: Na medicina real, pacientes graves sofrem de comorbidades concomitantes (ex.: um paciente com fibrilação atrial pode sofrer um infarto agudo e possuir bloqueio de ramo prévio). Forçar a escolha de uma única doença geraria omissão de diagnósticos secundários críticos e risco de erro médico (*malpractice*).

### 2.2. Treinamento e Validação Bi-Hospitalar (MIMIC-IV-ECG e PTB-XL)
* **Decisão de Negócio**: O desenvolvimento do produto utiliza obrigatoriamente duas das maiores bases clínicas mundiais: MIMIC-IV-ECG (EUA) e PTB-XL (Alemanha).
* **Justificativa de Negócio**: A inteligência artificial clínica sofre com a "síndrome de hospital único", onde algoritmos perdem precisão quando implementados em instituições com aparelhos ou populações diferentes. Treinar e validar em bases transcontinentais garante:
  * **Interoperabilidade Global**: O modelo aprende a lidar com diferentes marcas de eletrocardiógrafos e ruídos elétricos (50 Hz europeu e 60 Hz americano).
  * **Prontidão Regulatória (FDA / ANVISA / CE)**: Órgãos reguladores exigem validação multicêntrica externa para autorização de software como dispositivo médico (SaMD).

### 2.3. Transparência e Explicabilidade Clínica (XAI)
* **Decisão de Negócio**: Incorporação obrigatória de mecanismos de interpretabilidade (mapas de atenção e identificação de derivações críticas).
* **Justificativa de Negócio**: Médicos não confiam em algoritmos de "caixa-preta". O produto deve sinalizar visualmente qual batimento e qual derivação motivaram o alerta para assegurar adoção clínica e conformidade ética.

### 2.4. Armazenamento Eficiente e Escalabilidade (Migração para DuckDB)
* **Decisão de Negócio**: Metadados clínicos, históricos de pacientes e laudos estruturados são unificados em banco relacional colunar embutido (DuckDB), enquanto sinais físicos permanecem em arquivos padronizados.
* **Justificativa de Negócio**: Elimina custos recorrentes de infraestrutura com bancos de dados dedicados no estágio de desenvolvimento, garantindo resposta analítica instantânea para auditorias e telelaudos em alta escala.

---

## 3. Personas e Casos de Uso

### 3.1. Personas
1. **Médico Plantonista / Emergencista**: Precisa de triagem imediata no protocolo de "dor torácica" para acionar o laboratório de hemodinâmica (cateterismo) em casos de IAM com supra de ST.
2. **Central de Telemedicina**: Processa milhares de exames diários de cidades do interior e necessita de priorização inteligente na fila de laudos.
3. **Cardiologista Revisor**: Utiliza o sistema como copiloto para revisão rápida, reduzindo o tempo de emissão de laudo por exame.

### 3.2. Principais Casos de Uso
* **Caso de Uso 1 - Alerta Rápido de Infarto (Fast-Track IAM)**: Triagem prioritária de exames suspeitos em menos de 10 segundos após a realização do ECG.
* **Caso de Uso 2 - Fila de Laudos Automatizada e Priorizada**: Reordenação dinâmica de exames em centrais de telemedicina por grau de gravidade clínica.
* **Caso de Uso 3 - Auditoria de Qualidade Hospitalar**: Identificação retroativa de discrepâncias diagnósticas e monitoramento de tempos porta-eletrocardiograma.

---

## 4. Indicadores-Chave de Desempenho (KPIs de Negócio e Clínicos)

### 4.1. Métricas de Eficácia Clínica
* **Sensibilidade em Condições Críticas (Infarto / Arritmias Graves)**: $\ge 95\%$ (minimizar falsos negativos que coloquem vidas em risco).
* **Macro AUROC Global**: $\ge 0.90$ no conjunto de validação externa independente.
* **Macro AUPRC**: Superior às linhas de base comerciais do setor em patologias com baixa prevalência.

### 4.2. Métricas de Eficiência Operacional
* **Tempo Porta-Laudo**: Redução de $40\%$ no tempo de emissão de parecer prévio em prontos-atendimentos.
* **Latência de Inferência**: Inferência de ponta a ponta em menos de $100\text{ ms}$ por exame de 10 segundos.
* **Disponibilidade Operacional**: Arquitetura resiliente e local, operando sem dependência obrigatória de conexão externa à internet.

---

## 5. Roteiro de Entregas do Produto (Roadmap)

```
┌─────────────────────────────────┐
│ FASE 1: FUNDAÇÃO DE DADOS       │  Consolidação das bases MIMIC-IV-ECG e PTB-XL em DuckDB,
│ (Concluída)                     │  prevenção de data leakage e normalização multilabel.
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ FASE 2: MOTOR DE DEEP LEARNING  │  Treinamento e benchmark comparativo entre ResNet-1D,
│ (Em Andamento)                  │  Transformer-1D, Mamba-1D e Arquitetura Híbrida Integrada.
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ FASE 3: PILOTO & TELEMEDICINA   │  Integração do motor de inferência via API REST/gRPC
│ (Próxima Fase)                  │  com dashboard de telelaudo e suporte a visualização clínica.
└─────────────────────────────────┘
```
