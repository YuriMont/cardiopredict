# CardioPredict Server 🫀

Servidor de Inteligência Artificial e pipeline de processamento de biossinais para diagnóstico **Multilabel** em Eletrocardiogramas (ECG) de 12 derivações, alimentado pelas bases de dados clínicas **MIMIC-IV-ECG** e **PTB-XL**.

---

## 📌 Visão Geral

O **CardioPredict Server** foi projetado para atuar como o núcleo analítico de processamento e classificação de exames de eletrocardiograma de 12 derivações. Utilizando arquiteturas de aprendizado profundo (Deep Learning 1D), o sistema é capaz de identificar simultaneamente múltiplas patologias cardíacas (infartos, bloqueios de condução, hipertrofias ventriculares, arritmias e alterações isquêmicas de repolarização) em registros com taxa de amostragem padronizada de 500 Hz.

### Principais Destaques:
* **Abordagem Multilabel Realista**: Predição simultânea de múltiplos diagnósticos por exame utilizando saídas com `Sigmoid` e função de perda `BCEWithLogitsLoss` ponderada (`pos_weight`).
* **Bases de Dados Globais Unificadas**: Mais de **820 mil exames de ECG** e **180 mil pacientes** integrados das bases MIMIC-IV-ECG (EUA) e PTB-XL (Alemanha).
* **Arquitetura de Dados em DuckDB**: Metadados clínicos, laudos diagnósticos e medições fisiológicas consolidados em banco de dados colunar embutido de altíssima performance (`data/cardiopredict.duckdb`).
* **Prevenção Rígida de Vazamento de Dados (Zero Data Leakage)**: Particionamento em Treino (80%), Validação (10%) e Teste (10%) estritamente agrupado no nível do paciente.
* **Exploração Arquitetural Ampla**: Suporte a modelos ResNet-1D, Transformers 1D (Patch Attention), Mamba 1D (State Space Models) e Arquitetura Híbrida Integrada (CNN + Mamba + Transformer).

---

## 🏗️ Estrutura do Repositório

```text
├── config/                  # Arquivos de configuração do sistema
├── data/                    # Diretório local de dados (ignorado pelo git)
│   ├── cardiopredict.duckdb # Banco de dados relacional embutido com metadados e laudos
│   ├── mimic-iv-ecg/1.0/    # Arquivos brutos (.hea/.dat) e CSVs da base MIMIC
│   └── ptb-xl/              # Arquivos brutos (records500) e CSVs da base PTB-XL
├── src/                     # Código-fonte modularizado
│   ├── core/                # Núcleo de constantes, derivações e limites fisiológicos
│   │   ├── constants.py     # Definições de caminhos, STANDARD_LEADS e limites AHA
│   │   └── __init__.py
│   ├── data/                # Persistência relacional (DuckDB) e particionamento
│   │   ├── database.py      # Conexão DuckDB, esquemas relacionais e migração de CSVs
│   │   ├── reader.py        # Camada de consulta rápida e eficiente a pacientes
│   │   ├── audit.py         # Auditoria de integridade e regras fisiológicas
│   │   └── split.py         # Lógica de partição por paciente e extração multilabel
│   ├── processing/          # Processamento de biossinais e laudos médicos
│   │   ├── signal.py        # Filtros digitais (0.5-45 Hz), ganho (mV) e validação
│   │   ├── measurements.py  # Limpeza de intervalos clínicos derivados (PR, QRS, QT)
│   │   └── reports.py       # Normalização de linguagem natural e detecção de incerteza
│   ├── models/              # Arquiteturas de Deep Learning 1D (CNN, Mamba, Transformer)
│   └── visualization/       # Interfaces e gráficos
│       ├── dashboard.py     # Aplicação interativa em Streamlit com 5 abas analíticas
│       └── plot_patient.py  # Plotagem de exames específicos de 12 derivações via WFDB
├── .agents/                 # Habilidades e ferramentas especializadas dos agentes de IA
├── AGENTS.md                # Diretrizes operacionais e regras de desenvolvimento para agentes
├── PRD.md                   # Documento de Requisitos de Produto e Decisões de Negócio
├── pyproject.toml           # Configuração de dependências, Ruff e Pyright (Python 3.12 via uv)
└── uv.lock                  # Lockfile determinístico do gerenciador uv
```

---

## 🚀 Como Iniciar

### 1. Pré-Requisitos
* Python `>= 3.12`
* Gerenciador de pacotes [`uv`](https://github.com/astral-sh/uv)

### 2. Instalação do Ambiente
Clone o repositório e instale todas as dependências automaticamente com o `uv`:

```bash
# Sincroniza e cria o ambiente virtual isolado (.venv)
uv sync
```

### 3. Migração dos Dados para o DuckDB
Para consolidar os metadados dos CSVs das bases MIMIC-IV-ECG e PTB-XL no banco DuckDB:

```bash
uv run python src/database.py
```
*O processo leva ~25 segundos e cria o arquivo indexado `data/cardiopredict.duckdb`.*

### 4. Execução do Dashboard Streamlit
Para inicializar o painel interativo de análise dos traçados e distribuição diagnóstica:

```bash
uv run streamlit run src/dashboard.py
```

### 5. Inspeção de Exame com Sinais WFDB
Para visualizar graficamente as 12 derivações de um paciente específico:

```bash
uv run python src/visualise_patient.py
```

---

## 🔬 Engenharia de Sinais e Dados

1. **Taxa de Amostragem ($f_s$)**: 500 Hz padronizada (5.000 amostras por derivação em 10 segundos).
2. **Derivações**: Ordem universal padronizada: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
3. **Escala Física**: Conforme padrão WFDB convertido para milivolts (mV).
4. **Filtragem Digital**: Filtro passa-faixa (0,5 Hz a 45 Hz) para mitigação conjunta de oscilação da linha de base e eliminação de interferência das redes elétricas americana (60 Hz) e europeia (50 Hz).
5. **Documentação Completa da Esteira**: Consulte o guia detalhado em [**Pipeline de Processamento e Tratamento de Dados**](docs/obsidian/pipeline_processamento_dados.md).

---

## 🧪 Qualidade de Código e Validação Estrita

Conforme definido no [AGENTS.md](AGENTS.md), todo código deve cumprir as validações estáticas antes de ser integrado:

```bash
# Verificação de linting e boas práticas
uv run ruff check .

# Verificação de formatação de código
uv run ruff format --check .

# Verificação estática de tipos (modo standard)
uv run pyright

# Execução de todos os hooks do pre-commit
uv run pre-commit run --all-files
```

---

## 📄 Licença e Documentação Adicional
* **Pipeline de Processamento e Tratamento de Dados**: Consulte [docs/obsidian/pipeline_processamento_dados.md](docs/obsidian/pipeline_processamento_dados.md).
* **Portal de Documentação e Base de Conhecimento**: Consulte [docs/README.md](docs/README.md).
* **Decisões Estratégicas de Negócio e Produto**: Consulte o [PRD.md](PRD.md).
* **Diretrizes e Regras para Agentes**: Consulte o [AGENTS.md](AGENTS.md).
