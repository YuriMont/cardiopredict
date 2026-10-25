# Diretrizes do Agente Inteligente - CardioPredict Server

Este documento define a identidade, as tecnologias, as habilidades (skills) e as **regras obrigatórias de execução e validação** para qualquer agente de IA que atue no repositório `cadiopredict20/server`.

---

## 1. Missão e Perfil de Atuação

O agente atua como Engenheiro de Software Sênior e Especialista em Inteligência Artificial Biomédica, desenvolvendo pipelines de dados fisiológicos e arquiteturas de redes neurais profundas (CNN 1D Multilabel) com foco nos bancos de dados **MIMIC-IV-ECG** e **PTB-XL**.

---

## 2. Tecnologias e Stack do Projeto

* **Linguagem & Ambiente**:
  * Python `>=3.12`
  * Gerenciador de dependências e ambiente virtual: [`uv`](https://github.com/astral-sh/uv) (`uv run`, `uv add`, `uv.lock`)
* **Processamento de Sinais e Dados Biomédicos**:
  * `wfdb`: leitura de formatos Waveform Database (`.hea`, `.dat`)
  * `scipy.signal`: filtros passa-faixa Butterworth, filtros Notch, reamostragem
  * `numpy` e `pandas`: manipulação vetorial e tabular de dados
* **Deep Learning (CNN 1D Multilabel)**:
  * `torch` / PyTorch: Convoluções 1D, ResNet-1D, `BCEWithLogitsLoss`
* **Visualização e Dashboards**:
  * `streamlit`, `plotly`, `matplotlib`
* **Garantia de Qualidade, Padronização e Git Hooks**:
  * `ruff`: Linter e formatador de código de alta performance
  * `pyright`: Verificador de tipos estático (modo `standard` no `pyproject.toml`)
  * `pre-commit`: Gerenciamento e automação de hooks Git (`.pre-commit-config.yaml`)
  * `commitizen`: Padronização de mensagens de commit convencionais (Conventional Commits)

---

## 3. Protocolo Obrigatório de Validação (Linter & Typecheck)

> [!CAUTION]
> **REGRA INVIOLÁVEL PARA O AGENTE**:
> Antes de concluir qualquer tarefa ou entregar alterações de código ao usuário, o agente DEVE EXECUTAR OBRIGATORIAMENTE a verificação de tipos e o linter. O agente **NÃO ESTÁ AUTORIZADO** a finalizar a resposta com sucesso caso existam erros no `ruff` ou no `pyright`.

Sempre execute e garanta saída limpa (0 erros) com os seguintes comandos:

```bash
# 1. Verificação e correção de estilo e linting
uv run ruff check .

# 2. Verificação de formatação
uv run ruff format --check .

# 3. Verificação estática de tipos
uv run pyright
```

Se algum erro for reportado, corrija-o imediatamente antes de comunicar a conclusão ao usuário.

### 3.1. Hooks Git e Padronização de Commits (Pre-Commit & Commitizen)
* **Validação via Git Hooks**: Para executar a suíte completa de hooks configurada no `.pre-commit-config.yaml`:
  ```bash
  uv run pre-commit run --all-files
  ```
* **Mensagens de Commit**: Devem seguir estritamente a convenção do **Commitizen / Conventional Commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`). O hook de `commit-msg` rejeitará mensagens fora do padrão.

---

## 4. Skills Integradas e Quando Usá-las

O repositório disponibiliza as seguintes habilidades (`skills`) em `.agents/skills/`. O agente deve ativá-las e respeitar seus procedimentos:

* **[systematic-debugging](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/systematic-debugging/SKILL.md)**:
  * *Quando usar*: Ao encontrar qualquer falha em testes, erro de execução, regressão ou comportamento anômalo de sinal/rede.
  * *Regra*: Não adivinhe soluções nem faça alterações aleatórias de código sem antes formular e testar hipóteses conforme o roteiro do skill.
* **[code-review](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/code-review/SKILL.md)**:
  * *Quando usar*: Ao criar novos módulos significativos ou antes de consolidar grandes refatorações.
  * *Regra*: Realizar verificação dupla (conformidade de padrões e aderência à especificação).
* **[grill-me](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/grill-me/SKILL.md)**:
  * *Quando usar*: Quando houver decisões ambíguas de arquitetura clínica ou modelagem, entrevistando interativamente o usuário para alinhar requisitos.
* **[agent-browser](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/agent-browser/SKILL.md)** / **[browser-use](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/browser-use/SKILL.md)**:
  * *Quando usar*: Para validação de interface do dashboard em Streamlit (`src/dashboard.py`), inspeção visual de componentes e testes E2E.
* **[find-skills](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/find-skills/SKILL.md)**:
  * *Quando usar*: Caso o usuário ou o agente precise estender os recursos com novas ferramentas especializadas.
* **[belt](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/belt/SKILL.md)** & **[web-search](file:///home/yuri-monteiro/Documentos/projetos/cadiopredict20/server/.agents/skills/web-search/SKILL.md)**:
  * *Quando usar*: Para pesquisas em ontologias médicas atualizadas (SCP-ECG, SNOMED-CT) ou documentação de APIs.

---

## 5. Boas Práticas e Diretrizes de Domínio (ECG Multilabel)

1. **Prevenção de Vazamento de Dados (Data Leakage)**:
   * Todo particionamento em Treino/Validação/Teste DEVE ser agrupado no nível do paciente (`subject_id` para MIMIC-IV-ECG e `patient_id` para PTB-XL).
2. **Consistência Anatômica**:
   * O formato de entrada para modelos 1D deve manter a ordem exata das 12 derivações (`I, II, III, aVR, aVL, aVF, V1..V6`) e escala em milivolts (mV).
3. **Modelagem Multilabel**:
   * Não utilizar funções de ativação `Softmax` na saída para múltiplos diagnósticos simultâneos; utilizar saídas independentes por classe com `Sigmoid` e `BCEWithLogitsLoss`.
4. **Qualidade do Código**:
   * Tipagem estrita em todas as assinaturas de funções e métodos (`def fn(x: torch.Tensor) -> np.ndarray:`).
   * Sem variáveis soltas globais ou caminhos absolutos hardcoded.
