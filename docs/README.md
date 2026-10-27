# Documentação Técnica e Base de Conhecimento - CardioPredict Server 🫀

Bem-vindo à documentação do **CardioPredict Server**. Este diretório reúne as especificações técnicas, os guias de pipeline de dados e os artigos de decisões clínicas e de engenharia biomédica do projeto.

---

## 📑 Guias Principais

* 🚀 **[Pipeline de Processamento e Tratamento de Dados](obsidian/pipeline_processamento_dados.md)**:
  * Guia técnico e arquitetural completo sobre o tratamento de dados de ECG de ponta a ponta: ingestão das bases MIMIC-IV-ECG e PTB-XL, limpeza de códigos sentinela e limites AHA no DuckDB, filtragem digital de biossinais (0.5–45 Hz, notch 50/60 Hz, calibração em mV), motor ontológico multilabel (30 classes), particionamento no nível do paciente (80/10/10), persistência física em contêineres HDF5 e integração com PyTorch.

---

## 📚 Base de Conhecimento Temática (Obsidian Vault)

A pasta [`obsidian/`](obsidian/README.md) contém notas conceituais e aprofundadas sobre cada decisão clínica, fisiológica e de engenharia do sistema:

### ⚙️ Qualidade de Sinal e Limites Fisiológicos
* ⚡ **[Limitadores de Integridade Física do Sinal](obsidian/limitadores_qualidade_sinal.md)**: Eletrodo mudo (*flatline*), tolerância máxima de NaNs e eliminação de picos espúrios.
* 🫀 **[Limitadores Fisiológicos e Códigos de Hardware](obsidian/limites_fisiologicos_medicoes.md)**: Códigos sentinela (29999, 32767, 65535) e limites humanos de FC, PR, QRS e QT.

### 👥 Seleção de Pacientes e Persistência
* 👥 **[Decisões de Seleção e Particionamento por Paciente](obsidian/decisoes_selecao_particionamento.md)**: Prevenção de *data leakage* agrupando por paciente e mitigação da "Síndrome do Hospital Único".
* ❓ **[Decisões sobre Exames Sem Diagnóstico Alvo](obsidian/decisoes_exames_sem_rotulo.md)**: Tratamento de exames inespecíficos e exclusão do treino supervisionado.
* 💾 **[Decisões de Armazenamento: HDF5 vs. 800 Mil Arquivos](obsidian/decisoes_armazenamento_hdf5_arquivos.md)**: Prevenção do colapso de inodes e aceleração do *DataLoader*.
* 📦 **[Papel do WFDB vs HDF5](obsidian/wfdb_vs_hdf5_papeis.md)**: O formato médico de origem versus o formato de alto desempenho para treinamento.

### 🩺 Fisiopatologia e Modelagem Multilabel
* 🏷️ **[Decisões de Modelagem Multilabel e Taxonomia Clínica](obsidian/decisoes_taxonomia_multilabel.md)**: Por que a medicina real exige `Sigmoid` + `BCEWithLogitsLoss` e taxonomia com 5 superclasses e 25 patologias de emergência.
* 👁️ **[Anatomia Clínica das 12 Derivações: Guia Visual](obsidian/anatomia_12_derivacoes_clinica.md)**: As 12 visões elétricas do coração e sua correlação com as paredes miocárdicas e coronárias.
* 🎯 **[Decisões de Métricas Clínicas: Por que Acurácia é uma Armadilha](obsidian/decisoes_metricas_clinicas.md)**: O paradoxo do desbalanceamento em emergências, Sensibilidade $\ge 95\%$, AUROC e AUPRC.

---

## 🛠️ Especificações Técnicas de Engenharia

* 📐 **[Design Doc do Pipeline de Dados](superpowers/specs/2026-10-26-data-processing-pipeline-design.md)**: Especificação formal dos módulos, esquemas de dados e restrições.
* 📋 **[Plano de Implementação do Pipeline](superpowers/plans/2026-10-26-data-processing-pipeline-plan.md)**: Roteiro de tarefas e fases executadas no desenvolvimento da esteira.
