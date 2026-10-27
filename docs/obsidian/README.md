# Base de Conhecimento e Decisões de Projeto - CardioPredict (Obsidian Vault)

Este diretório reúne as explicações aprofundadas, conceituais e acessíveis sobre todas as decisões de engenharia biomédica, critérios de corte (limitadores), modelagem estatística e estratégias de seleção de dados adotadas no projeto **CardioPredict Server**.

---

## 📚 Índice Temático da Base de Conhecimento

### 🚀 Visão Geral e Arquitetura da Esteira
* 🫀 **[Pipeline de Processamento e Tratamento de Dados](pipeline_processamento_dados.md)**:
  * Guia técnico e arquitetural unificado de ponta a ponta: ingestão das bases MIMIC-IV-ECG e PTB-XL, limpeza de códigos sentinela e limites AHA no DuckDB, filtragem digital de biossinais (0.5–45 Hz, notch 50/60 Hz, escala em mV), motor ontológico multilabel (30 classes), particionamento no nível do paciente (80/10/10), contêineres HDF5 e integração com PyTorch.

### ⚙️ 1. Critérios de Qualidade e Limitadores Físicos/Biológicos
* ⚡ **[Limitadores de Integridade Física do Sinal](limitadores_qualidade_sinal.md)**:
  * Eletrodo mudo (*flatline* com desvio padrão $< 0,1\ \mu\text{V}$), tolerância máxima a dados corrompidos ($\le 5\%$ de NaNs) e eliminação de picos mecânicos espúrios ($> 15\text{ mV}$).
* 🫀 **[Limitadores Fisiológicos e Códigos de Hardware](limites_fisiologicos_medicoes.md)**:
  * Desmistificação dos "códigos sentinela" dos eletrocardiógrafos (29999, 32767, 65535) e os limites humanos reais para intervalos R-R (frequência cardíaca), onda P, PR, complexo QRS e QT.

### 👥 2. Seleção, Amostragem e Particionamento de Dados
* 👥 **[Decisões de Seleção e Particionamento por Paciente](decisoes_selecao_particionamento.md)**:
  * A regra de ouro contra a "ilusão dos 99% de acurácia" (prevenção de *data leakage* agrupando por paciente) e a fusão transcontinental entre MIMIC-IV-ECG (EUA) e PTB-XL (Alemanha) contra a "Síndrome do Hospital Único".
* ❓ **[Decisões sobre Exames Sem Diagnóstico Alvo (`is_supervised_target`)](decisoes_exames_sem_rotulo.md)**:
  * Por que laudos técnicos inespecíficos não podem ser forçados como "Normais" nem preenchidos com zeros falsos, justificando a exclusão supervisionada de 3,3% dos exames.
* 💾 **[Decisões de Armazenamento Físico: HDF5 vs. 800 Mil Arquivos](decisoes_armazenamento_hdf5_arquivos.md)**:
  * Por que não salvar centenas de milhares de arquivos soltos em disco (colapso de *inodes*), como o HDF5 acelera o DataLoader em tempo $O(1)$ e o desacoplamento entre metadados no DuckDB e sinais físicos locais.

### 🩺 3. Fisiopatologia Clínica e Modelagem Multilabel
* 🏷️ **[Decisões de Modelagem Multilabel e Taxonomia Clínica](decisoes_taxonomia_multilabel.md)**:
  * Por que a medicina real exige predição Multilabel independente (`Sigmoid` + `BCEWithLogitsLoss`) em vez de Softmax, a estruturação hierárquica (5 superclasses + 25 patologias de emergência), o escopo territorial de infartos e o tratamento de laudos duvidosos com peso 0,5.
* 👁️ **[Anatomia Clínica das 12 Derivações: Guia Visual](anatomia_12_derivacoes_clinica.md)**:
  * Como 10 eletrodos físicos criam 12 "câmeras imaginárias" que monitoram as diferentes paredes do coração (inferior, anterior, lateral e septal) e a correlação com as artérias coronárias no infarto.

### 🔬 4. Processamento de Sinais e Métricas de Validação
* 📦 **[Por que WFDB e HDF5? O Papel de Cada Formato](wfdb_vs_hdf5_papeis.md)**:
  * A analogia da cozinha industrial: por que o WFDB é o ingrediente bruto da medicina e o HDF5 é a dose processada e compactada a vácuo para acelerar o treinamento das redes em 30 a 50 vezes.
* 🔬 **[Decisões de Processamento e Filtragem Digital de Biossinais](decisoes_processamento_sinal.md)**:
  * A física por trás da taxa de 500 Hz (Nyquist), a escolha da faixa 0,5–45 Hz (eliminação de respiração e tremor muscular), o filtro Notch duplo (50/60 Hz para redes europeia e americana) e a calibração física em mV.
* 🎯 **[Decisões de Métricas Clínicas: Por que Acurácia é uma Armadilha](decisoes_metricas_clinicas.md)**:
  * O paradoxo do desbalanceamento de classes em emergências, o custo catastrófico do Falso Negativo contra o Falso Positivo e a priorização da Sensibilidade $\ge 95\%$, Macro AUROC $\ge 0,90$ e Macro AUPRC.
