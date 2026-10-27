# Decisões de Armazenamento Físico: Por que HDF5 e o Desacoplamento DuckDB vs Sinais

Este documento explica de forma acessível como o CardioPredict resolveu o problema de gerenciar quase 1 milhão de exames de ECG sem estourar o disco rígido e sem travar o treinamento das redes neurais.

---

## 1. Por que NÃO Salvar 800 Mil Arquivos Individuais (`.npy` ou `.csv`)?

Uma abordagem ingênua comum em projetos de dados é pré-processar cada exame e salvá-lo como um arquivo isolado em disco (ex.: `exame_001.npy`, `exame_002.npy`, etc.).

### 1.1. O Colapso dos Inodes no Sistema Operacional
- Em sistemas de arquivos Linux (como ext4), cada arquivo no disco exige uma estrutura de metadados chamada **inode** (que guarda data, permissões e blocos alocados).
- Criar mais de 800.000 pequenos arquivos individuais sobrecarrega a tabela de inodes do sistema, reduz o espaço livre utilizável e torna operações simples (como abrir uma pasta, fazer backup ou listar arquivos com `ls`) extremamente lentas e propensas a congelamentos do sistema operacional.

### 1.2. O Gargalo de I/O na Leitura Randômica
- Durante o treinamento de uma rede neural (Deep Learning), os exames são sorteados aleatoriamente (*shuffle*) para compor cada lote (*batch*).
- Se cada exame estiver em um arquivo separado, a cabeça de leitura do SSD/NVMe precisa abrir, ler e fechar 800.000 descritores de arquivos diferentes continuamente. Esse gargalo de I/O faz com que a GPU ou CPU passe 80% do tempo esperando o disco entregar os dados, desperdiçando poder computacional.

---

## 2. A Solução Adotada: Contêineres HDF5 Estruturados por Partição

Em vez de 800 mil arquivos soltos, criamos apenas **3 grandes contêineres binários** indexados:
* `data/processed/train.h5`
* `data/processed/val.h5`
* `data/processed/test.h5`

### 2.1. O que é o HDF5 (*Hierarchical Data Format*)?
O HDF5 funciona como um "sistema de arquivos dentro de um único arquivo". Ele foi desenvolvido pela NASA e pelo NCSA para armazenar dados científicos massivos em matrizes multidimensionais.

### 2.2. Vantagens Práticas:
1. **Acesso Aleatório em Tempo $O(1)$:** O PyTorch pode saltar instantaneamente para o exame de índice `45.892` e ler apenas seus 5.000 pontos em microssegundos, sem precisar abrir arquivos adicionais.
2. **Compressão Nativa Transparente (`lzf`):** Reduz o consumo de disco de quase 200 GB para cerca de 70–80 GB sem adicionar sobrecarga perceptível na CPU durante a descompactação.
3. **Leitura Segura em Múltiplos Núcleos:** Permite que 8 ou 16 *workers* do `torch.utils.data.DataLoader` leiam amostras concorrentes simultaneamente sem travar a memória.

---

## 3. O Desacoplamento DuckDB vs. Sinais Físicos em Disco

Uma dúvida frequente de desenvolvedores e auditores é:
> *"Por que o banco DuckDB contém 821 mil exames, mas o diretório de dados da máquina possui um volume menor de arquivos brutos `.hea`/`.dat`?"*

### 3.1. A Estratégia de Desenvolvimento
- **A Base DuckDB (`cardiopredict.duckdb`):** Contém os metadados relacionais completos, históricos clínicos, idades, sexos, medições fisiológicas e laudos médicos de **todos os 821.834 exames** das bases mundiais MIMIC-IV-ECG e PTB-XL. Isso permite auditar estatisticamente toda a população de pacientes instantaneamente via SQL.
- **Os Sinais Físicos Brutos (`data/`):** Os arquivos de ondas físicas brutas do MIMIC somam vários terabytes se baixados integralmente. Em ambientes locais de desenvolvimento, trabalha-se com a totalidade dos exames do PTB-XL (21.799 exames) somada a uma amostra estratificada maciça do MIMIC (~73.000 exames físicos já presentes no disco local, totalizando mais de 95 mil gravações completas).

### 3.2. Sincronização Inteligente via Código
O exportador [`src/data/exporter_h5.py`](src/data/exporter_h5.py) faz uma checagem rápida de existência do arquivo antes de processá-lo:
- Se o exame físico está presente no disco e passa no filtro de qualidade, ele é gravado no HDF5 e o DuckDB recebe seu índice posicional exato (`hdf5_index = N`).
- Se o exame físico ainda não foi baixado na máquina, o DuckDB mantém `hdf5_index = NULL` e o exame não entra na partição de treino daquela máquina, mantendo a integridade matemática de índices sem quebras.
