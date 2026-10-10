# Execução de ponta a ponta dos notebooks

Registro do card #303: os 12 notebooks de `notebooks/` executados do zero, na ordem da seção
"Execução dos projetos" do `README.md`, sobre a base real da Azul. O objetivo é responder com
evidência ao critério de publicação "Os notebooks executam todas as células de output
corretamente?", já que os notebooks são versionados sem saídas e o arquivo no repositório não prova
que eles executam.

## Como a execução foi feita

| Item | Valor |
|---|---|
| Código | `develop` em `1fb89ad` (05/10/2026, depois do merge das MRs !171 a !181) |
| Bases | os oito `.csv` da Azul em `data/raw/`, sem as cópias `.xlsx` |
| Python | 3.12.15, ambiente novo instalado só com `pip install -r requirements.txt` |
| Bibliotecas | pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1, statsmodels 0.15.0, matplotlib 3.11.2, pyarrow 25.0.1 |
| Máquina | macOS 15.7, Intel Core i5-8279U (4 núcleos, 8 threads), 8 GB de RAM |
| Execução | `nbclient` 0.11, um kernel novo por notebook, pasta de trabalho `notebooks/`, sem limite de tempo por célula, um notebook por vez |
| Testes | `pytest tests/ -q` antes da execução: 556 aprovados, 15 pulados, em 4 min 27 s |

As saídas executadas ficaram em `.execucao/`, que está no `.gitignore`. Nenhuma delas foi versionada,
porque contêm dados do parceiro.

## Resultado por notebook

| Ordem | Notebook | Células de código | Status | Tempo |
|---:|---|---:|---|---:|
| 1 | `pre-processamento.ipynb` | 15 | executou | 22 s |
| 2 | `exploracao_dados.ipynb` | 39 | executou | 32 s |
| 3 | `hipoteses_nps.ipynb` | 20 | executou | 1 min 9 s |
| 4 | `escalonamento_anexo_a1.ipynb` | 12 | executou | 10 s |
| 5 | `histogramas_anexo_a1.ipynb` | 5 | executou | 7 s |
| 6 | `definicao_predicao.ipynb` | 6 | executou | 5 s |
| 7 | `modelagem_nps.ipynb` | 4 | executou | 8 s |
| 8 | `modelagem.ipynb` | 25 | executou | 16 min 18 s |
| 9 | `regressao_logistica.ipynb` | 20 | executou | 2 h 39 min (ver achado 3) |
| 10 | `arvore_decisao.ipynb` | 12 | **falhou na célula 16**, corrigido no #324 | 4 min 32 s até a falha |
| 11 | `comparacao_modelos.ipynb` | 19 | executou | 8 min 5 s |
| 12 | `ensembles.ipynb` | 24 | **interrompido na busca da seção 8.1**, sem erro até ali | 52 min até a interrupção |

O `ensembles.ipynb` não foi executado até o fim. As seções 1 a 7 (ambiente, contrato de entrada, medição
de custo, decisão do boosting, espaços de busca e linha de base do Random Forest) rodaram sem erro
em cerca de 2 minutos. Em seguida começou a busca aleatória da seção 8.1, a primeira célula do notebook com
`n_jobs=-1`, e ela foi interrompida após 50 minutos com os oito processos ainda ajustando, para
fechar este card no prazo. A própria Seção 4.4.4 da documentação registra que essa busca, com 201
ajustes de 200 a 600 árvores sobre as 341.962 linhas do treino, "é a mais cara da comparação e não
coube no prazo desta entrega". Depois dela o notebook ainda executa a permutation importance
(seção 9), a linha de base do Gradient Boosting (seção 10) e a busca do Gradient Boosting (seção 11),
que a equipe registrou em cerca de 45 minutos efetivos. Executar o notebook inteiro
exige reservar algumas horas numa máquina dedicada; até lá, as seções 8 a 12 seguem sem evidência
de execução de ponta a ponta.

## Achados

### 1. `arvore_decisao.ipynb` parava na célula 16 (corrigido no #324)

A célula gravava `busca.best_params_` em `src/hiperparametros_arvore.json` por caminho relativo à
pasta de trabalho, e de `notebooks/` levantava `FileNotFoundError`. Rodando da raiz, ela teria
sobrescrito o JSON versionado com um formato mais pobre. As células seguintes tinham outros dois
problemas que a execução não chegou a alcançar: `criar_pipeline(..., **busca.best_params_)`
levantava `TypeError` pelo prefixo `modelo__`, e a seção 7 era um `# TODO`. O #324 (MR !203) corrige
os três. O notebook corrigido foi executado de ponta a ponta sem erro, a célula 16 confere que o
vencedor da busca é `entropy`, profundidade 5, `min_samples_leaf` 50 e `class_weight` `balanced`, e
o JSON versionado não muda.

### 2. Vencedor da busca da Regressão Logística é um empate técnico

Na busca da seção 4.2 de `regressao_logistica.ipynb`, as quatro combinações com `class_weight`
`balanced` ficam numa faixa de 0,0004 de F2, e a ordem entre elas mudou em relação ao que está
versionado:

| Combinação | F2 versionado (posição) | F2 nesta execução (posição) | Desvio entre folds |
|---|---:|---:|---:|
| `C` 1,0, L1 `liblinear` | 0,518916 (1º) | 0,518916 (2º) | 0,0035 |
| `C` 0,1, L2 `lbfgs` | 0,518777 (2º) | 0,518828 (3º) | 0,0033 |
| `C` 0,1, L1 `liblinear` | 0,518663 (3º) | 0,518663 (4º) | 0,0034 |
| `C` 1,0, L2 `lbfgs` | 0,518629 (4º) | 0,519003 (1º) | 0,0033 |

As duas combinações com `liblinear` reproduziram o F2 até a sexta casa. As duas com `lbfgs`
variaram na quarta casa, o que é esperado de um otimizador iterativo rodando sobre bibliotecas de
álgebra linear diferentes, e a de `C` 1,0 subiu do 4º para o 1º lugar. A distância entre o 1º e o
2º colocado é de 0,00009 de F2, cerca de quarenta vezes menor que o desvio entre os folds: é um
empate técnico, e o vencedor pode trocar de uma máquina para outra. Com o
vencedor trocado, o modelo reconstruído na seção 5.3 passa a ser o `lbfgs`, e os odds ratios das
seções 6 e 7, que a Seção 4.4.2 da documentação publica, mudam na segunda casa decimal: cancelamento com aviso no mesmo dia 7,75 contra 7,73
publicados, aviso com 48 dias 3,00 contra 2,99, tier Diamante 1,86 nos dois. A leitura da seção
não muda.

O mesmo efeito aparece no piso de Regressão Logística da seção 9 de `modelagem.ipynb`: Precisão
Média de 0,5014 contra 0,5019 versionados, ROC-AUC de 0,7446 contra 0,7442. O Gradient Boosting e
a classe majoritária da mesma tabela saíram idênticos.

Os números de `comparacao_modelos.ipynb` que a documentação cita se reproduziram: precisão de
0,5101 e cobertura de 0,4230 na fila diária do teste, Gradient Boosting calibrado com Precisão Média
de 0,5174 e ROC-AUC de 0,7330, linha de base do Random Forest com 0,4271 e 0,6716.

### 3. Tempo de parede da célula 2.3 de `regressao_logistica.ipynb`

A célula que mede o custo dos solvers levou 2 h 29 min de relógio, embora os oito ajustes que ela
mede somem cerca de 100 s no próprio cronômetro do notebook. O intervalo coincidiu com a tela
desligada e com um aviso de bateria baixa da máquina, e a explicação mais provável é o sistema ter
reduzido a prioridade do processo. O tempo registrado na tabela vale como limite superior; os
ajustes em si custaram o que a célula mostra.

### 4. Arquivos versionados que a execução regrava

Executar os notebooks regrava arquivos que estão no repositório. Todos foram comparados com a versão
versionada e restaurados depois, sem commit:

| Arquivo | Gravado por | Resultado da comparação |
|---|---|---|
| 8 PNGs `hist_*` de `assets/` | `histogramas_anexo_a1.ipynb` | pixels idênticos; muda só o metadado do arquivo |
| `assets/g10_curva_roc.png`, `assets/g11_precisao_cobertura.png` | `modelagem.ipynb` | pixels idênticos |
| `assets/linha-tempo-particionamento.png` | `modelagem.ipynb` | mesmo conteúdo (cortes, cores, cobertura de 91,5%); a fonte renderizada muda a largura de 1295 para 1256 px |
| `documents/extras/resultados/comparativo_modelos.json` e `documents/extras/comparativo-modelos.md` | `modelagem.ipynb` | muda só a linha da Regressão Logística (achado 2) |
| `documents/extras/resultados/hiperparametros_logistica.json` | `regressao_logistica.ipynb` | as duas primeiras linhas trocam de ordem (achado 2) |

## Conclusão

Dos 12 notebooks, 10 executaram de ponta a ponta sem erro na base real, com o ambiente instalado só
a partir do `requirements.txt`. O `arvore_decisao.ipynb` falhou e foi corrigido no #324 (MR !203);
a versão corrigida executou inteira sem erro. O `ensembles.ipynb` executou sem erro até a busca do
Random Forest da seção 8.1, que foi interrompida por tempo.

A execução também encontrou dois problemas de instrução, corrigidos em cards próprios: o README
pedia Python 3.10, mas o `requirements.txt` só instala a partir do 3.12 (#327, MR !198), e os
notebooks e o README ensinavam caminhos diferentes para o Colab (#329, MR !201).

Para quem for reexecutar: com Python 3.12 e as bases em `data/raw/`, a sequência do README leva
cerca de 1 hora sem `regressao_logistica.ipynb` e `ensembles.ipynb` (27 minutos nos nove notebooks
medidos acima, mais os cerca de 25 minutos da busca do `arvore_decisao.ipynb`). Esses dois têm células de
medição e de busca que levam horas, e a ordem de grandeza está na tabela acima.
