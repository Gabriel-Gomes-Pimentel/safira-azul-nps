# Parecer de reprodutibilidade e integração texto-código da subseção de modelos interpretáveis

Revisão cruzada do card 24A (#195), sobre a subseção de modelos interpretáveis da Seção 4.4: a
Regressão Logística (Seção 4.4.2 do documento e `notebooks/regressao_logistica.ipynb`, de Gabriel)
e a Árvore de Decisão (`notebooks/arvore_decisao.ipynb` e os textos em `documents/extras/`, de
Luiza). Escopo técnico: o notebook roda do zero, usa o contrato e a função `avaliar`, e cada número
do texto sai de uma célula. A coerência de texto e de métricas é do card 24B (#196) e não é tocada
aqui.

Revisado em 2026-09-25, sobre a `develop` em `5235f23`. É o parecer espelho do #216, que revisou a
subseção dos ensembles, e segue a mesma forma: veredito por critério, com arquivo e célula ou
parágrafo em cada apontamento.

## Escopo efetivamente coberto

| Modelo | Cards | MRs | O que foi revisado |
|---|---|---|---|
| Regressão Logística | #206 a #214 | !126 e anteriores | Seção 4.4.2, `regressao_logistica.ipynb`, `documents/extras/resultados/hiperparametros_logistica.json` |
| Árvore de Decisão | #228 a #234 | !122, !123, !124, !127, !128, !129 | `arvore_decisao.ipynb`, `src/hiperparametros_arvore.json`, `busca_arvore_decisao.md`, `folhas_arvore_decisao.md`, `interpretacao_arvore_decisao.md` |

**A Árvore de Decisão não tem subseção no documento.** A Seção 4.4.2 termina anunciando que a
árvore é *"apresentada na subseção seguinte"*, e a subseção seguinte é a 4.4.4, do Random Forest.
Não existe 4.4.3. Os resultados da árvore estão só em `documents/extras/`, então a revisão
texto-código da árvore foi feita contra esses arquivos. A ausência não pode ser lida como aprovação.

## Como a execução foi verificada

**O notebook não foi executado em sessão nova do Colab.** A base analítica não é versionada e não
estava disponível na máquina desta revisão, que também não tem `pyarrow` para ler o parquet. Os
vereditos do CR01 abaixo saem de duas fontes, e cada apontamento diz qual:

- **reprodução isolada**: o trecho que falha foi executado fora do notebook, com os módulos de
  `src/`, e o erro está transcrito;
- **leitura**: o encadeamento de células e o código de `src/` foram lidos em ordem, célula a
  célula.

Nenhum dos dois notebooks tem output salvo. Por isso a conferência do CR03 é feita contra o
markdown que transcreve cada output e contra os artefatos versionados em `assets/` e `src/`, e não
contra o output de uma execução. Quem rodar os notebooks no Colab fecha o CR01 à letra.

## CR01: o notebook roda do zero numa sessão nova?

### Regressão Logística: sem bloqueio encontrado

Lido célula a célula, o notebook se sustenta de ponta a ponta:

- o caminho de importação (`src/`) e o da base têm laço de alternativas para projeto local, Colab
  e Drive, e a célula da Seção 1.2 falha com a instrução do que fazer quando a base não existe;
- cada seção só consome variáveis definidas nas anteriores, em ordem;
- a Seção 5.1 regrava `documents/extras/resultados/hiperparametros_logistica.json`, mas sem o tempo de ajuste, de
  propósito (*"um artefato versionado que produz diff sem o resultado ter mudado deixa de servir
  como prova"*). Uma reexecução não suja o `git status`, que é exatamente o problema que o #216
  encontrou na busca do Gradient Boosting.

Dois pontos que não impedem a execução:

1. **Custo.** A busca da Seção 4.2 roda inteira em toda execução (579 s pelo próprio notebook), e
   as medições de custo das Seções 1.5 e 2.3 somam mais cerca de 6 minutos (111 s e 243 s pelas
   tabelas transcritas). A Seção 5.3 já sabe reconstruir o vencedor pelo JSON; uma guarda do tipo
   `REEXECUTAR_BUSCA = False` na Seção 4.2 evitaria refazer os 40 ajustes.
2. **Células sem `id`.** O notebook declara `nbformat_minor` 5, que exige `id` por célula, e
   nenhuma das 70 tem. O Jupyter emite `MissingIDFieldWarning` e segue; versões futuras podem
   recusar. Abrir e salvar no Jupyter preenche os ids.

### Árvore de Decisão: não roda

Quatro causas, em ordem de execução. As duas primeiras param o notebook.

**1. Seção 1, célula de carga: a base aponta para um marcador.** `CAMINHO_BASE =
'CAMINHO/PARA/base_analitica.parquet'  # TODO: ajustar`, seguido de `pd.read_parquet(CAMINHO_BASE)`.
Numa sessão nova, a primeira célula de dados levanta `FileNotFoundError`. A mesma célula faz
`sys.path.insert(0, 'src')` com caminho relativo, que só resolve se o diretório de trabalho for a
raiz do repositório; aberto a partir de `notebooks/`, o `import matriz` falha. *(leitura)*

Correção sugerida: copiar o laço de caminhos da Seção 1.2 da logística, que já cobre local, Colab
e Drive.

**2. Seção 6, primeira célula de código: `TypeError` antes de chegar à árvore.** A linha

```python
modelo_final = criar_pipeline(preparo['preprocessador'], **busca.best_params_.copy())
```

passa as chaves de `best_params_`, que vêm prefixadas por `modelo__`, direto para o
`DecisionTreeClassifier`, porque `pipeline_arvore.criar_pipeline` repassa `**hiperparametros` ao
estimador. Reproduzido:

```text
TypeError: DecisionTreeClassifier.__init__() got an unexpected keyword argument 'modelo__max_depth'
```

A linha seguinte (`modelo_final = busca.best_estimator_`) é a correta, mas nunca executa. O próprio
comentário `# TODO: remover o prefixo 'modelo__'` registra o problema. *(reprodução isolada)*

Correção sugerida: apagar a linha que chama `criar_pipeline` e ficar com o `best_estimator_`.

**3. Seção 5, segunda célula de código: a execução apaga o registro versionado.** A célula grava
`busca.best_params_` em `src/hiperparametros_arvore.json`. O arquivo versionado hoje tem outro
formato: os hiperparâmetros sem prefixo, o critério de busca, o F2 da validação cruzada, as três
métricas de validação, o número de folhas e a justificativa da escolha. Rodar o notebook substitui
tudo isso por um dicionário com chaves `modelo__...`, e nenhuma célula regenera o resto. *(leitura)*

Correção sugerida: a célula gravar o mesmo formato que está versionado. Vale também mover o arquivo
para `documents/extras/resultados/`, ao lado de `hiperparametros_logistica.json` e dos JSONs dos ensembles: é o único
artefato de resultado do projeto que mora em `src/`.

**4. Seção 3, segunda célula de código: a métrica de partida está comentada.**

```python
# TODO: substituir por 'from card_241 import avaliar' quando aquele card estiver em develop.
# metrica_de_partida(pipeline_base, preparo['x']['treino'], y_treino, avaliar)
```

O #241 já está em `develop`, como `src/avaliacao.py`. A célula não falha, mas não mede nada, e
nenhuma outra célula do notebook chama `avaliar`. *(leitura)*

**Markdown desatualizado, que não quebra a execução mas contradiz o repositório:**

- Seção 2.1: diz que `SEGUNDOS_POR_AJUSTE` *"está `None` de propósito, ninguém mediu ainda"*. Em
  `src/espaco_busca_arvore.py` ele vale `3.0979`.
- Seção 4: diz que `busca_arvore_decisao.md` e `src/hiperparametros_arvore.json` *"foram gerados
  com `average_precision` e ficam desatualizados"*. Os dois arquivos registram `scorer_f2` como
  critério.
- Seção 7: a célula de código é só `# TODO`, e o markdown remete a *"ver `.claude`/card #234 para
  o roteiro completo"*. A referência a `.claude` é do mesmo tipo que a !133 removeu de um
  comentário do #228.

## CR02: checklist técnico

| Item | Regressão Logística | Árvore de Decisão |
|---|---|---|
| Contrato do card 01 (`preparar_matriz`, cortes `2025-07-01` e `2026-01-01`) | **sim**, Seção 1.3 | **sim**, Seção 1, mas atrás do caminho `TODO` do CR01.1 |
| `avaliar` do card 05 (#241) | **sim**, Seção 9, para partida e otimizado | **não**: chamada comentada na Seção 3 e ausente no resto (CR01.4) |
| `GroupKFold` com `groups` | **sim**: `validacao.criar_folds(x, grupos)` na Seção 4.1 | **sim**: `validacao.criar_folds(x, grupos)` na Seção 4 |
| Conferência dos folds antes do `fit` | **não**: `validacao.conferir_folds` não é chamada | **não**: idem |
| `n_iter` | **não se aplica**: `GridSearchCV` exaustivo, 8 combinações declaradas | **não se aplica**: `GridSearchCV` exaustivo, 48 combinações (2 x 4 x 3 x 2) |
| `random_state` | **sim**, 42 (`congelamento.SEMENTE`) no pipeline, Seção 3.1 | **sim**, 42 (`espaco_busca_arvore.SEMENTE`) em `criar_pipeline` |
| `scoring` da busca | **ressalva**: `make_scorer(fbeta_score, beta=2)` montado na célula, marcado como *"provisório"*, e não o `scorer_f2` do #242. A Seção 9 mostra que coincide com `avaliar()["F2"]` na validação | **sim**, `scorer_f2` do #242 |
| Espaço de busca documentado | **sim**: Seção 2 do notebook, redução registrada na Seção 4.1 e no documento | **parcial**: justificado em `src/espaco_busca_arvore.py`; o notebook só resume, e o documento não tem subseção |
| Markdown antes e depois de cada bloco de código | **sim**, nas 20 células de código | **não**: 7 dos 12 blocos violam a regra. Contando a primeira célula como 0, ficam sem markdown antes as células 10, 15 e 18 (segundas células das Seções 3, 5 e 6) e sem markdown depois as células 9, 14, 17 e 20 (primeiras das Seções 3, 5 e 6, e a da Seção 7); nenhuma célula tem markdown de leitura do output |

Sobre a conferência dos folds: nos dois notebooks os folds vêm de `criar_folds`, então o
agrupamento está correto por construção. A recomendação é chamar `validacao.conferir_folds` antes
do `fit`, como fazem as buscas dos ensembles, para que um erro futuro apareça antes da busca e não
depois dela.

## CR03: cada número do texto sai de uma célula?

### Regressão Logística: todos os números conferem

| Número na Seção 4.4.2 | Conferido contra |
|---|---|
| `C` 1,0, `balanced`, L1 `liblinear`, `max_iter` 1600 | primeira linha de `documents/extras/resultados/hiperparametros_logistica.json`; tabela da Seção 4.2 do notebook |
| 8 combinações, 5 folds, 40 ajustes | Seção 4.2; 8 x 5 = 40 |
| 9,6 minutos | Seção 4.2: 579 s |
| 16 combinações e 58 minutos estimados | Seções 2.1 e 2.3: 57,7 min |
| F2 de 0,5189, desvio de 0,0035 | JSON: 0,5189160632 e 0,0034686472 |
| distância de 0,0003 entre a 1ª e a 4ª | JSON: 0,5189161 − 0,5186286 = 0,00029 |
| 0,519 com `balanced` contra 0,275 com `None` | JSON: 0,5189 e 0,2753 |
| Precisão Média 0,4996 → 0,5011, ROC-AUC 0,7257 → 0,7273, Sensibilidade 0,1976 → 0,4861 | tabela da Seção 9 |
| fila de 38.849 respostas no limiar 0,2934, contra 9.050 | Seção 9 |
| 38 coeficientes, 14 features, sete colunas colineares | Seções 3.2, 6 e 6.4 |
| odds ratio 7,73 e 2,99 no cancelamento | Seção 7.1 |
| 4,245, 1,986 em 35 min e 1,085 em 29 min | Seções 6 e 6.3 |
| `DIAMANTE` 1,86 contra `AZUL FIDELIDADE` | Seção 6.2; também 1,326 / 0,712 = 1,862 pelos valores brutos da Seção 6 |
| 69,2% e 24,6% da Hipótese 3 | Seção 7.1 do notebook e Seção 4.2.4 do documento |

**Uma contradição de datas no notebook, que o documento não herda.** A Seção 4.4 do notebook diz
que `AZUL ONE` e `DIAMANTE UNIQUE` aparecem a partir de **2025-10-24** e que as 1.353 linhas deles
*"caem inteiras na partição de teste: nem treino nem validação têm uma única linha delas"*. Pelos
cortes do registro de decisão, a validação vai de 2025-07-01 a 2025-12-31, e 2025-10-24 está dentro
dela. Ou a data está errada, ou a partição está. A Seção 4.4.2 do documento só afirma que os dois
tiers não existem no **treino**, o que vale nos dois casos. Precisa ser conferido na base e
corrigido no notebook.

### Árvore de Decisão: os números são coerentes entre si, mas não saem do notebook

Entre os arquivos versionados, tudo confere:

| Número | Conferido contra |
|---|---|
| 48 combinações | 2 critérios x 4 profundidades x 3 folhas mínimas x 2 pesos em `GRADE_ARVORE` |
| vencedor `entropy`, profundidade 5, folha mínima 50, `balanced` | `busca_arvore_decisao.md` e `src/hiperparametros_arvore.json` |
| F2 de 0,5078 na validação cruzada | JSON: 0,507832 |
| Sensibilidade 0,4759, Precisão Média 0,4415, ROC-AUC 0,6800, 32 folhas | JSON: 0,47589, 0,44155, 0,67998, 32 |
| folha 20 com 72,6% da base | `folhas_arvore_decisao.md`: 248.202 / 341.962 = 72,58% |
| faixas de atraso somando cerca de 90% | 83,3 + 1,7 + 1,8 + 1,1 + 2,0 = 89,9 |

**Nenhum desses números sai de uma célula do notebook.** O notebook não chama `avaliar` (CR01.4),
não compara profundidades, não imprime o tempo da busca e não calcula a taxa por folha; a Seção 7
está vazia. Especificamente, não têm célula de origem:

- as três métricas de validação da profundidade 5, e toda a tabela de profundidades 3, 4 e 5 de
  `busca_arvore_decisao.md`;
- o tempo de 1.526 s da busca;
- a taxa de Detrator e o `n` de cada folha em `folhas_arvore_decisao.md`, de que
  `interpretacao_arvore_decisao.md` diz tirar todos os seus números.

**Uma divergência numérica com a Regressão Logística.** `interpretacao_arvore_decisao.md`, item 3,
afirma que `ANTECEDENCIA_CANCELAMENTO` *"só existe preenchida para voos com cancelamento (16,3% da
base); para os demais 83,7% ela é nula"*. A Seção 6.4 da logística mede a mesma coluna no mesmo
treino: 307.567 de 341.962 linhas nulas, ou seja, **10,1% preenchida e 89,9% nula**. O próprio
arquivo da árvore, no item 1, diz que 10,1% das linhas têm `ATRASO_CHEGADA` nulo, que é exatamente
o efeito do contrato apagando os dados de voo nos cancelados. O 16,3% não bate com nenhuma das duas
medições e precisa ser refeito.

## Resumo dos vereditos

| Critério | Regressão Logística | Árvore de Decisão |
|---|---|---|
| CR01, roda do zero | **sem bloqueio por leitura**; execução no Colab pendente | **não**: caminho `TODO` e `TypeError` na Seção 6; a execução também apagaria o JSON versionado |
| CR02, checklist técnico | **cumprido**, com ressalva no `scoring` provisório | **parcial**: sem `avaliar` e sem markdown em torno de 7 blocos |
| CR03, número a número | **todos conferem**; uma contradição de datas no notebook | **coerentes entre arquivos, mas sem célula de origem**; uma divergência de 16,3% contra 10,1% |
| Subseção no documento | **sim**, 4.4.2 | **não existe**; a 4.4.2 anuncia uma subseção seguinte que é a do Random Forest |

A Regressão Logística é a referência de reprodutibilidade da Seção 4.4 inteira: é o único notebook
de modelagem desta subseção que grava artefato sem ruído de relógio, documenta cada output em
markdown e liga cada número do documento a uma seção. A Árvore de Decisão tem os resultados, mas
eles moram fora do notebook, e os quatro consertos do CR01 são o que falta para que a execução
produza de novo o que os arquivos de `documents/extras/` afirmam.
