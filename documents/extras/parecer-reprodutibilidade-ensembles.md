# Parecer de reprodutibilidade e integração texto-código da subseção dos ensembles

Revisão cruzada do card 24C (#216), sobre a Seção 4.4.4 do documento (`documents/documentacao.md`)
e o notebook `notebooks/ensembles.ipynb`. Escopo técnico: o notebook roda do zero, e cada número do
texto sai de uma célula dele. O checklist de texto é do card 22A e não é tocado aqui.

Revisado em 2026-09-25, sobre a `develop` em `5c55c68`.

## Escopo efetivamente coberto

O card previa revisar a subseção escrita nos cards **#193** (Felipe, Random Forest e permutation
importance) e **#194** (Fernanda, Gradient Boosting, PDPs e comparação entre ensembles).

Só a metade do #193 existe. A MR **!135** foi mergeada e entregou a Seção 4.4.4. O card **#194**
continua em `Backlog`, sem MR, e as subseções de Gradient Boosting, de dependência parcial e de
comparação entre os dois ensembles ainda não foram escritas. **Este parecer cobre o #193 e deixa o
#194 explicitamente fora**, para que a ausência não seja lida como aprovação.

## CR01: o notebook roda do zero numa sessão nova?

**Não.** Três causas independentes, em ordem de gravidade.

### 1. A Seção 9 depende de duas variáveis que nenhuma célula define

A célula da Seção 9.1 monta a comparação entre os dois ensembles assim:

```python
METRICAS_ENSEMBLES = {
    "Random Forest": globals().get("metricas_rf"),       # seção 8.2 (#189)
    "Gradient Boosting": globals().get("metricas_gb"),   # busca do #190
}
```

`metricas_rf` e `reconstruido_rf` são definidas na Seção 8.2. **`metricas_gb` e `reconstruido_gb`
não são definidas em nenhuma célula do notebook.** A Seção 11.3, que entrou pelo #190 e é a que
deveria expô-las, nomeia os objetos `metricas_melhor` e `melhor_gb`.

O `globals().get` engole a ausência sem erro, então a consequência não aparece como falha: numa
execução de cima para baixo a Seção 9.1 imprime `Gradient Boosting sem número medido por
avaliar()`, elege o Random Forest por falta de adversário e registra `comparação completa? não`.

E o problema não se resolve só renomeando as variáveis: a **Seção 9 vem antes da Seção 11** na
ordem do notebook, então no momento em que a célula roda o Gradient Boosting ainda não foi medido
em nenhuma hipótese.

**Correção sugerida:** mover a Seção 9 para depois da Seção 11, e na Seção 11.3 atribuir
`metricas_gb = metricas_melhor` e `reconstruido_gb = melhor_gb`, que é o contrato que a própria
Seção 9.1 declara em texto (*"quando ela entrar, a seção dela precisa expor `metricas_gb` e
`reconstruido_gb` com o mesmo papel"*).

### 2. Executar o notebook suja dois arquivos rastreados

A célula da Seção 11.2 chama `busca_gb.salvar_resultados`, que regrava
`documents/extras/resultados/hiperparametros_gradient_boosting.json` e `documents/extras/resultados/cv_resultados_gradient_boosting.json`.

O primeiro guarda `"tempo_total_s": 7146.9`. Esse campo é um cronômetro, não um resultado: ele muda
a cada execução mesmo com `random_state` fixo em 42. Então rodar o notebook de ponta a ponta deixa
`git status` sujo em dois arquivos versionados, e o diff que aparece não é do modelo, é do relógio.

**Correção sugerida:** ou o tempo sai do JSON e vai para o output da célula, ou a gravação passa a
ser condicional, como já é a leitura da Seção 8.2.

### 3. As duas buscas rodam sempre, sem checar se o resultado já existe

A célula da Seção 8.2 tem a guarda certa:

```python
if not busca_rf.ARQUIVO_HIPERPARAMETROS.exists():
```

A célula da Seção 8.1, que executa a busca, tem só a guarda de `busca is None`, que cobre a
montagem da busca e não a existência do resultado; falta nela a guarda de existência que a 8.2 tem.
A da Seção 11.2 não tem guarda alguma. São 201 ajustes de Random Forest, cada um com 200 a 600 árvores e `n_jobs=1`, mais 201 de
Gradient Boosting, cuja busca já foi medida em **7.146,9 s**. A própria Seção 4.4.4 registra que
esse é *"o principal risco para a execução numa sessão gratuita do Colab"*, e a estrutura atual
garante que o risco se realize em toda reexecução.

**Correção sugerida:** replicar na 8.1 e na 11.2 a guarda que a 8.2 já tem, com uma variável no
topo do notebook do tipo `REEXECUTAR_BUSCAS = False` para quem quiser forçar.

### Observação sobre o ambiente desta revisão

A execução por `jupyter nbconvert` não chegou a começar nesta máquina: falta `ipykernel` e
`xgboost` no `.venv` local, e a primeira célula importa `xgboost` sem proteção. **Isso é limitação
do ambiente do revisor, não defeito da subseção**: `xgboost==3.4.1` está no `requirements.txt` e a
primeira linha da célula é `!pip install -q -r requirements.txt`, que resolve a dependência no
Colab. As três causas acima foram estabelecidas por leitura do notebook e do código de `src/`, e
cada uma é verificável sem executar nada.

## CR02: cada número do texto sai de uma célula?

### O que confere

| Número na Seção 4.4.4 | Conferido contra |
|---|---|
| 407.139 Clientes em 484.915 respostas | soma das partições congeladas: 307.510 + 47.560 + 52.069 = 407.139; linhas 558 e 584 do documento |
| 341.962 linhas de treino | partição de treino congelada; linha 1109 do documento |
| 201 ajustes | 40 combinações x 5 folds + refit |
| log₂(341.962) ≈ 18,4 | 18,383 |
| 140 previsões da validação | 14 features x 10 repetições |
| `n_iter` 40, `random_state` 42, 5 folds | célula da Seção 8 do notebook |
| V de Cramér de 0,293 e 75,7% acima de 120 minutos | linhas 644 e 1022 do documento |
| correlação de 0,664 entre `ATRASO_CHEGADA` e `ESTATISTICA_ATRASOSAIDA` | linha 667 do documento |

Nenhuma divergência numérica. Os números que estão lá estão certos e rastreáveis.

### O que está pendente, e não é erro

As duas tabelas de resultado da subseção têm **14 marcadores de pendência**, e o registro da
permutation importance tem outros **5**. A própria 4.4.4 escreve "as quinze células"; a diferença de
um é do texto revisado e cabe ao checklist do card 22A, não a este parecer. Isso está declarado com honestidade no texto (*"Nenhum
valor acima foi estimado"*), e é a postura certa.

Vale registrar o alcance da pendência: `documents/extras/resultados/hiperparametros_random_forest.json` e
`documents/extras/resultados/importancia_permutacao.json` **não existem em nenhuma branch do repositório**. Conferi
varrendo todas as branches remotas. A busca do Random Forest nunca rodou, e o ranking de
permutation importance nunca foi calculado.

### Uma divergência entre o texto e o código

A Seção 4.4.4 afirma que a importância é calculada *"sobre o melhor ensemble, o de maior F2 de
`avaliar` na validação entre o Random Forest e o Gradient Boosting"*.

Pelo item 1 do CR01, essa comparação não acontece em nenhuma execução do notebook como ele está: o
Gradient Boosting nunca chega com número à Seção 9.1. Enquanto o notebook não for corrigido, o
texto descreve um procedimento que o código não executa.

**Correção sugerida:** corrigir o notebook, que é a origem do problema. Se a correção não couber
neste card, o texto precisa dizer que a comparação depende de a Seção 11 rodar antes da Seção 9.

## CR03: isolamento do teste e agrupamento por Cliente

**Veredito: as duas travas estão corretas, nos dois ensembles.**

**Agrupamento por Cliente.** `criar_busca_random_forest` e `criar_busca_gradient_boosting` recusam
`cv` passado como inteiro, o que impede o scikit-learn de substituí-lo por um particionador
estratificado que ignoraria `groups`. `executar_busca` chama `validacao.conferir_folds(busca.cv,
groups)` antes do `fit` e repassa `groups=groups` ao `fit`. A conferência é anterior ao ajuste, e
não um teste que se esquece de rodar.

**Isolamento do teste.** As buscas recebem `preparo["x"]["treino"]` e `x_treino_gb`. A partição de
teste não entra em nenhuma das duas. Os cortes são `2025-07-01` e `2026-01-01` nas três células que
os declaram (Seções 2, 10.1 e a do fim do notebook), iguais aos da Regressão Logística e aos do
registro de decisão.

Confirmo também que o conserto anunciado na !135 aconteceu: a Seção 2 usava `2025-06-01` e
`2025-12-01`, o que mediria o Random Forest numa validação diferente da dos outros candidatos. Não
há mais nenhuma ocorrência desses cortes no notebook.

## Resumo dos vereditos

| Critério | Veredito |
|---|---|
| CR01, roda do zero | **não**, por três causas independentes, todas com correção sugerida |
| CR02, número a número | **sem divergência numérica**; uma divergência entre texto e código, no procedimento da Seção 9 |
| CR03, teste isolado e folds por Cliente | **sim**, nos dois ensembles |
| CR05, revisou a subseção da outra dupla | **sim**, a 4.4.4 é do #193 |
| Metade do #194 | **fora do parecer**, porque ainda não foi escrita |

Nenhum dos três achados do CR01 é de conteúdo: a subseção está correta no que afirma e honesta no
que deixa pendente. Os três são de mecânica de execução, e os três têm conserto curto.
