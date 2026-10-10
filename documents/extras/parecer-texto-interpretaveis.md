# Parecer de texto e coerência de métricas da subseção de modelos interpretáveis

Revisão cruzada do card 24B (#196), sobre a subseção de modelos interpretáveis da Seção 4.4: a
Regressão Logística (Seção 4.4.2, de Gabriel, já em `develop`) e a Árvore de Decisão (Seção 4.4.3,
de Luiza, na MR do #235). Escopo de texto: clareza, leitura correta das métricas, ausência de
afirmação causal, "Cliente" e "Tripulantes" com inicial maiúscula, ausência de travessão, e se as
métricas dos interpretáveis e dos ensembles usam a mesma definição e o mesmo formato. A
reprodutibilidade e a conferência número a número são do card 24A (#195) e não são refeitas aqui.

Revisado em 2026-09-25, sobre a `develop` em `b30b6f5` e a branch
`docs/#235/escreve-subsecao-arvore-secao-4-4` em `d0ae19c`. Segue a forma do parecer do #195:
veredito explícito por item e, em cada apontamento, o parágrafo afetado, o trecho atual e uma
redação sugerida.

**Como os parágrafos são indicados.** Pela subseção, pelo subtítulo de quinto nível e pela posição
do parágrafo dentro dele, contando só parágrafos de texto (tabelas e listas não contam). Por
exemplo, "4.4.2, Métricas na validação, 4º parágrafo" é o que começa com *"O modelo supera a meta de
Precisão Média"*. O início de cada parágrafo citado vem transcrito para que ele seja encontrado
mesmo que a numeração de linhas mude.

## CR01: veredito por item do checklist

| Item | 4.4.2 Regressão Logística | 4.4.3 Árvore de Decisão |
|---|---|---|
| Leitura correta das métricas | **ajustar**: 4 apontamentos (L1 a L4) | **ajustar**: 4 apontamentos (L5 a L8) |
| Ausência de afirmação causal | **cumprido**, com 2 ajustes de vocabulário (C1, C2) | **cumprido**, com 3 ajustes de vocabulário (C3 a C5) |
| "Cliente" com inicial maiúscula | **cumprido**: nenhuma ocorrência em minúscula | **cumprido**: nenhuma ocorrência em minúscula |
| "Tripulantes" com inicial maiúscula | **não se aplica**: o termo não aparece | **não se aplica**: o termo não aparece |
| Ausência de travessão | **cumprido**: nenhum `—` nem `–` | **cumprido**: nenhum `—` nem `–` |

As duas subseções já tratam associação e causa com cuidado explícito: a 4.4.2 separa odds ratio de
"efeito de intervenção" e a 4.4.3 tem a ressalva *"As regras descrevem associação e não causa"*. Os
apontamentos C1 a C5 são palavras isoladas que contradizem essas ressalvas, e não leituras causais
do argumento.

Fora do escopo, só como registro: a Seção 4.4.1 (#244) tem seis travessões, nos parágrafos que
começam com *"A Seção 4.1.3 definiu"*, *"em que $P$ é a precisão"* e *"O F2 avalia o rótulo"*. Ela
não é da dupla revisada.

### Leitura das métricas

**L1. 4.4.2, Métricas na validação, 4º parágrafo** (*"O modelo supera a meta de Precisão
Média..."*). O texto compara a distância do ROC-AUC à meta com a do primeiro candidato (*"uma
distância bem maior que os 0,0008 do primeiro candidato"*), e o parágrafo seguinte afirma que os
dois números *"não são diretamente comparáveis"*, porque um é da validação e o outro do teste. A
comparação é feita e desautorizada em seguida.

Sugestão: *"O ROC-AUC não depende de limiar, então esse argumento não se aplica a ele: os 0,7273
ficam 0,0227 abaixo da meta, e essa distância indica que a capacidade de ordenar deste modelo é
menor, e não que a meta seja inatingível."*

**L2. 4.4.2, Métricas na validação, 3º parágrafo** (*"O ganho da otimização está inteiro..."*). A
frase *"reponderar a classe desloca o ponto de corte, e não a capacidade de ordenar"* é mais forte
que o dado. Na Regressão Logística, `class_weight` muda todos os coeficientes, e não só o
intercepto; o que a tabela mostra é que a ordenação quase não mudou.

Sugestão: *"reponderar a classe desloca sobretudo o ponto de corte: a capacidade de ordenar, que é o
que Precisão Média e ROC-AUC medem, praticamente não muda."*

**L3. 4.4.2, Configuração final e método de otimização, 5º parágrafo** (*"Esse desvio relativiza o
resultado..."*). *"`C` e a penalidade são indiferentes nesta base"* generaliza para a base o que foi
medido em dois valores de `C`, depois da redução da grade registrada no 3º parágrafo.

Sugestão: *"`C` e a penalidade são indiferentes na grade testada."*

**L4. 4.4.2, Métricas na validação, 2º parágrafo** (*"A Sensibilidade acima vale no limiar padrão
de 0,5..."*). *"cerca de quatro vezes a capacidade de 50 contatos por dia"* compara uma fila do
período inteiro com uma capacidade diária, e o leitor não tem a conversão. O "quatro vezes" é a
razão 38.849 / 9.050.

Sugestão: *"ele produziria uma fila de 38.849 respostas, cerca de quatro vezes a fila de 9.050 que a
capacidade de 50 contatos por dia comporta."*

**L5. 4.4.3, Métricas na validação, 2º parágrafo** (*"Nesta árvore não houve troca entre
desempenho e interpretabilidade."*). A frase de abertura é contradita pelas seguintes: a
profundidade 3, a mais legível, fica abaixo da meta de Precisão Média, e a vencedora tem quatro
vezes mais folhas. Há troca, e o parágrafo mostra exatamente qual.

Sugestão: *"Nesta árvore, legibilidade custa métrica. A profundidade 5, vencedora pelo F2, também é
a melhor nas três métricas de negócio, e a profundidade 3, que daria o conjunto de regras mais
curto, fica abaixo da meta de Precisão Média (0,3955 contra 0,40). Entre as três, a árvore mais
legível que atende à meta mínima de Precisão Média é a de 16 folhas, e a de 32 folhas é a que o
critério do grupo escolhe."*

**L6. 4.4.3, Métricas na validação, 4º parágrafo** (*"O modelo supera a meta de Precisão Média e
fica abaixo das outras duas."*). *"A causa é estrutural"* afirma como causa um mecanismo que não foi
medido. O mecanismo, 32 valores distintos de score com empate dentro da folha, é real, mas nenhuma
célula mostra que é ele que produz a diferença para a Regressão Logística.

Sugestão: *"Uma explicação provável é estrutural: com 32 folhas, a árvore atribui no máximo 32
valores distintos de probabilidade, e todas as respostas de uma mesma folha empatam no score."*

**L7. 4.4.3, Explicabilidade por árvore visual e regras, 5º parágrafo** (*"Três ressalvas
acompanham a leitura acima."*). *"nos 83,7% de respostas sem cancelamento ela é nula"* herda o
número que o parecer do #195 apontou como divergente: a Seção 6.4 da logística mede, no mesmo
treino, 89,9% de nulos em `ANTECEDENCIA_CANCELAMENTO`. O número vem de
`interpretacao_arvore_decisao.md` e entrou no documento principal sem correção.

Sugestão: refazer a medição e, se confirmar a logística, *"nos 89,9% de respostas sem cancelamento
ela é nula e recebe a mediana do treino"*.

**L8. 4.4.3, Explicabilidade por árvore visual e regras, tabela, linha "Atraso na chegada".** *"em
torno de 3,7 vezes a prevalência do treino (20,43%)"* está correto (76,5 / 20,43 = 3,74), mas a
Seção 4.4.1 cita 20,44% como proporção de Detratores **da base**. Sem dizer qual é qual, o leitor vê
dois números para a mesma coisa.

Sugestão: manter *"prevalência do treino (20,43%)"* e acrescentar na primeira menção *"contra
20,44% na base inteira (Seção 4.2.1)"*.

### Afirmação causal

**C1. 4.4.2, Explicabilidade por odds ratio, tabela, coluna "Leitura".** *"Maior efeito do
modelo"* e *"Segundo maior efeito"*, logo depois do parágrafo que avisa que odds ratio não é
*"efeito de intervenção"*.

Sugestão: *"Maior odds ratio do modelo"* e *"Segundo maior odds ratio"*.

**C2. 4.4.2, Explicabilidade por odds ratio, 5º parágrafo** (*"A principal limitação do modelo é
estrutural..."*). *"o efeito do atraso sobre a detração depende do tier de fidelidade"*.

Sugestão: *"a associação entre atraso e detração varia com o tier de fidelidade"*. A mesma frase
aparece na abertura da Seção 4.4.4 e deve mudar junto, para que as duas subseções citem a Hipótese
5 com as mesmas palavras.

**C3. 4.4.3, Explicabilidade por árvore visual e regras, 3º parágrafo** (*"O piso de risco é a
folha..."*). *"É essa folha que explica por que a fila priorizada funciona"* afirma um resultado da
fila que nenhuma seção mediu.

Sugestão: *"É essa concentração que torna a fila priorizada viável: a maior parte dos Clientes está
num grupo de risco baixo e homogêneo, e os Detratores se concentram em poucas combinações de
fatores que a árvore separa explicitamente."*

**C4. 4.4.3, Explicabilidade por árvore visual e regras, tabela, linha "Fidelidade", coluna
"Leitura para a operação".** *"O Cliente Diamante detrata mais, e não menos, quando a viagem tem
problema"* descreve comportamento do Cliente a partir de uma taxa observada.

Sugestão: *"Com o mesmo problema na viagem, a taxa de detração do Cliente Diamante é maior, e não
menor."*

**C5. 4.4.3, Explicabilidade por árvore visual e regras, 5º parágrafo**, última frase. *"é essa
instabilidade que os ensembles da Seção 4.4.4 corrigem"*. Agregar árvores reduz a variância, mas
não elimina a instabilidade de cada árvore.

Sugestão: *"é essa instabilidade que os ensembles da Seção 4.4.4 reduzem ao agregar muitas
árvores."*

### Termos fora do checklist, na mesma leitura

**T1. 4.4.2, Explicabilidade por odds ratio, 4º parágrafo** (*"O achado sobre a antecedência do
aviso..."*). *"69,2% de detratores"* em minúscula; nas Seções 4.4.3 e 4.4.4, e no resto da 4.4.2,
"Detrator" e "Detratores" têm inicial maiúscula. Sugestão: *"69,2% de Detratores"*.

## CR02: definição e formato das métricas entre interpretáveis e ensembles

Comparação da 4.4.2 e da 4.4.3 com a subseção dos ensembles (4.4.4, #193) e com a seção do Gradient
Boosting no notebook (`notebooks/ensembles.ipynb`, Seção 11, #190), que ainda não tem subseção no
documento (#194). Formato numérico: as quatro fontes usam vírgula decimal e quatro casas nas
métricas, sem divergência.

| # | Divergência | Onde aparece cada forma | Proposta de termo único |
|---|---|---|---|
| D1 | Peso do recall no F2 | **"duas vezes"**: 4.4.1, parágrafo *"em que $P$ é a precisão"*. **"quatro vezes"**: `protocolo-avaliacao-artefato7.md` (*"O termo `4P` no denominador dá ao recall um peso quatro vezes maior"*); `regressao_logistica.ipynb`, Seção 5.2; `ensembles.ipynb`, Seção 11.4 | A formulação da 4.4.1, que é a de Van Rijsbergen: *"com beta = 2, o recall é tratado como duas vezes mais importante que a precisão; o fator β² = 4 é a forma como esse peso entra na média harmônica"* |
| D2 | Seção de origem das métricas de negócio | 4.4.2: *"As três métricas de negócio da Seção 4.3.2"*. 4.4.3 e a coluna de meta das duas: *"Seção 4.1.3"*. A 4.4.1 diz que a 4.1.3 as definiu e a 4.3.2 as aplicou | *"as três métricas de negócio da Seção 4.1.3"* |
| D3 | Nome do critério da busca | 4.4.2 e 4.4.3: *"F2 médio ... na validação cruzada"*. 4.4.4: *"F2 médio nos 5 folds (critério da busca)"*. Notebook de ensembles: *"melhor F2 médio nos folds"* | *"F2 médio na validação cruzada (5 folds)"* |
| D4 | Desvio entre folds | 4.4.2 reporta (0,0035); 4.4.3 não reporta; 4.4.4 pendente; notebook do Gradient Boosting reporta (0,0035) | Reportar sempre o desvio ao lado do F2 médio. Sem ele, a 4.4.3 não permite ler se 0,5078 se separa da segunda combinação |
| D5 | F2 como linha da tabela de métricas | 4.4.4 tem as linhas *"F2 médio nos 5 folds"* e *"F2 na validação"*. 4.4.2 e 4.4.3 só têm as três métricas de negócio e dizem que o F2 *"não é reportado como desempenho"* | Tabela com as três métricas de negócio, como a 4.4.1 fixa para a tabela comparativa; o F2 fica no texto do método de otimização. A 4.4.4 retira as duas linhas de F2 da tabela |
| D6 | Ordem das linhas | 4.4.2 e 4.4.3: Precisão Média, ROC-AUC, Sensibilidade. 4.4.4: Sensibilidade, Precisão Média, ROC-AUC | Sensibilidade, Precisão Média, ROC-AUC, a ordem em que a 4.1.3 e a 4.4.1 as enumeram |
| D7 | Coluna de meta | Presente na 4.4.2 e na 4.4.3 (*"Meta (Seção 4.1.3)"*); ausente na 4.4.4 | *"Meta (Seção 4.1.3)"* em todas as tabelas de métricas |
| D8 | Referência antes da busca | 4.4.2: *"Partida"*. 4.4.4: *"Linha de base (padrão da biblioteca)"*. 4.4.3: sem linha de base; as colunas são as profundidades 3 e 4 | *"Linha de base (padrão da biblioteca)"*. Na 4.4.3, dizer que a linha de base não foi medida, porque a chamada de `avaliar` da Seção 3 do notebook está comentada (#195, CR01.4) |
| D9 | Limiar da Sensibilidade | 4.4.2: *"limiar padrão de 0,5"*. 4.4.3: *"limiar de `predict()`"*. 4.4.4: não diz | *"no limiar de `predict()` do estimador (Seção 4.4.1)"*, que é a regra que a 4.4.1 fixa para todos os candidatos |
| D10 | Função que mede as métricas | 4.4.2: *"função `avaliar()` do card 05A.1"*. 4.4.4: *"`avaliar` (`src/avaliacao.py`)"*. 4.4.3: não nomeia a função | *"`avaliar` (`src/avaliacao.py`)"*. Ver a observação abaixo sobre a 4.4.3 |

**Observação sobre a comparabilidade da 4.4.3 (D10).** O parecer do #195 registrou que o notebook da
árvore não chama `avaliar` em nenhuma célula, e que as três métricas da profundidade 5 vêm de
`busca_arvore_decisao.md` e do JSON, sem célula de origem. A 4.4.3 apresenta esses números como
*"medidas na partição de validação"*, ao lado dos da Regressão Logística, sem dizer por qual função.
Enquanto a árvore não for medida por `avaliar`, nada garante que Precisão Média e ROC-AUC dela foram
calculadas da mesma forma que as dos outros candidatos, e é essa garantia que torna válida a tabela
comparativa da Seção 4.4.6. Proposta: medir o vencedor da árvore com `avaliar` antes da 4.4.6 e,
até lá, a 4.4.3 dizer de onde vêm os números.

**D1 atinge também a dupla revisora.** A forma "quatro vezes" está na Seção 11.4 do notebook de
ensembles, escrita no #190, além dos dois lugares dos interpretáveis e do protocolo. A correção é a
mesma nos quatro.

## Resumo

| Critério | 4.4.2 Regressão Logística | 4.4.3 Árvore de Decisão |
|---|---|---|
| CR01, checklist de texto | Cliente, Tripulantes e travessão **cumpridos**; métricas e vocabulário causal com **6 ajustes** (L1 a L4, C1 e C2), mais o termo T1 | Cliente, Tripulantes e travessão **cumpridos**; métricas e vocabulário causal com **7 ajustes** (L5 a L8, C3 a C5) |
| CR02, mesma definição e formato | **6 divergências** a ajustar: D1 (no notebook), D2, D6, D8, D9 e D10 | **4 divergências** a ajustar: D4, D6, D8 e D10, esta com as métricas medidas sem `avaliar` |
| CR03, parágrafo e redação | Todos os apontamentos acima indicam parágrafo e redação sugerida | Idem |

Das dez divergências de métrica, **D1, D3, D5, D7 e D9** pedem mudança na subseção dos ensembles
(4.4.4) ou no notebook de ensembles, e não na subseção revisada. A uniformização é das duas duplas,
e não só da revisada.

## Aplicado do lado dos ensembles neste card

A parte que cabe à dupla revisora foi aplicada no próprio #196, para que os comentários às duplas
revisadas apontem para um lado já uniformizado:

| Item | Onde | O que mudou |
|---|---|---|
| C2 | 4.4.4, 1º parágrafo | *"o efeito do atraso sobre a detração depende do tier"* passa a *"a associação entre atraso e detração varia com o tier"* |
| D1 | `ensembles.ipynb`, Seção 11.4, parágrafo "Leitura" | *"pesa o recall quatro vezes mais"* passa a *"trata o recall como duas vezes mais importante que a precisão (Seção 4.4.1 da documentação)"* |
| D3 | 4.4.4, texto e tabela de rastreabilidade; `ensembles.ipynb`, tabelas de registro das Seções 8.3 e 11.4 e parágrafo "Leitura" da 11.4 | *"F2 médio nos 5 folds"* e *"Melhor F2 médio nos folds"* passam a *"F2 médio na validação cruzada (5 folds)"*. As células que transcrevem o output impresso (`melhor F2 médio nos folds`) ficam como estão, para o texto continuar batendo com o output |
| D5 | 4.4.4, Hiperparâmetros vencedores e métricas | As duas linhas de F2 saem da tabela de métricas e o F2 da validação cruzada, com o desvio entre folds, vai para o texto |
| D7 | 4.4.4, mesma tabela | Coluna *"Meta (Seção 4.1.3)"* acrescentada |
| D9 | 4.4.4, depois da mesma tabela | Frase dizendo que a Sensibilidade vale *"no limiar de `predict()` do estimador (Seção 4.4.1)"* |

D6 já estava na ordem proposta na 4.4.4. A forma "quatro vezes" de
`protocolo-avaliacao-artefato7.md` (D1) é do protocolo da dupla de Métricas e Decisões e fica
registrada aqui sem alteração.
