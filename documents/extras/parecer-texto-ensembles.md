# Parecer de texto e coerência de métricas da subseção dos ensembles

Revisão cruzada do card #236, sobre a subseção dos ensembles da Seção 4.4 (`documents/documentacao.md`).
Escopo de texto: clareza, ortografia e gramática, citações no padrão ABNT e se as métricas seguem a
mesma definição e o mesmo formato do protocolo de avaliação do card 09
([`protocolo-avaliacao-artefato7.md`](protocolo-avaliacao-artefato7.md)). A reprodutibilidade e a
conferência técnica são do #216 ([`parecer-reprodutibilidade-ensembles.md`](parecer-reprodutibilidade-ensembles.md))
e não são refeitas aqui.

Revisado em 2026-09-25, sobre a `develop` em `06c1e5a`. Segue a forma dos pareceres do #195 e do
#196: veredito explícito por item e, em cada apontamento, o parágrafo afetado, o trecho atual e uma
redação sugerida.

**Como os parágrafos são indicados.** Pelo subtítulo de quinto nível da Seção 4.4.4 e pela posição
do parágrafo dentro dele, contando só parágrafos de texto (tabelas não contam). O início de cada
parágrafo citado vem transcrito para que ele seja encontrado mesmo que a numeração de linhas mude.

## Escopo efetivamente coberto

O card previa revisar a subseção escrita nos cards **#193** (Felipe, Random Forest e permutation
importance) e **#194** (Fernanda, Gradient Boosting, PDPs e comparação entre ensembles).

Só a parte do #193 existe na `develop`: a MR **!135** entregou a Seção 4.4.4. O #194 continua em
`Backlog`, sem MR, e a subseção do Gradient Boosting ainda não foi escrita. **Este parecer cobre a
Seção 4.4.4 e deixa o #194 explicitamente fora**, para que a ausência não seja lida como aprovação,
como o #216 já fez no parecer técnico.

A MR **!146** (#196, ainda aberta) altera a tabela de métricas e o 1º parágrafo da 4.4.4. Os
apontamentos abaixo são sobre a `develop`; quando a !146 já resolve um deles, isso está indicado,
para que a correção não seja feita duas vezes.

## Veredito por item

| Item | Seção 4.4.4 Random Forest | Seção 4.4.5 Gradient Boosting (#194) |
|---|---|---|
| Clareza do texto | **ajustar**: 5 apontamentos (T1 a T5) | **não avaliável**: não escrita |
| Ortografia e gramática | **cumprido**, com 1 ajuste de pontuação (G1) | **não avaliável** |
| Citações ABNT | **cumprido** no texto; **ajustar** a identificação das tabelas (A1) | **não avaliável** |
| Consistência de métricas com o card 09 | **ajustar**: 3 apontamentos (M1 a M3) | **não avaliável** |

O texto da 4.4.4 é claro na maior parte: cada decisão vem com o motivo (busca aleatória contra grade,
`GroupKFold` por Cliente, `n_jobs=1`), a permutation importance é apresentada com a limitação de
leitura não causal e com o efeito de features correlacionadas, e todo número tem rastreabilidade.
Os apontamentos são pontuais.

## Clareza do texto

**T1. Menções a MR e card dentro do texto acadêmico.** Hiperparâmetros vencedores e métricas,
3º parágrafo (*"Esta MR sozinha não fecha o card."*) e 4º parágrafo (*"Condição para que os números
sejam comparáveis (corrigida nesta MR)."*). O documento é lido pelo parceiro e pela banca sem o
contexto do GitLab; "esta MR", "o card" e "bloqueia a revisão cruzada dos cards #195 e #196" são
registro de processo, que já está no board e no parecer do #216.

Sugestão para o 3º parágrafo: *"As duas tabelas acima ainda não têm valores de resultado: nenhum
hiperparâmetro vencedor nem métrica existe até a execução das Seções 7.1, 8.1 e 8.2 do notebook, que
estão prontas e cobertas por `tests/test_busca_random_forest.py`, mas dependem da base analítica
real, que não é versionada no repositório por compromisso com o parceiro. Nenhum valor foi estimado:
os campos serão preenchidos com o output dessas células, número a número."* No 4º parágrafo, trocar o
título por *"Condição para que os números sejam comparáveis."* e manter o restante.

**T2. "Treino inteiro de cada fold".** Limitações, 2º parágrafo (*"Custo computacional. O Random
Forest é o candidato mais caro..."*). O trecho *"sobre o treino inteiro de cada fold"* é
contraditório: dentro de um fold a floresta é ajustada só sobre as linhas de ajuste daquele fold
(quatro quintos do treino), como a própria subseção explica em Pipeline. Só o reajuste final usa o
treino inteiro.

Sugestão: *"cada uma das 201 execuções treina de 200 a 600 árvores com `n_jobs=1`, sobre as linhas de
ajuste de cada fold ou, no reajuste final, sobre o treino inteiro."*

**T3. De onde vem o mínimo de 40 combinações.** Método de otimização, 1º parágrafo (*"Os
hiperparâmetros não foram escolhidos manualmente."*). *"Quarenta é o mínimo definido para os dois
ensembles"* não diz quem definiu nem onde. O leitor não encontra essa regra em outra seção.

Sugestão: *"Quarenta é o mínimo que a dupla fixou para os dois ensembles, e
`busca_random_forest.criar_busca_random_forest` recusa qualquer valor menor"*, ou apontar o registro
da decisão, se houver.

**T4. Afirmação sem medição sobre `n_repeats`.** Tabela de configuração da permutation importance,
linha `n_repeats` (*"Com menos, o desvio da queda fica instável demais para separar a terceira da
quarta feature"*). A frase afirma um resultado que ainda não foi medido (a importância está pendente
da execução). Lida hoje, ela parece uma observação empírica.

Sugestão: *"Dez repetições dão um desvio da queda estimável para cada feature; com menos, posições
vizinhas do ranking tendem a não se separar."*

**T5. "Seções 10 e 11" sem dizer de quê.** Hiperparâmetros vencedores e métricas, 4º parágrafo
(*"...os mesmos cortes usados pela Regressão Logística e pelo Gradient Boosting nas Seções 10 e
11."*). Num parágrafo que também cita a Seção 4.3 da documentação, "Seções 10 e 11" pode ser lido
como seções do documento, que não existem.

Sugestão: *"...nas Seções 10 e 11 do notebook."*

## Ortografia e gramática

Não há erro de ortografia nem de concordância na 4.4.4. "Cliente" aparece sempre com inicial
maiúscula. Um ajuste de pontuação:

**G1. Travessão.** Hiperparâmetros vencedores e métricas, 4º parágrafo (*"...que fixa `2025-07-01` e
`2026-01-01` — os mesmos cortes usados..."*). É o único travessão da 4.4.4, e a convenção de texto
do grupo, conferida pelo #196 na subseção dos interpretáveis, é não usá-lo.

Sugestão: *"...que fixa `2025-07-01` e `2026-01-01`, os mesmos cortes usados..."*

Registro, sem pedido de mudança: 1º parágrafo da subseção, *"o efeito do atraso sobre a detração
depende do tier"*, é leitura causal de uma associação. A !146 já troca por *"a associação entre
atraso e detração varia com o tier de fidelidade"*, e o apontamento não é repetido aqui.

## Citações ABNT

As três citações do texto seguem a NBR 10520 no formato autor-data em caixa alta: (BREIMAN, 2001),
(PEDREGOSA et al., 2011) e (BERGSTRA; BENGIO, 2012). As três têm entrada correspondente na Seção 6,
Referências, no formato da NBR 6023. Nenhuma afirmação que pediria citação ficou sem fonte: a
permutation importance é atribuída a Breiman (2001), onde ela foi proposta, e a busca aleatória a
Bergstra e Bengio (2012). **Cumprido.**

**A1. Tabelas sem identificação.** As seis tabelas da 4.4.4 têm a fonte embaixo (*"Fonte: Autoria
própria."*), mas nenhuma tem o título acima, que a norma de apresentação tabular adotada pelo
documento exige e que a Seção 4.2 já usa (*"Tabela 1 — Diagnóstico de valores extremos na base
integrada"*). Sem título, o texto também não consegue se referir a uma tabela pelo número e usa
"a tabela acima".

Sugestão: acrescentar título acima de cada tabela, no formato já usado no documento, por exemplo
*"Tabela N — Espaço de busca do Random Forest"*, *"Tabela N — Hiperparâmetros vencedores do Random
Forest"*, *"Tabela N — Métricas do Random Forest na validação"*, com a numeração seguindo a do
documento. A mesma falta existe na 4.4.2; como é de outra dupla, fica só o registro.

## Consistência de métricas com o card 09

Nomes: a 4.4.4 usa *"Sensibilidade (Recall) na classe Detrator"*, *"Precisão Média (Average
Precision)"* e *"ROC-AUC"*, exatamente os nomes do protocolo do card 09 e da Seção 4.4.1. O F2 é
apresentado como critério de busca e não como métrica de negócio, como o protocolo define.
**Nomes cumpridos.** Os três apontamentos são de formato.

**M1. F2 misturado com as métricas de negócio na mesma tabela.** Tabela de métricas do vencedor (a
que tem as colunas *"Linha de base"* e *"Vencedor da busca"*). As duas primeiras linhas são F2, e as
três seguintes são as métricas de negócio. O protocolo do card 09 diz que o F2 *"não substitui
Sensibilidade, Precisão Média ou ROC-AUC na tabela"*, e a 4.4.2 reporta só as três. Com o F2 na
mesma tabela, a 4.4.6 herda duas formas diferentes de ler os candidatos.

**Resolvido pela !146**, que tira o F2 da tabela e o registra num parágrafo à parte. Basta a !146
entrar.

**M2. Tabela sem a coluna de meta e em ordem diferente da 4.4.2.** Mesma tabela. A 4.4.2 tem a coluna
*"Meta (Seção 4.1.3)"* com ≥ 0,70, ≥ 0,40 e ≥ 0,75, e a 4.4.4 não tem; sem ela, o leitor não vê se o
Random Forest atinge as metas sem voltar à 4.1.3. A !146 acrescenta a coluna. A ordem das linhas
continua diferente entre as duas subseções (4.4.2: Precisão Média, ROC-AUC, Sensibilidade; 4.4.4:
Sensibilidade, Precisão Média, ROC-AUC). A ordem da 4.4.4 é a do protocolo e da 4.4.1, então o
ajuste de ordem cabe à 4.4.2, e fica registrado para a tabela consolidada da 4.4.6 usar a ordem do
protocolo.

**M3. Formato numérico e limiar da Sensibilidade ainda não declarados.** Os valores estão pendentes,
mas o texto já pode fixar como eles entram: **quatro casas decimais com vírgula**, o formato do card
09 e da 4.4.2 (0,4464, 0,5011), e a Sensibilidade acompanhada do limiar em que foi medida, porque sem
limiar ela não tem definição (4.4.2, *"A Sensibilidade acima vale no limiar padrão de 0,5"*). A !146
acrescenta a frase do limiar; o número de casas decimais continua sem registro. O mesmo vale para a
queda de F2 da permutation importance, que deve sair com o desvio entre as 10 repetições, no mesmo
formato.

Sugestão, logo abaixo da tabela de métricas: *"Os valores são reportados com quatro casas decimais,
no formato das demais subseções desta seção."*

## Resumo para a dupla revisada

| Apontamento | Onde | Já resolvido por outra MR? |
|---|---|---|
| T1 | Hiperparâmetros vencedores e métricas, 3º e 4º parágrafos | não |
| T2 | Limitações, 2º parágrafo | não |
| T3 | Método de otimização, 1º parágrafo | não |
| T4 | Tabela da permutation importance, linha `n_repeats` | não |
| T5 | Hiperparâmetros vencedores e métricas, 4º parágrafo | não |
| G1 | Hiperparâmetros vencedores e métricas, 4º parágrafo | não |
| A1 | As seis tabelas da 4.4.4 | não |
| M1 | Tabela de métricas do vencedor | sim, !146 |
| M2 | Tabela de métricas do vencedor | coluna de meta sim, !146; ordem fica para a 4.4.2 e a 4.4.6 |
| M3 | Tabela de métricas do vencedor e ranking da permutation importance | limiar sim, !146; casas decimais não |

Nenhum apontamento muda número, método ou conclusão da 4.4.4. Quando o #194 for entregue, a
subseção do Gradient Boosting precisa passar pelos mesmos quatro itens antes da tabela consolidada
da 4.4.6.
