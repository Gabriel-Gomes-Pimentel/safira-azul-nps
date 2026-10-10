# Decisão sobre explicabilidade: por que o grupo não adota SHAP

Registro de decisão sobre a técnica de explicabilidade usada na comparação de modelos da Seção 4.4.

| | |
|---|---|
| Card | #215 |
| Situação | Decidido |
| Data | 24/09/2026 |
| Decidido por | Grupo G01, em conversa registrada no canal da equipe |
| Justificativa revista em | 24/09/2026, na revisão do !121, que trocou o motivo principal em relação ao plano original do card |
| Consumido por | #214 (subseção da Regressão Logística), #193 e #194 (subseção dos ensembles), #251 (leitura da tabela comparativa) |

Este documento **decide e justifica**. Ele não redige a Seção 4.4 nem implementa explicabilidade
nenhuma: as implementações são as dos cards listados na seção 2.

## 1. A decisão

O grupo não adota SHAP na entrega do ART.7. A explicabilidade exigida pelo barema é atendida por
duas vias já implementadas, descritas na seção 2, e a ausência de SHAP é escolha de método
registrada aqui, não lacuna de execução.

A decisão foi tomada em 24/09/2026, depois de os odds ratio do card #212 estarem prontos. A ordem
importa: só com a explicabilidade intrínseca medida foi possível avaliar o que SHAP acrescentaria
sobre ela.

## 2. As duas vias de explicabilidade que o grupo entrega

| Via | O que produz | Onde está |
|---|---|---|
| **Intrínseca** | Odds ratio por feature original da Regressão Logística, com nível de referência declarado e suporte por coluna | Seções 6 e 7 de `notebooks/regressao_logistica.ipynb`, cards #212 e #213 |
| **Intrínseca** | Árvore de decisão visual | Cards #228 e #229, em execução |
| **Post-hoc** | Permutation importance do melhor ensemble | Card #191, em `develop` |
| **Post-hoc** | Gráficos de dependência parcial | Card #192, em execução |

Na data desta decisão, só a explicabilidade intrínseca da Regressão Logística e a permutation
importance do #191 estão em `develop`. A árvore visual e a dependência parcial estão sendo
entregues, e este documento não as dá como concluídas.

O barema do ART.7 exige que **ao menos um** dos modelos supervisionados apresente explicabilidade.
A via intrínseca da Regressão Logística já cumpre esse requisito sozinha.

## 3. O motivo: o que o SHAP acrescentaria, e o que entregá-lo exigiria

O requisito do ART.7 é que **ao menos um** dos modelos supervisionados apresente explicabilidade, e
a via intrínseca da Regressão Logística já o atende sozinha, com os odds ratio por feature original
do card #212. A pergunta que resta não é se a entrega fica sem explicabilidade, e sim o que o SHAP
acrescentaria sobre o que já existe.

**O que ele acrescentaria é a atribuição por observação.** As vias adotadas explicam o
comportamento do modelo sobre o conjunto: quais variáveis ele usa para ordenar e quanto. Nenhuma
responde por que um Cliente específico ocupa a posição que ocupa na fila. O SHAP responde.

**Entregar isso com responsabilidade exigiria duas coisas que não couberam nesta sprint.**

A primeira é somar os blocos de colunas colineares antes de reportar qualquer atribuição. O
cancelamento ocupa sete colunas da matriz, e uma atribuição individual que mostrasse "faltou o
tempo de voo" em vez de "o voo foi cancelado" estaria tecnicamente correta e operacionalmente
inútil. O SHAP é aditivo, então a soma do bloco é bem definida e resolve isso, e a biblioteca ainda
oferece o `PartitionExplainer` para features agrupadas. O trabalho existe, é o mesmo mapeamento de
blocos que o card #212 já fez, mas precisa ser feito e conferido.

A segunda, e a que pesa mais, é **validar com a área de Experiência do Cliente como cada
atribuição vira um motivo operacional**. Entregar a quem faz o contato uma lista de variáveis com
pesos não é entregar um motivo: alguém precisa definir, com a área, qual vocabulário faz sentido no
telefone e o que fazer quando a atribuição aponta para algo sobre o qual a operação não tem ação.
Sem essa validação, uma explicação por Cliente que parece precisa e não é sai pior do que a ausência
de explicação individual, porque quem opera a lista não tem como duvidar dela.

A decisão é adiar o SHAP até que essas duas condições existam, e não descartá-lo como técnica.

## 4. A colinearidade limita todas as vias, inclusive as adotadas

Este documento não usa a colinearidade como motivo para rejeitar o SHAP, e é importante registrar
por quê: **ela atinge mais as vias que o grupo adotou do que atingiria o SHAP.**

A permutation importance do card #191 permuta uma coluna crua por vez. Como o contrato apaga
`TEMPO_VOO`, `ESTATISTICA_ATRASOSAIDA`, `ATRASO_CHEGADA` e `N_TRECHOS` em toda linha de voo
cancelado, o cancelamento está em seis colunas cruas, e permutar uma delas deixa as outras cinco
carregando o mesmo fato. Medindo na Regressão Logística, sobre a validação, com cinco repetições:

| Coluna permutada | Queda de ROC-AUC |
|---|---|
| `CANCELAMENTO_VOO` | -0,0001 |
| `ANTECEDENCIA_CANCELAMENTO` | +0,0055 |
| `TEMPO_VOO` | +0,0215 |
| `ESTATISTICA_ATRASOSAIDA` | +0,0660 |
| `ATRASO_CHEGADA` | +0,0079 |
| `N_TRECHOS` | +0,0002 |
| **soma das individuais** | **+0,1010** |
| **as seis juntas, mesma permutação** | **+0,1353** |

<div align="center"><sup>Fonte: Autoria própria.</sup></div>

Esta medição é reproduzível na **Seção 8 de `notebooks/regressao_logistica.ipynb`**, que é a
célula de onde os números acima saem.

A coluna que nomeia o cancelamento aparece com importância **zero**, e o bloco vale 1,34 vez a soma
das partes, o que mostra que a permutation importance não é aditiva: somar as importâncias
individuais depois também não recupera o efeito. O comportamento é documentado pelo `scikit-learn`
como *"Misleading values on strongly correlated features"*.

**Esta medição é da Regressão Logística, com ROC-AUC.** O card #191 mede o melhor ensemble, com
F2. O mecanismo é o mesmo, porque o pré-processador e os indicadores de ausência são os mesmos para
todos os candidatos, mas **o número que os cards #193, #194 e #251 citarem precisa sair do modelo
deles**, com a permutação conjunta do bloco. Reusar os valores desta tabela para descrever os
ensembles seria trocar um erro de leitura por outro.

A dependência parcial do card #192 tem limitação própria: ela supõe independência entre features, e
variar `TEMPO_VOO` numa linha de voo cancelado produz uma combinação que não existe na base.

**A leitura de qualquer das três vias deve ser feita por bloco**, e não coluna a coluna. Os blocos
mapeados pelo card #212 são: as sete colunas de cancelamento, o par `VOO_TIPO` = CONEXÃO com
`TIPO_ENTRETENIMENTO` = NAO_INFORMADO, e os dois indicadores de ausência de `HIST_DETRATOU_ANTES` e
`HIST_TAXA_DETRACAO_ANTERIOR`.

## 5. O custo computacional, dito com precisão

A base tem **484.915 registros**, e é sobre esse número que o argumento de custo costuma ser
construído. Ele não sustenta a decisão, e vale registrar por quê, para que ninguém o reapresente:

- **O KernelSHAP explica uma amostra, não a base inteira.** O uso descrito na seção 6, dar a quem
  faz o contato o motivo daquele caso, precisaria apenas das respostas efetivamente contatadas.
  Pela premissa de capacidade do documento, são 50 contatos por dia, ou cerca de 9.050 no semestre.
  Nessa escala o KernelSHAP cabe.
- **Para os ensembles existe o TreeSHAP**, exato e de custo polinomial baixo
  (LUNDBERG et al., 2020), e todos os candidatos da outra dupla são baseados em árvore.
- **Para a Regressão Logística existe o `LinearExplainer`**, que é exato e praticamente gratuito.

O custo só impediria o KernelSHAP aplicado às 484.915 linhas, que é um uso que ninguém propôs.
**Ele não é motivo da decisão.**

## 6. O que se perde com a decisão

A entrega fica **sem atribuição por observação individual**. As duas vias adotadas explicam o
comportamento do modelo sobre o conjunto: quais variáveis ele usa para ordenar e quanto. Nenhuma
delas responde por que um Cliente específico ocupa a posição que ocupa na lista priorizada.

Para a operação de Experiência do Cliente isso tem custo concreto: quem faz o contato recebe a
posição na fila, mas não o motivo daquele caso.

Vale ser preciso sobre o tamanho da perda. **Os dois modelos interpretáveis já permitem a leitura
individual sem SHAP nenhum**, e por vias diferentes: a Regressão Logística decompõe cada observação
pela própria forma linear, somando a contribuição de cada coluna por bloco, e a Árvore de Decisão
explica cada observação pelo caminho que ela percorre até a folha.

**Nenhuma das duas foi entregue nesta sprint.** Então a decisão custa duas coisas: a atribuição por
observação **nos ensembles**, onde ela exigiria SHAP e não tem alternativa; e a padronização de uma
mesma leitura individual entre todos os candidatos, que hoje não existe para nenhum deles.

## 7. Divergência a resolver no documento

A matriz de riscos da Seção 4.1.5 registra, no risco R04, a mitigação *"apresentar os resultados do
SHAP como associação, não causalidade"*. Esse texto foi escrito quando o uso de SHAP era previsto e
**ficou inconsistente com esta decisão**.

O ajuste não é feito aqui, porque a Seção 4.1.5 é de outra autoria e este card não reescreve seção
de colega. Fica registrado para o card de revisão editorial da entrega.

## 8. Referências

LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. In: CONFERENCE
ON NEURAL INFORMATION PROCESSING SYSTEMS, 31., 2017, Long Beach. **Advances in Neural Information
Processing Systems 30**. Red Hook: Curran Associates, 2017. p. 4765-4774.

LUNDBERG, S. M. et al. From local explanations to global understanding with explainable AI for
trees. **Nature Machine Intelligence**, v. 2, n. 1, p. 56-67, 2020.

SHAPLEY, L. S. A value for n-person games. In: KUHN, H. W.; TUCKER, A. W. (org.). **Contributions
to the theory of games**, v. 2. Princeton: Princeton University Press, 1953. p. 307-317.
