# Política de particionamento temporal dos conjuntos

Registro de decisão da divisão da base analítica em treino, validação e teste.

| | |
|---|---|
| Card | #127 |
| Situação | Decidido |
| Data | 01/09/2026 |
| Consumido por | #113 (implementação do split), #116 (item (a) da Seção 4.3), #131 (congelamento dos splits) |
| Implementação que recebe estes parâmetros | `src/split.py`, entregue em #128 |

Este documento **decide e justifica**. Ele não implementa o corte nem redige a
Seção 4.3. Ele é a fonte única das datas de corte: `src/split.py` recebe essas
datas como parâmetro e deliberadamente não as fixa em constante, para que não
exista uma segunda fonte de verdade capaz de divergir desta.

---

## 1. A decisão

A base analítica é dividida por **data prevista de partida do voo** (`DATA_STD`),
em três conjuntos contíguos e sem sobreposição:

| Conjunto | Início | Fim | Meses datados | Estação coberta |
|---|---|---|---:|---|
| Treino | 01/07/2023 | 30/06/2025 | 24 | dois ciclos anuais completos |
| Validação | 01/07/2025 | 31/12/2025 | 6 | pico de julho e pico de dezembro |
| Teste | 01/01/2026 | 30/06/2026 | 6 | pico de janeiro e baixa de fevereiro a maio |

As duas datas de corte são **01/07/2025** e **01/01/2026**. O treino não tem data
de início própria: é tudo o que precede o primeiro corte. A menor data
efetivamente presente na base é 01/07/2023, confirmando o que a Seção 4.2.1
declara; ver a nota de correção na Seção 4 sobre o registro anterior desta
política, que dizia o contrário.

**Convenção de intervalo:** fechado à esquerda e aberto à direita. Um voo datado
exatamente em 01/07/2025 pertence à validação, não ao treino. Sem essa convenção
declarada, o dia do corte cairia em dois conjuntos ou em nenhum, conforme o
operador de comparação que o implementador escolhesse.

**Chamada correspondente em `src/split.py`:**

```python
particoes, metadados = dividir(
    df,
    corte_validacao="2025-07-01",
    corte_teste="2026-01-01",
    sem_data="treino",   # condicionado à verificação da Seção 5
)
```

## 2. Proporções-alvo

As proporções abaixo não são meta escolhida a priori: são a composição **medida**
sobre a base analítica real com estas duas datas de corte, reportada em !52.

| Conjunto | n | Proporção | Clientes | Período | Taxa do alvo | Faixa de alarme |
|---|---:|---:|---:|---|---:|---|
| Treino | 341.962 | 77,1% | 307.510 | 01/07/2023 a 30/06/2025 | 20,43% | 72% a 82% |
| Validação | 48.301 | 10,9% | 47.560 | 01/07/2025 a 31/12/2025 | 21,60% | 8% a 14% |
| Teste | 53.486 | 12,1% | 52.069 | 01/01/2026 a 30/06/2026 | 20,41% | 9% a 15% |

**O que é vinculante são as datas, não as proporções.** As proporções são
consequência das datas e não podem ser ajustadas para caber na faixa: o volume
mensal de respostas varia, e a regra de desempate por Cliente da Seção 6 remove
linhas do treino em quantidade que só se conhece em execução. A faixa serve como
alarme, não como meta. Se a composição observada cair fora dela, a resposta é
reabrir este registro e mover uma data de corte com justificativa escrita, nunca
reamostrar até a proporção fechar.

**O treino concentra 77,1% da base com 24 dos 36 meses do horizonte (66,7%)**,
uma diferença menor que a de qualquer partição puramente proporcional aos meses
teria, porque o volume mensal de respostas não é uniforme ao longo dos três
anos. Não há bloco sem data a explicar essa diferença: a base atual tem
`DATA_STD` preenchida em 100% dos registros (ver a nota de correção na Seção 4).

**A evidência de que os cortes são adequados está na taxa do alvo, não no
tamanho.** Um corte temporal pode produzir partições de tamanho correto e
prevalências muito diferentes, e é a diferença de prevalência, não a de tamanho,
que impede comparar métricas entre conjuntos. As três taxas ficam entre 20,41% e
21,60%, num intervalo de 1,2 ponto percentual, e todas próximas dos 20,44% do
período completo registrados na Seção 4.2.1. As métricas de validação e de teste
são, portanto, lidas sobre populações de risco comparáveis.

A composição efetiva de cada execução fica registrada nos metadados devolvidos
por `dividir` e na tabela de `resumo`, que reporta n, intervalo de datas,
Clientes distintos e taxa do alvo por partição.

## 3. Por que o corte é temporal, e não aleatório

**Porque a divisão aleatória mede uma tarefa que a operação nunca enfrenta.** O
Safira pontua o risco de detração no intervalo entre a realização do voo e a
resposta à pesquisa, que é enviada um dia após o voo e permanece aberta por sete
dias (Seção 4.1.3). O modelo sempre prevê para frente, a partir do que se sabe
até hoje. Embaralhar as linhas colocaria voos de 2026 no treino e voos de 2024 no
teste, e a métrica resultante descreveria a habilidade de prever o passado
conhecendo o futuro. Essa métrica seria melhor que a real e não teria contraparte
em produção.

**Porque o efeito de período vazaria para os três conjuntos.** A Seção 4.2.1
documenta um choque em 2024Q4, quando a taxa de detratores chegou a 32,58% contra
20,44% no período completo, e mostra que ele não é explicado pela composição
operacional: a detração subiu inclusive entre voos pontuais, de 16,6% em 2024Q3
para 25,4% em 2024Q4. É conjuntura reputacional, não sinal do voo. Sob divisão
aleatória, o mesmo trimestre atípico apareceria nos três conjuntos e o modelo
seria avaliado sobre um período que já viu.

**Porque a dependência intracliente sobreviveria ao corte por data.** A Seção
4.2.1 registra que 26,8% das respostas vêm de Clientes que responderam mais de
uma vez, e a Hipótese 4 da Seção 4.2.4 estima, sobre 77.673 respostas de Clientes
com histórico, que a chance de detratar entre quem já detratou é 4,61 vezes a
chance entre quem não detratou (IC 95% de 4,43 a 4,79; p < 0,001), já descontados
atraso na chegada e cancelamento. Um mesmo Cliente presente no treino e no teste faria o
modelo reconhecer a pessoa em vez de aprender o fenômeno. Por isso o corte é
temporal **e** por Cliente, conforme a Seção 6.

Esta política atende ao plano de resposta do risco **R01** da matriz da Seção
4.1.5, que trata o desempenho artificialmente alto como sinal de alerta e não
como resultado.

## 4. Por que estas datas

> **Correção (após o #92):** a versão anterior desta seção afirmava que a base
> tinha só 30 meses datados, com 120.000 linhas (cerca de um quarto do total)
> sem `DATA_STD`, e concluía daí que o horizonte real divergia do declarado pela
> Seção 4.2.1. Essa leitura estava correta sobre a base medida em !52, mas a
> causa era um bug de concatenação em `normalizar_data_std` — a primeira
> partição de `NPS_01` chegava como `Timestamp` do Excel e as demais como texto
> de CSV, e a mistura de tipos produzia `NaT` para 120 mil linhas que tinham
> data de origem. O fix(#92) corrigiu a normalização por fonte antes da
> concatenação. Com a base regerada, `DATA_STD` está 100% preenchida e a menor
> data é 01/07/2023: **a Seção 4.2.1 estava certa desde o início**, e o parágrafo
> abaixo, que antes contradizia o documento principal, foi reescrito para
> confirmá-lo. As datas de corte (01/07/2025, 01/01/2026) não mudam: a
> composição das partições que a Seção 2 mede é a mesma de antes, porque as
> linhas que ganharam data já estavam sendo enviadas ao treino pela política da
> Seção 5, só que por um caminho diferente.

**Horizonte real, confirmando o declarado.** A base analítica tem 484.915
registros e 407.139 Clientes distintos, com cobertura de 01/07/2023 a
30/06/2026, 36 meses completos — exatamente o que a Seção 4.2.1 declara. As
datas de corte foram escolhidas sobre esse horizonte, dividindo os 36 meses em
24 + 6 + 6.

**Sazonalidade da demanda.** O mercado doméstico brasileiro tem alta em janeiro,
julho e dezembro, e baixa entre fevereiro e maio. Isso restringe as datas de duas
maneiras que uma escolha por proporção pura ignoraria:

1. **Validação e teste não podem cair dentro de uma única estação.** Este é o
   erro que a janela mais curta convida. Um teste de três meses no fim da base
   cobriria abril, maio e junho de 2026, ou seja, baixa temporada inteira, e
   mediria o modelo num regime de demanda que não representa o ano. Com seis
   meses, o teste contém janeiro, mês de pico, e o bloco de baixa de fevereiro a
   maio; a validação contém julho e dezembro, os outros dois picos, e a baixa de
   agosto a novembro. **Cada conjunto de avaliação vê pico e vale**, que é a
   condição para a métrica não ser artefato de estação.
2. **O treino cobre exatamente dois ciclos anuais completos.** Seus 24 meses vão
   de julho de 2023 a junho de 2025: cada mês do calendário aparece duas vezes,
   sem desequilíbrio entre estações. Isso não era garantido pela escolha das
   datas de corte — é consequência de o horizonte real ter, de fato, os 36 meses
   que a Seção 4.2.1 sempre declarou.

**Posição do choque de 2024Q4.** O trimestre atípico cai inteiramente dentro do
treino. É o lugar correto: o modelo aprende que períodos de choque reputacional
existem, sem que a métrica de teste seja inflada ou deprimida por um evento único
e irrepetível.

**Recência do teste.** O teste ocupa os seis meses mais recentes da base. É a
melhor aproximação disponível do regime em que o modelo vai operar, e é a única
posição que permite ler a métrica de teste como estimativa de desempenho futuro
em vez de retrospectiva.

## 5. Registros sem data válida

> **Correção (após o #92):** a base analítica atual não tem nenhuma linha sem
> `DATA_STD` — `linhas_sem_data = 0`. O bloco de 120.000 linhas descrito
> originalmente nesta seção era efeito do bug de concatenação corrigido pelo
> fix(#92) (ver a nota da Seção 4), não uma característica permanente da fonte.
> A regra e o mecanismo abaixo continuam em vigor como salvaguarda: nada impede
> que uma fonte futura volte a chegar sem a coluna, e a política precisa
> continuar valendo se isso acontecer. A antiga "correção pendente na
> documentação" que ficava aqui listava três pontos; dois eram efeito deste
> bug e foram retirados, porque a Seção 4.2.1 sempre esteve correta. O
> terceiro — a Seção 4.1.3 descreve a fonte com 98.414 respostas, e a Seção
> 4.2.1 descreve a base analítica com 484.915 registros, sem explicar a
> diferença — não é efeito do #92 e segue sem solução, registrado em #184.

**Regra:** vão para o **treino**, condicionada à verificação de anterioridade
descrita abaixo. Se a verificação falhar, são **excluídos** dos três conjuntos.

Nunca vão para validação ou teste. Nesses dois conjuntos a data é necessária para
situar cada linha dentro do período avaliado, e não apenas antes dele.

Uma linha sem `DATA_STD` não é data corrompida: `normalizar_data_std` levanta
exceção diante de data inválida, então o que ficaria ausente é ausência de
origem, não erro de conversão. Descartar linhas sem ganho de rigor é caro o
bastante para exigir prova antes de simplesmente excluir, e a prova está
implementada em `verificar_anterioridade_sem_data`, que exige duas condições
simultâneas sobre qualquer bloco sem data que aparecer:

1. `RESPONDENT_ID` acompanha a ordem cronológica nas linhas datadas, com
   correlação de Spearman de no mínimo 0,99;
2. o maior `RESPONDENT_ID` sem data é menor que o menor `RESPONDENT_ID` datado.

Valendo as duas, os blocos não se sobrepõem e as linhas sem data precedem todo o
período datado. Nesse caso elas são as observações mais antigas disponíveis, que
é exatamente o lugar de dados de treino. A função levanta `AssertionError` quando
qualquer condição falha, de modo que uma base futura com outro padrão de ausência
não herda em silêncio uma conclusão que valia para esta.

**A verificação é obrigatória a cada execução, e não uma vez.** `sem_data="treino"`
só pode ser usado depois de `verificar_anterioridade_sem_data` passar na mesma
execução — ou, como é o caso hoje, depois de ela confirmar que o bloco sem data
está vazio e a verificação não se aplica.

Antes do fix(#92), a verificação foi executada contra a base então vigente (medida
em !52) e passou: correlação de Spearman de 0,9999, maior identificador sem data
31.673.808 contra o menor datado 31.673.817. Essa medição está preservada na
Seção 5.1 como exemplo de como o mecanismo se comporta quando existe de fato um
bloco sem data — não como descrição da base atual.

### 5.1 Se a verificação falhar em uma execução futura

> Esta subseção ilustra o mecanismo com a medição pré-#92, quando a base ainda
> tinha um bloco sem data real. Hoje, com `linhas_sem_data = 0`, aplicar
> `sem_data="excluir"` não muda nada em relação a `sem_data="treino"` — não há
> bloco para excluir. O exemplo fica como referência de como ler uma falha, se
> e quando uma fonte futura voltar a chegar sem `DATA_STD`.

**O alarme é a própria falha, e ela interrompe a execução.** `dividir` chama
`verificar_anterioridade_sem_data` internamente quando `sem_data="treino"` e
existe linha sem data, sem deixar isso a critério de quem chama. A função levanta
`AssertionError`, que sobe por `dividir`: não há partição parcial, não há retorno
com aviso, e não existe caminho em que a execução siga com as linhas sem data no
treino sem a prova. A mensagem da exceção nomeia qual das duas condições falhou e
com que valores medidos.

**O que a exclusão custava, na base pré-#92.** Aplicando `sem_data="excluir"` à
composição medida naquela base, as 120.000 linhas saíam do treino e o total
particionado caía na mesma medida:

| Conjunto | Com `sem_data="treino"` | Com `sem_data="excluir"` |
|---|---:|---:|
| Treino | 341.962 (77,1%) | 221.962 (68,6%) |
| Validação | 48.301 (10,9%) | 48.301 (14,9%) |
| Teste | 53.486 (12,1%) | 53.486 (16,5%) |
| Total particionado | 443.749 | 323.749 |

**Naquele cenário, as três partições saíam da faixa de alarme da Seção 2 ao mesmo
tempo.** O treino caía abaixo do piso de 72%, a validação passava do teto de 14%
e o teste passava do teto de 15%. Isso não seria efeito colateral a tolerar: pela
regra da própria Seção 2, composição fora da faixa exige reabrir este registro e
mover uma data de corte com justificativa escrita. Uma falha da verificação,
portanto, não é caso de trocar um parâmetro e seguir, e sim de reabrir a decisão
das datas — o raciocínio continua valendo para uma falha futura, mesmo que os
números de hoje não a ilustrem mais.

**O que só se sabe medindo.** A taxa do alvo do treino sem o bloco não é
derivável dos números acima: 20,43% é a taxa do treino já com as linhas sem data
dentro, e a taxa do bloco isolado não está registrada. Como o que sustenta os
cortes é a comparabilidade de prevalência, e não o tamanho das partições (Seção
2), a primeira medição depois de uma falha é a taxa do alvo do treino sem o
bloco. Se ela sair do intervalo de 1,2 ponto percentual hoje observado entre as
três partições, o problema deixou de ser de volume.

**Trocar para `excluir` é barato, e é por isso que a troca precisa de registro.**
O padrão de `dividir` é `sem_data="excluir"`, então silenciar o alarme custa uma
palavra na chamada. Os metadados tornam o resultado auditável, com
`politica_sem_data`, `linhas_sem_data` e `linhas_sem_data_excluidas`, e `conferir`
soma as linhas excluídas ao total esperado. A **decisão** de excluir, porém, não
fica auditável em lugar nenhum: nada no código distingue "excluí porque a
verificação falhou" de "excluí sem tentar". **Regra:** nenhuma execução passa de
`treino` para `excluir` sem que a mudança venha acompanhada de registro nesta
seção, com a data, qual condição falhou e os valores medidos.

**Qual das duas condições falhou muda o diagnóstico.**

1. **Falha na condição 1**, correlação de Spearman abaixo de 0,99:
   `RESPONDENT_ID` deixou de acompanhar a cronologia. O argumento de anterioridade
   cai inteiro, porque não há mais ordem em que situar o bloco sem data. Não
   existe regra alternativa, a exclusão é a única saída, e o que precisa ser
   investigado é a mudança na origem dos dados que quebrou a sequência.
2. **Falha na condição 2**, maior `RESPONDENT_ID` sem data não menor que o menor
   datado: os blocos se sobrepõem. Parte das linhas sem data é contemporânea ao
   período datado, e pode ser contemporânea à validação ou ao teste, que é
   exatamente o vazamento que a Seção 3 existe para impedir. Excluir o bloco
   inteiro é conservador e correto. Uma regra por faixa de `RESPONDENT_ID`, que
   aproveitasse só o trecho anterior ao primeiro corte, é concebível, mas exige
   decisão escrita e prova nova, não ajuste na chamada.

**Monitoramento esperado.** Não há serviço em produção a instrumentar: o
particionamento roda em lote, dentro do notebook ou do script que treina. O que
se espera é que a falha seja legível e comparável:

- a verificação roda **a cada execução**, dentro de `dividir`, sem cache e sem
  resultado herdado de execução anterior, como a Seção 5 exige;
- `metadados["anterioridade_sem_data"]` traz correlação de Spearman, maior
  identificador sem data, menor identificador datado e data mais antiga da
  execução que passou. Esses valores são o que deve ser persistido junto do split
  congelado em #131, para que uma falha futura seja lida contra o último
  resultado bom em vez de no vácuo;
- `resumo` reporta n, intervalo de datas, Clientes distintos e taxa do alvo por
  partição a cada execução, que é onde a saída da faixa da Seção 2 aparece.

## 6. Registros sem Cliente e desempate por recorrência

**Registros sem `ID_GOLDENRECORD`:** excluídos dos três conjuntos. São 103
registros, 0,02% da base (Seção 4.2.1). O nulo ocorre simultaneamente na tabela
de pesquisa e na de perfil, então não há como afirmar que duas dessas linhas
pertencem a Clientes diferentes; tratá-las como grupos unitários reabriria
justamente o vazamento que a partição agrupada existe para impedir.

**Cliente que aparece em mais de um período:** mantido apenas no conjunto mais
recente em que ocorre, e suas linhas anteriores são descartadas. A direção
importa. Preservar o conjunto mais recente mantém teste e validação intactos, que
são os que medem desempenho, e concentra a perda no treino, que é o mais
abundante. A perda correspondente é reportada nos metadados em
`linhas_removidas_por_recorrencia` e `linhas_removidas_do_treino_por_validacao`,
para que seja auditável em vez de silenciosa.

## 7. Registros fora do horizonte

**Regra:** registros com data posterior a 30/06/2026 são excluídos dos três
conjuntos.

A Seção 4.1.3 menciona respostas até 26/07/2026, o que colocaria um julho
incompleto no fim da base. Um mês parcial no conjunto de teste distorceria tanto
a comparação sazonal quanto a taxa do alvo da partição, e julho é justamente mês
de pico: meio julho pesaria na métrica como se fosse julho inteiro. O horizonte
fecha em mês cheio.

Não há regra simétrica no início do horizonte: a base atual não tem registro
anterior a 01/07/2023, que já é o início declarado pela Seção 4.2.1. Se uma base
futura trouxer um registro anterior a essa data, ele entra no treino pela mesma
lógica — é a observação mais antiga disponível —, e se vier sem `DATA_STD`, é
tratado como o bloco sem data da Seção 5.

## 8. Como conferir

A conferência é por asserção, não por inspeção visual. `conferir` verifica quatro
falhas que passam despercebidas na leitura de uma tabela:

- nenhuma linha em duas partições;
- nenhum `ID_GOLDENRECORD` em duas partições;
- maior data do treino anterior à menor data da validação, e maior data da
  validação anterior à menor data do teste, aferido sobre as datas reais e não
  sobre os parâmetros de corte, porque um filtro errado deixaria o parâmetro
  coerente;
- soma das três partições igual ao total esperado, quando informado.

Um terceiro reproduz esta divisão com o que está escrito aqui: as duas datas de
corte da Seção 1, a convenção de intervalo, e as regras das Seções 5, 6 e 7.

## 9. O que esta política não decide

- Não define features, nem o tratamento do desbalanceamento, nem métricas.
- Não decide o uso dos pesos de pós-estratificação da Seção 4.2.1 no treinamento,
  na avaliação ou apenas na comunicação. Essa decisão continua em aberto.
- Não substitui `criar_folds_validacao_por_cliente`: a validação cruzada por
  Cliente dentro do treino é assunto de #102 e opera sobre a partição de treino
  definida aqui.
- Não congela os splits. O congelamento com semente, hash e artefato de índices é
  #131, e consome estas datas.
