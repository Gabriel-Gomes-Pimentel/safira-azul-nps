# Contrato de entrega operacional do Safira

## 1. Estado e fontes

**Contrato proposto:** `safira-batch-v1`. **Responsável:** @kaylan.sathler. **Revisor:** @pedro.leal. Card [#318](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/318).

Este documento fixa as decisões técnicas para empacotar o modelo final como pontuador batch por CSV. Não declara implantação, homologação pela Azul ou implementação dos cards seguintes. A referência examinada é a `develop` no commit `ce278f5bc104ca2bcdc222f11e200e6e822c2bb6`, que já contém a avaliação do #265 e a reconciliação das métricas do #278. Mudança do estimador, features ou política de fila exige nova versão e registro de avaliação.

| Fonte canônica | O que fundamenta |
|---|---|
| [Documentação principal](../documentacao.md), Seções 4.1.3, 4.2.3 e 4.5 | Uso pós-viagem, janela temporal, modelo recomendado, capacidade e limitações |
| [Comparação dos modelos](../../notebooks/comparacao_modelos.ipynb), Seções 3, 5.5, 6.1, 9 e 10.1 | Preparação, ajuste, calibração, avaliação final e encadeamento de objetos já ajustados |
| [Hiperparâmetros vencedores](resultados/hiperparametros_gradient_boosting.json) | Parâmetros da busca #190, semente e cortes registrados |
| [Contrato da base analítica](contrato-dados-score-pos-viagem.md) e [pré-processamento](../../scripts/preprocessamento_nps.py) | Feature Set V1, tipos, exclusões e máscara de cancelamento |
| [Matriz](../../src/matriz.py), [histórico](../../src/features.py), [calibração](../../src/calibracao.py) e [split](../../src/split.py) | Pré-processador efetivo, três features de histórico, limitações da calibração e partições |
| [Política temporal](politica-de-particionamento-temporal.md) e [dependências](../../requirements.txt) | Cortes, agrupamento por Cliente e versões declaradas |

Nas seções seguintes, **existente** significa verificado nessas fontes; **decisão V1** significa regra proposta para os cards de implementação; **pendência Azul** significa confirmação necessária antes do uso operacional. A revisão deste contrato não substitui essa confirmação.

## 2. Modelo e artefato

### Modelo existente

O único produtor da probabilidade e da fila é `HistGradientBoostingClassifier` calibrado por Platt (`sigmoid`). O alvo positivo é `DETRATOR = 1`, derivado de `NPS_PRINCIPAL == -100`; Neutros e Promotores são a classe negativa. O score estima detração entre respondentes, não insatisfação de toda a população de passageiros.

| Parâmetro | Valor canônico |
|---|---|
| `learning_rate` | `0.049833191601257244` |
| `max_iter` | `200` |
| `max_leaf_nodes` | `61` |
| `min_samples_leaf` | `150` |
| `l2_regularization` | `1.774425485152653` |
| `class_weight` | `balanced` |
| `random_state` | `42` |
| `early_stopping` | `False`, fixado para evitar holdout aleatório interno |

Os seis primeiros valores e a semente vêm do JSON da busca de 40 combinações, selecionadas por F2 em cinco folds agrupados por Cliente. Os demais parâmetros mantêm os defaults da biblioteca usados na geração; o manifesto deve registrar `get_params()` completo e as versões realmente utilizadas, não somente os parâmetros buscados. O notebook integrado ainda repete os valores em seu construtor: o #317 deve conferir sua igualdade com o JSON e falhar em caso de divergência.

A Regressão Logística e a Árvore de Decisão permanecem materiais complementares de interpretação agregada. Não compõem ensemble, não alteram a probabilidade e não substituem automaticamente o Gradient Boosting em caso de falha. As recomendações de contingência da Seção 4.5.3 não são implementação de fallback deste pacote.

### Decisão V1: preservar o ajuste avaliado

1. Reproduzir as partições canônicas de `split.dividir`: treino anterior a `2025-07-01`, validação de `2025-07-01` até antes de `2026-01-01`, teste a partir de `2026-01-01`, com `DATA_STD` e separação por `ID_GOLDENRECORD`. Preservar a política vigente para linhas sem data e exclusão de Clientes recorrentes; registrar os metadados e a impressão digital local da base.
2. Construir a matriz com o mesmo Feature Set V1 e histórico de `matriz.preparar_matriz`, sem alterar a semântica usada no treinamento histórico nesta versão de reprodução.
3. Ajustar imputação, escala e encoding apenas no treino; ajustar o Gradient Boosting apenas na matriz transformada de treino.
4. Ajustar `CalibratedClassifierCV(FrozenEstimator(estimador_ajustado), method="sigmoid")` somente sobre o treino, reproduzindo a Seção 6.1. A calibração existente é **in-sample**: `FrozenEstimator` impede reajuste do classificador, mas não torna a calibração disjunta do treino. Essa limitação deve constar no manifesto e no model card.
5. Encadear os objetos já ajustados como `Pipeline([("preparo", preprocessador), ("modelo", modelo_gb_calibrado)])`, conforme a Seção 10.1. Serializar uma única unidade; não executar novo `fit` sobre esse pipeline após encapsulá-lo.
6. A geração do artefato não escolhe modelo, limiar ou calibração pelo teste. A validação já fundamentou a escolha documentada e a aceitação da calibração; não integra nenhum `fit`. O teste não é exigido para pontuar e não deve ser reexecutado para cada empacotamento.

O pré-processador canônico é o de `src/matriz.py`: numéricas com `SimpleImputer(strategy="median", add_indicator=True)` e `RobustScaler`; categóricas com imputação constante `NAO_INFORMADO` e `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Preservar estatísticas, categorias, indicadores de ausência e ordem de saída já aprendidos. Não substituir pelo construtor alternativo de `scripts/preprocessamento_nps.py`, cujo sentinel e configuração de imputação diferem.

O reajuste com treino e validação sugerido na Seção 4.5.1 fica **fora da V1**. Ele mudaria o estimador avaliado e não autoriza reutilizar o mapa de Platt antigo sobre um classificador novo. Calibração disjunta, reconstrução histórica point-in-time e retreino requerem uma versão de modelo própria e nova avaliação independente, sem otimização sobre o teste já consultado.

O contrato identifica a receita, não comprova equivalência de um binário ainda não gerado: o #317 deve demonstrar equivalência antes/depois de serializar e registrar origem, ambiente e versão da base. Uma mudança de base ou ambiente não herda automaticamente os resultados publicados na Seção 4.5.

## 3. Entrada do modelo: 14 features

**Existente:** o Feature Set V1 contém 11 atributos, mas o modelo final utiliza também as três features de `FEATURES_HISTORICO`. Entregar somente as 11 colunas produziria uma entrada incompatível. A ordem de `X` é a ordem do V1 seguida da ordem do histórico, conforme a tabela.

| Ordem | Coluna | Tipo após leitura | Ausência e regra |
|---|---|---|---|
| 1 | `TIER_VIAGEM` | Texto/categoria | Nulo passa pelo imputador |
| 2 | `VOO_TIPO` | Texto/categoria | Nulo passa pelo imputador |
| 3 | `TIPO_ENTRETENIMENTO` | Texto/categoria | Nulo passa pelo imputador |
| 4 | `CANAL_COMPRA` | Texto/categoria | Nulo passa pelo imputador |
| 5 | `SEGMENTO` | Texto/categoria | Nulo passa pelo imputador |
| 6 | `ESTATISTICA_ATRASOSAIDA` | Numérico, não booleano | Nulo permitido; infinito proibido |
| 7 | `ATRASO_CHEGADA` | Numérico, não booleano | Nulo permitido; infinito proibido |
| 8 | `CANCELAMENTO_VOO` | Booleano | Obrigatório e não nulo |
| 9 | `ANTECEDENCIA_CANCELAMENTO` | Numérico, não booleano | Preservar ausência como ausência, não preencher com zero |
| 10 | `TEMPO_VOO` | Numérico, não booleano | Nulo permitido; infinito proibido |
| 11 | `N_TRECHOS` | Numérico; inteiro quando informado na V1 operacional | Nulo permitido; derivação aprovada abaixo |
| 12 | `HIST_RESPOSTAS_ANTERIORES` | Inteiro | Contagem anterior; zero somente quando ausência de histórico foi comprovada |
| 13 | `HIST_DETRATOU_ANTES` | Numérico `0`/`1`, ou nulo | Nulo se contagem zero; caso contrário, informa detração passada |
| 14 | `HIST_TAXA_DETRACAO_ANTERIOR` | Numérico em `[0, 1]`, ou nulo | Nulo se contagem zero; caso contrário, proporção passada |

**Decisão V1:** todas as colunas devem existir mesmo quando o valor pode ser nulo. Validar contagem histórica não negativa, taxa finita e coerência entre contagem, indicador e taxa; não inferir ausência de histórico de falha na fonte ou de identificador ausente. Categorias novas seguem o encoder ajustado, sem incluir categorias no artefato ou refazer encoding; informar a contagem agregada de ocorrências desconhecidas nos metadados locais. Não arredondar, reescalar ou normalizar valores antes do pré-processador; manter unidades e semântica da fonte usadas no treino.

Antes de chamar o pipeline ajustado, o leitor batch deve reproduzir duas regras de preparação da base analítica, implementadas em `padronizar_categoricas` e `preparar_base_analitica`: nas cinco features textuais, remover espaços nas extremidades, converter para maiúsculas e substituir sequências de espaços em branco por um espaço; em `TEMPO_VOO`, converter valor numérico menor ou igual a zero em nulo, contabilizando a correção. Preservar nulos e não inventar categorias. Essas transformações anteriores ao `ColumnTransformer` não estão incorporadas no binário fitted descrito na Seção 2. Não aplicá-las a IDs de passagem, timestamps ou metadados, nem tratar erro de parsing numérico como duração ausente.

`N_TRECHOS` conta separadores `/` em `ORIGEM[/CONEXAO...]/DESTINO`, isto é, aeroportos menos um, e não quantidade de aeroportos. O adaptador de dados autorizado deve entregar a coluna já materializada; o CSV operacional não recebe rota bruta. Recusar valor informado não inteiro ou menor que um. Esta restrição adicional é decisão V1, não garantia já implementada pelo validador numérico.

Aplicar a máscara existente de cancelamento antes da transformação: para `CANCELAMENTO_VOO=True`, `ESTATISTICA_ATRASOSAIDA`, `ATRASO_CHEGADA`, `TEMPO_VOO` e `N_TRECHOS` ficam nulos, mesmo quando preenchidos retrospectivamente. Não aplicar `fillna(0)` global nem converter “não aplicável” em atraso zero.

### Histórico e anterioridade

**Limitação existente:** `features.adicionar_historico` ordena por `RESPONDENT_ID` e acumula respostas anteriores; não compara a disponibilidade de cada resposta com o `t_score` da jornada atual. A Seção 4.3.2.6 reconhece essa lacuna. O cálculo sobre a base inteira antes do split também não comprova disponibilidade temporal.

**Decisão V1 de inferência:** receber as três features históricas pré-calculadas por uma fonte autorizada da Azul, considerando somente respostas registradas e disponíveis **estritamente antes de `t_score`**. Nunca usar a resposta da jornada atual, seu alvo, previsões do próprio lote ou a função histórica de treino sobre jornadas novas sem rótulo. Histórico desconhecido deve bloquear a linha/lote para investigação; não criar `(0, nulo, nulo)` por conveniência.

Validar metadados de uma fotografia não prova sozinho que o seu conteúdo foi reconstruído corretamente. A Azul precisa fornecer evidência do corte as-of. Mesmo com entrada operacional correta, o treinamento histórico atual pode ter usado informação indisponível no instante correspondente; por isso esta V1 é candidata a **piloto controlado**, não modelo aprovado para produção. Retirar as features de histórico exigiria retreino, e não é fallback permitido.

## 4. CSV operacional e janela temporal

### Decisão V1 de transporte

CSV UTF-8, delimitador `,`, primeira linha como cabeçalho e decimal `.`. Aspas seguem CSV padrão; campo vazio representa nulo. Não tratar textos como `NA` ou `NULL` automaticamente como ausentes. IDs são lidos como texto, preservando zeros à esquerda. `CANCELAMENTO_VOO` aceita somente `true`/`false`; numéricas usam conversão explícita, sem coerção silenciosa de textos inválidos para nulo. Rejeitar cabeçalhos repetidos, colunas ausentes, infinidades e campos extras não contratados.

Além das 14 features, são obrigatórias as seguintes colunas de passagem, **nenhuma delas preditora**:

| Coluna | Tipo | Regra V1 e finalidade |
|---|---|---|
| `ID_CLIENTE` | Texto não vazio | Chave técnica estável que a Azul pode resolver internamente; nunca nome, CPF, telefone ou e-mail |
| `ID_JORNADA` | Texto não vazio | Chave única do par passageiro/jornada; não número de voo ou localizador compartilhado |
| `T_EVENTO_ELEGIBILIDADE` | Timestamp ISO 8601 com offset | Instante do encerramento operacional; para cancelamento, instante do registro do cancelamento |
| `T_DISPONIBILIDADE_FEATURES` | Timestamp ISO 8601 com offset | Maior instante de disponibilidade das versões das features efetivamente utilizadas, inclusive histórico |

Os nomes desses IDs são **proposta técnica**, não campos confirmados da base Azul. A base acadêmica usa `RESPONDENT_ID` para respostas e `ID_GOLDENRECORD` para Clientes. Não presumir que `RESPONDENT_ID` existe antes da resposta, nem criar a chave operacional a partir da posição da linha. A associação a uma jornada elegível precisa ser fornecida e validada pela Azul.

Parâmetros obrigatórios da execução: `t_score` com offset, `fuso_operacional` IANA, `versao_fontes` da fotografia utilizada e `id_execucao` não vazio. A capacidade diária é inteiro não negativo; o default proposto é `50`. O fuso não é presumido pelo programa. Um lote utiliza um único corte temporal e uma única fotografia declarada; fontes diferentes devem compor uma versão de fotografia rastreável antes do envio.

Validar `T_EVENTO_ELEGIBILIDADE <= t_score` e `T_DISPONIBILIDADE_FEATURES <= t_score`. A fotografia deve garantir que a jornada ainda não tem resposta NPS em `t_score`, e que versões cadastrais e operacionais não incorporam atualizações futuras. O timestamp máximo é um resumo de auditoria: a proveniência detalhada por feature permanece na fonte autorizada, e sua ausência bloqueia uso operacional, ainda que o schema sintético passe.

O marco de elegibilidade é pós-encerramento/pré-resposta, ou pós-registro do cancelamento. Executar antes do convite à pesquisa é uma preferência operacional mencionada na documentação, não um SLA já confirmado. Atrasos de consolidação, janela máxima e política para lotes atrasados devem ser acordados com a Azul.

O lote deve conter a população elegível completa de cada dia operacional incluído, sem envio de fragmentos como se cada um recebesse nova capacidade. Derivar `DIA_OPERACIONAL` do evento de elegibilidade no fuso declarado. `DATA_STD`, usada para agrupar a avaliação histórica, é data programada e não comprova o dia de encerramento; a mudança deve permanecer identificada na política da fila.

### Exclusões

Recusar `NPS_*`, `SUB_*`, `DETRATOR`, `CATEGORIA_NPS`, `CLASSE_NPS` e qualquer resposta da pesquisa atual. Identificadores, timestamps e metadados nunca entram em `X`. Não receber `QTDE_VIAGENS_12M`, rota/equipamento bruto ou novas features por inferência de tipo. O envelope operacional não é a base bruta de 46 colunas nem a base analítica de treino com alvo; os validadores existentes que exigem `RESPONDENT_ID` não devem ser reutilizados integralmente para este envelope.

Em caso de violação, falhar o lote antes da escrita da saída, sem produzir fila parcial ou excluir linhas silenciosamente. Mensagens apontam regra, coluna e quantidade, sem imprimir IDs, valores individuais ou registros reais. Lote válido vazio gera somente cabeçalho e metadados locais com zero pontuações.

## 5. Fila e saída

### Decisão V1 de priorização

1. Pontuar uma linha por `ID_JORNADA`, exigindo unicidade dessa chave. Repetição de Cliente em jornadas diferentes não altera a quantidade de previsões nem permite somar scores.
2. Para cada `DIA_OPERACIONAL`, ordenar probabilidade decrescente, depois `ID_CLIENTE` e `ID_JORNADA` em ordem textual crescente para desempatar. Não arredondar score antes de ordenar.
3. Para respeitar contatos por Cliente, selecionar primeiro a jornada de maior prioridade de cada `ID_CLIENTE` no dia. Jornadas adicionais do mesmo Cliente continuam na saída com score, mas sem posição elegível nem priorização.
4. Numerar as jornadas representantes de `1` a `n` por dia; marcar `PRIORIZADO=True` somente para posições até `min(capacidade_diaria, n)`. Capacidade zero não prioriza ninguém. Não transferir vagas entre dias, não admitir excedente por empate e não usar limiar fixo `0.5` ou o limiar do período histórico.
5. Ordenar o CSV por dia crescente e, dentro de cada dia, score decrescente e as chaves de desempate. Manter uma linha de saída por jornada válida, inclusive jornadas não representantes.

Essas regras de desempate e contato único são **propostas V1**. A fila histórica da Seção 9 agrupa respostas por `DATA_STD` e inclui empates (`rank(method="min")`), sem deduplicar Clientes. Logo, suas métricas não demonstram o desempenho da fila operacional aqui proposta. A matriz de confusão do período também não é a matriz da fila diária. O #321 deve separar homologação funcional de nova avaliação estatística; nenhuma das duas pode ser alegada antecipadamente.

O pacote produz uma fila, não realiza contato. Customer Insights e Customer Experience precisam controlar jornadas já pontuadas, contatos efetivados, consentimentos/restrições e reexecuções. `ID_EXECUCAO` deve permanecer o mesmo no retry do mesmo lote; não reenviar a fila como nova capacidade diária. Estado entre execuções e supressão de contatos entre dias pertencem à integração interna, ainda não fornecida pela Azul.

### Schema da saída CSV

| Coluna | Tipo e significado |
|---|---|
| `ID_CLIENTE`, `ID_JORNADA` | Chaves de passagem preservadas |
| `DIA_OPERACIONAL` | Data `YYYY-MM-DD` do agrupamento no fuso declarado |
| `PROBABILIDADE_DETRACAO` | Numérico finito em `[0, 1]`, da coluna de `predict_proba` correspondente a `classes_ == 1` |
| `FAIXA_RISCO` | `NAO_VALIDADA` na V1, até existir política de faixas aprovada |
| `POSICAO_FILA` | Inteiro positivo para jornada representante do Cliente no dia; vazio nas demais |
| `PRIORIZADO` | `true`/`false`, conforme capacidade diária |
| `VERSAO_MODELO` | Identificação imutável do artefato carregado |
| `VERSAO_CONTRATO` | `safira-batch-v1` |
| `VERSAO_FONTES` | Identificação da fotografia declarada |
| `VERSAO_POLITICA_FILA` | `contato-unico-topk-v1` |
| `FUSO_OPERACIONAL` | Fuso utilizado na derivação do dia |
| `T_SCORE` | Timestamp de corte declarado e utilizado, igual em todas as linhas |
| `ID_EXECUCAO` | Identificação rastreável da execução/lote |

O programa não inventa cortes de “baixo/médio/alto”. A menção a faixas na Seção 4.1.3 não define limites; `NAO_VALIDADA` é status explícito, não classe de risco. Faixas futuras precisam de limites, inclusividade das bordas, versão e validação própria; não substituem ranking por capacidade. A decisão humana de recuperação não deve ser inferida de `PRIORIZADO` ou da faixa.

## 6. Pacote, segurança e reprodução

**Decisão V1:** gerar `artifacts/modelo-safira-v1.joblib` e `artifacts/modelo-safira-v1.json` em diretório local protegido pelo `.gitignore` **antes de gerar o primeiro arquivo**. O binário contém o pipeline ajustado; o manifesto real registra metadados, não passageiros. Mesmo o manifesto real permanece local; somente seu schema e exemplo inteiramente sintético podem entrar no GitLab.

Campos mínimos do manifesto: versão do modelo/contrato/política, commit fonte, SHA-256 do binário, lista ordenada e tipos das 14 features, classes e classe positiva, nomes das features transformadas, parâmetros completos, semente, método/partição de calibração e limitação in-sample, cortes e política de split, impressão digital local da base, versão das fontes de treino, timestamp de geração, versões efetivas de Python/scikit-learn/pandas/numpy/scipy/joblib e proveniência da avaliação de referência. Não incluir índices de registros, caminhos confidenciais, IDs, rótulos individuais ou logs de dados.

`requirements.txt` declara versões de algumas bibliotecas, mas não fixa toda a pilha. O #322 deve selecionar e verificar um ambiente operacional executável, com versões exatas compatíveis com a geração do binário. Divergência de ambiente deve bloquear o carregamento, sem adaptação silenciosa ou alegação de equivalência. Este documento não certifica que os pins declarados já foram instalados e testados.

Verificar hash e manifesto antes de desserializar. SHA-256 detecta corrupção, mas não autentica fornecedor: obter ambos por canal interno aprovado e de origem confiável. `joblib`/pickle pode executar código no carregamento; não aceitar artefato externo ou enviado por passageiro. Distribuição: código e documentos seguros via GitLab; modelo, manifesto real, CSVs e evidências operacionais somente no ambiente/canal autorizado da Azul/Inteli.

Testes devem usar dados artificiais. Nunca versionar base, binário real, CSV real, amostra, saída de notebook, credenciais ou logs identificáveis. A solução não envia informações a APIs, serviços de IA ou telemetria externos. Relatórios de homologação públicos ao projeto descrevem procedimentos e resultados sintéticos; evidência real fica interna. Privacidade, autorização, retenção, devolução/descarte e resolução de identidade precisam de validação da Azul antes do piloto.

## 7. Rastreabilidade de implementação e liberação

| Card | Entrega contratada | Evidência mínima |
|---|---|---|
| [#317](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/317) | Artefato e manifesto, sem retreino no teste | Configuração igual à fonte; pipeline fitted; equivalência antes/depois de serializar e repetibilidade sintética |
| [#320](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/320) | Validação das 14 features e envelope | Falha para schema inválido, resposta atual, histórico incoerente e timestamps posteriores ao corte; preparação canônica das categorias/duração e máscara de cancelamento |
| [#319](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/319) | Scoring e CSV de fila | Nenhum `fit`; resultado independente da ordem das linhas; desempate e contato único; capacidade por dia; escrita atômica |
| [#322](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/322) | Ambiente e manual | Instalação limpa, versões verificadas, comando reproduzível, limitações e recuperação de versão |
| [#321](https://git.inteli.edu.br/graduacao/2026-2a/t28/g01/-/issues/321) | Homologação funcional integrada | Fluxo sintético completo, erros sem saída parcial, metadados rastreáveis e ausência de conteúdo confidencial |

O objetivo do #318 é a especificação, não a implementação dessas evidências. Não corrigir outros notebooks ou reabrir a escolha do algoritmo dentro deste card.

### Pendências Azul e bloqueadores de produção

| Confirmação necessária | Conduta enquanto pendente |
|---|---|
| Chaves operacionais e mapeamento passageiro/jornada para contato interno | Usar somente chaves artificiais em testes; não presumir que `RESPONDENT_ID` identifica a jornada pré-resposta |
| Snapshots as-of, disponibilidade de cada feature/histórico e ausência de resposta atual | Bloquear piloto com dados reais sem evidência; registrar a lacuna do histórico de treino |
| Capacidade real, contato único, supressão entre dias, fuso e janela de processamento | Usar regras V1 somente em homologação sintética; 50 é premissa do grupo |
| População diária completa, retry e jornadas atrasadas | Não executar fragmentos como novas filas independentes |
| Faixas e ações associadas | Manter `FAIXA_RISCO=NAO_VALIDADA`; nenhuma ação automática |
| Ambiente, armazenamento, responsáveis internos e canal de distribuição | Não declarar API, dashboard, Snowflake ou serviço implantado |
| Privacidade operacional, controle de acesso e prazo de retenção/descarte | Não resolver identidade fora da Azul; validar com as orientações Azul/Inteli |
| Qualificação estatística com histórico point-in-time e nova política de fila | Não transferir métricas do #265 para o novo fluxo; produção exige avaliação independente e aceite explícito |

As pendências não impedem implementar e testar o pacote com dados sintéticos. Impedem declarar que o modelo está aprovado para operação real. Esta versão preserva a escolha documentada do algoritmo e torna explícitas as interfaces e limitações necessárias para a continuidade por outra equipe.
