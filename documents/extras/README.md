# Documentos complementares

Esta pasta guarda o material de apoio do projeto Safira, ou seja, o que não faz parte de
`documents/documentacao.md` mas precisa ficar versionado e localizável.

## Apresentações

Ficam em [`apresentacoes/`](apresentacoes), uma por sprint.

| Sprint | Arquivo | Conteúdo |
|---|---|---|
| Sprint 01 | [sprint-01.pdf](apresentacoes/sprint-01.pdf) | Sprint Review 1: entendimento do negócio, contexto de mercado, SWOT, 5 Forças de Porter, proposta de solução, personas e timeline das sprints |
| Sprint 02 | [sprint-02.pdf](apresentacoes/sprint-02.pdf) | Sprint Review 2: perfil da base de 484.915 respostas, viés de resposta e ponderação, tratamento de nulos e outliers, resultado das hipóteses H1 a H4, testes de normalidade, escolha das escalas de normalização e próximos passos |
| Sprint 03 | [sprint-03.pdf](apresentacoes/sprint-03.pdf) | Sprint Review 3: base integrada de 484.915 respostas, contrato temporal que só admite features disponíveis em `t_score`, primeiro modelo (Gradient Boosting) com split temporal e por Cliente, métricas da classe Detrator contra as metas, leitura de atraso, cancelamento no mesmo dia e tier, rastreabilidade entre notebook e documentação e próximos passos |
| Sprint 04 | [sprint-04.pdf](apresentacoes/sprint-04.pdf) | Sprint Review 4: custo do falso negativo e escolha do F2 para a busca, candidatos comparados na mesma base, Grid Search na Regressão Logística e Random Search no Gradient Boosting, placar por F2, decisão de usar o Gradient Boosting para a fila e a Regressão Logística para explicar, efeito da fila de 50 contatos por dia, limite da meta de recall pela capacidade, principais odds ratio e próximos passos |

## Dados e particionamento

| Documento | Conteúdo |
|---|---|
| [Contrato de dados do score pós-viagem](contrato-dados-score-pos-viagem.md) | Contrato da base analítica: unidade da linha, chaves, colunas obrigatórias, target e as validações de schema que `scripts/preprocessamento_nps.py` executa |
| [Política de particionamento temporal](politica-de-particionamento-temporal.md) | Registro de decisão do #127: cortes de treino, validação e teste no tempo, separação por Cliente e tratamento das linhas sem data |
| [Composição dos conjuntos](composicao-dos-conjuntos.md) | Tabela e figura da composição de treino/validação/teste, material de apoio do #132 para a redação do #116 |
| [Relatório de validação do corte de leakage](relatorio-validacao-leakage.md) | Resultado do #198 sobre a base dummy: datas de treino < validação < teste e nenhum `RESPONDENT_ID`/`ID_GOLDENRECORD` em dois conjuntos. Regerar com `python scripts/validar_leakage.py` |
| [Auditoria de reuso do split e da validação cruzada](auditoria_reuso_split_validacao.md) | Conferência do #239 de que os quatro pipelines de modelagem usam `split.dividir` e os folds por Cliente de `validacao`, em vez de partição própria |
| [Hipóteses](Hipoteses.md) | Hipóteses levantadas e ainda não confirmadas, com tipo e status de validação |

## Entrega operacional

| Documento | Conteúdo |
|---|---|
| [Contrato de entrega operacional](contrato-entrega-operacional.md) | Especificação do #318 para empacotar o Gradient Boosting calibrado: 14 features, ajuste preservado, CSV de entrada/saída, fila por capacidade, artefato privado e pendências de validação pela Azul |

## Avaliação e comparação dos modelos

| Documento | Conteúdo |
|---|---|
| [Protocolo de avaliação do Artefato 7](protocolo-avaliacao-artefato7.md) | Escolha do F2 como critério único da busca de hiperparâmetros dos quatro candidatos (#238) e compatibilidade com as metas da Seção 4.3.2 |
| [Referência técnica do protocolo de avaliação](protocolo-avaliacao-referencia-tecnica.md) | Consulta rápida do #240 para as duplas de modelagem: o que usar para tunar e medir cada modelo, e onde está o código |
| [Tabela comparativa entre o candidato e os pisos](comparativo-modelos.md) | Tabelas geradas pela seção 9 de `notebooks/modelagem.ipynb` sobre a partição de teste, base do item (d) da Seção 4.3 (#108) |
| [Matrizes de confusão dos candidatos](matrizes_confusao_candidatos.md) | Matrizes no limiar por capacidade de contato, na partição de teste, para os candidatos da Seção 4.4 (#250) |

## Árvore de Decisão e explicabilidade

| Documento | Conteúdo |
|---|---|
| [Busca de hiperparâmetros da Árvore de Decisão](busca_arvore_decisao.md) | Resultado do `GridSearchCV` dos #231 e #232: 48 combinações, vencedor por F2 e métricas na partição de validação |
| [Regras da Árvore de Decisão](regras_arvore_decisao.txt) | Regras da árvore final exportadas com `export_text` (#233) |
| [Folhas da Árvore de Decisão](folhas_arvore_decisao.md) | Taxa de Detrator observada em cada folha da árvore final, no treino (#233) |
| [Interpretação das regras da Árvore de Decisão](interpretacao_arvore_decisao.md) | Leitura das folhas da árvore final traduzida para decisões da área de Experiência do Cliente da Azul (#234) |
| [Decisão sobre explicabilidade: por que não adotamos SHAP](decisao-explicabilidade-shap.md) | Registro da decisão do grupo de não usar SHAP na Seção 4.4, com a justificativa, o que se perde e as duas vias de explicabilidade entregues (#215) |

## Revisões cruzadas e reprodutibilidade

| Documento | Conteúdo |
|---|---|
| [Parecer de reprodutibilidade dos interpretáveis](parecer-reprodutibilidade-interpretaveis.md) | Revisão cruzada da Seção 4.4.2 e dos notebooks da Regressão Logística e da Árvore de Decisão pelo #195: veredito de execução do zero, checklist técnico e conferência número a número |
| [Parecer de texto dos interpretáveis](parecer-texto-interpretaveis.md) | Revisão cruzada das Seções 4.4.2 e 4.4.3 pelo #196: veredito do checklist de texto, apontamentos com parágrafo e redação sugerida, e divergências de definição e formato de métrica em relação aos ensembles |
| [Parecer de reprodutibilidade dos ensembles](parecer-reprodutibilidade-ensembles.md) | Revisão cruzada da Seção 4.4.4 e do `ensembles.ipynb` pelo #216: veredito de execução de ponta a ponta, conferência número a número e verificação do isolamento do teste e dos folds por Cliente |
| [Parecer de texto dos ensembles](parecer-texto-ensembles.md) | Revisão cruzada da Seção 4.4.4 pelo #236: veredito de clareza, ortografia/gramática, citações ABNT e consistência de métricas com o card 09, apontamentos com parágrafo e redação sugerida; o #194 fica fora por ainda não estar escrito |
| [Reprodutibilidade do notebook integrado](reprodutibilidade-notebook-integrado/relatorio-reprodutibilidade-base-real.md) | Duas execuções consecutivas de `notebooks/comparacao_modelos.ipynb` na base real, com a mesma semente: métricas idênticas, logs iguais linha a linha ([execução 1](reprodutibilidade-notebook-integrado/execucao-1.log), [execução 2](reprodutibilidade-notebook-integrado/execucao-2.log)) e variação máxima de 0%. Regerar com `NOTEBOOK_BASE_REAL=1 pytest tests/test_reprodutibilidade_notebook.py` |
| [Comparação das duas execuções](reprodutibilidade-notebook-integrado/comparacao.json) | Critérios de aceite e métricas das duas execuções do notebook integrado, no formato que `tests/test_reprodutibilidade_notebook.py` grava |
| [Execução de ponta a ponta dos notebooks](execucao-ponta-a-ponta.md) | Os 12 notebooks executados do zero na base real com Python 3.12 e o `requirements.txt` (#303): status, tempo e célula de falha de cada um, diferenças de número encontradas e arquivos versionados que a execução regrava |

## Resultados das buscas e comparações

Ficam em [`resultados/`](resultados), em JSON e não em CSV, porque o `.gitignore` bloqueia `*.csv` para proteger os dados do parceiro. Os arquivos guardam só parâmetros e métricas agregadas, sem nenhuma linha de Cliente, e são regravados pelas células ou módulos indicados.

| Arquivo | Conteúdo | Gravado por |
|---|---|---|
| [hiperparametros_candidato.json](resultados/hiperparametros_candidato.json) | As 12 combinações da busca do primeiro candidato (Gradient Boosting), com precisão média, desvio entre folds e ROC-AUC | `notebooks/modelagem.ipynb`, seção da busca |
| [contorno_hiperparametros.json](resultados/contorno_hiperparametros.json) | Verificação de contorno de `max_iter` em volta do melhor ponto da busca acima | `notebooks/modelagem.ipynb`, conclusão do ajuste |
| [comparativo_modelos.json](resultados/comparativo_modelos.json) | Métricas sem arredondamento do candidato e dos pisos na partição de teste | `notebooks/modelagem.ipynb`, seção 9 |
| [hiperparametros_logistica.json](resultados/hiperparametros_logistica.json) | As 8 combinações da busca da Regressão Logística, com F2 médio e desvio entre folds | `notebooks/regressao_logistica.ipynb`, seção 5.1 |
| [hiperparametros_gradient_boosting.json](resultados/hiperparametros_gradient_boosting.json) | Hiperparâmetros vencedores do Gradient Boosting, semente, número de iterações e de folds, melhor F2 e cortes temporais | `src/busca_gradient_boosting.py` |
| [cv_resultados_gradient_boosting.json](resultados/cv_resultados_gradient_boosting.json) | Resumo das 40 combinações da busca do Gradient Boosting: parâmetros, F2 médio, desvio, posição e tempo de ajuste | `src/busca_gradient_boosting.py` |

## Convenções desta pasta

- Nome de pasta e de arquivo em minúsculas, sem acento e com hífen separando as palavras. O Git
  escapa caminho com acento na saída padrão (o `assets/instrução.txt` do template aparecia como
  `"assets/instru\303\247\303\243o.txt"`), o que atrapalha script, link e `git add` pelo nome.
- Apresentação de sprint entra como `apresentacoes/sprint-NN.pdf`, em PDF e não em `.pptx`, porque o
  PDF abre no próprio GitLab e não depende da versão do editor de origem.
- Um arquivo novo aqui entra também na tabela correspondente acima, senão ele fica invisível para
  quem não abriu a pasta.
