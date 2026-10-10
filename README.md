<div align="center">

# ✈️ Safira: previsão de detratores de NPS para a Azul Linhas Aéreas

**Modelo de machine learning que avisa, antes da resposta à pesquisa, quais Clientes tendem a dar nota de Detrator, para a Azul contatar primeiro quem está em maior risco.**

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest" />

<a href="https://youtu.be/c-2v88OHld4"><img src="https://img.shields.io/badge/V%C3%ADdeo%20demonstrativo-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Vídeo demonstrativo" /></a>

</div>

> Projeto do Módulo 3 de Engenharia da Computação do [Inteli](https://www.inteli.edu.br/), feito em grupo (G01, turma T28) com a **Azul Linhas Aéreas** como empresa parceira, no 2º semestre de 2026. Este repositório é a vitrine pública do projeto. **Nenhum dado da Azul está aqui**: as bases ficam fora do repositório, conforme o termo firmado com o parceiro, e os notebooks são publicados sem saídas de célula.

## O problema

A Azul só sabe quem é Detrator **depois** que a nota foi dada. A pesquisa de NPS chega um dia após o voo e fica aberta por sete dias, e é nessa janela que ainda daria para recuperar uma experiência ruim. Mas a equipe de Experiência do Cliente atende algumas centenas de casos por dia, enquanto um único fim de semana de mau tempo afeta milhares de passageiros. Faltava um critério objetivo para decidir **quem contatar primeiro**.

## A solução

Um **score pós-viagem de risco de detração**: para cada jornada encerrada, o modelo estima a probabilidade de o Cliente responder como Detrator, usando só dados operacionais e de perfil disponíveis antes da resposta. Todo dia, os 50 Clientes de maior risco viram a fila de contato.

| Papel | Modelo |
|---|---|
| **Ordena a fila de contato** | Gradient Boosting (`HistGradientBoostingClassifier`) com calibração de Platt |
| **Explica os fatores de risco** | Regressão Logística, lida por *odds ratio* |
| **Leitura complementar em regras** | Árvore de Decisão |

## Resultados

Medidos uma única vez na partição de teste: o 1º semestre de 2026, com **53.486 respostas** que não participaram de nenhuma escolha de modelo, hiperparâmetro ou calibração.

| Métrica | Meta | Resultado no teste |
|---|---|---|
| Precisão Média (Average Precision) | ≥ 0,40 | **0,5211** ✅ (2,55× uma ordenação ao acaso) |
| ROC-AUC | ≥ 0,75 | 0,7483 (0,0017 abaixo da meta) |
| Recall na fila diária de 50 contatos | ≥ 0,70 | 0,4230 (o teto com essa capacidade é 0,829) |
| Brier, depois da calibração | menor é melhor | 0,1313 |

**Na prática:** na fila diária, **51 de cada 100 Clientes contatados eram Detratores**, contra 20 numa lista montada ao acaso. O desempenho se manteve num semestre que o modelo nunca viu, e a distância no Recall vem mais da capacidade de contato do que do modelo.

O que os dados mostraram (Regressão Logística):

- **Cancelamento avisado no mesmo dia** é o maior fator de risco (*odds ratio* de 7,73).
- **Quem já detratou tende a detratar de novo**: 42,9% voltam a detratar, contra 14,3% de quem não detratou.
- Atrasos de mais de duas horas pesam mais do que uma tripulação mal avaliada.

## Destaques de engenharia

- **Pipeline do dado à métrica** em módulos Python reutilizáveis ([src/](src/)), importados pelos notebooks em vez de reimplementados.
- **Sem vazamento de dados (leakage)**: contrato temporal de features com corte estrito no momento do score, verificado por testes ([relatório](documents/extras/relatorio-validacao-leakage.md)).
- **Divisão temporal e por Cliente**, congelada num artefato de índices com hash, para que todo mundo compare modelos sobre as mesmas partições ([política](documents/extras/politica-de-particionamento-temporal.md)).
- **Validação cruzada agrupada por Cliente** (GroupKFold) na busca de hiperparâmetros de seis candidatos ([comparativo](documents/extras/comparativo-modelos.md)).
- **Reprodutibilidade**: versões fixadas no `requirements.txt` e duas execuções do notebook integrado comparadas métrica a métrica ([relatório](documents/extras/reprodutibilidade-notebook-integrado/relatorio-reprodutibilidade-base-real.md)).
- **Mais de 870 testes automatizados com pytest**, que rodam sobre dados sintéticos, sem depender das bases da Azul.

## Minha participação

Fui o integrante com mais contribuições no repositório do grupo. Trabalhei principalmente em:

- **Hipótese H4** (recorrência de detração) e escalonamento das variáveis na análise exploratória;
- **Modelagem**: o primeiro modelo candidato, as curvas ROC e de precisão-cobertura e a tabela comparativa;
- **Modelos interpretáveis**: Regressão Logística, leitura dos *odds ratio* para a operação e a decisão sobre explicabilidade (SHAP);
- **Avaliação final (Seção 4.5)**: justificativa do modelo escolhido, confronto com as metas de negócio, plano de contingência e verificação das hipóteses.

## Documentação e apresentações

- 📄 [Documentação completa do projeto](documents/documentacao.md): entendimento do negócio, exploração, modelagem, avaliação e conclusões.
- 🛠️ [Guia de execução](GUIA_EXECUCAO.md): como instalar, reproduzir as partições, rodar os testes e os notebooks.
- 📊 Apresentações das sprints: [Sprint 1](documents/extras/apresentacoes/sprint-01.pdf) · [Sprint 2](documents/extras/apresentacoes/sprint-02.pdf) · [Sprint 3](documents/extras/apresentacoes/sprint-03.pdf) · [Sprint 4](documents/extras/apresentacoes/sprint-04.pdf)

## Como rodar os testes

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Os notebooks precisam das bases da Azul, que não são públicas. O [notebook do modelo final](notebooks/pipeline_modelo_final.ipynb) tem um modo com base sintética, descrito no [guia de execução](GUIA_EXECUCAO.md).

## Estrutura

```
├── documents/   documentação, apresentações e relatórios complementares
├── notebooks/   exploração, hipóteses, modelagem e comparação de modelos
├── src/         módulos do pipeline: limpeza, features, partições, modelos, avaliação
├── scripts/     contrato de features e geração de artefatos
├── tests/       suíte pytest sobre dados sintéticos
├── examples/    entrada operacional de exemplo, inteiramente artificial
└── assets/      figuras usadas na documentação
```

## Equipe

Arthur Augusto Proença Gonçalves · Cassio Reis Costa · Felipe Menossi Estrada · Fernanda Jawetz Steiner · **Gabriel Gomes Pimentel** · Kaylan Alexandre de Paula Sathler · Luiza Chaccur de Cresci · Pedro Estellita Leal

Orientação: Prof. Marcelo Gonçalves · Inteli, Instituto de Tecnologia e Liderança

<sub>O repositório oficial do grupo fica no GitLab do Inteli. Esta cópia reúne o código e a documentação sem o histórico de commits e sem nenhum dado do parceiro.</sub>
