# Tabela comparativa entre o modelo candidato e os pisos

Material de apoio do card #108, pronto para a redação do item (d) da Seção 4.3 e para a Seção 4.4.
Este arquivo não redige a seção: entrega as tabelas já geradas pela seção 9 de
`notebooks/modelagem.ipynb`, sem valor digitado à mão, para que a prosa seja escrita ao redor.

Todos os números saem da **partição de teste** (2026-01-01 a 2026-06-30, 53.486 respostas, 10.919
Detratores, 20,41%). Nenhum vem dos folds de validação das seções 3 e 5.

As duas tabelas entre os marcadores abaixo são **regeneradas pela célula da seção 9** a partir do
mesmo DataFrame que produz o JSON. Não editar à mão: trocar o modelo candidato e reexecutar as
atualiza, e uma edição manual seria sobrescrita na execução seguinte. A prosa fora dos marcadores
é escrita à mão e não é tocada pela célula.

<!-- TABELAS-GERADAS:inicio -->

## Tabela comparativa

| Modelo | Precisão média | ROC-AUC | Brier | Limiar | Fila | Fila comparável | Precisão no topo | Cobertura | F1 |
|---|---:|---:|---:|---:|---:|:-:|---:|---:|---:|
| classe majoritária | 0,2041 | 0,5000 | 0,2041 | 0,0000 | 53.486 | não | 0,2041 | 1,0000 | 0,3391 |
| regressão logística | 0,5019 | 0,7442 | 0,1877 | 0,5993 | 9.050 | sim | 0,5283 | 0,4379 | 0,4788 |
| **gradient boosting** | **0,5212** | **0,7492** | **0,1312** | **0,2934** | **9.050** | **sim** | **0,5386** | **0,4464** | **0,4882** |

## Ganho do candidato sobre cada piso

| Métrica | Piso | Valor do piso | Candidato | Ganho absoluto | Razão |
|---|---|---:|---:|---:|---:|
| Precisão média | classe majoritária | 0,2041 | 0,5212 | +0,3171 | 2,55x |
| Precisão média | regressão logística | 0,5019 | 0,5212 | +0,0193 | 1,04x |
| ROC-AUC | classe majoritária | 0,5000 | 0,7492 | +0,2492 | 1,50x |
| ROC-AUC | regressão logística | 0,7442 | 0,7492 | +0,0050 | 1,01x |
| Precisão no topo | classe majoritária | 0,2041 | 0,5386 | +0,3344 | 2,64x |
| Precisão no topo | regressão logística | 0,5283 | 0,5386 | +0,0103 | 1,02x |

<!-- TABELAS-GERADAS:fim -->

## Como ler a coluna do limiar

As três primeiras métricas não dependem de ponto de corte. As cinco últimas dependem, e o mesmo
orçamento de `k = 50 x 181 = 9.050` contatos foi pedido aos três modelos, em vez do mesmo valor
numérico de limiar. Igualar o número seria a comparação errada: cada modelo emite score na sua
própria escala, e 0,29 na escala do candidato não significa o mesmo que 0,29 na escala da
logística.

**Pedir o mesmo orçamento não garante capacidade efetiva igual, e a coluna `Fila comparável` é
onde isso se lê.** O piso trivial emite a mesma probabilidade para toda linha, o limiar cai sobre
essa constante e a seleção alcança a partição inteira, 53.486 e não 9.050. As cinco métricas
dependentes de corte da linha dele descrevem uma fila de outro tamanho e **não são comparáveis**
às das outras duas neste orçamento. As três primeiras seguem comparáveis, porque não dependem de
corte.

Não há desempate embutido para cortar a fila degenerada em exatamente 9.050. Qualquer regra desse
tipo teria de escolher quais empatados entram, e sobre um score constante essa escolha é
arbitrária: o resultado dependeria da ordem das linhas, e não do modelo.

## O que a tabela sustenta, e o que não sustenta

**O ganho sobre o piso trivial é grande e não é a notícia.** Superar um modelo que não olha
nenhuma feature é o mínimo exigido, e é para isso que o piso linear existe ao lado.

**O ganho sobre a regressão logística é pequeno e consistente.** Na mesma fila de 9.050 contatos,
o candidato encontra 4.874 Detratores contra 4.781 da logística: 93 Detratores a mais no semestre,
cerca de meio por dia de operação. É ganho real nas três métricas, mas apresentá-lo como salto
seria distorcer.

**O Brier é a maior diferença entre os dois, e o que ele sustenta tem limite.** O escore cai de
0,1877 para 0,1312, e o candidato é o único dos três melhor do que o preditor constante igual à
prevalência (0,1625, seção 6.2 do notebook). Isso sustenta que ele tem o melhor erro
probabilístico dos três e que a média do score fica próxima da frequência observada, 0,1976 contra
0,2041.

Não sustenta, porém, que a probabilidade possa ser lida como risco faixa a faixa. O Brier agrega
calibração e discriminação num único número, e uma melhora dele pode vir de qualquer um dos dois
componentes; a conferência da seção 6.2 é de média global, e média próxima não demonstra
calibração por faixa. **A curva de calibração prevista na Seção 4.1.3 é o que fecharia essa
afirmação, e ainda não existe no projeto.** Quem redigir a 4.3 ou a 4.4 não deve escrever que o
score é lido como risco por faixa enquanto essa curva não estiver no documento.

**Uma meta da Seção 4.1.3 não foi atingida.** O ROC-AUC ficou em 0,7492 contra a meta de 0,75,
uma diferença de 0,0008. Não muda conclusão prática nenhuma, mas arredondar para 0,75 na redação
afirmaria que a meta foi cumprida quando não foi. A precisão média, meta de 0,40, ficou em 0,5212.

## Como os números foram gerados

`notebooks/modelagem.ipynb`, seção 9, por `modelo.tabela_comparativa` e `modelo.ganhos_do_candidato`
(`src/modelo.py`), sobre os mesmos objetos `trivial`, `logistica` e `candidato` das seções 2 e 4.1.
A mesma célula formata as tabelas acima a partir desse DataFrame e as reescreve entre os
marcadores, e grava os valores sem arredondamento em `documents/extras/resultados/comparativo_modelos.json`. Para
reproduzir, rodar o notebook de ponta a ponta contra `data/processed/base_analitica.parquet`.
