# Auditoria de reuso do split e da validação cruzada (card #239)

Registro da conferência pedida pelo card #240 (referência técnica): os quatro pipelines de
modelagem, produzidos pelas duas duplas (Modelos Interpretáveis e Ensembles), reusam
`split.dividir` e `validacao.criar_folds`/`criar_folds_validacao_por_cliente`, em vez de
reimplementar partição ou validação cruzada próprias.

## O que foi conferido

Para cada um dos quatro candidatos, dois pontos: (a) o código referencia `validacao.criar_folds`
na própria documentação/uso, e (b) **onde a busca de hiperparâmetro de fato roda** — porque é ali
que um `cv` inteiro ou um particionador solto vazaria Cliente entre treino e validação, não no
módulo que só monta o pipeline.

**A força da garantia não é a mesma nos quatro candidatos, e a tabela abaixo marca essa
diferença** (achado da revisão do Arthur em !141, incorporado nesta versão):

| Candidato | Onde a busca roda de fato | Referencia `validacao.criar_folds` | Garantia contra partição própria |
|---|---|---|---|
| Random Forest | `src/busca_random_forest.py` (`RandomizedSearchCV(cv=list(folds))`) | Sim — recusa `cv` inteiro explicitamente | **Automatizada**: `test_busca_random_forest.py::test_card_nao_cria_particao_nem_cv_inteiro` audita o mesmo arquivo que chama a busca de verdade |
| Gradient Boosting | `src/busca_gradient_boosting.py` (`RandomizedSearchCV(cv=list(folds))`) | Sim — recusa `cv` inteiro explicitamente | **Automatizada**: `test_busca_gradient_boosting.py::test_card_nao_cria_particao_nem_cv_inteiro` audita o mesmo arquivo que chama a busca de verdade |
| Regressão Logística | Célula da Seção 4 de `notebooks/regressao_logistica.ipynb` (fora do `pytest`) | Sim, na célula | **Manual**: `src/pipeline_logistica.py` (testado) só monta o pipeline e não faz a busca; `test_pipeline_nao_importa_selecao_de_modelo` prova que esse arquivo não importa `model_selection`, mas não alcança a célula do notebook onde o `GridSearchCV(cv=folds)` de fato executa. Conferido por leitura manual da célula hoje; nenhum teste protegeria uma regressão amanhã |
| Árvore de Decisão | Célula da Seção 4 de `notebooks/arvore_decisao.ipynb` (fora do `pytest`) | Sim, na célula | **Manual**, mesmo motivo da Regressão Logística: `src/pipeline_arvore.py` só monta o pipeline; a busca real está na célula do notebook |

Nenhum dos quatro candidatos foi encontrado reimplementando partição ou validação cruzada
própria. Mas só Random Forest e Gradient Boosting têm essa garantia **automatizada de ponta a
ponta**: a busca com hiperparâmetro (`RandomizedSearchCV`) roda dentro do mesmo módulo Python que
o teste audita, e o teste recusa explicitamente `cv` inteiro — a salvaguarda mais forte, porque
um inteiro faria o scikit-learn converter sozinho para um particionador estratificado que ignora
o agrupamento por Cliente. Para Regressão Logística e Árvore de Decisão, a chamada real do
`GridSearchCV` mora numa célula de notebook, que o `pytest` não coleta; a conferência para esses
dois é uma leitura manual do código da célula, válida hoje, mas sem trava automatizada contra uma
edição futura que trocasse `cv=folds` por um inteiro sem querer.

## Como foi conferido

`grep` nos quatro módulos testados por `criar_folds`, `KFold`, `train_test_split` e `cv=`,
seguido da leitura de cada teste correspondente para confirmar exatamente o que ele audita — e,
para Regressão Logística e Árvore, leitura direta da célula do notebook, já que o módulo
`src/` não cobre a chamada real da busca. Comando reprodutível, a partir da raiz do repositório:

```
grep -n "criar_folds\|KFold\|train_test_split\|cv=" src/pipeline_logistica.py src/pipeline_arvore.py src/busca_random_forest.py src/busca_gradient_boosting.py
```

Suíte completa executada (497 testes, 0 falhas) para confirmar que os quatro testes-guarda
citados passam de fato, não só existem no código.

## Conclusão

O reuso exigido pelo card #240 está confirmado nos quatro candidatos — nenhum reimplementa
partição ou validação cruzada própria. A força dessa garantia, porém, é desigual: automatizada
para Random Forest e Gradient Boosting, e manual (sem trava de teste) para Regressão Logística e
Árvore de Decisão, porque a busca real destes dois roda em célula de notebook. Nenhuma ação
corretiva foi necessária hoje, mas fica registrado como risco: uma dupla de modelos interpretáveis
poderia mover a chamada real de `GridSearchCV` pra fora de `criar_folds` numa edição futura do
notebook sem que nenhum teste acusasse.
