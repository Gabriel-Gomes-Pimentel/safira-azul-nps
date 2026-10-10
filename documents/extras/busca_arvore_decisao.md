# Busca de hiperparâmetros da Árvore de Decisão (card #231/#232)

48 combinações, GridSearchCV com validação cruzada agrupada por Cliente (5 folds), scoring=scorer_f2 (F2, protocolo #238/#242 -- critério único das quatro duplas). Tempo real da busca: 1526s (~25min).

## Vencedor

`{'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 50, 'class_weight': 'balanced'}` -- F2 (validação cruzada) = 0,5078

## Métricas de negócio na partição de validação (não é o que a busca otimiza, é o que decide a leitura final)

| Profundidade | Folhas | Sensibilidade | Precisão Média | ROC-AUC |
|---|---|---|---|---|
| 3 | 8 | 0,3845 | 0,3955 (abaixo da meta 0,40) | 0,6460 |
| 4 | 16 | 0,4310 | 0,4162 | 0,6597 |
| **5 (final)** | **32** | **0,4759** | **0,4415** | **0,6800** |

Profundidade 5 vence nas quatro leituras (F2 da busca e as três métricas de negócio) -- não houve troca entre métrica e interpretabilidade neste caso.
