# Composição dos conjuntos de treino, validação e teste

Material de apoio do card #132, pronto para a redação do item (a) da Seção 4.3 no #116. Este
arquivo não redige a seção — entrega a tabela e a figura já geradas pelo notebook
`notebooks/modelagem.ipynb` (seção 1.7), sem valor digitado à mão, para que #116 escreva a prosa
ao redor.

A tabela e a figura abaixo estão prontas para colar em `documents/documentacao.md`; os caminhos
de imagem já seguem a convenção usada lá (`../assets/...`, relativos a `documents/`).

## Tabela de composição

| Conjunto | n | Período | Taxa de detração | % do total |
|---|---|---|---|---|
| Treino | 341.962 | 2023-07-01 a 2025-06-30 | 20,43% | 70,5% |
| Validação | 48.301 | 2025-07-01 a 2025-12-31 | 21,60% | 10,0% |
| Teste | 53.486 | 2026-01-01 a 2026-06-30 | 20,41% | 11,0% |

Os três conjuntos somam 443.749 das 484.915 linhas da base (91,5%). Os 41.166 restantes (8,5%)
não entram em nenhum dos três, por dois motivos distintos: 41.063 linhas de Cliente recorrente,
removidas pela regra de desempate que mantém cada `ID_GOLDENRECORD` em um único conjunto (a mais
recente em que ocorre), e 103 linhas sem `ID_GOLDENRECORD`, que a divisão por grupo exclui por
não poder confirmar se pertencem à mesma pessoa. Nenhuma linha fica de fora por falta de data —
a base atual não tem nenhuma (`linhas_sem_data = 0`); a política de anterioridade em
`verificar_anterioridade_sem_data` (`src/split.py`) segue como salvaguarda para uma fonte futura
que volte a chegar sem a coluna.

## Figura da linha do tempo

Pela ordem de leitura do documento, esta figura entra dentro da Seção 4.3, que começa antes do
Anexo A.1.1 — logo, ela vem depois da Figura 8 (a última antes da 4.3) e antes das Figuras 9 a 11
do anexo, que precisam subir para 10, 11 e 12. Ela é, portanto, a **Figura 9**, não a 12.

**Atenção ao mesclar:** a MR !65 insere uma figura na Seção 3, antes da 4.3; se ela mergear
depois desta, os números de Figuras 1 a 8 também deslocam, e este número precisa ser conferido
de novo contra a versão de `develop` no momento do merge do #116.

A figura já traz, no rodapé, que os três conjuntos cobrem 91,5% da base — o mesmo número do
parágrafo acima — para que o leitor não tente reconciliar os 100% do eixo visual com a tabela.

```html
<div align="center">
  <sub>Figura 9 – Linha do tempo do particionamento temporal</sub><br>
  <img src="../../assets/linha-tempo-particionamento.png" width="100%" alt="Linha do tempo mostrando os cortes de validação em 2025-07-01 e de teste em 2026-01-01 sobre o horizonte da base, com os três conjuntos em cores distintas, e nota de rodapé informando que os três conjuntos cobrem 91,5% da base"><br>
  <sup>Fonte: Autoria própria.</sup>
</div>
```

## Como os números foram gerados

`notebooks/modelagem.ipynb`, seção 1.7, reaproveita `tabela_particoes` (já calculada na seção
1.5, a partir de `split.dividir` sobre a base analítica real) para montar a tabela acima e a
figura. Nenhum split é recalculado nesta seção. Para reproduzir, rodar o notebook de ponta a
ponta contra `data/processed/base_analitica.parquet`.
