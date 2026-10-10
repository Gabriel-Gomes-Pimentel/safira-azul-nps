# Interpretação das regras da Árvore de Decisão para a operação (card #234)

Leitura das 32 folhas da árvore final (`criterion=entropy, max_depth=5, min_samples_leaf=50,
class_weight=balanced` — vencedora da busca por `scorer_f2`, ver #232/#233), traduzida para
decisões que a área de Experiência do Cliente da Azul reconhece. Todos os números vêm de
`documents/extras/folhas_arvore_decisao.md` (taxa observada) e da estrutura real da árvore —
os thresholds numéricos foram desfeitos do `RobustScaler` de volta para minutos/dias, conforme
o dicionário de dados da Seção 4.2. Com 32 folhas, esta interpretação resume os **padrões
dominantes**, não enumera folha por folha.

**Associação, não causa.** As regras abaixo descrevem o que o modelo encontrou associado à
detração na base histórica, não o que causa a detração.

## O que domina a árvore

**1. Atraso de chegada (`ATRASO_CHEGADA`) continua sendo o primeiro corte, agora em 43,5 min.**
Acima disso a árvore mantém cortando em 61,5 / 99,5 / 151,5 minutos — quanto mais atraso, maior
o risco, de forma consistente:

| Faixa de atraso na chegada | % da base de treino | Taxa média na faixa | Taxa observada por folha (min–max) |
|---|---|---|---|
| ≤ 43,5 min | 83,3% | 14,6% | 12,7% a 87,1%* |
| 43,5–61,5 min | 1,7% | 41,5% | 35,7% a 77,0% |
| 61,5–99,5 min | 1,8% | 52,4% | 42,3% a 56,3% |
| 99,5–151,5 min | 1,1% | 65,7% | 58,5% a 94,8% |
| > 151,5 min | 2,0% | 76,5% | 73,4% a 95,8% |

\* dentro de ≤43,5min, a única folha sem nenhum outro fator de risco (sem cancelamento
recente, sem histórico de detração) já responde por **72,6% de toda a base** e tem taxa de
12,7% — é o piso de risco da maioria dos Clientes. A variação até 87,1% dentro dessa mesma
faixa de atraso vem do cancelamento com pouca antecedência e do histórico do Cliente — itens
2 e 3 abaixo.

**Leitura para a operação:** o piso de risco (a maioria dos clientes, sem problema de atraso ou
histórico) fica em ~13-15%; a partir de ~2h30 de atraso combinado com outros fatores o risco
chega a 4-5x a prevalência geral. (As percentuais da tabela somam ~90%, não 100%: os ~10,1%
restantes têm `ATRASO_CHEGADA` nulo na base bruta — o pipeline imputa a mediana, 0 minutos, e
essas linhas entram operacionalmente na faixa ≤43,5min.)

**2. Histórico de detração do próprio Cliente (`HIST_TAXA_DETRACAO_ANTERIOR`,
`HIST_DETRATOU_ANTES`) é o segundo sinal mais forte — e aparece nas folhas de risco mais
extremo da árvore inteira.**

As cinco folhas com maior taxa de detração (87,1%, 85,7%, 86,9%, 94,8%, 95,8%) têm em comum um
valor alto de histórico de detração anterior do Cliente, combinado com atraso relevante. Isso é
coerente com o que a Seção 4.2.4 já havia medido na exploração de dados: quem detratou uma vez
tem chance bem maior de detratar de novo.

**Leitura para a operação:** Cliente com histórico de detração que enfrenta um novo problema
operacional é o segmento de maior risco identificado — candidato natural a contato preventivo,
não só reativo.

**3. Cancelamento com pouca antecedência (`ANTECEDENCIA_CANCELAMENTO ≤ 9,5 dias`) continua
sendo um sinal forte, isolado do atraso.**

Mesma ressalva de antes: essa coluna só existe preenchida para voos com cancelamento (16,3% da
base); para os demais 83,7% ela é nula e o pré-processador imputa a mediana (10 dias) — o corte
em 9,5 dias separa, na prática, cancelamento recente do restante da base. Dentro desse ramo, as
taxas vão de 51,1% a 87,1%, subindo ainda mais quando combinado com tier Diamante ou histórico
de detração.

**4. Fidelidade (`TIER_VIAGEM_DIAMANTE`) mantém o mesmo padrão da versão anterior: Cliente
Diamante detrata mais dentro do mesmo contexto de atraso/cancelamento, não menos.**

Nos pares de folhas comparáveis desta árvore:

| Contexto | não-Diamante | Diamante | Diferença |
|---|---|---|---|
| Cancelamento 1,5–5,5 dias de antecedência | 51,1% | 64,5% | +13,4 pp |
| Atraso 99,5–151,5 min, sem detração prévia | 63,0% | 75,0% | +12,0 pp |

**Leitura para a operação:** confirma o achado da versão anterior desta análise — Cliente
Diamante reclama proporcionalmente mais quando a viagem tem problema, o oposto da intuição de
"cliente fiel é mais tolerante".

## Limitações da leitura

- Com 32 folhas a árvore já não é uma lista curta de regras — esta interpretação prioriza os
  padrões que se repetem em várias folhas em vez de descrever cada uma. A tabela completa está
  em `folhas_arvore_decisao.md` para quem precisar do detalhe folha a folha.
- O rótulo `class: 0/1` do `export_text` reflete o voto ponderado por `class_weight="balanced"`,
  não a maioria observada — todas as taxas citadas aqui vêm da contagem real por folha.
- `ANTECEDENCIA_CANCELAMENTO` é condicional a `CANCELAMENTO_VOO`, como registrado na Seção 4.2.
- Esta árvore foi escolhida pelo critério oficial de busca do protocolo do ART.7 (F2, card
  #242) — e, ao contrário da primeira tentativa deste card (baseada em `average_precision`,
  descartada), ela também vence nas três métricas de negócio (Sensibilidade, Precisão Média,
  ROC-AUC) medidas na validação, então não houve troca entre desempenho e interpretabilidade
  desta vez.
