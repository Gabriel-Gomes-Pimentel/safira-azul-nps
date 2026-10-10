# Entrada sintética safira-batch-v1

O contrato normativo está em [contrato-entrega-operacional.md](../documents/extras/contrato-entrega-operacional.md).
Não colocar dados reais, artefatos reais ou identificadores pessoais neste diretório.

## Interfaces

Com `src/` no caminho de importação:

```python
from contrato_inferencia import COLUNAS_MODELO, validar_entrada
from dados_sinteticos_operacionais import criar_entrada_sintetica

entrada = criar_entrada_sintetica()
normalizada, correcoes = validar_entrada(
    entrada,
    t_score="2026-10-06T12:00:00Z",
    fuso_operacional="America/Sao_Paulo",
)
x = normalizada.loc[:, list(COLUNAS_MODELO)]
```

- `ler_entrada(caminho: str | Path) -> pd.DataFrame`: CSV UTF-8, vírgula, cabeçalho único e schema exato, registros com a mesma quantidade de campos. Todos os campos são lidos como texto, inclusive vazios; `NA`/`NULL` e zeros iniciais permanecem literais.
- `validar_entrada(df, *, t_score: str, fuso_operacional: str) -> tuple[pd.DataFrame, dict]`: cópia com as quatro colunas de passagem, as 14 features na ordem canônica e `DIA_OPERACIONAL` (`YYYY-MM-DD`). Preserva índice, linhas, IDs e timestamps; não modifica a entrada.
- `ErroContrato(ValueError)`: bloqueia o lote inteiro com regra/coluna/quantidade, sem valores individuais, nomes arbitrários de colunas extras ou caminhos.
- `COLUNAS_MODELO`: tupla do V1 seguida do histórico. `COLUNAS_PASSAGEM`: lista dos dois IDs e dos dois timestamps. Nenhuma coluna de passagem ou dia entra em `x`.

Campos vazios/em branco e nulos reais são ausência; textos numéricos inválidos não são convertidos silenciosamente. Booleano aceita somente `true`/`false` ou booleanos reais. Categorias novas são aceitas como texto e normalizadas conforme treino, sem aprender categorias. Números usam decimal ponto, sem booleanos ou infinitos. O parsing numérico precede a máscara de cancelamento; `N_TRECHOS` é validado depois da máscara. A contagem histórica é obrigatória, inteira e não negativa; zero exige indicador/taxa nulos, positivo exige indicador binário e taxa finita em `[0,1]`, com indicador 1 exatamente quando a taxa é positiva.

O resumo contém somente contagens inteiras: `categorias_normalizadas` (células textuais alteradas), `tempo_voo_nao_positivo` (durações convertidas em nulo) e `campos_mascarados_cancelamento` (células ainda preenchidas removidas pela máscara). Lote vazio com schema correto retorna cópia válida e contagens zero.

## Limites

Não há `fit`, imputação, encoding, recálculo de histórico, carregamento de artefato ou scoring nesta interface. Sem encoder ajustado, não se contabilizam categorias desconhecidas. Timestamps declarados até `t_score` não comprovam reconstrução as-of, ausência de resposta atual, proveniência autorizada ou população diária completa. Essas garantias continuam sob responsabilidade da fonte e da integração; validação sintética não autoriza uso de dados reais.

Os casos inteiramente artificiais e executáveis estão em `tests/test_contrato_inferencia.py`; nenhum CSV real é necessário.
