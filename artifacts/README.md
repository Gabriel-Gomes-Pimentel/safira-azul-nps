# Artefatos locais

Modelos, manifestos reais e outputs ficam apenas no ambiente autorizado. Todo o
conteúdo desta pasta, exceto este guia, está ignorado pelo Git.

`scripts/gerar_artefato_modelo.py` produz um pipeline ajustado e seu manifesto.
O modo `--sintetico` serve somente para demonstração funcional. Nenhum artefato
gerado por este pacote tem aprovação automática para produção.

## Geração sintética

Na raiz do repositório, usando o ambiente de treinamento:

```bash
python scripts/gerar_artefato_modelo.py --sintetico \
  --modelo artifacts/modelo-sintetico-v1.joblib \
  --manifesto artifacts/manifesto-sintetico-v1.json \
  --versao-modelo sintetico-v1 --versao-fontes artificial
```

Os destinos não podem existir: o script recusa sobrescrever uma versão.
Para fontes reais autorizadas, substituir `--sintetico` por `--base` com o
caminho local da base analítica Parquet. Não publicar os arquivos gerados.

## Manifesto

O JSON gerado registra:

| Campos | Conteúdo |
| --- | --- |
| `versao_modelo`, `versao_contrato`, `versao_politica_fila`, `commit_fonte` | Identificação da entrega e do código |
| `sha256`, `sha256_configuracao` | Integridade do binário e da configuração |
| `versoes` | Python e bibliotecas efetivamente utilizadas |
| `features`, `tipos`, `features_transformadas` | Schema bruto e saída do pré-processador |
| `classes`, `classe_positiva` | Classes do modelo; Detrator corresponde a 1 |
| `parametros`, `semente` | Configuração efetiva do estimador |
| `cortes`, `split`, `periodo_treino` | Particionamento e período observado no treino |
| `calibracao` | Sigmoid no treino, explicitamente in-sample |
| `versao_fontes`, `fingerprint_base`, `gerado_em` | Proveniência e instante de geração |
| `dados_sinteticos`, `aprovado_producao` | Demonstração artificial e ausência de aprovação automática |
| `avaliacao_referencia`, `limitacoes` | Referência histórica e limites da entrega |

O exemplo completo é produzido localmente pelo comando acima. Os testes de
`tests/test_artefato_modelo.py` verificam manifesto, hash, equivalência após
serialização e determinismo exclusivamente com dados artificiais.

O manifesto registra o ambiente efetivo e o SHA-256. Carregar apenas arquivos de
origem confiável: hash não autentica o fornecedor e Joblib pode executar código.
