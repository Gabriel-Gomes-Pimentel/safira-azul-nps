# Inteli - Instituto de Tecnologia e Liderança 

<p align="center">
<a href= "https://www.inteli.edu.br/"><img src="assets/inteli.png" alt="Inteli - Instituto de Tecnologia e Liderança" border="0"></a>
</p>

# Safira

## Avatares

O **Safira** é desenvolvido pelo grupo **Avatares** (G01), em parceria com a Azul Linhas Aéreas.

## :student: Integrantes: 
- <a href="https://www.linkedin.com/in/arthur-proen%C3%A7a-87522b355">Arthur Augusto Proença Gonçalves</a>
- <a href="https://www.linkedin.com/in/cassio-reis-costa-0989803b9/">Cassio Reis Costa</a>
- <a href="https://www.linkedin.com/in/felipe-menossi-estrada/">Felipe Menossi Estrada</a>
- <a href="https://www.linkedin.com/in/fernanda-steiner-938806313/">Fernanda Jawetz Steiner</a>
- <a href="https://www.linkedin.com/in/gabriel-gomes-pimentel/">Gabriel Gomes Pimentel</a>
- <a href="https://www.linkedin.com/in/kaylan-alexandre/">Kaylan Alexandre de Paula Sathler</a>
- <a href="https://www.linkedin.com/in/luiza-chaccur-de-cresci-450015397/">Luiza Chaccur de Cresci</a>
- <a href="https://www.linkedin.com/in/pedro-leal-5b8788341/">Pedro Estellita Leal</a>

## :teacher: Professores:
### Orientador(a) 
- <a href="https://www.linkedin.com/in/marcelo-gon%C3%A7alves-phd/">Marcelo Gonçalves</a>
### Instrutores
- <a href="https://www.linkedin.com/in/zotovici/">Andréa Zotovici</a>
- <a href="https://www.linkedin.com/in/bruna-mayer/">Bruna Mayer Costa</a>
- <a href="https://www.linkedin.com/in/camilanarantes/">Camila Naves Arantes</a>
- <a href="https://www.linkedin.com/in/leandromundim/">Leandro Resende Mundim</a>
- <a href="https://www.linkedin.com/in/natalia-k-37a62052/">Natalia Kloeckner</a>


## 📝 Descrição

O **Safira** é o modelo preditivo que o G01 desenvolve com a Azul Linhas Aéreas para antecipar a detração de Clientes antes da resposta à pesquisa de NPS.

Hoje a companhia só sabe quem detratou depois que a nota foi dada. A pesquisa chega um dia após o voo, para metade dos Clientes domésticos, e fica aberta por sete dias. É nesse intervalo que ainda seria possível reverter uma experiência ruim, mas a área de Experiência do Cliente atua sem critério objetivo de priorização: trata algumas centenas de casos por dia, enquanto um único fim de semana de mau tempo afeta milhares de passageiros. Quando o Detrator aparece no indicador, a informação já deixou de ser acionável.

A solução proposta é um modelo de classificação que estima, para cada jornada encerrada, a probabilidade de o Cliente se tornar Detrator, usando apenas dados operacionais e de perfil disponíveis antes da resposta à pesquisa. A saída tem dois usos complementares: a **pontuação individual de risco**, que ordena a lista de contato da equipe de Experiência do Cliente dentro da janela em que a recuperação ainda é possível, e o **diagnóstico agregado**, que hierarquiza os fatores de insatisfação e orienta onde investir em melhoria operacional.

A base analisada reúne 484.915 respostas de 36 meses de operação doméstica, de julho de 2023 a junho de 2026, integradas pela chave `RESPONDENT_ID` às informações de voo e ao perfil do Cliente. A exploração dos dados, o pré-processamento e as hipóteses testadas estão na seção 4.2 da documentação e nos notebooks deste repositório. O modelo preditivo em si é construído a partir da Sprint 3.

<b>Link para vídeo demonstrativo:</b><a href="https://youtu.be/c-2v88OHld4"> Vídeo Demonstrativo</a>

## 📁 Estrutura de pastas

Dentre os arquivos presentes na raiz do projeto, definem-se:

- <b>README.md</b>: arquivo que serve como guia e explicação geral sobre o projeto (o mesmo que você está lendo agora).

- <b>requirements.txt</b>: dependências Python do projeto, com as versões de scikit-learn, scipy e xgboost fixadas para que a comparação de modelos seja reproduzível. É o arquivo instalado na seção Execução dos projetos.

- <b>.gitignore</b>: lista o que nunca é versionado, a começar pelas bases da Azul (`data/` e os arquivos `.csv`, `.xlsx`, `.pkl` e `.parquet`), conforme o compromisso de proteção de dados com o parceiro.

- <b>.gitattributes</b>: normaliza o fim de linha dos arquivos de texto e marca PDFs, imagens e `.parquet` como binários.

- <b>assets</b>: todas as imagens e mídias utilizadas nos notebooks e documentação são posicionadas aqui.

- <b>documents</b>: o documento do projeto, [documentacao.md](documents/documentacao.md), e a pasta <b>extras</b>, com os documentos complementares: as apresentações de sprint (`apresentacoes/`), os resultados das buscas e comparações em JSON (`resultados/`) e as execuções do notebook integrado (`reprodutibilidade-notebook-integrado/`). O índice dessa pasta está em <a href="documents/extras/README.md">documents/extras</a>.

- <b>notebooks</b>: todos os Jupyter Notebooks criados para desenvolvimento do projeto.

- <b>src</b>: módulos Python importados pelos notebooks, do dado à métrica: integração e limpeza das bases, estatística descritiva, figuras, particionamento e congelamento dos conjuntos, matriz do modelo, pipelines e espaços de busca dos modelos, avaliação, calibração e explicabilidade. É a implementação canônica: os notebooks importam essas funções em vez de reimplementá-las.

- <b>scripts</b>: o módulo `preprocessamento_nps.py`, que define o contrato do score pós-viagem (allowlist do Feature Set V1, derivação de `N_TRECHOS`, validação de schema e preparação da base analítica), documentado nas Seções 4.2.2 a 4.3.2.5 de [documents/documentacao.md](documents/documentacao.md), e utilitários executados pela linha de comando, como `gerar_dummy.py`, que gera a base sintética usada nos testes.

- <b>tests</b>: testes automatizados executáveis com `pytest`, sem dependência das bases do parceiro: cobrem as travas de integridade, o contrato de dados, o particionamento e os módulos de modelagem, e executam o notebook integrado sobre a base sintética.

- <b>artifacts</b>: guia de geração do modelo calibrado e do manifesto. Os artefatos gerados permanecem locais e ignorados pelo Git.

- <b>examples</b>: exemplo executável de validação da entrada operacional com dados inteiramente artificiais.

- <b>requirements-operacional.txt</b>: dependências fixadas do ambiente batch proposto, sem Jupyter; homologação pendente no #322.

- <b>requirements-treinamento.txt</b>: dependências do ambiente operacional e leitura de Parquet para geração do artefato.

O [notebook-pipeline do modelo final](notebooks/pipeline_modelo_final.ipynb) carrega o Joblib e percorre avaliacao, graficos, importancia por permutacao e fila diaria simulada. Suas cinco cenas seguem o roteiro de video de dois minutos; o modo de gravacao usa somente dados artificiais e nao declara homologacao.

A divisão entre `src/` e `scripts/` é histórica, e não por tipo de código: os dois guardam módulos importáveis, e os testes de `tests/` colocam ambos no caminho de importação. O grupo optou por não mover arquivos entre as pastas nesta etapa, porque notebooks, testes e documentação referenciam os caminhos atuais.

Três pastas aparecem só na máquina de quem executa o projeto e **não estão no repositório**, porque o `.gitignore` as exclui: `data/`, com as bases da Azul em `data/raw/` e a base analítica e as partições geradas em `data/processed/`; `figuras/`, com os PNGs gravados por `exploracao_dados.ipynb` e `escalonamento_anexo_a1.ipynb`; e `.execucao/`, com as cópias executadas pelo `jupyter nbconvert`.

## 💻 Execução dos projetos

### Modelo final: execução principal

O ponto de entrada da entrega é [notebooks/pipeline_modelo_final.ipynb](notebooks/pipeline_modelo_final.ipynb). O fluxo é **Joblib e manifesto -> avaliação -> gráficos -> importância por permutação -> pontuação -> fila diária simulada**. O notebook carrega o preprocessador e o Gradient Boosting calibrado já ajustados; não executa treinamento ou recalibração.

A demonstração padrão usa exclusivamente dados artificiais. **Não exige bases da Azul nem a execução prévia dos notebooks de exploração, pré-processamento ou treinamento.** As métricas sintéticas demonstram funcionamento, não o desempenho real publicado na [Seção 4.5 da documentação](documents/documentacao.md). O ambiente abaixo é proposto para a demonstração local; não representa homologação operacional pela Azul.

#### 1. Preparar o ambiente

Na raiz do repositório, use **CPython 3.12.3** para gerar e carregar o artefato no mesmo ambiente. Confirme `python --version` antes de criar o ambiente; no Linux/macOS, use `python3.12` no lugar de `python` se necessário.

```bash
python -m venv .venv-operacional
```

Ative o ambiente conforme seu sistema:

| Sistema | Comando de ativação |
|---|---|
| Windows, PowerShell | `.venv-operacional\Scripts\Activate.ps1` |
| Windows, Prompt de Comando | `.venv-operacional\Scripts\activate.bat` |
| Windows, Git Bash | `source .venv-operacional/Scripts/activate` |
| Linux ou macOS | `source .venv-operacional/bin/activate` |

Instale as dependências do artefato e as ferramentas do notebook:

```bash
python -m pip install -r requirements-treinamento.txt
python -m pip install jupyterlab ipykernel nbconvert nbclient matplotlib pytest
python -m pip check
```

`requirements-treinamento.txt` inclui os pins de `requirements-operacional.txt` e a leitura Parquet. Para esta demonstração, não é necessário instalar todos os modelos alternativos de `requirements.txt`. O manifesto registra as versões efetivas de Python e das bibliotecas do artefato; qualquer divergência bloqueia a carga. As ferramentas de visualização não substituem esses pins.

#### 2. Gerar ou reutilizar o artefato sintético

Se ainda não existir o par abaixo, gere-o na raiz do repositório:

```bash
python scripts/gerar_artefato_modelo.py --sintetico --modelo artifacts/modelo-sintetico-v1.joblib --manifesto artifacts/manifesto-sintetico-v1.json --versao-modelo sintetico-v1 --versao-fontes artificial
```

Resultado esperado: mensagem `Artefato gerado`, SHA-256 e `aprovado_producao=false`. Esse script **treina e serializa** o candidato usando dados artificiais; o notebook da etapa seguinte apenas carrega o resultado. Os dois arquivos ficam locais em `artifacts/`, protegidos pelo `.gitignore`, e não são publicados no GitLab.

O script não sobrescreve arquivos. Se o par já existir e tiver origem confiável, reutilize-o e pule a geração. Para uma nova versão, escolha novos nomes e uma nova `--versao-modelo`; na célula de configuração do notebook, ajuste `CAMINHO_MODELO` e `CAMINHO_MANIFESTO` para esse novo par. Não combine binário e manifesto de versões diferentes.

SHA-256 verifica integridade, não autentica o fornecedor. Carregue somente Joblib e manifesto obtidos por canal confiável: a desserialização pode executar código.

#### 3. Abrir e executar o notebook final

Registre o kernel do ambiente ativo e abra o notebook:

```bash
python -m ipykernel install --user --name safira-final --display-name "Safira final (CPython 3.12.3)"
python -m jupyter lab notebooks/pipeline_modelo_final.ipynb
```

No JupyterLab, selecione **Kernel > Change Kernel > Safira final (CPython 3.12.3)** e execute **Run > Run All Cells**. Mantenha `MODO_VIDEO=True` e `USAR_BASE_SINTETICA=True`, os padrões da célula de configuração. O notebook encontra a raiz pelo diretório `src/` e usa os dois arquivos gerados na etapa 2.

Ao final, são exibidos:

- Precisão Média, ROC-AUC e Brier na partição de teste, com identificação de demonstração sintética.
- Curvas ROC, Precisão-Recall e calibração, além da matriz de confusão no limiar diagnóstico de `0.5`. Esse limiar não controla a fila diária.
- Importância por permutação na validação, com tabela completa e gráfico. O alvo artificial depende do atraso, então uma única feature relevante é possível; não representa fatores reais da Azul nem prova causalidade.
- Fila simulada de um dia com **120 jornadas, 60 Clientes e 50 contatos sugeridos**, contato único por Cliente, desempate determinístico e capacidade de 50 como premissa do projeto.

Os gráficos são mostrados no notebook, sem exportação obrigatória. A fila não realiza contatos e não comprova homologação do CLI operacional. A exportação opcional da fila artificial permanece em `outputs/`, ignorado pelo Git. Os notebooks versionados ficam sem outputs; não versione cópias executadas nem `notebooks/Roteiro.ipynb`.

#### 4. Verificar sem interface gráfica

No mesmo ambiente ativo, o teste cria um artefato artificial temporário, executa todas as células em um kernel desse ambiente e verifica os gráficos, o schema e as regras da fila:

```bash
python -m pytest tests/test_pipeline_modelo_final.py -q
```

Resultado esperado: `2 passed`, sem acesso a dados reais. Para executar diretamente o par da etapa 2, com uma cópia local do notebook executado:

```bash
python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.kernel_name=safira-final --ExecutePreprocessor.timeout=180 --output-dir=.execucao notebooks/pipeline_modelo_final.ipynb
```

O arquivo executado fica em `.execucao/`, protegido pelo `.gitignore`; o notebook fonte não é alterado. O comando termina com código zero quando todas as células executam. Não envie esse output ao GitLab.

#### Uso real local e erros de carga

Avaliação real é restrita ao ambiente autorizado: exige modelo e manifesto reais confiáveis, a base analítica original em `data/processed/base_analitica.parquet`, fingerprint correspondente e ambiente idêntico ao registrado. Para esse modo, desative `MODO_VIDEO` e `USAR_BASE_SINTETICA` e ajuste os caminhos do artefato na configuração. A fila de demonstração continua artificial. Não publique dados, identificadores, gráficos ou outputs reais no vídeo ou no repositório.

Se a carga falhar por hash, versão, schema ou origem incompatível, interrompa a execução e confira o par e o ambiente contra o manifesto. Não remova as verificações nem edite o hash para contornar o erro. Em caso de destino já existente, reutilize a versão confiável ou gere outra com novos nomes. Consulte o [guia de artefatos](artifacts/README.md) e o [contrato operacional](documents/extras/contrato-entrega-operacional.md) para os limites de calibração in-sample, histórico temporal e liberação para produção.

### Fluxo de desenvolvimento: análise e treinamento

As instruções a seguir preservam a reprodução dos estudos que originaram o modelo. **Não são pré-requisitos da demonstração sintética do notebook final.** A execução com bases reais depende de autorização e de ambiente local protegido.

### Bases de dados

As bases fornecidas pela Azul **não são versionadas neste repositório**, conforme o Termo de Abertura de Projeto de Inovação, que veda a publicação de dados do parceiro. O diretório `data/` está no `.gitignore`, assim como qualquer arquivo `.csv`, `.xlsx`, `.pkl` ou `.parquet`.

Para executar os notebooks de desenvolvimento com dados reais, é preciso obter com o grupo os oito arquivos da Azul e colocá-los em `data/raw/`, na raiz do repositório:

```
data/raw/PROJETO_INTELI.NPS_01.csv ... PROJETO_INTELI.NPS_04.csv
data/raw/PROJETO_INTELI.PERFIL_CLIENTE_01.csv e PROJETO_INTELI.PERFIL_CLIENTE_02.csv
data/raw/PROJETO_INTELI.INFORMACAO_VIAGEM.csv
data/raw/PROJETO_INTELI.DISTRIBUICAO_PAX_NORMALIZADO.csv
```

Deixe fora de `data/raw/` as cópias `.xlsx` das mesmas bases: o pré-processamento compara os dois formatos de uma mesma parte e interrompe a execução se eles divergirem. A pasta `data/processed/` é criada pelos notebooks e não precisa de nenhum arquivo colocado à mão.

### Desenvolvimento local (VS Code com Python)

Requer **Python 3.12 ou superior**: as versões de `scipy` e `xgboost` fixadas no `requirements.txt` não instalam em versões anteriores. Confira a versão antes de criar o ambiente:

```bash
python --version
```

Se aparecer 3.11 ou menos, instale o Python 3.12 pelo [python.org](https://www.python.org/downloads/) e use `python3.12` (no Windows, `py -3.12`) no lugar de `python` no comando abaixo. Na raiz do repositório, crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente com o comando do seu sistema:

| Sistema | Comando de ativação |
|---|---|
| Windows, PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows, Prompt de Comando | `.venv\Scripts\activate.bat` |
| Windows, Git Bash | `source .venv/Scripts/activate` |
| Linux ou macOS | `source .venv/bin/activate` |

No Linux e no macOS, use `python3` no lugar de `python` se o segundo não existir. Se o PowerShell recusar o script de ativação por causa da política de execução, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e ative de novo.

Com o ambiente ativo, instale as dependências e abra o Jupyter:

```bash
pip install -r requirements.txt
jupyter lab
```

Execute os notebooks nesta ordem, cada um com todas as células em sequência:

1. **`notebooks/pre-processamento.ipynb`**: lê as bases de `data/raw/` e grava a base analítica em `data/processed/base_analitica.parquet`. Vem antes de todos os outros, porque eles leem essa base.
2. **Notebooks de análise**: `hipoteses_nps.ipynb`, `histogramas_anexo_a1.ipynb` e `escalonamento_anexo_a1.ipynb`. O `exploracao_dados.ipynb` é o único que lê direto de `data/raw/` e pode rodar a qualquer momento; ele grava as figuras em `figuras/`.
3. **Notebooks de modelagem**: `definicao_predicao.ipynb`, `modelagem_nps.ipynb`, `modelagem.ipynb`, `regressao_logistica.ipynb`, `arvore_decisao.ipynb`, `ensembles.ipynb` e `comparacao_modelos.ipynb`. Cada um monta a própria divisão em treino, validação e teste a partir da base analítica, então não dependem uns dos outros; a ordem acima segue a das Seções 4.3 e 4.4 da documentação.

Os módulos de `src/` também podem ser executados isoladamente. `python src/clean.py` lê as bases de `data/raw/` por padrão; a variável de ambiente `SAFIRA_DATA_DIR` troca essa pasta.

### Reprodução dos conjuntos de treino, validação e teste

Os cards de modelagem da Seção 4.3 comparam métricas entre si, e isso só faz sentido se todos rodarem sobre exatamente as mesmas partições. A divisão é temporal e por Cliente, portanto determinística — mas determinismo não basta: uma base analítica regerada com uma linha a mais produziria outros conjuntos sem aviso. Por isso a divisão é **congelada** num artefato de índices.

O artefato é gerado na primeira execução da seção 1.5 de `notebooks/modelagem.ipynb` e reutilizado em todas as seguintes:

```
data/processed/particoes_modelagem.json
```

Ele fica sob `data/`, coberto pelo `.gitignore`, e **não é versionado**: índices são posições de linhas de uma base do parceiro. Cada pessoa gera o seu localmente, e o hash é o que prova que todos chegaram ao mesmo lugar.

Para reproduzir os conjuntos do zero numa pasta limpa:

```bash
git clone <url-do-repositorio>
cd g01
# copie as bases da Azul para data/raw/ (ver "Bases de dados")
python -m venv .venv
# ative o ambiente com o comando do seu sistema (ver "Localmente")
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --output-dir=.execucao notebooks/pre-processamento.ipynb
jupyter nbconvert --execute --to notebook --output-dir=.execucao notebooks/modelagem.ipynb
```

O primeiro notebook grava `data/processed/base_analitica.parquet`; o segundo cria o artefato de índices e imprime, na seção 1.5, a semente fixada, a impressão digital da base de origem, as datas de corte e o hash dos índices. Compare esse hash com o de outra pessoa: iguais, as partições são as mesmas.

Para conferir que o congelamento se mantém, apague o artefato e reexecute a seção 1.4 seguida da 1.5 — o hash impresso tem de ser o mesmo:

```bash
rm data/processed/particoes_modelagem.json
```

A conferência recusa, com erro explícito, um artefato editado à mão, gerado sobre outra versão da base, com outras datas de corte, ou partições que não sejam as congeladas. Mudar a política de particionamento é decisão registrada na Seção 4.3, não efeito colateral de uma execução: exige passar `regerar=True` a `congelar_ou_conferir`.

A divisão em si não é refeita aqui. Ela acontece uma única vez, na seção 1.4, dentro de `matriz.preparar_matriz`; a seção 1.5 apenas congela o que a 1.4 produziu e, nas execuções seguintes, confere que nada mudou. `congelamento.obter_particoes` continua disponível para quem precisar das partições fora do notebook, sem montar a matriz.

### Verificação dos notebooks de desenvolvimento

Os notebooks de desenvolvimento são versionados **sem saídas de célula**, por proteção dos dados do parceiro. As verificações abaixo são dos estudos; para a entrega final, use os comandos da seção de execução principal.

**Testes das travas de integridade.** Não dependem das bases da Azul: usam dados sintéticos. A suíte completa leva alguns minutos (4 min 27 s numa máquina de 8 núcleos, com 556 testes aprovados e 15 pulados).

```bash
pytest -v
```

Cobrem o que precisa falhar quando deve: duplicata com conteúdo divergente, cobertura incompleta da chave antes da junção, violação da cardinalidade 1:1, `ID_GOLDENRECORD` divergente entre tabelas, e estrato de pós-estratificação sem contrapartida populacional. Cobrem também a partição temporal por Cliente: nenhum Cliente nos dois conjuntos e exclusão registrada em log dos registros sem `ID_GOLDENRECORD`.

**Execução de ponta a ponta do notebook.** Requer as bases em `data/raw/`. Termina com código de saída zero apenas se todas as células executarem sem erro.

```bash
jupyter nbconvert --execute --to notebook --output-dir=.execucao notebooks/exploracao_dados.ipynb
```

O notebook executado, com as saídas, fica em `.execucao/`, e as sete figuras em `figuras/`. Ambos os diretórios estão no `.gitignore`: a execução serve para verificar, não para versionar saídas que contenham dados do parceiro.

Para levar as figuras à documentação, copie os PNGs de `figuras/` para `assets/`, preservando os nomes.

### Google Colab: desenvolvimento e limite do modelo final

O fluxo principal do modelo final descrito acima é local. Sua carga exige correspondência exata com o Python e as bibliotecas do manifesto; a compatibilidade desse fluxo no Colab ainda não foi verificada. Não envie um Joblib local ao Colab supondo equivalência, nem desative a verificação de versões. Até existir evidência dessa compatibilidade, execute `pipeline_modelo_final.ipynb` no ambiente local documentado.

Os passos abaixo são o fluxo de desenvolvimento anterior, não uma instrução de implantação do modelo final. Não autorizam transferir dados reais a um serviço externo; utilize somente material cujo processamento nesse ambiente tenha sido expressamente autorizado.

Os notebooks importam módulos de `src/` e `scripts/` e leem as bases de `data/`, então o Colab precisa da pasta inteira do projeto, e não só do notebook. Além disso, cada notebook aberto no Colab roda numa máquina própria, e o que um grava em `/content` não aparece para o outro. Por isso o projeto fica no Google Drive: a base analítica que o pré-processamento grava ali continua disponível para os notebooks seguintes.

1. Baixe o repositório pelo GitLab (**Code > Download source code > zip**), descompacte e envie a pasta para o seu Google Drive.
2. Somente com autorização específica para processamento no Colab, coloque as bases em `data/raw/`, dentro dessa pasta, seguindo as regras de [Bases de dados](#bases-de-dados). Não transfira bases confidenciais sem essa autorização.
3. No Drive, abra o notebook com **Abrir com > Google Colaboratory**.
4. Insira uma célula no topo do notebook e execute-a antes de qualquer outra, trocando `<pasta-do-projeto>` pelo caminho da pasta do repositório dentro do seu Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd "/content/drive/MyDrive/<pasta-do-projeto>/notebooks"
   %pip install -q -r ../requirements.txt
   ```

   Com a pasta atual em `notebooks/`, cada notebook encontra `src/`, `scripts/` e `data/` sozinho, como na execução local. Se o Colab pedir para reiniciar a sessão depois da instalação, reinicie e execute essa célula de novo.
5. Execute com **Ambiente de execução > Executar tudo**, na mesma ordem da execução local: primeiro `pre-processamento.ipynb`, que grava `data/processed/base_analitica.parquet`; depois os de análise (`hipoteses_nps.ipynb`, `histogramas_anexo_a1.ipynb` e `escalonamento_anexo_a1.ipynb`; o `exploracao_dados.ipynb` lê direto de `data/raw/` e pode rodar a qualquer momento); por último os de modelagem (`definicao_predicao.ipynb`, `modelagem_nps.ipynb`, `modelagem.ipynb`, `regressao_logistica.ipynb`, `arvore_decisao.ipynb`, `ensembles.ipynb` e `comparacao_modelos.ipynb`).

> A célula de montagem e as saídas ficam só na sua cópia do Drive. Não as leve de volta ao repositório: os notebooks são versionados sem saídas, por proteção dos dados do parceiro.

## 🗃 Histórico de lançamentos

* 1.0.0 - 09/10/2026 (fechamento previsto)
    * [sprint 5] Avaliação do modelo final na partição de teste (metas contra resultado, resposta a falsos negativos e falsos positivos, monitoramento e plano B, importância por permutação e veredito de cada hipótese), objetivos, proposta de solução e justificativa (Seção 2) e preparação do repositório para publicação.
* 0.4.0 - 26/09/2026
    * [sprint 4] Comparação de modelos: custo do falso negativo e escolha do F2 como critério de busca, Grid Search na Regressão Logística e Random Search no Gradient Boosting, candidatos comparados na mesma base, Gradient Boosting escolhido para ordenar a fila de contato e Regressão Logística para explicar o risco, e limite da meta de recall pela capacidade de 50 contatos por dia.
* 0.3.0 - 12/09/2026
    * [sprint 3] Preparação dos dados e primeiro modelo preditivo: base integrada de 484.915 respostas, contrato temporal que só admite features disponíveis no momento do score, divisão temporal e por Cliente e Gradient Boosting avaliado contra as metas da classe Detrator.
* 0.2.0 - 29/08/2026
    * [sprint 2] Análise exploratória e hipóteses: perfil da base, viés de resposta e ponderação, tratamento de nulos e outliers, testes de normalidade, escolha das escalas de normalização e teste das hipóteses H1 a H4.
* 0.1.0 - 14/08/2026
    * [sprint 1] Entendimento do negócio: contexto de mercado, análise SWOT, 5 Forças de Porter, proposta de solução, personas e cronograma das sprints.

## 📋 Licença/License

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">Safira</span> by <span property="cc:attributionName">Inteli, Avatares</span> is licensed under <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
