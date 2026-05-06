# Projeto MLOps 1

## Problema Proposto: 

- Atacar o problema de gerar recomendações para usuários levando-se em conta a interação deles com produtos no decorrer dos últimos 600 dias. Limite o tamanha do cesta como acharem válido.
- Utilizar os dados transacionais gerados artificialmente. (Pensem em features derivadas que vocês podem gerar) 
- Documentar as estratégias que vocês pensaram para atacar o problema, mesmo as que vocês descartaram.
- Comparar pelo menos dois modelos utilzando métricas de sucesso: Um bem burrinho (ou até mesmo aleatório) para servir de baseline e um que seja um bom candidato para esse tipo de problema. (não se preocupem tanto com o resultado aqui; o processo é mais importante)
- Não queremos nesse ponto trabalhar com automatização de retreinamento ou algo assim. A seleção das versões champion dos modelos será manual.
- Tentem organizar o código de vocês de maneira que vocês possam rodar separadamente cada etapa do pipeline da geração de modelos. Vocês podem se basear em uma versão simplificada de um pipeline ideal (por ideal digo parecido com o que vimos na reunião do dia 09/04/2026).
- A orquestração fica a cargo de vocês. Vocês podem rodar como acharem mais fácil; usando o airflow; criando um bash; ou como acharem melhor.

## Tecnologias Utilizadas

- Python 3.10+: linguagem principal do projeto, usada para geração de dados, análise e desenvolvimento do pipeline de machine learning.
- Pandas: biblioteca para manipulação tabular dos dados transacionais, leitura de arquivos CSV e criação de features.
- NumPy: suporte a operações numéricas e geração de distribuições aleatórias utilizadas na simulação dos dados.
- Faker: geração de dados artificiais para compor o dataset transacional do projeto.
- Scikit-learn: biblioteca que será utilizada para treinar e comparar dois modelos supervisionados de classificação probabilística, ambos capazes de devolver a probabilidade de compra de um produto por um usuário em uma determinada data.
- Seaborn: apoio à análise exploratória e visualização de dados nas etapas de estudo do comportamento das transações.
- Jupyter Notebook + ipykernel: ambiente interativo para experimentação, exploração inicial dos dados e registro de análises.
- Ruff: ferramenta de linting e padronização de código Python.
- Docker Compose: orquestração local dos serviços de infraestrutura necessários ao projeto.
- MLflow: rastreamento de experimentos, métricas, parâmetros, artefatos e versionamento manual de modelos.
- PostgreSQL: banco utilizado como backend store do servidor MLflow.
- S3 compatível via RustFS: armazenamento de artefatos do MLflow, como modelos, métricas exportadas e outros arquivos gerados no processo.

Além dessas tecnologias, o projeto também prevê uma organização modular para separar as etapas de dados, treinamento, avaliação e execução do pipeline, mantendo o fluxo reproduzível e simples de operar.

## Etapas de Desenvolvimento

### Etapa 1 - Definição e Estudo do Problema:

Nesta primeira etapa, o objetivo é deixar explícito qual problema queremos resolver e qual será o recorte adotado para o projeto.

#### 1. Contexto

Temos um conjunto de dados transacionais artificiais contendo interações entre usuários e produtos ao longo dos últimos 600 dias. Cada registro representa uma compra, com informações como identificador do usuário, identificador do produto, data da transação, quantidade e preço.

#### 2. Problema de negócio

O problema que vamos atacar é o de recomendação de produtos. A ideia é apoiar a tomada de decisão sobre quais itens mostrar para cada usuário com base no seu comportamento histórico.

Em termos práticos, queremos responder à seguinte pergunta:

"Dado o histórico de interações de um usuário nos últimos 600 dias, quais produtos devemos recomendar para ele agora?"

#### 3. Formulação do problema em Machine Learning

Vamos tratar o desafio como um problema de recomendação do tipo top-N. Para cada usuário, o modelo deverá produzir uma lista ordenada com os produtos mais relevantes segundo os padrões observados no histórico.

O foco deste projeto não será prever uma nota explícita dada pelo usuário, mas sim inferir preferência a partir de sinais implícitos de interação, como:

- frequência de compra;
- recorrência de produtos;
- recência da última interação;
- quantidade comprada;
- valor movimentado por produto.

#### 4. Entradas e saídas esperadas

As entradas principais do sistema serão os dados transacionais históricos, enriquecidos com features derivadas construídas a partir do comportamento dos usuários e dos produtos.

A saída esperada será, para cada usuário, uma cesta de recomendação com tamanho limitado, contendo os produtos mais prováveis de gerar interesse ou nova compra.

Exemplo conceitual de saída:

- usuário `user_17` -> [`product_4`, `product_11`, `product_2`, `product_9`]

#### 5. Premissas e restrições do projeto

Para manter o escopo adequado ao objetivo didático de MLOps, vamos assumir as seguintes restrições:

- os dados utilizados serão apenas os dados sintéticos fornecidos no projeto;
- a seleção do modelo campeão será manual;
- não haverá, nesta etapa, automação de retreinamento contínuo;
- o pipeline deve ser organizado em etapas independentes e reproduzíveis;
- precisaremos comparar ao menos um baseline simples com um modelo mais adequado ao problema.

#### 6. Estratégia inicial de modelagem

Para a primeira versão do projeto, vamos utilizar o scikit-learn para comparar dois modelos que retornem probabilidade de compra para um par usuário-produto em uma data de referência.

O objetivo será estimar algo conceitualmente equivalente a:

P(compra do produto x | usuário u, data d, histórico anterior)

Os modelos escolhidos serão:

- Modelo baseline: `DummyClassifier`, que servirá como referência inicial. Mesmo sendo um modelo simples, ele permite gerar probabilidades de forma controlada e estabelecer um piso de comparação para as métricas.
- Modelo candidato mais robusto: `DecisionTreeClassifier`, que consegue capturar regras não lineares a partir das variáveis derivadas do histórico transacional e também produzir probabilidades por meio de `predict_proba`.

Com essa escolha, garantimos uma comparação entre:

- um modelo base extremamente simples, usado como benchmark mínimo;
- um modelo mais robusto que o baseline, mas ainda simples de interpretar, capaz de aprender padrões mais complexos a partir das features construídas.

#### 7. Análise inicial dos dados

Antes da construção das features e do treinamento dos modelos, faremos uma análise inicial do dataset para entender sua estrutura, verificar sua qualidade e identificar padrões importantes para a modelagem.

Essa etapa terá como objetivo responder perguntas como:

- os dados possuem valores ausentes ou registros inconsistentes?
- os tipos das colunas estão corretos para uso analítico?
- a distribuição de compras entre usuários e produtos é equilibrada ou concentrada?
- existem padrões temporais relevantes ao longo da janela de 600 dias?
- há sinais de outliers ou comportamentos artificiais que precisem ser tratados?

Os principais pontos de verificação serão:

- Qualidade estrutural: checagem de colunas obrigatórias, tipos de dados, valores nulos, duplicidades e integridade básica dos registros.
- Consistência de negócio: validação de regras simples, como a manutenção de um preço único por produto no dataset gerado e a coerência de quantidades compradas.
- Distribuição das variáveis numéricas: análise de estatísticas descritivas para `quantity` e `price`, observando média, dispersão, mínimos, máximos e possíveis valores extremos.
- Distribuição das variáveis categóricas: avaliação da quantidade de usuários, quantidade de produtos e concentração de interações por usuário e por produto.
- Distribuição temporal: inspeção da coluna `timestamp` para entender recência, volume de transações por período e possíveis concentrações artificiais no tempo.
- Sinais úteis para recomendação: observação de frequência de compra, repetição de produtos, recorrência por usuário e intensidade de consumo.

Com base no que já foi explorado no notebook inicial e no gerador de dados, essa análise deverá confirmar pelo menos:

- o volume total de transações disponíveis para estudo;
- a quantidade de usuários e produtos distintos;
- a ausência, ou presença, de problemas básicos de preenchimento;
- a consistência de preço por produto;
- o comportamento geral das variáveis `quantity` e `price`.

Essa análise será importante para orientar as próximas decisões do projeto, especialmente:

- quais features derivadas serão criadas;
- como construir a variável alvo para classificação probabilística;
- quais tratamentos de dados serão necessários antes do treinamento;
- quais limitações do dataset sintético precisam ser explicitadas no jornal.

#### 8. Respostas encontradas na análise inicial (2026-04-14)

Com base na execução do notebook de EDA, os principais resultados observados nesta etapa foram:

- Volume e estrutura: o dataset possui 99.557 linhas e 5 colunas.
- Completude e duplicidade: não foram encontrados valores ausentes (0 missing values) nem linhas duplicadas (0 duplicatas).
- Cobertura de entidades: foram identificados 100 usuários distintos e 30 produtos distintos.
- Janela temporal observada: as transações estão entre `2024-10-03 08:39:40.086187` e `2026-04-14 08:39:40.649101`, cobrindo aproximadamente a janela esperada de 600 dias.
- Consistência de preço por produto: a validação retornou `True` para consistência, indicando que cada produto manteve um único preço no dataset gerado.
- Comportamento de `quantity`: média de `1,9998`, com mínimo `1` e máximo `3`, coerente com a regra de geração dos dados.
- Comportamento de `price`: média de `48,4507`, com mínimo `11,85` e máximo `97,29`, sem anomalias estruturais aparentes na faixa de valores.

Conclusão desta etapa: os dados apresentam boa qualidade estrutural para seguir para a próxima fase (engenharia de atributos e definição da variável alvo), mantendo em mente que se trata de um dataset sintético e, portanto, com limitações de realismo que deverão ser documentadas ao longo do projeto.

---

### Etapa 2 - Engenharia de Atributos (2026-05-05)

#### 1. Estratégia adotada

Optamos por uma abordagem de **feature store simples baseada em arquivos CSV**. A ideia é manter cada conjunto de features separado para que cada etapa do pipeline possa carregar apenas o que precisa, sem depender de todos os dados ao mesmo tempo.

A data de referência usada para calcular recência foi **2026-04-14** (data máxima do dataset). Isso simula o momento em que o modelo seria chamado para gerar recomendações.

#### 2. Feature sets gerados

Os arquivos ficam em `data/feature_store/` e são gerados pelo script `src/feature_enginer/feature_enginer.py`.

---

**`transaction_features.csv`** — 99.557 linhas (uma por transação original)

Enriquece cada transação com colunas derivadas simples:

| Feature | Descrição |
|---|---|
| `total_value` | `quantity * price` — valor total da transação |
| `day_of_week` | dia da semana (0=seg … 6=dom) |
| `month` | mês da compra |
| `is_weekend` | 1 se sábado ou domingo, 0 caso contrário |
| `days_since_prev_purchase` | dias desde a compra anterior do mesmo usuário (lag 1); NaN na primeira compra |

---

**`user_features.csv`** — 100 linhas (uma por usuário)

Agrega o comportamento histórico de cada usuário:

| Feature | Descrição |
|---|---|
| `total_transactions` | número total de compras |
| `total_spend` | gasto total acumulado |
| `avg_order_value` | ticket médio por transação |
| `unique_products` | quantidade de produtos distintos comprados |
| `avg_quantity` | quantidade média por transação |
| `days_since_last_purchase` | dias desde a última compra até a data de referência |
| `customer_lifetime_days` | dias entre primeira e última compra |
| `avg_days_between_purchases` | frequência média de compra em dias |
| `purchases_last_30d` | número de compras nos últimos 30 dias |
| `spend_last_30d` | gasto nos últimos 30 dias |
| `purchases_last_90d` | número de compras nos últimos 90 dias |
| `spend_last_90d` | gasto nos últimos 90 dias |

Estatísticas relevantes observadas:
- Média de ~996 transações por usuário, com grande variação (min 804, max 1189).
- Gasto médio acumulado: R$ 96.404, variando entre R$ 75.158 e R$ 117.018.
- `days_since_last_purchase` com média de ~6 dias — usuários muito ativos, coerente com dados sintéticos.

---

**`user_product_features.csv`** — 3.000 linhas (100 usuários × 30 produtos)

Agrega o relacionamento de cada usuário com cada produto:

| Feature | Descrição |
|---|---|
| `purchase_count` | quantas vezes o usuário comprou esse produto |
| `total_spend` | gasto total do usuário nesse produto |
| `avg_quantity` | quantidade média por compra desse produto |
| `days_since_last_purchase` | dias desde a última compra desse produto pelo usuário |
| `last_quantity` | quantidade comprada na transação mais recente |

Estatísticas relevantes:
- Todos os 100 usuários compraram todos os 30 produtos pelo menos uma vez (dataset sintético balanceado).
- Média de ~33 compras por par usuário-produto.
- `days_since_last_purchase` com média de ~26 dias — produto não comprado recentemente é um sinal útil de candidato a recomendação.

#### 3. Decisões e descartes

- **Lags mais longos (lag 2, lag 3)**: descartados por ora. Com dados sintéticos muito regulares, provavelmente não acrescentam muito e aumentam complexidade.
- **Features de produto global** (ex: produto mais vendido do catálogo, produto mais caro): pensadas mas não implementadas nessa rodada — podem entrar se os modelos precisarem de mais contexto sobre os itens.
- **Normalização / encoding**: não feitos aqui; ficam a cargo do script de treinamento conforme a necessidade de cada modelo.
- **Variável alvo**: não gerada nessa etapa. Será construída junto com o script de treinamento, provavelmente como um indicador binário de recompra em uma janela futura.

---

#### 4. Refatoração do pipeline de features (2026-05-06)

O script `src/feature_enginer/feature_enginer.py` foi refatorado para corrigir um problema de **data leakage** que existia na abordagem anterior, onde todas as features eram calculadas sobre o dataset completo antes de qualquer divisão treino/teste.

##### Nova estrutura

O pipeline agora segue o padrão `fit / transform`:

1. **Split temporal** — os dados são divididos por tempo (não aleatoriamente), preservando a causalidade da série. O ponto de corte é calculado como `t_min + (t_max - t_min) * 0.8`, o que garante que os primeiros 80% do **intervalo de tempo** vão para treino e os 20% finais para teste.
2. **`FeatureBuilder.fit(df_train)`** — aprende todas as estatísticas (médias, agregações, data de referência) **exclusivamente** com os dados de treino.
3. **`FeatureBuilder.transform(df)`** — aplica as estatísticas congeladas do treino sobre qualquer split, sem reaprender nada.

A saída agora são dois arquivos flat em `data/feature_store/`: `train.csv` e `test.csv`, cada um com granularidade de transação e todas as features já unidas em uma única tabela.

##### Tradeoff: um arquivo por grupo de features vs. tabela flat única

Consideramos manter a estrutura anterior de três arquivos separados (`transaction_features.csv`, `user_features.csv`, `user_product_features.csv`), o que facilitaria o reuso isolado de cada grupo de features em diferentes experimentos.

**Optamos pela tabela flat única** por simplicidade: com o número atual de features e grupos, o overhead de gerenciar múltiplos arquivos e fazer os joins manualmente em cada experimento não se justifica. Caso o número de feature sets cresça, essa decisão pode ser revisada.

##### Features geradas

Todas as features abaixo estão presentes em ambas as tabelas (`train.csv` e `test.csv`).

**Features de transação** (calculadas por linha no `transform`):

| Feature | Descrição |
|---|---|
| `total_value` | `quantity * price` |
| `day_of_week` | dia da semana (0=segunda … 6=domingo) |
| `month` | mês da transação |
| `is_weekend` | 1 se sábado ou domingo, 0 caso contrário |
| `days_since_prev_purchase` | dias desde a transação anterior do mesmo usuário; na fronteira treino/teste, usa o último timestamp do treino para evitar NaN |

**Features de usuário** (agregadas no `fit`, por `user_id`):

| Feature | Descrição |
|---|---|
| `total_transactions` | total de transações no treino |
| `user_total_spend` | gasto total acumulado |
| `user_avg_order_value` | ticket médio por transação |
| `unique_products` | quantidade de produtos distintos comprados |
| `user_avg_quantity` | quantidade média por transação |
| `days_since_last_purchase` | dias desde a última compra até a `reference_date_` |
| `customer_lifetime_days` | dias entre primeira e última compra |
| `avg_days_between_purchases` | frequência média de compra em dias (NaN se apenas 1 transação) |
| `purchases_last_30d` | número de compras nos últimos 30 dias do treino |
| `spend_last_30d` | gasto nos últimos 30 dias do treino |
| `purchases_last_90d` | número de compras nos últimos 90 dias do treino |
| `spend_last_90d` | gasto nos últimos 90 dias do treino |

**Features de usuário × produto** (agregadas no `fit`, por `user_id` + `product_id`):

| Feature | Descrição |
|---|---|
| `up_purchase_count` | vezes que o usuário comprou o produto |
| `up_total_spend` | gasto total do usuário naquele produto |
| `up_avg_quantity` | quantidade média por compra do par |
| `up_days_since_last_purchase` | dias desde a última compra daquele produto pelo usuário |
| `up_last_quantity` | quantidade comprada na transação mais recente do par |

