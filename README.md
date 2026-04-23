# Projeto1: Sistema de recomendação

## Objetivos

O objetivo desse projeto é termos contato com algumas práticas e ferramentas de MLOps. Queremos colocar em prática as partes teóricas que vimos nos nossos encontros:

- Definição do problema
- Métricas de sucesso
- Decisão sobre tratamento de dados
- Hipóteses sobre algorítmos
- Pipeline de treinamento
- Log de informações
- Versionamento de modelos

## Proposta

Irei propor para vocês o seguinte:

- Atacar o problema de gerar recomendações para usuários levando-se em conta a interação deles com produtos no decorrer dos últimos 600 dias. Limite o tamanha do cesta como acharem válido.
- Utilizar os dados transacionais gerados artificialmente. (Pensem em features derivadas que vocês podem gerar) 
- Documentar as estratégias que vocês pensaram para atacar o problema, mesmo as que vocês descartaram.
- Comparar pelo menos dois modelos utilzando métricas de sucesso: Um bem burrinho (ou até mesmo aleatório) para servir de baseline e um que seja um bom candidato para esse tipo de problema. (não se preocupem tanto com o resultado aqui; o processo é mais importante)
- Não queremos nesse ponto trabalhar com automatização de retreinamento ou algo assim. A seleção das versões champion dos modelos será manual.
- Tentem organizar o código de vocês de maneira que vocês possam rodar separadamente cada etapa do pipeline da geração de modelos. Vocês podem se basear em uma versão simplificada de um pipeline ideal (por ideal digo parecido com o que vimos na reunião do dia 09/04/2026).
- A orquestração fica a cargo de vocês. Vocês podem rodar como acharem mais fácil; usando o airflow; criando um bash; ou como acharem melhor.

## Papéis

Esse projeto visa ajudar a Alícia a se desenvolver como Engenheira de ML e o Utagawa como Engenheiro de MLOps. Em muitas empresas, principalmente as menores, esses cargos podem se sobrepor ou até uma mesma pessoa pode executar as duas funções.

O engenheiro de ML tem seu foco na construção dos modelos e sua aplicação em um projeto; aprofunda em ML e tem noções de infra.

O engenheiro de MLOps foca em fazer a execução dos modelos de maneira eficiente, confiável, reprodutível e automatizada; aprofunda em infra e tem noções de ML.

A troca entre vocês pode ser muito interessante nesse primeiro momento. Ambos MLE e engenheiros MLOps nesse projeto devem colabora em:

- Treinar os modelos
- Versionar
- Obter métricas
- Executar o pipe

# Contruindo seu ambiente

Aqui é a hora de vocês decidirem a infra de vocês. Darei apenas algumas poucas sugestões e deixarei alguns arquivos já engatilhados.

## Gestão de dependências

Para fazer a gestão de dependências utilizei o pacote **uv**. Ele consegue fazer a criação de ambientes virtuais, instalação de versões do python e dependências de maneira muito eficiente.

Vocês podem encontrar a documentação [nesse link](https://docs.astral.sh/uv/)

## Servidor do MLflow

Na pasta infra vocês encontrarão um arquivo **docker-compose** que sobe localmente um servidor do MLflow em **localhost:5000**. Além disso subimos também um emulador de S3 para que vocês possam escrever seus artefatos; acessível pela porta **9001**.

Para subir os containers use

```bash
docker compose up -d
```

na pasta de infra, vocês encontrarão um arquivo .env de exemplo com as variáveis de ambiente necessárias.

Para derrubar os containers use

```bash
docker compose down
```

A documentação completa está [nesse link](https://github.com/mlflow/mlflow/tree/master/docker-compose)

## Orquestração

Como nesse projeto não estamos preocupados com um fluxo que gera loops no nosso pipe; vocês podem orquestrar como quiserem a execução das etapas. O Airflow é uma ferramenta muito poderosa de orquestração. Avaliem se vocês querem ir por esse caminho ou fazer algo mais simples.

## Padrão de design

Busquem implementar a solução de vocês de uma forma simples. Não é o objetivo aqui nos preocuparmos com design patterns complicados. tentem garantir só a modularização do código para cada etapa. Para rodar os códigos vocês podem usar libs como o **argparse** ou **click** por exemplo. Ambas são boas para criar CLIs.

## Sugestão para o repositório

Vocês podem criar uma estrutura como acharem melhor. Sugiro algo simples como a árvore abaixo.

```bash
big-data-mlops-project1/
│
├── README.md
├── pyproject.toml
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py          # ingest + preprocess + features (all together)
│   ├── model.py         # baseline + candidate models
│   ├── train.py         # training logic + MLflow logging
│   ├── evaluate.py      # metrics
│   └── pipeline.py      # glue everything together
│
├── scripts/
│   ├── run_train.py
│   └── run_pipeline.py
│
├── infra/
│   └── docker-compose.yml   # MLflow + S3 (what you already gave them)
│
└── experiments/
    └── notes.md
```

# Dicas

- Façam as coisas da maneira mais simples que conseguirem. Nosso foco é o pipe simplificado e usar o MLflow.
- Pesquisem os algorítmos com carinho. Como aqui queremos fazer algo didático, procurem algorítmos simples.
- Documentem o processo de vocês. Seria legal vocês apresentarem o projeto pro pessoal depois :)
