<h1 align="center">Nba-stats-prediction</h2>


### Introdução

Este repositório foi criado para o desafio de projeto proposto no curso Microsoft - Fundamentos de IA (Azure AI-900), promovido pela DIO.

O objetivo do desafio é desenvolver um modelo de previsão utilizando o Azure Machine Learning e o recurso AutoML. Como proposta prática, decidi criar um modelo para prever a pontuação média por jogo de jogadores da NBA, utilizando um conjunto de dados encontrado no Kaggle.

- 📊 Fonte dos Dados:
    - [Nba Traditional Stats (Kaggle)](https://www.kaggle.com/datasets/tilii7/nba-traditional-stats/data)
    - [Download do Dataset](https://github.com/user-attachments/files/18876175/NBA_02122025_Traditional.csv)

O conjunto de dados contém estatísticas dos jogadores da temporada 2024-2025, fornecendo uma base sólida para treinar o modelo de IA.

### Para o modelo eu utilizei o modelo de regressão com a coluna `PTS` como destino.

##

Estrutura da tabela:

| Player                  | Team | Age | GP | W  | L  | Min | PTS | FGM | FGA | FG% | 3PM | 3PA | 3P% | FTM | FTA | FT% | OREB | DREB | REB | AST | TOV | STL | BLK | PF | FP  | DD2 | TD3 | Plus/Minus |
|-------------------------|------|-----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|-----|-----|-----|-----|-----|----|-----|-----|-----|------------|
| Shai Gilgeous-Alexander| OKC  | 26  | 51 | 43 | 8  | 34  | 32.6| 11.4| 21.5| 52.8| 2   | 5.7 | 35.5| 7.9 | 8.8 | 89.9| 0.9  | 4.2  | 5.1 | 6   | 2.6 | 1.9 | 1   | 2.2| 53.9| 4   | 0   | 13.4       |
| Giannis Antetokounmpo  | MIL  | 30  | 41 | 23 | 18 | 34.9| 31.8| 12.7| 20.9| 60.8| 0.1 | 0.8 | 18.8| 6.2 |10.7 | 57.7| 2.3  | 9.9  |12.2 | 5.9 | 3.4 | 0.8 | 1.3 | 2.4| 58  | 37  | 5   | 3.5        |
| Nikola Jokic           | DEN  | 29  | 48 | 33 | 15 | 36  | 29.8| 11.4| 19.6| 57.8| 2.1 | 4.5 | 45.6| 5.1 | 6.2 | 82.1| 3    | 9.5  |12.5 |10.3 | 3.1 | 1.8 | 0.7 | 2.1| 64.6| 40  | 24  | 9.7        |
| Tyrese Maxey           | PHI  | 24  | 46 | 19 | 27 | 38  | 27.6| 9.6 | 21.5| 44.7| 3.3 | 9.6 | 34.6| 5   | 5.8 | 87.2| 0.3  | 3.2  | 3.5 | 6.1 | 2.5 | 1.9 | 0.4 | 2.3| 45.2| 8   | 1   | -1.2       |
| Anthony Edwards        | MIN  | 23  | 52 | 29 | 23 |36.7 | 27.5| 9.2 | 20.6| 44.6| 4.2 | 10  | 42.4| 4.9 | 5.8 | 84.1| 0.8  | 5    | 5.8 | 4.5 | 3.5 | 1.1 | 0.7 | 1.9| 43.1| 3   | 0   | 3.4        |
| Luka Doncic            | LAL  | 25  | 23 | 14 | 9  |35.1 | 27.5| 9.6 | 20.9| 46  | 3.3 | 9.5 | 34.7| 5   | 6.5 | 77.2| 0.7  | 7.4  | 8.2 | 7.7 | 3.3 | 1.9 | 0.4 | 2.7| 52.4| 10  | 3   | 8          |
| LaMelo Ball            | CHA  | 23  | 33 | 12 | 21 |32.7 | 27.3| 9.5 | 22.8| 41.6| 4   |11.9 | 33.8| 4.3 | 5.3 | 82.2| 0.9  | 4.2  | 5.1 | 7.2 | 3.6 | 1.3 | 0.3 | 3.3| 45.6| 9   | 0   | -1.5       |
| Kevin Durant           | PHX  | 36  | 40 | 24 | 16 |36.3 | 27.1| 9.8 | 18.6| 52.8| 2.4 | 5.9 | 40.4| 5.1 | 6.2 | 82.2| 0.4  | 5.6  | 6   | 4.1 | 3   | 0.8 | 1.4 | 1.7| 44.3| 3   | 0   | 0.1        |
| Jayson Tatum           | BOS  | 26  | 51 | 36 | 15 |36.4 | 26.9| 9.1 | 20  | 45.4| 3.6 |10.1 | 36  | 5.2 | 6.4 | 80.2| 0.6  | 8    | 8.6 | 5.5 | 2.8 | 1.2 | 0.5 | 2.2| 47.9| 22  | 1   | 7          |



Pra criar minha predição eu segui o passo à passo disponivel pela Microsoft: [mslearn-ai-fundamentals]

##

### Criar um espaço de trabalho do Azure Machine Learning

  1. O primeiro passo é acessar o [portal da Azure](https://portal.azure.com) utilizando as credencias usadas para criar a conta no Azure.
  
  2. Já tendo acessado o portal, clique em ***Criar um serviço***, depois pesquise por Machine Learning na ferramenta de pesquisa, e crie um Azure ***Machine Learning*** e use as seguintes configurações:

      - ***Assinatura***: Sua assinatura do Azure.
      - ***Grupo de recursos***: Crie ou selecione um grupo de recursos.
      - ***Nome***: Insira um nome único para seu espaço de trabalho.
      - ***Região***: East US.
      - ***Conta de armazenamento***: Observe a conta de armazenamento nova que será criada para seu espaço de trabalho.
      - ***Key vault***: Observe o cofre de chaves novo que será criado para seu espaço de trabalho.
      - ***Application Insights***: Observe o recurso de insights de aplicativo novo que será criado para seu espaço de trabalho.
      - ***Registro de contêiner***: Nenhum (um será criado automaticamente na primeira vez que você implantar um modelo em um contêiner).
    
3. Selecione ***Revisar + criar***, depois selecione Criar. Aguarde a criação do seu espaço de trabalho (pode levar alguns minutos) e depois vá até o recurso implantado.


### Iniciar o Estúdio

1. No recurso do seu espaço de trabalho do Azure Machine Learning, selecione **Iniciar Estúdio**.  
   (Ou abra uma nova aba do navegador e navegue até [https://ml.azure.com](https://ml.azure.com), e faça login no Estúdio do Azure Machine Learning usando sua conta Microsoft). Feche qualquer mensagem exibida.

2. No Estúdio do Azure Machine Learning, você deverá ver seu espaço de trabalho recém-criado.  
   Se não visualizar, selecione **Todos os espaços de trabalho** no menu à esquerda e, em seguida, selecione o espaço de trabalho que você acabou de criar.


### Criar um job automatizo para treinar o modelo

1. No Estúdio do Azure Machine Learning, visualize a página **Automated ML** (sob **Authoring**).

2. Crie um novo job de Machine Learning Automatizado com as seguintes configurações, utilizando **Next** conforme necessário para avançar pela interface:

### Configurações Básicas:

Preencher os campos:
- **Nome do trabalho**: O campo já deve estar pré-preenchido com um nome único. Mantenha o nome como está.
- **Nome do novo experimento**: `O nome do seu experimento`
- **Descrição**: `Descrição do experimento`
- **Tags**: Nenhum

### Tipo de Tarefa e Dados:

Preencher os campos:
- **Selecionar tipo de tarefa**: **Regressão**
- **Selecionar o ativo de dados**: Crie um novo ativo de dados com as seguintes configurações:
  - **Tipo de dados**: 
      - **Nome**: `nome-do-dataset`
      - **Descrição**: `Descrição do ativo de dados`
      - **Tipo**: Tabela (mltable)

- **Fonte de dados**: Selecione **De arquivos locais**
  - **Tipo de armazenamento de destion**:
      - **Tipo de armazenamento de dados**: **Azure Blob Storage**
      - **Nome**: `workspaceblobstore`

- **Seleção de MLTable**:
  - **Carregar o caminho** ou ***Carregar a pasta*** de destino contendo o arquivo .csv e o MLTable.

### 

> ***Dica 💡***:
> Se o arquivo MLTable não estiver presente na sua pasta, você pode cria-lo de maneira simples, apenas alterando as infomações relacionadas a sua base de dados (Nome do arquivo; O tipo de separador)

Exemplo de arquivo MLTable:

    # MLTable definition file
    
    paths:
      - file: ./nome_do_arquivo.csv
    transformations:
      - read_delimited:
            delimiter: ','
            encoding: 'ascii' # O tipo de separador utilizado
    

- Selecione **Create**. Após o ativo de dados ser criado, selecione-o para continuar e submeter o job de Machine Learning Automatizado.

### Configurações da Tarefa:

- **Tipo de tarefa**: **Regressão**
- **Ativo de dados**: `bike-rentals`
- **Coluna de destino**: `nome_da_coluna_de_destion` (integer ou decimal)
  
- **Exibir definições de configuração adicionair**:
  - **Métrica primária**: `NormalizedRootMeanSquaredError`
  - **Explicar o melhor modelo**: Não selecionado
  - **Habilitar empilhamento de ensemble**: Não selecionado
  - **Usar todos os modelos selecionados**: Não selecionado
  - **Modelos bloqueados**: Selecione apenas **RandomForest** e **LightGBM**

### Limites:

- Expanda esta seção:
  - **Máximo de avaliações**: 3
  - **Máximo de avaliações simultâneas**: 3
  - **Máximo de nós**: 3
  - **Limite de pontuação da métrica**: 0.085 (se um modelo alcançar um erro quadrático médio normalizado menor ou igual a 0.085, o job será finalizado).
  - **Tempo limite do experimento (minutos)**: 15
  - **Tempo limite de iteração (minutos)**: 15
  - **Habilitar encerramento antecipado**: Selecionado. O trabalho será encerrado se a pontuação não estiver melhorando no curto prazo.

### Validação e Teste:

- **Tipo de validação**: Divisão de validação de treinamento
- **Validação de percentual de dados**: 10
- **Dados de teste**: Nenhum

### Computação:

- **Selecione o tipo de computação**: **Sem servidor**
- **Tipo de máquina virtual**: **CPU**
- **Tipo de máquina virtua**: **Dedicado**
- **Tamanho da máquina virtual**: **Standard_DS3_V2**
- **Number of instances**: 1

3. Submeta o job de treinamento. Ele começará automaticamente.

4. Aguarde o job ser finalizado. Pode levar um tempo — agora é um bom momento para um intervalo para o café!


### Revisar o melhor modelo

Quando o trabalho de aprendizado de máquina automatizado for concluído, você pode revisar o melhor modelo que ele treinou.

1. Na aba **Visão Geral** do trabalho de aprendizado de máquina automatizado, observe o resumo do melhor modelo.

    ### - [Ver imagem de exemplo](https://microsoftlearning.github.io/mslearn-ai-fundamentals/Instructions/Labs/media/use-automated-machine-learning/complete-run.png) ###

2. Selecione o texto sob **Nome do algoritmo** para o melhor modelo para visualizar seus detalhes.

3. Selecione a aba **Métricas** e selecione os gráficos de `residuals` e `predicted_true`, se ainda não estiverem selecionados.

Revise os gráficos que mostram o desempenho do modelo. O gráfico de `residuals` mostra os resíduos (as diferenças entre os valores previstos e reais) como um histograma. O gráfico de `predicted_true` compara os valores previstos com os valores reais.


### Implantar e testar o modelo

1. Na aba **Modelo** para o melhor modelo treinado pelo seu trabalho de aprendizado de máquina automatizado, selecione Implantar e use a opção Ponto de extremidade em tempo real (pode estar traduzido como: Terminal em tempo real) para implantar o modelo com as seguintes configurações:

**Máquina virtual**: Standard_DS3_v2 (Por questões de região e cota essa máquina virtual pode estar indispónivel. recomendo usar a **Standard_E2s_v3**)
**Contagem de instâncias**: 3
**Ponto de extremidade**: Novo
**Nome do ponto de extremidade**: Deixe o padrão ou certifique-se de que seja globalmente único
**Nome da implantação**: Deixe o padrão
**Coleta de dados de inferência**: Desativado
**Pacote do Modelo**: Desativado


>**Importante ⚠**:
> Existe um bug em que o número de instâncias é alterado automaticamente para 1 e o tipo da máquina virtual é modificado. Para evitar esse problema, clique em **Mais opções** e preencha apenas os campos necessários.


2. Aguarde o início do deploy – isso pode levar alguns segundos. O status de Deploy para o endpoint predict-rentals será indicado na parte principal da página como Em execução.

3. Aguarde até que o status de Deploy mude para Concluído. Isso pode levar de 5 a 10 minutos.


> **⚠ Importante:**  
> Se o endpoint falhar e o erro **"Subscription is not registered with [N/A]"** for exibido, isso pode indicar que alguns provedores de recursos não estão registrados para a sua assinatura.  
> Verifique se os seguintes provedores estão habilitados:  
> - `Microsoft.Bash`  
> - `Microsoft.Network`  
> - `Microsoft.PolicyInsights`  
> - `Microsoft.StreamAnalytics`
> - `Microsoft.Advisor`  
>   
> Para resolver esse problema, consulte os links abaixo:  
> - [Solução no Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/1983847/error-while-creating-a-managed-online-endpoint-in)  
> - [Registrar provedores de recursos no Azure](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/resource-providers-and-types#register-resource-provider)  
> - [Discussão no Stack Overflow](https://stackoverflow.com/questions/76066942/azure-deployment-error-subscription-is-not-registered-with-nrp)


### Testar o serviço implantado
Agora você pode testar o serviço implantado.

1. No Azure Machine Learning Studio, no menu lateral esquerdo, selecione Endpoints e abra o endpoint em tempo real predict-rentals.

2. Na página do endpoint predict-rentals, acesse a aba Test.

2. No painel Input data to test endpoint (Dados de entrada para testar o endpoint), substitua o JSON de modelo pelos seguintes dados de entrada:


```JSON
{
  "input_data": {
    "columns": [
      Aqui irá conter as colunas do ativo de dados importado
    ],
    "index": [],
    "data": [] Aqui será passado os dados para predição
  }
}
```
