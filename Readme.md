# Introdução

Reserva não honrada é um projeto de portfólio dedicado à análise e ciência de dados, com o propósito de aplicar meus conhecimentos em situações mais realistas. É claro que no Kaggle muitas vezes encontramos conjuntos já tratados, poupando-nos do desafio de lidar com dados mais complexos. Além disso, é comum deparar-se com conjuntos de dados fictícios, gerados artificialmente por usuários e disponibilizados na plataforma.

Ao iniciar os estudos em ciência de dados, a utilização de conjuntos de dados previamente tratados pode ser extremamente útil, funcionando como um ponto de partida. No entanto, à medida que avançamos em nossos estudos, torna-se crucial buscar conjuntos de dados mais realistas, que reproduzam a complexidade e diversidade dos desafios encontrados em problemas do mundo real.

Pensando nisso, tive a ideia de trazer um novo projeto em ciência de dados utilizando uma base de dados real de um salão Canadense.

# Contextualizando


Em um modelo de negócios de salão de beleza, é uma prática comum que os clientes agendem horários para receberem serviços. No entanto, uma parcela desses clientes acaba não honrando a reserva, seja faltando ou cancelando o serviço em cima da hora. Essa conduta resulta em custos para o salão, uma vez que o horário poderia ter sido disponibilizado para outra pessoa, gerando assim receita para o estabelecimento.

Diante desse cenário, torna-se crucial para o salão adotar estratégias para mitigar esses custos.

# Dados

Os dados desse salão foram obtidos no Kaggle através do link: [Hair Salon No-Show](https://www.kaggle.com/datasets/frederickferguson/hair-salon-no-show-data-set?select=Service+Listing0.csv) 

Esse banco de dados é constituído por 7 tabelas que estão listadas a seguir:


>* **Future Bookings (All Clients)**
>    * Reservas futuras é o conjunto de todas as reservas não canceladas.

>* **Receipt Transactions**
>    * Esta é uma lista de todas as transações com recibos.

>* **Product Listing (Retail)**
>    * Esta é uma lista de todos os produtos de varejo.

>* **hair_salon_no_show_wrangled_df**
>    * É um único dataset já organizado pelo dono do projeto no Kaggle. Este dataset foi desconsiderado do meu projeto, pois a ideia é desenvolvermos o dataset final de treinamento e teste.

>* **Client Cancellations**
>    * Cancelamentos de clientes são reservas que foram canceladas.

>* **No-Show Report0**
>    * Esta é uma lista de reservas de não comparecimento que não foram canceladas.
>* **Service Listing0**
>    * Esta é uma lista de todos os serviços.


# Organização

Além de buscar por dados mais realistas, também optei por organizar melhor os meus projetos. Desse modo, todos os diretórios foram previamente pensados para uma maior organização do projeto.


>* **datasets**
>    * É um diretório que contém todos as tabelas originais obtidas no Kaggle.

>* **datasets_for_ml**
>    * Consiste de um diretório que armazena tabelas tratadas além do dataset para treinamento e teste de modelos.

>* **images**
>   * Um diretório que armazena as figuras desenvolvidas no decorrer das análises.

>* **notebooks**
>   * É o diretório principal, contém todos os notebooks utilizados neste projeto.

>* **utils**
>   * Este diretório contém alguns scripts que auxiliam em tarefas no decorrer do projeto.


Como mencionado, todos os notebooks utilizados neste projeto estão disponíveis no diretório **notebooks**. No total, foram desenvolvidos três notebooks, os quais estão listados abaixo:

>* **analises.ipynb**
>   * Contém toda a análise de dados realizada com os dados brutos do salão.

>* **confeccionando_dataset_treino_teste.ipynb**
>   * Neste notebook foi desenvolvido o passo a passo da confecção do dataset de treino e teste a partir dos dados brutos.  

>* **modelagem.ipynb**
>   * Contém o desenvolvimento dos modelos de aprendizado de maquina.

# Objetivo / problema de negócio

Algumas vezes, clientes faltam aos compromissos ou cancelam seus serviços em cima da hora, o que resulta em custos para o salão. Isso ocorre porque o horário reservado poderia ser ocupado por outro cliente comprometido. Clientes que não comparecem ou cancelam com menos de dois dias de antecedência podem ser considerados como não seguindo as políticas do salão.

Assim, o objetivo deste projeto é desenvolver um modelo preditivo capaz de identificar clientes que possivelmente não seguirão as políticas do salão. A intenção é encontrar uma maneira de antecipar esses comportamentos, proporcionando ao salão a oportunidade de mitigar os custos associados a tais situações.

# Estimativas

Ao analisar os dados históricos do salão, identificou-se que, na mediana, os clientes geram uma receita de 55 CAD (dólares canadenses). Portanto, o salão deixa de receber 55 CAD em receita para cada cliente que não honra os compromissos agendados.

Com base nos dados históricos, pode-se determinar que o salão deixou de faturar 3245 CAD com clientes que faltaram aos serviços e 7700 CAD com pessoas que cancelaram o serviço com menos de dois dias de antecedência. Somando esses valores, o salão acumulou uma perda de receita total de 10945 CAD devido ao não cumprimento das políticas por parte dos clientes.

# Solução

Com base nos dados históricos, conseguimos construir um novo conjunto de dados para ser utilizado no treinamento. Para a criação desse conjunto de dados, utilizamos a última data de interação do cliente com o salão. Definimos três cenários possíveis para a interação:

1. O cliente vai ao salão e recebe o serviço, ou realiza o cancelamento de acordo com as políticas estabelecidas pelo salão.
2. O cliente cancela o compromisso faltando menos de dois dias.
3. O cliente falta ao compromisso.

Esses cenários refletem as diversas maneiras como os clientes interagem com o salão.


Ao utilizar a data da última interação, conseguimos construir o alvo (target) para o treinamento do modelo. Dessa forma, atribuímos a flag 1 a todos os clientes que não seguiram as políticas do salão, enquanto os demais receberam a flag 0. Essa abordagem permite categorizar os clientes com base em seu comportamento em relação às políticas de agendamento e cancelamento do salão.


Para desenvolver as features, utilizamos as datas anteriores à última interação de cada cliente. A partir desse histórico, foram geradas diversas features. O novo dataset resultante pode ser acessado no diretório **datasets_for_ml** para uma análise mais detalhada.

O dataset resultante possui 24 colunas e 798 clientes. Dentre esses clientes, 751 (94%) pertencem à classe 0, enquanto 47 (5,9%) estão na classe 1. É notável um desbalanceamento significativo nos dados, com a maioria dos clientes pertencendo à classe 0. Além disso, observa-se que há um número limitado de dados disponíveis para o treinamento e validação dos modelos, o que pode ser um desafio no desenvolvimento.


O problema identificado é que dos 798 clientes, 442 são considerados novos clientes, ou seja, não possuem histórico prévio, enquanto 356 são clientes com algum histórico registrado. Neste contexto, não faz sentido utilizar clientes sem histórico, pois não podemos extrair informações significativas desses novos clientes com base na nossa base de dados. Portanto, optamos por restringir ainda mais o nosso dataset, considerando apenas os clientes com histórico.

Com essa restrição, o dataset resultante passa a ter 356 instâncias, com 92% pertencendo à classe 0 e 8% à classe 1. Além do desbalanceamento considerável entre as classes, observamos também a presença de um número reduzido de instâncias disponíveis para o treinamento dos modelos.


Devemos estabelecer as métricas que vamos utilizar para a avaliação dos modelos. Vamos utilizar as três métricas listadas a seguir:

1. **Precision Score;**
2. **Recall Score;**
3. **F1 Score;**

A questão central deste problema consiste em identificar faltas ou cancelamentos que violam as políticas estabelecidas pelo salão. Tais ações resultam em custos para o salão, e, portanto, é crucial determinar se um cliente seguirá ou não essas políticas.

Embora as ações específicas para lidar com clientes propensos a descumprir as políticas não sejam conhecidas, é necessário tomar medidas para mitigar os custos associados. Uma abordagem seria alertar os clientes sobre a possibilidade de adiantar o atendimento, liberando assim novos horários. Dessa forma, os funcionários não ficariam sem trabalho, e haveria a oportunidade de preencher os horários vagos com outros clientes.

Embora os custos exatos dessas ações de mitigação sejam desconhecidos, podemos, por enquanto, desconsiderá-los. Imagine uma situação em que o contato com esses clientes seja feito por meio de ligação ou WhatsApp, onde podemos considerar esse custo igual zero, pelo menos inicialmente. Essa abordagem visa otimizar a utilização do tempo disponível e minimizar os impactos negativos causados por faltas ou cancelamentos.


Essas métricas serão convertidas para o problema de negócio através da matriz de confusão.

**Verdadeiro Positivo** - Existe a possibilidade de mitigar o custo.

**Verdadeiro Negativo** - Nenhum custo envolvido, teremos receita devido ao comparecimento do cliente.

**Falso Positivo** - O modelo estima que o cliente não vai cumprir a politica mas na verdade ele vai. Neste caso não teremos um custo real.

**Falso Negativo** - O modelo estima que o cliente vai cumprir a politica mas na verdade ele não vai. Neste caso vamos deixar de faturar.


Após o treinamento e validação obtemos os seguintes resultados:

A tabela a seguir indica a contabilidade real (referente ao teste)

<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>Rendimento</th>      <th>Custo</th>    </tr>  </thead>  <tbody>    <tr>      <th></th>      <td>5445</td>      <td>440</td>    </tr>  </tbody></table>


A seguir uma tabela que indica os modelos com seus respectivos f1 score de treino e teste e também métricas relacionadas ao negócio.


<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>modelo</th>      <th>f1_treino</th>      <th>f1_teste</th>      <th>rendimento</th>      <th>rendimento_n_contabilizado</th>      <th>custos_mitigar</th>      <th>gastos</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>regressao_logistica</td>      <td>0.296296</td>      <td>0.210526</td>      <td>2915.0</td>      <td>2530.0</td>      <td>275.0</td>      <td>165.0</td>    </tr>    <tr>      <th>1</th>      <td>svc</td>      <td>0.325203</td>      <td>0.200000</td>      <td>NaN</td>      <td>NaN</td>      <td>NaN</td>      <td>NaN</td>    </tr>    <tr>      <th>2</th>      <td>knn</td>      <td>0.482759</td>      <td>0.000000</td>      <td>5335.0</td>      <td>110.0</td>      <td>0.0</td>      <td>440.0</td>    </tr>    <tr>      <th>3</th>      <td>arvore</td>      <td>0.461538</td>      <td>0.181818</td>      <td>4785.0</td>      <td>660.0</td>      <td>110.0</td>      <td>330.0</td>    </tr>    <tr>      <th>4</th>      <td>floresta</td>      <td>0.358974</td>      <td>0.000000</td>      <td>4400.0</td>      <td>1045.0</td>      <td>220.0</td>      <td>220.0</td>    </tr>    <tr>      <th>5</th>      <td>adaboost</td>      <td>0.307692</td>      <td>0.000000</td>      <td>5390.0</td>      <td>55.0</td>      <td>55.0</td>      <td>385.0</td>    </tr>    <tr>      <th>6</th>      <td>gradientboost</td>      <td>0.516129</td>      <td>0.285714</td>      <td>5225.0</td>      <td>220.0</td>      <td>110.0</td>      <td>330.0</td>    </tr>    <tr>      <th>7</th>      <td>xgboost</td>      <td>0.342857</td>      <td>0.166667</td>      <td>5170.0</td>      <td>275.0</td>      <td>165.0</td>      <td>275.0</td>    </tr>  </tbody></table>



É interessante notar que o nosso melhor modelo foi a Regressão Logística. Observamos que o F1 Score de treino e teste estão razoavelmente próximos, indicando uma boa generalização do modelo para novos dados.

Além do F1 Score, podemos analisar o desempenho do modelo considerando métricas que relacionam a matriz de confusão ao problema de negócio. Abaixo, estão descritas as interpretações de algumas dessas métricas:


* **Rendimento** - Representa o faturamento que o salão terá de acordo com a classificação do modelo.

* **Rendimento não contabilizado** - Está relacionado com os Falsos Positivos que de acordo com o modelo o cliente geraria um custo, mas na realidade esse cliente vai gerar uma receita.

* **Custos mitigar** - Corresponde aos verdadeiros positivos, o modelo conseguiu inferir corretamente a quantidade de dinheiro que o salão deixará de receber e abre a possibilidade para uma possível mitigação desses custos.

* **Gastos** - Representam os falsos negativos, o cliente não vai cumprir com a politica do salão e o modelo falhou em detectar, desse modo o salão deixará de faturar.





1. Rendimento:

    * Representa o faturamento estimado com base nas previsões do modelo. Corresponde à receita projetada considerando as classificações corretas dos clientes.

2. Rendimento não contabilizado:

    * Refere-se aos Falsos Positivos do modelo, ou seja, clientes que, de acordo com a previsão, gerariam custos para o salão, mas na realidade contribuíram para a receita.

3. Custos a mitigar:

    * Relacionado aos Verdadeiros Positivos do modelo, onde o modelo prevê corretamente que um cliente não seguirá as políticas. Isso permite ao salão tomar ações para mitigar os custos associados a clientes que não honrariam os compromissos agendados.

4. Gastos:

    * Representa os Falsos Negativos, ou seja, casos em que o modelo não detecta corretamente que um cliente não seguirá as políticas. Isso implica que o salão pode perder receita devido a faltas ou cancelamentos não previstos.


Observamos que a regressão logística conseguiu capturar muito bem os custos que o salão teria. De 440 CAD de custo o modelo capturou 275 CAD, representando dessa forma uma possível mitigação de 62,5% dos custos.


# Conclusão

Com este trabalho, conseguimos simular uma situação mais próxima de uma condição real, utilizando diversos datasets que interagem entre si. Realizamos uma análise exploratória a fim de compreender como o salão obtém lucro, identificar seus funcionários e avaliar seu desempenho ao longo dos meses. Investigamos preferências por dias da semana, a distribuição dos clientes e conduzimos outras análises relevantes. Durante esse processo, detectamos algumas inconsistências nas bases, o que é comum na prática, e desenvolvemos estratégias para lidar com esses desafios.

Após a análise dos dados, construímos o dataset de treino e validação do zero, integrando informações das diversas tabelas presentes em nossa base de dados.

Ao lidar com o dataset resultante, nos deparamos com desafios, incluindo a escassez de dados para treino e teste, bem como um desbalanceamento significativo entre as amostras.

No desfecho, conseguimos desenvolver um modelo inicial que resultou em uma mitigação de 62,5% nos gastos.

# Considerações finais

 
Um dos desafios cruciais enfrentados no desfecho do projeto foi a escassez de dados e o notável desbalanceamento, o que complicou significativamente o desenvolvimento de um modelo robusto. Infelizmente, a continuidade do projeto tornou-se inviável devido a não atualização dos dados.

Considere em visitar os notebooks desenvolvidos, nos mesmos contêm  muito conteúdo legal que não foi abordado de forma detalhada neste README. Busquei manter os notebooks o mais explicativos possível. Além disso, no notebook de análises, abordo diversas situações com a base de dados que refletem desafios comuns encontrados na vida real, como valores questionáveis, erros e assim por diante.
