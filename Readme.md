# Introdução

Reserva não honrada é um projeto de portfólio em análise/ciência de dados com o objetivo de aplicar os meus conhecimentos em dados mais realistas.  É claro para todos nós que no Kaggle existem muitos datasets, entretanto, vários deles já estão tratados, nos poupando de quebrar a cabeça com dados mais complexos. Há também muitos dados fakes, isto é, pessoas geram dados fake e os colocam no Kaggle. 

Ao iniciar os estudos em dados esses datasets previamente tratados são como uma mão na roda, entretanto, com o avanço de nossos estudos é necessário que procuremos dados mais realistas, que visam representar a realidade que geralmente vamos encontrar nos problemas reais. 

Pensando nisso, tive a ideia de trazer um novo projeto em ciência de dados utilizando uma base de dados real de um salão Canadense.

# Contextualizando


Em um modelo de negócio de salão de beleza é muito comum que os clientes reservem um horário para receberem um serviço. Entretanto, uma parcela desses clientes não honram a reserva, seja apenas faltando ou seja desmarcando o compromisso em cima da hora. Tal atitude gera custos para o salão, visto que aquele horário poderia ter sido ocupado por uma pessoa que honraria a reserva e, portanto, geraria receita ao salão.

Desse modo, é interessante ao salão ter meios para mitigar esses custos, otimizando assim sua receita e seus lucros.

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
>    * Consiste de um diretório que armazena tabelas tratadas além do dataset para treinamento e teste de modelos de aprendizado de maquina.

>* **images**
>   * Um diretório que armazena algumas figuras desenvolvidas no decorrer das análises.

>* **notebooks**
>   * É o diretório principal, contém todos os notebooks utilizados neste projeto.

>* **utils**
>   * Este diretório contém alguns scripts que auxiliam em algumas tarefas no decorrer do projeto.


Como dito, no diretório **notebooks** se encontram todos os notebooks utilizados neste projeto. Ao todos foram desenvolvidos três notebooks que estão listados a seguir:

>* **analises.ipynb**
>   * Contém toda a análise de dados realizada com os dados brutos do salão.

>* **confeccionando_dataset_treino_teste.ipynb**
>   * Neste notebook foi desenvolvido o passo a passo da confecção do dataset de treino e teste a partir dos dados brutos.  

>* **modelagem.ipynb**
>   * Contém o desenvolvimento de alguns modelos de aprendizado de maquina.

# Objetivo / problema de negócio

Alguns clientes faltam ou simplesmente desmarcam o compromisso com o salão em cima da hora, essa ação gera despesas ao salão, uma vez que o horário poderia ter sido preenchido por outro cliente que honraria o compromisso. Clientes que faltam ou cancelam o serviço com um prazo menor do que dois dias podem ser configuradas como clientes que não seguiram as politicas do salão.

Desse modo, o nosso objetivo nesse projeto consiste em desenvolver um modelo preditivo que seja capaz de identificar clientes que possivelmente não vão seguir as políticas do salão.

Espera-se conseguir com isso uma maneira de mitigar os custos envolvidos.

# Estimativas

Analisando os dados históricos do salão é possível determinar que na mediana os clientes geram de receita 55 CAD (dólares canadenses). Logo, o salão vai estar deixando de receber 55 CAD de receita para cada cliente que não honra com os compromissos marcados com o salão.

Pode-se determinar com os dados históricos que o salão deixou de faturar 3245 CAD com clientes que faltaram ao serviço e 7700 CAD com pessoas que cancelaram o serviço com um prazo menor do que dois dias. Ao todo o salão deixou de arrecadar 10945 CAD devido aos clientes não respeitarem as políticas.

# Solução

Com os dados históricos conseguimos construir um novo dataset para utilizarmos no treinamento. Para a construção desse dataset nós utilizamos a ultima data de interação do cliente com o salão. Determino por interação três possíveis cenários, o cliente vai ao salão e recebe o serviço ou realiza o cancelamento de acordo com as políticas do salão, o cliente cancela o compromisso faltando menos de dois dias e o cliente falta o compromisso.

Com essa data de ultima interação foi possível construir o target, desse modo, todos os clientes que não respeitaram as políticas do salão receberam uma flag 1 e os demais receberam a flag 0. 

Para a elaboração das features utilizamos as datas anteriores a ultima interação do cliente. Foram geradas muitas features a partir desse histórico, citar todas aqui nesse espaço seria inadequado, porém, o dataset com essas variáveis pode ser encontrado no diretório **datasets_for_ml**.

O dataset resultante conta com 24 colunas e 798 clientes. Onde 751 (94%) clientes são da classe 0 e 47 (5.9%) são da classe 1. Percebe-se duas coisas, um desbalanceamento considerável dos dados.

Um problema encontrado são que desses 798 clientes 442 deles são novos clientes, isto é, eles não tem nenhum histórico e 356 são clientes com algum histórico. Neste ponto, não faz sentido utilizarmos clientes sem histórico, pois com a base que temos acesso não temos como tirarmos nenhuma informação desses novos clientes, dessa forma, devemos limitar mais ainda nosso dataset considerando somente os clientes com histórico. 

O dataset resultante ao consideramos esses clientes cai para 356 instâncias, onde 92%  são da classe 0 e 8% são da classe 1. Além do desbalanceamento considerável das classes observamos também a existência de poucas instâncias para o treinamento de modelos.


Devemos estabelecer as métricas que vamos utilizar para a avaliação dos modelos. Vamos utilizar as três métricas citadas a seguir:

1. **Precision Score;**
2. **Recall Score;**
3. **F1 Score;**

A questão central do problema consiste em identificar faltas ou cancelamentos que fujam das políticas do salão. Tais ações acarretam em custos para o salão, desse modo, devemos selecionar se um cliente vai ou não seguir as políticas.

As ações para os possíveis clientes que vão descumprir as políticas não são conhecidas, entretanto, algo deve ser feito para mitigar os custos envolvidos. Uma maneira de fazer isso seria deixar determinados clientes avisados que existe a possibilidade de adiantar o atendimento deles, liberando assim novos horários. Desse modo, o funcionário não vai deixar de trabalhar e vai existir a possibilidade de marcar clientes nos horários que ficaram vagos.

Os custos envolvidos nessas ações de mitigação são desconhecidos, entretanto, podemos desconsidera-lo por enquanto. Podemos imaginar uma situação onde o contato com esses clientes é feito via ligação ou whatsapp. Desse modo, podemos considerar esse custo como sendo zero, ao menos em um primeiro momento.


Essas métricas serão convertidas para o problema de negócio através da matriz de confusão.

**Verdadeiro Positivo** - Existe a possibilidade de mitigar o custo.

**Verdadeiro Negativo** - Nenhum custo envolvido, teremos lucro devido ao comparecimento do cliente.

**Falso Positivo** - O modelo estima que o cliente não vai cumprir a politica mas na verdade ele vai. Neste caso não teremos um custo real.

**Falso Negativo** - O modelo estima que o cliente vai cumprir a politica mas na verdade ele não vai. Neste caso vamos deixar de faturar.


Após o treinamento e validação obtemos os seguintes resultados:

A tabela a seguir indica a contabilidade real (referente ao teste)

<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>Rendimento</th>      <th>Custo</th>    </tr>  </thead>  <tbody>    <tr>      <th></th>      <td>5445</td>      <td>440</td>    </tr>  </tbody></table>


A tabela a seguir indica os modelos com seus respectivos f1 score de treino e teste e também métricas relacionadas ao negócio.


<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>modelo</th>      <th>f1_treino</th>      <th>f1_teste</th>      <th>rendimento</th>      <th>rendimento_n_contabilizado</th>      <th>custos_mitigar</th>      <th>gastos</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>regressao_logistica</td>      <td>0.296296</td>      <td>0.210526</td>      <td>2915.0</td>      <td>2530.0</td>      <td>275.0</td>      <td>165.0</td>    </tr>    <tr>      <th>1</th>      <td>svc</td>      <td>0.325203</td>      <td>0.200000</td>      <td>NaN</td>      <td>NaN</td>      <td>NaN</td>      <td>NaN</td>    </tr>    <tr>      <th>2</th>      <td>knn</td>      <td>0.482759</td>      <td>0.000000</td>      <td>5335.0</td>      <td>110.0</td>      <td>0.0</td>      <td>440.0</td>    </tr>    <tr>      <th>3</th>      <td>arvore</td>      <td>0.461538</td>      <td>0.181818</td>      <td>4785.0</td>      <td>660.0</td>      <td>110.0</td>      <td>330.0</td>    </tr>    <tr>      <th>4</th>      <td>floresta</td>      <td>0.358974</td>      <td>0.000000</td>      <td>4400.0</td>      <td>1045.0</td>      <td>220.0</td>      <td>220.0</td>    </tr>    <tr>      <th>5</th>      <td>adaboost</td>      <td>0.307692</td>      <td>0.000000</td>      <td>5390.0</td>      <td>55.0</td>      <td>55.0</td>      <td>385.0</td>    </tr>    <tr>      <th>6</th>      <td>gradientboost</td>      <td>0.516129</td>      <td>0.285714</td>      <td>5225.0</td>      <td>220.0</td>      <td>110.0</td>      <td>330.0</td>    </tr>    <tr>      <th>7</th>      <td>xgboost</td>      <td>0.342857</td>      <td>0.166667</td>      <td>5170.0</td>      <td>275.0</td>      <td>165.0</td>      <td>275.0</td>    </tr>  </tbody></table>


Observamos que o nosso melhor modelo foi a Regressão logística. Conseguimos observar que o f1 de treino e teste estão razoavelmente próximos e que esse modelo é aquele que melhor identifica os clientes que não vão seguir as políticas do salão.  Além do f1 score, podemos analisar o modelo a partir das métricas que associam a matriz de confusão ao problema de negócio. Segue listado o que cada métrica dessa representa:


* **Rendimento** - Representa o rendimento que o salão terá de acordo com a classificação do modelo.

* **Rendimento não contabilizado** - Está relacionado com os Falsos Positivos, de acordo com o modelo o cliente geraria um custo, mas na realidade esse cliente vai gerar uma receita.

* **Custos mitigar** - Corresponde aos verdadeiros positivos, o modelo conseguiu inferir corretamente a quantidade de dinheiro que o salão deixará de receber e abre a possibilidade para uma possível mitigação desses custos.

* **Gastos** - Representam os falsos negativos, o cliente não vai cumprir com a politica do salão e o modelo falhou em detectar, desse modo o salão deixará de faturar.


Novamente observamos que a regressão logística conseguiu capturar muito bem os custos que o salão teria. De 440 CAD de custo o modelo capturou 275 CAD, representando dessa forma uma possível mitigação de 62,5% dos custos.


# Conclusão

Com este trabalho, conseguimos simular uma situação mais próxima de uma condição real, utilizando diversos datasets que interagem entre si. Realizamos uma análise exploratória a fim de compreender como o salão obtém lucro, identificar seus funcionários e avaliar seu desempenho ao longo dos meses. Investigamos preferências por dias da semana, a distribuição dos clientes e conduzimos outras análises relevantes. Durante esse processo, detectamos algumas inconsistências nas bases, o que é comum na prática, e desenvolvemos estratégias para lidar com esses desafios.

Após a análise dos dados, construímos o dataset de treino e validação do zero, integrando informações das diversas tabelas presentes em nossa base de dados.

Ao lidar com o dataset resultante, nos deparamos com desafios, incluindo a escassez de dados para treino e teste, bem como um desbalanceamento significativo entre as amostras.

No desfecho, conseguimos desenvolver um modelo inicial que resultou em uma mitigação de 62,5% nos gastos.

# Considerações finais

 
Um dos desafios cruciais enfrentados no desfecho do projeto foi a escassez de dados e o notável desbalanceamento, o que complicou significativamente o desenvolvimento de um modelo robusto. Infelizmente, a continuidade do projeto tornou-se inviável devido a não atualização dos dados.

Considere em visitar os notebooks desenvolvidos, nos mesmos contêm  muito conteúdo legal que não foi abordado de forma detalhada neste README. Busquei manter os notebooks o mais explicativos possível. Além disso, no notebook de análises, abordo diversas situações com a base de dados que refletem desafios comuns encontrados na vida real, como valores questionáveis, erros e assim por diante.
