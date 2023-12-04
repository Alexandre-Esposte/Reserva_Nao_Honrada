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