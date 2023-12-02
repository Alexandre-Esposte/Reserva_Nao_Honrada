# Introdução

Reserva não honrada é um projeto de portfólio em análise/ciência de dados com o objetivo de aplicar os meus conhecimentos em datasets mais realistas.  É claro para todos nós que no Kaggle existem muitos datasets, entretanto, vários deles já estão tratados, nos poupando de quebrar a cabeça com dados mais complexos. Em outras situações cruzamos com dados fakes, isto é, pessoas geram dados fake e os colocam no Kaggle. 

Ao iniciar os estudos em dados esses datasets previamente tratados soa como uma mão na roda, entretanto, com o avanço de nossos estudos é necessário que procuremos dados mais realistas, que visam representar a realidade que geralmente vamos encontrar nos problemas reais. 

Pensando nisso, tive a ideia de trazer um novo projeto em ciência de dados utilizando uma base de dados real de um salão Canadense.

# Contextualizando


Em um modelo de negócio de salão de beleza é muito comum que os clientes reservem um horário para receberem um serviço. Entretanto, uma parcela desses clientes não honram a reserva, seja apenas faltando ou seja desmarcando o compromisso em cima da hora. Tal atitude gera custos para o salão, visto que aquele horário poderia ter sido ocupado por uma pessoa que honraria a reserva e, portanto, geraria receita ao salão.

Desse modo, é interessante ao salão ter meios para mitigar esses custos, otimizando assim sua receita e seus lucros.

# Dados

Os dados desse salão foram obtidos no Kaggle, o projeto lá  se chama  <a href =https://www.kaggle.com/datasets/frederickferguson/hair-salon-no-show-data-set?select=Service+Listing0.csv >Hair Salon No-Show</a>. 

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
>    * Esta é uma lista de reservas de não comparecimento que não foram canceladas antes da data da reserva. Este conjunto não inclui cancelamentos fora da política do salão.

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