import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns

from sklearn.metrics 	     import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose 	     import ColumnTransformer
from sklearn.pipeline 	     import Pipeline
from sklearn.model_selection import KFold, train_test_split

class ModelCrafter:

    def __init__(self,folds=5) -> None:
        
        self.models = dict()

        self.results = dict()

        self.results_per_fold = dict()

        self.kf = KFold(n_splits=folds, random_state = 42 ,shuffle = True)
        
        self.folds = folds

    def AddModel(self, modelos : list = []) -> None:
        """Método para adicionar modelos ao objeto. A estrutura é uma lista de tuplas onde a tupla segue o seguinte esquema: (nome do modelo, modelo instanciado)"""
        
        for modelo in modelos:
            self.models[modelo[0]] = modelo[1]

    def RemoveModel(self, nome: str = None, tipo: str = None) -> None:
        """Remove modelos do objeto"""
        
        if tipo == 'all':
            self.models=dict()
            return

        del self.models[nome]

    def _Viabilidade(self,model,x,y, limiar):
    
        prob = model.predict_proba(x)[:,1]

        pred = np.where(prob >= limiar,1,0)

        matrix = confusion_matrix(y, pred)

        tn, fp, fn, tp = matrix.ravel()

        custo_mitigar = tp*55

        rendimento = (tn*55)

        rendimento_n_contabilizado = fp*55

        gastos = fn*55

        print(tn,fp,fn,tp)

        return rendimento, rendimento_n_contabilizado, custo_mitigar, gastos



    def Validacao(self, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None , y_train: pd.Series = None, y_test: pd.Series = None, pipe: Pipeline = None, limiar: float = 0.5):
       
        if len(self.models) == 0:
            return "Nenhum modelo adicionado na estrutura"

        resultados = {'modelo':[],'f1_treino':[], 'f1_teste':[],'rendimento':[],'rendimento_n_contabilizado':[],'custos_mitigar':[],'gastos':[]}

        for aux in self.models.items():
            nome_modelo = aux[0]
            modelo = aux[1]

            if len(pipe.steps) > 1:
                pipe.steps.pop()

            
            print(f"-----{nome_modelo}-----")
            pipe.steps.append(("Model",modelo))
            modelo = pipe

            modelo.fit(X_train,y_train)

            pred_train = modelo.predict(X_train)
            pred_test = modelo.predict(X_test)


            f1_train  =  f1_score(y_train,pred_train)
            f1_test = f1_score(y_test,pred_test)
            
            try:
                rendimento, rendimento_n_contabilizado, custos_mitigar, gastos = self._Viabilidade(modelo,X_test,y_test,limiar)
            except:
                rendimento = custos_mitigar = gastos = rendimento_n_contabilizado = np.nan

            resultados['modelo'].append(nome_modelo)
            resultados['f1_treino'].append(f1_train)
            resultados['f1_teste'].append(f1_test)
            resultados['rendimento'].append(rendimento)
            resultados['custos_mitigar'].append(custos_mitigar)
            resultados['gastos'].append(gastos)
            resultados['rendimento_n_contabilizado'].append(rendimento_n_contabilizado)

        return pd.DataFrame(resultados)
        

    def ValidacaoCruzada(self, X: np.ndarray, y: np.array, pipe: Pipeline = None) -> None:
        """Treina todos os modelos inseridos no objeto através de validação cruzada"""
		
        if len(self.models) == 0:
            return "Nenhum modelo adicionado na estrutura"
        
        for aux in self.models.items():
            nome_modelo = aux[0]
            modelo = aux[1]

            if len(pipe.steps) > 1:
                pipe.steps.pop()

            precision = 0
            recall = 0
            f1 = 0

            resultados_aux = []

            print(f"-----{nome_modelo}-----")
            pipe.steps.append(("Model",modelo))
            modelo = pipe
            for i, (train_index, test_index) in enumerate(self.kf.split(X)):
                #print(f"Fold {i}:")
                
                #print(f"  Train: index={train_index}")
                #print(f"  Test:  index={test_index}")

                X_train = X.loc[train_index,:]
                y_train = y.loc[train_index]

                
                X_test = X.loc[test_index,:]
                y_test = y.loc[test_index]

                
                modelo.fit(X_train,y_train)

                predito = modelo.predict(X_test)
  
                
                resultados_aux.append((y_test,predito)) 
                
                precision  += precision_score(y_test,predito)
                recall += recall_score(y_test,predito)
                f1 += f1_score(y_test,predito)
                
                
            self.results[nome_modelo]=[precision/self.folds, recall/self.folds, f1/self.folds]
            self.results_per_fold[nome_modelo] = resultados_aux

        return self._gerar_resultado()

    def _gerar_resultado(self) -> None:
        """Gera os resultados em uma estrutura DataFrame"""
        
        indices = ['precision','recall','f1']
        #display(pd.DataFrame(self.results,index=indices).T)
        return pd.DataFrame(self.results,index=indices).T
